"""
Integration test for the Silver read-path + cleaning operator.

Mirrors the projection used by SilverProcessor in cleaning mode (raw columns,
no SQL cast/filter) and proves dirty rows survive the read so the operator can
charge them to the right rule, then replays the batch to confirm idempotency.
Exercises duckdb (in requirements) but not S3/Delta, which are not available
in the unit-test environment.
"""
import json

import duckdb
import pyarrow as pa

from src.cleaning import DataCleaner, REASON_COLUMN


# Same projection as SilverProcessor.process when a cleaner is supplied.
RAW_PROJECTION = """
    SELECT
        CAST(transaction_id AS VARCHAR) AS transaction_id,
        customer_id,
        product_id,
        quantity,
        unit_price,
        total_amount,
        transaction_timestamp,
        payment_method,
        CAST(store_id AS BIGINT) AS store_id,
        status
    FROM read_json_auto('{path}')
"""

CONFIG = {
    "dedup": {"keys": ["transaction_id"], "on_missing_keys": "fail"},
    "null_policy": {
        "transaction_id": "reject",
        "customer_id": "reject",
        "total_amount": "reject",
        "transaction_timestamp": "reject",
    },
    "casts": [
        {"column": "customer_id", "type": "int64"},
        {"column": "product_id", "type": "int64"},
        {"column": "quantity", "type": "int64"},
        {"column": "total_amount", "type": "float64"},
        {"column": "transaction_timestamp", "type": "timestamp"},
    ],
    "ranges": [
        {"column": "total_amount", "min": 0, "min_strict": True, "action": "reject"},
        {"column": "quantity", "min": 1, "action": "reject"},
    ],
}

DIRTY_BATCH = [
    {
        "transaction_id": "T1", "customer_id": 100, "product_id": 1,
        "quantity": 2, "unit_price": 10.0, "total_amount": 20.0,
        "transaction_timestamp": "2024-01-01T10:00:00",
        "payment_method": "credit_card", "store_id": 1, "status": "completed",
    },
    # exact replay duplicate
    {
        "transaction_id": "T1", "customer_id": 100, "product_id": 1,
        "quantity": 2, "unit_price": 10.0, "total_amount": 20.0,
        "transaction_timestamp": "2024-01-01T10:00:00",
        "payment_method": "credit_card", "store_id": 1, "status": "completed",
    },
    # negative amount
    {
        "transaction_id": "T2", "customer_id": 101, "product_id": 2,
        "quantity": 1, "unit_price": 15.5, "total_amount": -15.5,
        "transaction_timestamp": "2024-01-02T10:00:00",
        "payment_method": "cash", "store_id": 1, "status": "completed",
    },
    # unparseable date
    {
        "transaction_id": "T3", "customer_id": 102, "product_id": 3,
        "quantity": 1, "unit_price": 9.99, "total_amount": 9.99,
        "transaction_timestamp": "not-a-date",
        "payment_method": "cash", "store_id": 1, "status": "completed",
    },
]


def _write_and_read(tmp_path):
    json_path = tmp_path / "sales_batch.json"
    json_path.write_text(json.dumps(DIRTY_BATCH), encoding="utf-8")
    conn = duckdb.connect()
    try:
        # con.sql(...).to_arrow_table() is the no-copy path on duckdb 2.x;
        # fall back to .arrow() for the pinned duckdb 1.0 in production.
        if hasattr(conn, "sql"):
            return conn.sql(
                RAW_PROJECTION.format(path=json_path.as_posix())
            ).to_arrow_table()
        return conn.execute(
            RAW_PROJECTION.format(path=json_path.as_posix())
        ).arrow()
    finally:
        conn.close()


def test_silver_read_path_feeds_dirty_rows_to_operator(tmp_path):
    raw = _write_and_read(tmp_path)
    # The bad date must still be present as a string after the raw projection;
    # a SQL CAST here would have failed or nulled it before cleaning.
    timestamps = raw.column("transaction_timestamp").to_pylist()
    assert "not-a-date" in timestamps

    result = DataCleaner(CONFIG, dataset_name="sales").clean(raw)

    cleaned_ids = result.cleaned.column("transaction_id").to_pylist()
    assert cleaned_ids == ["T1"]  # duplicate collapsed, T2/T3 quarantined

    quarantine = result.quarantined.to_pylist()
    reasons = {row["transaction_id"]: row[REASON_COLUMN] for row in quarantine}
    assert reasons["T1"] == "dedup"
    assert reasons["T2"] == "range:total_amount"
    assert reasons["T3"] == "cast:transaction_timestamp"


def test_silver_read_path_replay_is_deterministic(tmp_path):
    cleaner = DataCleaner(CONFIG, dataset_name="sales")

    first = cleaner.clean(_write_and_read(tmp_path))
    # Re-read the same batch file and re-clean, exactly like an Airflow retry.
    replayed = cleaner.clean(_write_and_read(tmp_path))

    assert replayed.cleaned.to_pylist() == first.cleaned.to_pylist()
    assert replayed.quarantined.to_pylist() == first.quarantined.to_pylist()
    assert replayed.stats == first.stats
    assert isinstance(first.cleaned, pa.Table)
