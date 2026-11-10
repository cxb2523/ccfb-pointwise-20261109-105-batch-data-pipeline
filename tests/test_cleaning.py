"""
Tests for the configurable cleaning operator (src/cleaning.py).

Covers the failure modes behind the doubled Metabase sales figures:
- duplicate rows replayed by Airflow retries (deterministic dedup)
- negative-amount dirty rows (range quarantine with correct reason)
- unparseable dates (cast quarantine, cast attributed before range)
- replay determinism and first-hit reason attribution
"""
import logging

import pyarrow as pa
import pytest

from src.cleaning import (
    DataCleaner,
    CleaningConfigError,
    MissingDedupKeyError,
    REASON_COLUMN,
    _row_hash,
    clean_dataset,
)


COLUMNS = [
    "transaction_id",
    "customer_id",
    "product_id",
    "quantity",
    "total_amount",
    "transaction_timestamp",
    "payment_method",
]


def base_config(**overrides):
    config = {
        "dedup": {
            "keys": ["transaction_id"],
            "on_missing_keys": "fail",
        },
        "null_policy": {
            "transaction_id": "reject",
            "customer_id": "reject",
            "total_amount": "reject",
            "transaction_timestamp": "reject",
        },
        "casts": [
            {"column": "customer_id", "type": "int64"},
            {"column": "quantity", "type": "int64"},
            {"column": "total_amount", "type": "float64"},
            {"column": "transaction_timestamp", "type": "timestamp"},
        ],
        "ranges": [
            {"column": "total_amount", "min": 0, "min_strict": True, "action": "reject"},
            {"column": "quantity", "min": 1, "action": "reject"},
        ],
    }
    config.update(overrides)
    return config


def make_table(rows):
    return pa.Table.from_pylist(rows)


DIRTY_ROWS = [
    # clean row
    {
        "transaction_id": "T1",
        "customer_id": "100",
        "product_id": 1,
        "quantity": "2",
        "total_amount": 20.0,
        "transaction_timestamp": "2024-01-01T10:00:00",
        "payment_method": "credit_card",
    },
    # exact duplicate of T1 (Airflow replay)
    {
        "transaction_id": "T1",
        "customer_id": "100",
        "product_id": 1,
        "quantity": "2",
        "total_amount": 20.0,
        "transaction_timestamp": "2024-01-01T10:00:00",
        "payment_method": "credit_card",
    },
    # negative amount -> range rejection
    {
        "transaction_id": "T2",
        "customer_id": "101",
        "product_id": 2,
        "quantity": "1",
        "total_amount": -15.5,
        "transaction_timestamp": "2024-01-02T10:00:00",
        "payment_method": "cash",
    },
    # bad date -> cast rejection
    {
        "transaction_id": "T3",
        "customer_id": "102",
        "product_id": 3,
        "quantity": "1",
        "total_amount": 9.99,
        "transaction_timestamp": "not-a-date",
        "payment_method": "cash",
    },
    # null required field -> null rejection
    {
        "transaction_id": None,
        "customer_id": "103",
        "product_id": 4,
        "quantity": "1",
        "total_amount": 5.0,
        "transaction_timestamp": "2024-01-03T10:00:00",
        "payment_method": "cash",
    },
    # second clean row whose quantity is also range-invalid if cast passed;
    # cast must win because casts execute before ranges
    {
        "transaction_id": "T4",
        "customer_id": "104",
        "product_id": 5,
        "quantity": "abc",
        "total_amount": 8.0,
        "transaction_timestamp": "2024-01-04T10:00:00",
        "payment_method": "cash",
    },
]


# ---------------------------------------------------------------------------
# End-to-end cleaning + reason attribution
# ---------------------------------------------------------------------------
def test_dirty_rows_are_quarantined_with_first_hit_reasons():
    result = DataCleaner(base_config(), dataset_name="sales").clean(
        make_table(DIRTY_ROWS)
    )

    cleaned_ids = result.cleaned.column("transaction_id").to_pylist()
    # Only one T1 survives dedup; T2 (negative amount) must not survive.
    assert "T2" not in cleaned_ids
    assert cleaned_ids == ["T1"]

    reasons = dict(
        zip(
            result.quarantined.column("transaction_id").to_pylist(),
            result.quarantined.column(REASON_COLUMN).to_pylist(),
        )
    )
    # The None transaction_id row is keyed by None; look up by reason set.
    reason_values = set(result.quarantined.column(REASON_COLUMN).to_pylist())
    assert "range:total_amount" in reason_values       # T2 negative amount
    assert "cast:transaction_timestamp" in reason_values  # T3 bad date
    assert "cast:quantity" in reason_values            # T4 cast beats range
    assert "dedup" in reason_values                    # T1 duplicate
    assert any(r.startswith("null:") for r in reason_values)  # null txn id

    stats = result.stats
    assert stats["input_rows"] == 6
    assert stats["cleaned_rows"] == 1
    assert stats["quarantined_rows"] == 5
    assert stats["rule_hits"]["dedup"] == 1
    assert stats["rule_hits"]["range:total_amount"] == 1
    assert stats["rule_hits"]["cast:transaction_timestamp"] == 1
    assert stats["rule_hits"]["cast:quantity"] == 1


def test_quarantine_holds_raw_pre_cast_values():
    result = DataCleaner(base_config(), dataset_name="sales").clean(
        make_table(DIRTY_ROWS)
    )
    # Quarantine is built from the raw table, so the bad date string is
    # preserved verbatim rather than being replaced by a casted null.
    q = result.quarantined.to_pylist()
    bad_date_rows = [r for r in q if r["transaction_id"] == "T3"]
    assert len(bad_date_rows) == 1
    assert bad_date_rows[0]["transaction_timestamp"] == "not-a-date"
    assert bad_date_rows[0][REASON_COLUMN] == "cast:transaction_timestamp"


# ---------------------------------------------------------------------------
# Replay determinism (the idempotency guarantee)
# ---------------------------------------------------------------------------
def test_replay_produces_identical_output_regardless_of_row_order():
    cleaner = DataCleaner(base_config(), dataset_name="sales")
    table = make_table(DIRTY_ROWS)

    first = cleaner.clean(table)
    # Re-feed the same batch in a different order, as a retried task would.
    shuffled = make_table(list(reversed(DIRTY_ROWS)))
    replayed = cleaner.clean(shuffled)

    assert replayed.cleaned.to_pylist() == first.cleaned.to_pylist()
    assert replayed.quarantined.to_pylist() == first.quarantined.to_pylist()
    assert replayed.stats == first.stats


def test_dedup_tiebreaker_is_stable_across_replays():
    # Same business key, conflicting payloads: the winner must be selected by
    # the explicit row-hash tiebreaker, never input position.
    rows = [
        {
            "transaction_id": "K1",
            "customer_id": "1",
            "product_id": 1,
            "quantity": "1",
            "total_amount": 10.0,
            "transaction_timestamp": "2024-01-01T10:00:00",
            "payment_method": "cash",
        },
        {
            "transaction_id": "K1",
            "customer_id": "1",
            "product_id": 1,
            "quantity": "1",
            "total_amount": 99.0,
            "transaction_timestamp": "2024-01-01T10:00:00",
            "payment_method": "cash",
        },
    ]
    cleaner = DataCleaner(base_config(), dataset_name="sales")

    forward = cleaner.clean(make_table(rows)).cleaned.to_pylist()
    backward = cleaner.clean(make_table(list(reversed(rows)))).cleaned.to_pylist()

    assert forward == backward
    assert len(forward) == 1
    expected_winner_amount = min(
        (r["total_amount"] for r in rows),
        key=lambda amt: _row_hash(
            next(r for r in rows if r["total_amount"] == amt)
        ),
    )
    assert forward[0]["total_amount"] == expected_winner_amount


# ---------------------------------------------------------------------------
# Missing dedup key policy: fail vs whole-row hash degradation
# ---------------------------------------------------------------------------
def test_missing_dedup_key_fails_when_policy_is_fail():
    config = base_config()
    config["dedup"]["keys"] = ["transaction_id", "nonexistent_column"]
    cleaner = DataCleaner(config, dataset_name="sales")
    with pytest.raises(MissingDedupKeyError):
        cleaner.clean(make_table(DIRTY_ROWS))


def test_missing_dedup_key_degrades_to_hash_dedup(caplog):
    config = base_config()
    config["dedup"]["keys"] = ["transaction_id", "nonexistent_column"]
    config["dedup"]["on_missing_keys"] = "hash"
    cleaner = DataCleaner(config, dataset_name="sales")

    with caplog.at_level(logging.WARNING):
        result = cleaner.clean(make_table(DIRTY_ROWS))

    assert any("whole-row hash" in rec.message for rec in caplog.records)
    # The exact duplicate T1 pair is still collapsed even without the key.
    assert result.stats["rule_hits"]["dedup"] == 1


def test_hash_dedup_is_computed_before_casts():
    # Two rows that become identical AFTER casting (quantity "02" vs "2") but
    # differ in their raw representation must be treated as distinct under
    # hash dedup, because the hash is taken on raw rows before casts.
    rows = [
        {
            "transaction_id": "A",
            "customer_id": "1",
            "product_id": 1,
            "quantity": "02",
            "total_amount": 10.0,
            "transaction_timestamp": "2024-01-01T10:00:00",
            "payment_method": "cash",
        },
        {
            "transaction_id": "A",
            "customer_id": "1",
            "product_id": 1,
            "quantity": "2",
            "total_amount": 10.0,
            "transaction_timestamp": "2024-01-01T10:00:00",
            "payment_method": "cash",
        },
    ]
    config = base_config()
    config["dedup"] = {"keys": ["ghost_key"], "on_missing_keys": "hash"}
    result = DataCleaner(config, dataset_name="sales").clean(make_table(rows))
    # Both survive: distinct raw hashes, even though casts make them equal.
    assert result.cleaned.num_rows == 2

    # Whereas truly identical raw rows collapse.
    identical = make_table([rows[0], dict(rows[0])])
    result_identical = DataCleaner(config, dataset_name="sales").clean(identical)
    assert result_identical.cleaned.num_rows == 1
    assert result_identical.stats["rule_hits"]["dedup"] == 1


def test_invalid_on_missing_keys_policy_is_rejected_at_construction():
    config = base_config()
    config["dedup"]["on_missing_keys"] = "warn"
    with pytest.raises(CleaningConfigError):
        DataCleaner(config, dataset_name="sales")


# ---------------------------------------------------------------------------
# Missing columns on ordinary rules: warn and skip, keep cleaning
# ---------------------------------------------------------------------------
def test_ordinary_rule_on_missing_column_is_skipped_with_warning(caplog):
    config = base_config()
    config["ranges"] = [
        {"column": "ghost_column", "min": 0, "action": "reject"},
        {"column": "total_amount", "min": 0, "min_strict": True, "action": "reject"},
    ]
    config["casts"].append({"column": "other_ghost", "type": "int64"})
    config["null_policy"]["null_ghost"] = "reject"
    cleaner = DataCleaner(config, dataset_name="sales")

    with caplog.at_level(logging.WARNING):
        result = cleaner.clean(make_table(DIRTY_ROWS))

    skip_messages = [rec.message for rec in caplog.records if "skipping rule" in rec.message]
    assert len(skip_messages) == 3  # range, cast, null policy each warn once
    # The real range rule still fires; cleaning did not get disabled.
    assert result.stats["rule_hits"]["range:total_amount"] == 1


# ---------------------------------------------------------------------------
# Null fill default + range clipping
# ---------------------------------------------------------------------------
def test_null_default_fill_keeps_row_and_range_clip_adjusts_value():
    rows = [
        {
            "transaction_id": "T1",
            "customer_id": "100",
            "product_id": None,
            "quantity": "5",
            "total_amount": 250.0,
            "transaction_timestamp": "2024-01-01T10:00:00",
            "payment_method": "credit_card",
        },
    ]
    config = {
        "dedup": {"keys": ["transaction_id"], "on_missing_keys": "fail"},
        "null_policy": {
            "product_id": {"strategy": "default", "default": 7},
        },
        "casts": [
            {"column": "customer_id", "type": "int64"},
            {"column": "quantity", "type": "int64"},
            {"column": "total_amount", "type": "float64"},
            {"column": "transaction_timestamp", "type": "timestamp"},
        ],
        "ranges": [
            {"column": "total_amount", "min": 0, "max": 100, "action": "clip"},
        ],
    }
    result = DataCleaner(config, dataset_name="sales").clean(make_table(rows))
    assert result.quarantined.num_rows == 0
    out = result.cleaned.to_pylist()[0]
    assert out["product_id"] == 7
    assert out["total_amount"] == 100.0
    assert result.stats["clip_hits"]["range:total_amount"] == 1


def test_null_default_requires_explicit_default_value():
    config = base_config()
    config["null_policy"] = {"product_id": {"strategy": "default"}}
    with pytest.raises(CleaningConfigError):
        DataCleaner(config, dataset_name="sales")


# ---------------------------------------------------------------------------
# Empty input and config-driven passthrough
# ---------------------------------------------------------------------------
def test_empty_table_returns_empty_outputs_without_error():
    result = DataCleaner(base_config(), dataset_name="sales").clean(
        make_table([])
    )
    assert result.cleaned.num_rows == 0
    assert result.quarantined.num_rows == 0
    assert REASON_COLUMN in result.quarantined.column_names


def test_clean_dataset_passthrough_when_no_cleaning_section(monkeypatch):
    import src.config as cfg

    monkeypatch.setattr(cfg, "get_config", lambda *keys, default=None: None)
    table = make_table(DIRTY_ROWS)
    result = clean_dataset(table, "sales")

    assert result.stats.get("passthrough") is True
    assert result.cleaned.to_pylist() == table.to_pylist()
    assert result.quarantined.num_rows == 0


def test_clean_dataset_uses_config_when_present(monkeypatch):
    import src.config as cfg

    fake = {"cleaning": {"datasets": {"sales": base_config()}}}

    def fake_get_config(*keys, default=None):
        value = fake
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value if value is not None else default

    monkeypatch.setattr(cfg, "get_config", fake_get_config)
    result = clean_dataset(make_table(DIRTY_ROWS), "sales")
    assert result.cleaned.num_rows == 1
    assert result.quarantined.num_rows == 5


def test_each_rule_hit_is_logged(caplog):
    cleaner = DataCleaner(base_config(), dataset_name="sales")
    with caplog.at_level(logging.INFO, logger="src.cleaning"):
        cleaner.clean(make_table(DIRTY_ROWS))
    hit_logs = [rec.message for rec in caplog.records if "rule hit" in rec.message]
    # null, cast (x2), range, dedup all fired at least one hit line.
    assert len(hit_logs) >= 4
