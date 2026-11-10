"""
Configurable row-level cleaning operator.

The operator is intentionally framework-neutral (pyarrow only) so it can be
unit-tested without S3/Delta/Airflow and reused by every layer that needs to
quarantine dirty rows before writing to the warehouse.

Fixed rule execution order (deliberately independent of YAML key order,
because the order decides which rule a dirty row is charged to):

    1. row hash  - computed on the *raw* row, before any cast, so changing a
                   cast rule never changes the dedup identity of a historical
                   batch. Used as the explicit dedup tiebreaker.
    2. nulls     - reject to quarantine or fill with a configured default
    3. casts     - safe type coercion, failures go to quarantine
    4. ranges    - numeric/timestamp bounds, reject or clip in place
    5. dedup     - stable sort by (dedup keys, row hash), keep first row

A row is charged to the *first* rule that rejects it only. Cleaned and
quarantined outputs are both sorted by row hash, so replaying the same input
produces byte-for-byte identical rows regardless of input row order.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed rule order. Do not reorder: reason attribution depends on it.
# ---------------------------------------------------------------------------
RULE_NULL = "null"
RULE_CAST = "cast"
RULE_RANGE = "range"
RULE_DEDUP = "dedup"

REASON_COLUMN = "cleaning_reason"
HASH_COLUMN = "_row_hash"
ORIG_IDX_COLUMN = "_cleaning_orig_idx"

_CAST_TYPES = {
    "int32": pa.int32(),
    "int64": pa.int64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "string": pa.string(),
    "bool": pa.bool_(),
    "timestamp": pa.timestamp("us"),
}

VALID_ON_MISSING_KEYS = ("fail", "hash")


class CleaningConfigError(ValueError):
    """Raised when a cleaning configuration block is internally invalid."""


class MissingDedupKeyError(KeyError):
    """Raised when a configured dedup key column is absent and policy is fail."""


@dataclass
class CleaningResult:
    """Output of a cleaning run."""

    cleaned: pa.Table
    quarantined: pa.Table
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def rule_hits(self) -> Dict[str, int]:
        return self.stats.get("rule_hits", {})


def _row_hash(row: Dict[str, Any]) -> str:
    """Stable hash of a raw row.

    Canonical JSON (sorted keys, string fallback) so the value only depends on
    the row content, never on dict insertion order or the platform.
    """
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_null_or_nan(array: pa.ChunkedArray) -> pa.ChunkedArray:
    mask = pc.is_null(array)
    if pa.types.is_floating(array.type):
        mask = pc.or_(mask, pc.is_nan(array))
    return mask


class DataCleaner:
    """Apply one dataset's cleaning rules to a pyarrow Table.

    Args:
        config: per-dataset mapping under ``cleaning.datasets.<name>``.
        dataset_name: name used in log lines / future statsd tags.
    """

    def __init__(self, config: Dict[str, Any], dataset_name: str = "default"):
        self.dataset_name = dataset_name
        self.config = config or {}
        self._validate_config()

    # ------------------------------------------------------------------ setup
    def _validate_config(self) -> None:
        dedup = self.config.get("dedup")
        if dedup is not None:
            keys = dedup.get("keys")
            if not keys or not isinstance(keys, list):
                raise CleaningConfigError(
                    f"cleaning.{self.dataset_name}.dedup.keys must be a non-empty list"
                )
            policy = dedup.get("on_missing_keys")
            if policy not in VALID_ON_MISSING_KEYS:
                raise CleaningConfigError(
                    f"cleaning.{self.dataset_name}.dedup.on_missing_keys must be "
                    f"explicitly one of {VALID_ON_MISSING_KEYS}, got {policy!r}"
                )

        for spec in self.config.get("ranges", []) or []:
            if spec.get("action", "reject") not in ("reject", "clip"):
                raise CleaningConfigError(
                    f"range rule on {spec.get('column')!r}: action must be "
                    f"'reject' or 'clip'"
                )

        for column, policy in (self.config.get("null_policy") or {}).items():
            strategy = policy if isinstance(policy, str) else policy.get("strategy")
            if strategy not in ("reject", "default"):
                raise CleaningConfigError(
                    f"null_policy.{column}: strategy must be 'reject' or 'default'"
                )
            if strategy == "default" and (
                not isinstance(policy, dict) or "default" not in policy
            ):
                raise CleaningConfigError(
                    f"null_policy.{column}: strategy 'default' requires a default value"
                )

    # ------------------------------------------------------------------- main
    def clean(self, table: pa.Table) -> CleaningResult:
        rule_hits: Dict[str, int] = {}
        input_rows = table.num_rows

        if input_rows == 0:
            return CleaningResult(
                cleaned=table,
                quarantined=self._empty_quarantine(table),
                stats=self._stats(input_rows, 0, 0, rule_hits, {}),
            )

        raw_table = table
        reasons: Dict[int, str] = {}

        # Step 1: hash on the RAW rows, before any transformation.
        rows = table.to_pylist()
        hashes = pa.array([_row_hash(r) for r in rows], type=pa.string())
        work = table.append_column(
            HASH_COLUMN, hashes
        ).append_column(
            ORIG_IDX_COLUMN, pa.array(range(input_rows), type=pa.int64())
        )

        # Step 2: null policy
        work = self._apply_null_policy(work, reasons, rule_hits)
        # Step 3: casts
        work = self._apply_casts(work, reasons, rule_hits)
        # Step 4: range clipping / rejection
        clip_hits: Dict[str, int] = {}
        work = self._apply_ranges(work, reasons, rule_hits, clip_hits)
        # Step 5: deterministic dedup
        work = self._apply_dedup(work, reasons, rule_hits)

        cleaned = (
            work.sort_by([(HASH_COLUMN, "ascending")])
            .drop_columns([HASH_COLUMN, ORIG_IDX_COLUMN])
        )

        quarantined = self._build_quarantine(raw_table, reasons)
        stats = self._stats(
            input_rows,
            cleaned.num_rows,
            quarantined.num_rows,
            rule_hits,
            clip_hits,
        )

        for rule, count in rule_hits.items():
            logger.info(
                "cleaning rule hit dataset=%s rule=%s rows=%d",
                self.dataset_name, rule, count,
            )
        for rule, count in clip_hits.items():
            logger.info(
                "cleaning rule clipped dataset=%s rule=%s rows=%d",
                self.dataset_name, rule, count,
            )
        logger.info(
            "cleaning complete dataset=%s input=%d cleaned=%d quarantined=%d",
            self.dataset_name, input_rows, cleaned.num_rows, quarantined.num_rows,
        )
        return CleaningResult(cleaned=cleaned, quarantined=quarantined, stats=stats)

    # ------------------------------------------------------------- rule steps
    def _apply_null_policy(
        self,
        table: pa.Table,
        reasons: Dict[int, str],
        rule_hits: Dict[str, int],
    ) -> pa.Table:
        null_policy = self.config.get("null_policy") or {}
        if not null_policy:
            return table

        reject_idxs: List[int] = []
        for column, policy in null_policy.items():
            if column not in table.column_names:
                logger.warning(
                    "cleaning dataset=%s null policy references missing column %r; "
                    "skipping rule", self.dataset_name, column,
                )
                continue
            strategy = policy if isinstance(policy, str) else policy.get("strategy")
            array = table.column(column)
            null_mask = _is_null_or_nan(array)

            if strategy == "default":
                default = policy["default"]
                # An all-null column has Arrow's null type, which cannot hold
                # a scalar; promote it to a concrete type inferred from default.
                if pa.types.is_null(array.type):
                    array = pc.cast(array, pa.array([default]).type)
                    table = table.set_column(
                        table.schema.get_field_index(column), column, array
                    )
                filled = pc.if_else(
                    null_mask, pa.scalar(default, type=array.type), array
                )
                table = table.set_column(
                    table.schema.get_field_index(column), column, filled
                )
                continue

            hit = [
                table.column(ORIG_IDX_COLUMN)[i].as_py()
                for i in range(table.num_rows)
                if null_mask[i].as_py()
            ]
            new_rejects = [idx for idx in hit if idx not in reasons]
            for idx in new_rejects:
                reasons[idx] = f"{RULE_NULL}:{column}"
            if new_rejects:
                rule_hits[f"{RULE_NULL}:{column}"] = (
                    rule_hits.get(f"{RULE_NULL}:{column}", 0) + len(new_rejects)
                )
                reject_idxs.extend(new_rejects)

        if reject_idxs:
            table = self._drop_original_idxs(table, set(reject_idxs))
        return table

    def _apply_casts(
        self,
        table: pa.Table,
        reasons: Dict[int, str],
        rule_hits: Dict[str, int],
    ) -> pa.Table:
        for spec in self.config.get("casts", []) or []:
            column = spec["column"]
            type_name = spec["type"]
            if column not in table.column_names:
                logger.warning(
                    "cleaning dataset=%s cast references missing column %r; "
                    "skipping rule", self.dataset_name, column,
                )
                continue
            target_type = _CAST_TYPES.get(type_name)
            if target_type is None:
                raise CleaningConfigError(
                    f"cast rule on {column!r}: unsupported type {type_name!r}"
                )
            if table.schema.field(column).type == target_type:
                continue

            cast_array, bad_positions = self._safe_cast(
                table.column(column), target_type
            )
            if bad_positions:
                reason = f"{RULE_CAST}:{column}"
                new_rejects = []
                for pos in bad_positions:
                    orig_idx = table.column(ORIG_IDX_COLUMN)[pos].as_py()
                    if orig_idx not in reasons:
                        reasons[orig_idx] = reason
                        new_rejects.append(orig_idx)
                rule_hits[reason] = rule_hits.get(reason, 0) + len(new_rejects)
                table = self._drop_original_idxs(table, set(new_rejects))
                # Re-cast the surviving rows; the offending values are gone.
                cast_array, remaining_bad = self._safe_cast(
                    table.column(column), target_type
                )
                if remaining_bad:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"cast failures remain on {column!r} after quarantine"
                    )
            table = table.set_column(
                table.schema.get_field_index(column), column, cast_array
            )
        return table

    def _apply_ranges(
        self,
        table: pa.Table,
        reasons: Dict[int, str],
        rule_hits: Dict[str, int],
        clip_hits: Dict[str, int],
    ) -> pa.Table:
        for spec in self.config.get("ranges", []) or []:
            column = spec["column"]
            if column not in table.column_names:
                logger.warning(
                    "cleaning dataset=%s range rule references missing column %r; "
                    "skipping rule", self.dataset_name, column,
                )
                continue
            action = spec.get("action", "reject")
            array = table.column(column)
            if not (
                pa.types.is_integer(array.type)
                or pa.types.is_floating(array.type)
                or pa.types.is_temporal(array.type)
                or pa.types.is_boolean(array.type)
            ):
                logger.warning(
                    "cleaning dataset=%s range rule on %r needs a numeric/"
                    "temporal column but got %s; skipping rule",
                    self.dataset_name, column, array.type,
                )
                continue
            out_of_range, low, high = self._range_mask(array, spec)
            if out_of_range is None:
                continue
            hit_count = pc.sum(pc.cast(out_of_range, pa.int64())).as_py() or 0
            if hit_count == 0:
                continue
            reason = f"{RULE_RANGE}:{column}"

            if action == "clip":
                # clamp via element-wise min/max (pc.clamp is unavailable on
                # newer pyarrow). Bounds are scalar; nulls stay null.
                clipped = array
                if low is not None:
                    clipped = pc.max_element_wise(clipped, low)
                if high is not None:
                    clipped = pc.min_element_wise(clipped, high)
                table = table.set_column(
                    table.schema.get_field_index(column), column, clipped
                )
                clip_hits[reason] = clip_hits.get(reason, 0) + hit_count
                continue

            new_rejects = []
            for pos in range(table.num_rows):
                if not out_of_range[pos].as_py():
                    continue
                orig_idx = table.column(ORIG_IDX_COLUMN)[pos].as_py()
                if orig_idx not in reasons:
                    reasons[orig_idx] = reason
                    new_rejects.append(orig_idx)
            rule_hits[reason] = rule_hits.get(reason, 0) + len(new_rejects)
            table = self._drop_original_idxs(table, set(new_rejects))
        return table

    def _apply_dedup(
        self,
        table: pa.Table,
        reasons: Dict[str, str],
        rule_hits: Dict[str, int],
    ) -> pa.Table:
        dedup = self.config.get("dedup")
        if dedup is None:
            return table

        configured_keys: List[str] = dedup["keys"]
        policy = dedup["on_missing_keys"]
        present_keys = [k for k in configured_keys if k in table.column_names]

        if len(present_keys) != len(configured_keys):
            missing = [k for k in configured_keys if k not in present_keys]
            if policy == "fail":
                raise MissingDedupKeyError(
                    f"dedup key column(s) {missing} absent from dataset "
                    f"{self.dataset_name!r} and on_missing_keys='fail'"
                )
            logger.warning(
                "cleaning dataset=%s dedup key column(s) %s absent; falling back "
                "to whole-row hash dedup (computed before casts)",
                self.dataset_name, missing,
            )
            effective_keys = [HASH_COLUMN]
        else:
            effective_keys = present_keys

        # Stable, order-independent selection: sort by keys then row hash and
        # keep the first row of every key group.
        sort_keys = [(k, "ascending") for k in effective_keys]
        sort_keys.append((HASH_COLUMN, "ascending"))
        sorted_table = table.sort_by(sort_keys)

        seen = set()
        keep_positions: List[int] = []
        dup_idxs: List[int] = []
        key_arrays = [sorted_table.column(k).to_pylist() for k in effective_keys]
        for pos in range(sorted_table.num_rows):
            key = tuple(arr[pos] for arr in key_arrays)
            if key in seen:
                dup_idxs.append(sorted_table.column(ORIG_IDX_COLUMN)[pos].as_py())
            else:
                seen.add(key)
                keep_positions.append(pos)

        if dup_idxs:
            new_dups = [idx for idx in dup_idxs if idx not in reasons]
            for idx in new_dups:
                reasons[idx] = RULE_DEDUP
            rule_hits[RULE_DEDUP] = rule_hits.get(RULE_DEDUP, 0) + len(new_dups)
            logger.warning(
                "cleaning dataset=%s removed %d duplicate rows (keys=%s)",
                self.dataset_name, len(new_dups), effective_keys,
            )
        return sorted_table.take(pa.array(keep_positions, type=pa.int64()))

    # ------------------------------------------------------------ primitives
    @staticmethod
    def _safe_cast(
        array: pa.ChunkedArray, target_type: pa.DataType
    ) -> Tuple[pa.ChunkedArray, List[int]]:
        """Safe cast; return (cast_array, positions_of_uncastable_values).

        Per-value pc.cast is used (rather than pa.scalar(value, type=...)) so
        timestamp strings and numeric strings coerce uniformly across pyarrow
        versions, with uncastable values isolated instead of poisoning the
        whole array.
        """
        try:
            return pc.cast(array, target_type, safe=True), []
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            bad_positions: List[int] = []
            cast_values: List[Any] = []
            for pos, value in enumerate(array.to_pylist()):
                if value is None:
                    cast_values.append(None)
                    continue
                try:
                    cast_values.append(
                        pc.cast(pa.scalar(value), target_type).as_py()
                    )
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
                    bad_positions.append(pos)
                    cast_values.append(None)
            return pa.chunked_array([pa.array(cast_values, type=target_type)]), bad_positions

    def _range_mask(
        self, array: pa.ChunkedArray, spec: Dict[str, Any]
    ) -> Tuple[Optional[pa.ChunkedArray], Any, Any]:
        # Bound value (min/max) is separate from the strictness flag
        # (min_strict/max_strict): strict means the bound itself is excluded.
        low = self._bound_scalar(spec.get("min"), array.type)
        high = self._bound_scalar(spec.get("max"), array.type)
        mask = None
        if low is not None:
            low_mask = (
                pc.less_equal(array, low)
                if spec.get("min_strict")
                else pc.less(array, low)
            )
            mask = low_mask
        if high is not None:
            high_mask = (
                pc.greater_equal(array, high)
                if spec.get("max_strict")
                else pc.greater(array, high)
            )
            mask = high_mask if mask is None else pc.or_(mask, high_mask)
        if mask is None:
            logger.warning(
                "cleaning dataset=%s range rule on %r has no bounds; skipping",
                self.dataset_name, spec.get("column"),
            )
            return None, low, high
        # Nulls are not range violations.
        mask = pc.and_(mask, pc.invert(pc.is_null(array)))
        return mask, low, high

    @staticmethod
    def _bound_scalar(raw: Any, column_type: pa.DataType) -> Any:
        if raw is None:
            return None
        if pa.types.is_timestamp(column_type) and isinstance(raw, str):
            raw = datetime.fromisoformat(raw)
        return pa.scalar(raw, type=column_type)

    @staticmethod
    def _drop_original_idxs(table: pa.Table, orig_idxs: set) -> pa.Table:
        if not orig_idxs:
            return table
        keep = [
            pos
            for pos in range(table.num_rows)
            if table.column(ORIG_IDX_COLUMN)[pos].as_py() not in orig_idxs
        ]
        return table.take(pa.array(keep, type=pa.int64()))

    def _build_quarantine(
        self, raw_table: pa.Table, reasons: Dict[int, str]
    ) -> pa.Table:
        if not reasons:
            return self._empty_quarantine(raw_table)
        ordered_idxs = sorted(reasons)
        reason_array = pa.array(
            [reasons[idx] for idx in ordered_idxs], type=pa.string()
        )
        quarantined = (
            raw_table.take(pa.array(ordered_idxs, type=pa.int64()))
            .append_column(REASON_COLUMN, reason_array)
        )
        # Deterministic quarantine ordering, independent of rule hit order.
        hashes = pa.array(
            [_row_hash(r) for r in quarantined.drop_columns([REASON_COLUMN]).to_pylist()],
            type=pa.string(),
        )
        quarantined = quarantined.append_column(HASH_COLUMN, hashes)
        quarantined = (
            quarantined.sort_by([(HASH_COLUMN, "ascending")])
            .drop_columns([HASH_COLUMN])
        )
        return quarantined

    @staticmethod
    def _empty_quarantine(raw_table: pa.Table) -> pa.Table:
        empty = raw_table.slice(0, 0)
        return empty.append_column(
            REASON_COLUMN, pa.array([], type=pa.string())
        )

    def _stats(
        self,
        input_rows: int,
        cleaned_rows: int,
        quarantined_rows: int,
        rule_hits: Dict[str, int],
        clip_hits: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "input_rows": input_rows,
            "cleaned_rows": cleaned_rows,
            "quarantined_rows": quarantined_rows,
            "rule_hits": dict(sorted(rule_hits.items())),
            "clip_hits": dict(sorted(clip_hits.items())),
        }


# ---------------------------------------------------------------------------
# Config factory. Imported lazily so this module stays loadable in tests
# without an /opt/airflow config file.
# ---------------------------------------------------------------------------
def build_cleaner(dataset_name: str) -> Optional[DataCleaner]:
    """Build a cleaner from pipeline_config.yaml, or None when unconfigured.

    Absence of a ``cleaning`` section (or of the dataset under it) means
    passthrough: callers must leave their existing behaviour untouched.
    """
    from src.config import get_config

    datasets = get_config("cleaning", "datasets", default=None)
    if not datasets:
        return None
    dataset_config = datasets.get(dataset_name)
    if not dataset_config:
        return None
    return DataCleaner(dataset_config, dataset_name=dataset_name)


def clean_dataset(table: pa.Table, dataset_name: str) -> CleaningResult:
    """Convenience wrapper preserving passthrough semantics."""
    cleaner = build_cleaner(dataset_name)
    if cleaner is None:
        return CleaningResult(
            cleaned=table,
            quarantined=DataCleaner._empty_quarantine(table),
            stats={
                "dataset": dataset_name,
                "input_rows": table.num_rows,
                "cleaned_rows": table.num_rows,
                "quarantined_rows": 0,
                "rule_hits": {},
                "clip_hits": {},
                "passthrough": True,
            },
        )
    return cleaner.clean(table)
