"""Vectorized P0 trajectory coverage metrics for ``observations_v2`` batches.

This module deliberately works on one Arrow record batch (at most 65,536
rows).  It flattens Arrow list columns to NumPy arrays and uses indexed
reductions; it never materializes the observation pieces as Python objects.

``valid_bits`` uses the same P0 rule as :mod:`experiments.trajectory_mlp_v1.data`:
a present bin is valid only when *every* piece in that bin has ``valid=True``.

For an exact, scalable hidden-target reference rate, write this row-level
output partition-by-partition, then aggregate it by the intended strata with
DuckDB/Arrow SQL.  For the expected rate of a group of size ``K``, use the
exact weight ``floor(K / 2) / K`` on each member's ``n_valid``; no mask needs
to be generated.  For one concrete training epoch, generate and persist one
small identity-plus-hidden table using the reader's
``default_rng(_seed(group_id, epoch)).choice(...)`` rule, then join it to this
output and sum ``n_valid``.  That produces the realized exact count without
repeatedly building training tensors or running a random-mask loop during the
census.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


N_BINS = 50
MAX_BATCH_ROWS = 65_536
_IDENTIFIER_COLUMNS = (
    "cell_id", "sample_id", "window", "target_link_id", "map_version", "seg_idx", "dt",
)
_PIECE_COLUMNS = ("T_diff", "ratio_pct", "valid", "bin_pos")


def _column(batch: pa.RecordBatch | pa.Table, name: str) -> pa.Array:
    """Return one contiguous Arrow array, without converting it to Python."""
    try:
        column = batch.column(name)
    except (KeyError, IndexError) as exc:
        raise ValueError(f"missing required column {name!r}") from exc
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


def _list_lengths(array: pa.Array, name: str) -> np.ndarray:
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or
            pa.types.is_fixed_size_list(array.type)):
        raise ValueError(f"{name} must be an Arrow list column, got {array.type}")
    if array.null_count:
        raise ValueError(f"{name} contains a null list")
    # A null piece cannot participate in a P0 all-pieces-valid decision.
    if array.values.null_count:
        raise ValueError(f"{name} contains a null list element")
    return pc.list_value_length(array).to_numpy(zero_copy_only=False).astype(np.int64,
                                                                               copy=False)


def _flat(array: pa.Array, name: str) -> np.ndarray:
    values = pc.list_flatten(array)
    if values.null_count:
        raise ValueError(f"{name} contains a null list element")
    return values.to_numpy(zero_copy_only=False)


def _numeric(values: np.ndarray, name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _bad_row(mask: np.ndarray, rows: np.ndarray, message: str) -> None:
    """Raise a useful error without formatting the full batch."""
    if mask.any():
        raise ValueError(f"row {int(rows[np.flatnonzero(mask)[0]])}: {message}")


def _bad_piece(mask: np.ndarray, offsets: np.ndarray, message: str) -> None:
    """Report the source row for a bad flattened piece without ``repeat``."""
    if mask.any():
        piece = int(np.flatnonzero(mask)[0])
        row = int(np.searchsorted(offsets, piece, side="right") - 1)
        raise ValueError(f"row {row}: {message}")


def summarize_batch(batch: pa.RecordBatch | pa.Table) -> pa.Table:
    """Summarize one bounded ``observations_v2`` Arrow batch.

    The output preserves ``cell_id, sample_id, window, target_link_id,
    map_version, seg_idx, dt`` with their incoming Arrow types and adds:

    ``n_present`` / ``n_valid`` (int16), ``has_internal_gap`` (bool),
    ``covered_m`` (float64), ``present_bits`` / ``valid_bits`` (uint64),
    ``usable`` (bool), and ``n_pieces`` (int32).

    Inputs are rejected when list lengths differ, bin positions are outside
    0..49, ``ratio_pct`` is non-finite or negative, a valid ``T_diff`` is
    non-finite or negative, or scalar ``dt`` differs from the reader's finite
    [0, 600] contract.
    """
    if not isinstance(batch, (pa.RecordBatch, pa.Table)):
        raise TypeError("batch must be a pyarrow.RecordBatch or pyarrow.Table")
    rows_n = batch.num_rows
    if rows_n > MAX_BATCH_ROWS:
        raise ValueError(f"batch has {rows_n} rows; maximum is {MAX_BATCH_ROWS}")

    identifiers = [_column(batch, name) for name in _IDENTIFIER_COLUMNS]
    pieces = {name: _column(batch, name) for name in _PIECE_COLUMNS}
    lengths = {name: _list_lengths(array, name) for name, array in pieces.items()}
    expected = lengths["T_diff"]
    for name in _PIECE_COLUMNS[1:]:
        unequal = expected != lengths[name]
        _bad_row(unequal, np.arange(rows_n),
                 f"T_diff and {name} have different list lengths")
    if expected.size and expected.max() > np.iinfo(np.int32).max:
        raise ValueError("n_pieces exceeds int32")

    dt_array = identifiers[-1]
    if dt_array.null_count:
        raise ValueError("dt contains null values")
    dt = _numeric(dt_array.to_numpy(zero_copy_only=False), "dt")
    bad_dt = ~np.isfinite(dt) | (dt < 0) | (dt > 600)
    _bad_row(bad_dt, np.arange(rows_n), "dt outside finite [0, 600]")

    t = _numeric(_flat(pieces["T_diff"], "T_diff"), "T_diff")
    ratio = _numeric(_flat(pieces["ratio_pct"], "ratio_pct"), "ratio_pct")
    valid_array = pieces["valid"]
    if not pa.types.is_boolean(valid_array.type.value_type):
        raise ValueError(f"valid must have boolean list values, got {valid_array.type.value_type}")
    valid = np.asarray(_flat(valid_array, "valid"), dtype=bool)
    bin_array = pieces["bin_pos"]
    if not (pa.types.is_integer(bin_array.type.value_type) and
            not pa.types.is_boolean(bin_array.type.value_type)):
        raise ValueError(f"bin_pos must have integer list values, got {bin_array.type.value_type}")
    bin_pos = np.asarray(_flat(bin_array, "bin_pos"), dtype=np.int64)

    offsets = np.empty(rows_n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(expected, out=offsets[1:])
    _bad_piece(~np.isfinite(ratio) | (ratio < 0), offsets,
               "ratio_pct must be finite and nonnegative")
    _bad_piece(valid & (~np.isfinite(t) | (t < 0)), offsets,
               "valid T_diff must be finite and nonnegative")
    _bad_piece((bin_pos < 0) | (bin_pos >= N_BINS), offsets,
               "bin_pos outside 0..49")

    present_bits = np.zeros(rows_n, dtype=np.uint64)
    invalid_bits = np.zeros(rows_n, dtype=np.uint64)
    covered_m = np.zeros(rows_n, dtype=np.float64)
    if bin_pos.size:
        piece_bits = np.left_shift(np.uint64(1), bin_pos.astype(np.uint64, copy=False))
        nonempty_pieces = expected > 0
        starts = offsets[:-1][nonempty_pieces]
        present_bits[nonempty_pieces] = np.bitwise_or.reduceat(piece_bits, starts)
        invalid_bits[nonempty_pieces] = np.bitwise_or.reduceat(
            np.where(valid, np.uint64(0), piece_bits), starts
        )
        covered_m[nonempty_pieces] = np.add.reduceat(ratio, starts)
    valid_bits = present_bits & ~invalid_bits

    n_present = np.bitwise_count(present_bits).astype(np.int16, copy=False)
    n_valid = np.bitwise_count(valid_bits).astype(np.int16, copy=False)
    # A nonzero bitmap has an internal gap precisely when it differs from the
    # contiguous run between its lowest and highest set bits.  All source bins
    # are in 0..49, so the shifts are safely below uint64's sign bit.
    nonempty = present_bits != 0
    safe_bits = np.where(nonempty, present_bits, np.uint64(1))
    lowest = safe_bits & (np.uint64(0) - safe_bits)
    spread = safe_bits.copy()
    for shift in (1, 2, 4, 8, 16, 32):
        spread |= spread >> np.uint64(shift)
    highest = spread ^ (spread >> np.uint64(1))
    before_first = lowest - np.uint64(1)
    through_last = (highest << np.uint64(1)) - np.uint64(1)
    has_internal_gap = nonempty & (present_bits != (through_last ^ before_first))

    return pa.Table.from_arrays(
        identifiers + [
            pa.array(n_present, type=pa.int16()),
            pa.array(n_valid, type=pa.int16()),
            pa.array(has_internal_gap, type=pa.bool_()),
            pa.array(covered_m, type=pa.float64()),
            pa.array(present_bits, type=pa.uint64()),
            pa.array(valid_bits, type=pa.uint64()),
            pa.array(n_valid > 0, type=pa.bool_()),
            pa.array(expected.astype(np.int32, copy=False), type=pa.int32()),
        ],
        names=list(_IDENTIFIER_COLUMNS) + [
            "n_present", "n_valid", "has_internal_gap", "covered_m", "present_bits",
            "valid_bits", "usable", "n_pieces",
        ],
    )
