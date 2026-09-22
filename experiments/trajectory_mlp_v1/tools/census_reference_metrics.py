"""Exact reference metrics for reduced trajectory-cell partitions.

The input is the compact per-trajectory form produced by the census job.  It
keeps the grouping and MAE-mask contracts in :mod:`..data`, without building
the ``[member, 50, 3]`` tensors used by training.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


N_BINS = 50
_VALID_BITS_MASK = (1 << N_BINS) - 1
_REQUIRED_COLUMNS = ("cell_id", "sample_id", "valid_bits", "n_valid", "usable")


def _seed(*parts: object) -> int:
    """Stable seed, kept byte-for-byte equivalent to ``data._seed``."""
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")


def _column(table: pa.Table, name: str) -> pa.Array:
    column = table[name]
    return column.combine_chunks() if hasattr(column, "combine_chunks") else column


def _as_numpy(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return _column(table, name).to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _group_id(m_max: int, day: str, bucket: str, cell_id: int, group_index: int) -> str:
    return "groups-v1-m%d/%s/%s/%s/%d" % (m_max, day, bucket, cell_id, group_index)


def _metrics_for_members(bits: np.ndarray, group_id: str, epoch: int) -> tuple[int, int, int]:
    """Return hidden trajectories, valid targets, and visible-supported targets."""
    size = len(bits)
    hidden_count = size // 2
    if not hidden_count:
        return 0, 0, 0
    hidden_at = np.random.default_rng(_seed(group_id, int(epoch))).choice(
        size, size=hidden_count, replace=False
    )
    is_hidden = np.zeros(size, dtype=bool)
    is_hidden[hidden_at] = True
    visible_bits = 0
    for value in bits[~is_hidden]:
        visible_bits |= int(value)
    supervised = sum(int(value).bit_count() for value in bits[is_hidden])
    supported = sum((int(value) & visible_bits).bit_count() for value in bits[is_hidden])
    return hidden_count, supervised, supported


def reference_metrics(table: pa.Table, day: str, bucket: str, seed: int = 42,
                      epoch: int = 0, m_max: int = 64) -> dict:
    """Compute exact grouping and masking counts for one reduced partition.

    ``n_valid > 0`` is applied before grouping, as in the source reduction
    contract.  Of those candidates, only ``usable`` rows can be members.  The
    returned counters are plain Python integers and may therefore be safely
    accumulated across partitions without approximation.
    """
    if not isinstance(table, pa.Table):
        raise TypeError("table must be a pyarrow.Table")
    if m_max < 3:
        raise ValueError("m_max must be at least 3")
    missing = [name for name in _REQUIRED_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))

    cell_id = _as_numpy(table, "cell_id", np.int64)
    valid_bits = _as_numpy(table, "valid_bits", np.uint64)
    n_valid = _as_numpy(table, "n_valid", np.int16)
    usable = _as_numpy(table, "usable", bool)
    if np.any(valid_bits > np.uint64(_VALID_BITS_MASK)):
        raise ValueError("valid_bits contains bits outside 0..49")

    # A malformed reduced row cannot be allowed to enter a group merely because
    # its supplied usability flag is true.  The census reducer normally makes
    # this condition impossible, but applying the source candidate rule here is
    # both cheap and explicit.
    member = (n_valid > 0) & usable
    member_count = int(member.sum())
    result = {
        "retained_observations": 0,
        "dropped_no_valid": int(len(cell_id) - member_count),
        "dropped_tail": 0,
        "groups": 0,
        "hidden_trajectories": 0,
        "supervised_bins": 0,
        "supported_supervised_bins": 0,
    }
    if not member_count:
        return result

    # Stable ordering makes physical parquet order irrelevant, exactly as in
    # ``CellDataset._cells``.
    order = np.argsort(cell_id, kind="stable")
    member_order = order[member[order]]
    member_cell_id = cell_id[member_order]
    starts = np.r_[0, np.flatnonzero(member_cell_id[1:] != member_cell_id[:-1]) + 1]
    ends = np.r_[starts[1:], len(member_order)]
    counts = ends - starts
    full_groups, tails = np.divmod(counts, m_max)
    keep_tail = tails >= 3
    hidden_counts = full_groups * (m_max // 2) + np.where(keep_tail, tails // 2, 0)

    # The common path is entirely vectorized: reduceat finds whether every
    # usable member in a cell has the same coverage bitmap.  This avoids a
    # Python loop over the roughly 200k cells in a million-row partition.
    member_bits = valid_bits[member_order]
    minimum_bits = np.minimum.reduceat(member_bits, starts)
    maximum_bits = np.maximum.reduceat(member_bits, starts)
    homogeneous = minimum_bits == maximum_bits
    bit_counts = np.bitwise_count(minimum_bits[homogeneous]).astype(np.int64, copy=False)
    homogeneous_hidden = hidden_counts[homogeneous]

    result["retained_observations"] = int(np.sum(counts - np.where(keep_tail, 0, tails)))
    result["dropped_tail"] = int(np.sum(np.where(keep_tail, 0, tails)))
    result["groups"] = int(np.sum(full_groups + keep_tail))
    result["hidden_trajectories"] = int(np.sum(hidden_counts))
    result["supervised_bins"] = int(np.sum(bit_counts * homogeneous_hidden))
    result["supported_supervised_bins"] = result["supervised_bins"]

    # Only cells whose member coverage differs need the exact member shuffle,
    # sample-ID ordering, and per-group epoch mask.
    heterogeneous_cells = np.flatnonzero(~homogeneous)
    if not len(heterogeneous_cells):
        return result
    heterogeneous_counts = counts[heterogeneous_cells]
    # Arrow's take is cheap when called once, but invoking it once per cell
    # would repeatedly walk chunk metadata.  Gather exactly the exceptional
    # rows and their IDs in one vectorized operation, then retain the small
    # exact loop needed for each distinct RNG stream.
    heterogeneous_rows = member_order[np.repeat(~homogeneous, counts)]
    heterogeneous_ids = pc.take(
        table["sample_id"], pa.array(heterogeneous_rows, type=pa.int64())
    ).to_pylist()
    id_offsets = np.empty(len(heterogeneous_cells) + 1, dtype=np.int64)
    id_offsets[0] = 0
    np.cumsum(heterogeneous_counts, out=id_offsets[1:])
    for position, cell_index in enumerate(heterogeneous_cells):
        lo, hi = int(id_offsets[position]), int(id_offsets[position + 1])
        rows = heterogeneous_rows[lo:hi]
        usable_k = len(rows)
        cell = int(member_cell_id[starts[cell_index]])
        ids = heterogeneous_ids[lo:hi]
        packed = sorted(zip(ids, rows.tolist()), key=lambda item: str(item[0]))
        perm = np.random.default_rng(_seed(seed, day, bucket, cell, "members")).permutation(usable_k)
        shuffled_rows = np.fromiter((packed[int(i)][1] for i in perm), dtype=np.int64,
                                    count=usable_k)
        for group_index, begin in enumerate(range(0, usable_k, m_max)):
            group_rows = shuffled_rows[begin:begin + m_max]
            if len(group_rows) < 3:
                continue
            group_bits = valid_bits[group_rows]
            hidden, supervised, supported = _metrics_for_members(
                group_bits, _group_id(m_max, day, bucket, cell, group_index), epoch
            )
            result["supervised_bins"] += supervised
            result["supported_supervised_bins"] += supported
    return result
