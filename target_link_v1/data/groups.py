"""Group profiles into (sub-link, window) aggregation units (spec §8).

A group is (link_id, sub_id, window_id) — sub_id is only unique within a link,
so the link id must be part of the key. All profiles of one group have to reach
``scatter_mean`` in the same forward, hence the group-aligned chunker that
slices profile rows on group boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GroupIndex:
    """Contiguous group ids for each profile row + one row of keys per group."""

    group_idx: np.ndarray  # [n_profiles] int64, ids are 0..G-1 in sorted key order
    keys: pd.DataFrame     # columns link_id, sub_id, window_id, n_trajs


def build_group_index(
    link_id: np.ndarray, sub_id: np.ndarray, window_id: np.ndarray
) -> GroupIndex:
    """Factorize (link_id, sub_id, window_id) into contiguous group ids.

    Ids follow sorted key order (groupby sort=True), so the result is
    deterministic and reproducible across runs on the same profile table.
    """
    keys = pd.DataFrame(
        {
            "link_id": np.asarray(link_id),
            "sub_id": np.asarray(sub_id, dtype=np.int64),
            "window_id": np.asarray(window_id, dtype=np.int64),
        }
    )
    grouped = keys.groupby(["link_id", "sub_id", "window_id"], sort=True)
    group_idx = grouped.ngroup().to_numpy(dtype=np.int64)
    table = grouped.size().rename("n_trajs").reset_index()
    return GroupIndex(group_idx=group_idx, keys=table)


def sort_by_group(group_idx: np.ndarray) -> np.ndarray:
    """Row permutation that makes group_idx non-decreasing (stable)."""
    return np.argsort(group_idx, kind="stable")


def group_aligned_chunks(group_idx: np.ndarray, max_rows: int) -> list[np.ndarray]:
    """Slices of row positions, each holding whole groups and <= max_rows rows.

    ``group_idx`` must be sorted (see :func:`sort_by_group`). A single group
    larger than max_rows becomes its own chunk — the bound is soft.
    """
    if len(group_idx) and np.any(np.diff(group_idx) < 0):
        raise ValueError("group_idx must be sorted; apply sort_by_group first")
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(group_idx)) + 1, [len(group_idx)]))
    chunks: list[np.ndarray] = []
    head = 0
    for g in range(len(bounds) - 1):
        end = bounds[g + 1]
        if head < bounds[g] and end - head > max_rows:
            chunks.append(np.arange(head, bounds[g]))
            head = bounds[g]
    if head < bounds[-1]:
        chunks.append(np.arange(head, bounds[-1]))
    return chunks
