"""Read complete road snapshots, preserving partial passages and spatial gaps.

Only build_windows_spark v1 sorted/hash-partitioned output is accepted. One
snapshot may span Parquet record batches, but must never span output files.
Exact times remain in the artifact for audit; model input gets coarse ages.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


class WindowDataset(IterableDataset):
    def __init__(self, directory, start_ts=None, end_ts=None, seed=0,
                 age_bucket_seconds=60, max_curves_per_snapshot=512):
        super().__init__()
        hits = glob.glob(str(Path(directory) / "window_meta.json.d/part-*.txt"))
        if len(hits) != 1:
            raise ValueError("Require a completed window corpus with one metadata file")
        self.meta = json.loads(Path(hits[0]).read_text())
        if self.meta.get("format") != "target_link_windows_v2":
            raise ValueError("Unsupported window format; v1 snapshots are physical-link "
                             "level and cannot be regrouped into modeling units")
        self.files = sorted(glob.glob(str(Path(directory) / "window_curves/part-*.parquet")))
        if not self.files:
            raise ValueError("No window_curves Parquet files")
        if age_bucket_seconds <= 0 or max_curves_per_snapshot <= 0:
            raise ValueError("age bucket and curve limit must be positive")
        self.start_ts, self.end_ts, self.seed = start_ts, end_ts, seed
        self.age_bucket_seconds = age_bucket_seconds
        self.max_curves = max_curves_per_snapshot

    def _pack(self, rows):
        n, m = len(rows), self.meta["max_bins"]
        arrays = {k: np.zeros((n, m), dtype=np.float32)
                  for k in ["duration", "distance", "position", "age", "observed"]}
        valid = np.zeros((n, m), dtype=bool)
        lengths = np.zeros(n, dtype=np.int64)
        unit_starts = np.zeros(n, dtype=np.float32)
        unit_lengths = np.zeros(n, dtype=np.float32)
        anchor = rows[0]["anchor_ts"]
        w = self.meta["lookback_seconds"]
        pass_ids = {}
        curve_pass = []
        for j, row in enumerate(rows):
            if row["sample_id"] not in pass_ids:
                pass_ids[row["sample_id"]] = len(pass_ids)
            curve_pass.append(pass_ids[row["sample_id"]])
            bins = row["bins"]
            if not 0 < len(bins) <= m:
                raise ValueError("Invalid curve length; refusing silent truncation")
            lengths[j] = len(bins)
            unit_starts[j] = row["unit_start_m"]
            unit_lengths[j] = row["unit_length_m"]
            for i, b in enumerate(bins):
                if not (b["bin_start_ts"] >= anchor - w and b["bin_end_ts"] < anchor
                        and b["available_ts"] <= anchor):
                    raise ValueError("Window artifact violates causal time contract")
                if not (b["duration"] > 0 and b["distance_m"] > 0):
                    raise ValueError("Invalid reconstruction target")
                arrays["duration"][j, i] = b["duration"]
                arrays["distance"][j, i] = b["distance_m"]
                arrays["position"][j, i] = b["position_m"]
                arrays["age"][j, i] = np.floor((anchor - b["bin_end_ts"]) /
                                                self.age_bucket_seconds) * self.age_bucket_seconds / w
                arrays["observed"][j, i] = b["observed"]
                valid[j, i] = True
        return dict(arrays, valid=valid, lengths=lengths,
                    unit_starts=unit_starts, unit_lengths=unit_lengths,
                    curve_pass=np.asarray(curve_pass, dtype=np.int64),
                    snapshot_id=rows[0]["snapshot_id"], anchor_ts=anchor,
                    map_version=rows[0].get("map_version"), target_link_id=rows[0].get("target_link_id"),
                    sub_id=rows[0].get("sub_id"))

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
        # Every worker uses the SAME permutation, then takes disjoint indices.
        order = np.random.default_rng(self.seed).permutation(len(self.files))[wid::nw]
        for i in order:
            pending, key = [], None
            for batch in pq.ParquetFile(self.files[i]).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    t = row["anchor_ts"]
                    if (self.start_ts is not None and t < self.start_ts or
                            self.end_ts is not None and t >= self.end_ts):
                        continue
                    sid = row["snapshot_id"]
                    if key is not None and sid < key:
                        raise ValueError("Window file is not sorted by snapshot_id")
                    if sid != key and pending:
                        yield self._pack(pending)
                        pending = []
                    key = sid
                    pending.append(row)
                    if len(pending) > self.max_curves:
                        raise ValueError("Snapshot exceeds reader limit; use window-level cap upstream")
            if pending:
                yield self._pack(pending)


def collate_windows(items):
    """Flatten curves, but retain complete snapshot and passage membership."""
    tensors = {}
    for k in ["duration", "distance", "position", "age", "observed", "valid", "lengths",
              "unit_starts", "unit_lengths"]:
        tensors[k] = torch.from_numpy(np.concatenate([x[k] for x in items], axis=0))
    tensors["curve_group"] = torch.repeat_interleave(
        torch.arange(len(items)), torch.tensor([len(x["lengths"]) for x in items]))
    offsets, passes = 0, []
    for x in items:
        passes.append(x["curve_pass"] + offsets)
        offsets += int(x["curve_pass"].max()) + 1
    tensors["curve_pass"] = torch.from_numpy(np.concatenate(passes))
    tensors["n_groups"] = len(items)
    tensors["snapshot_ids"] = [x["snapshot_id"] for x in items]
    tensors["anchor_ts"] = [x["anchor_ts"] for x in items]
    tensors["map_versions"] = [x.get("map_version") for x in items]
    tensors["link_ids"] = [x.get("target_link_id") for x in items]
    tensors["sub_ids"] = [x.get("sub_id") for x in items]
    return tensors
