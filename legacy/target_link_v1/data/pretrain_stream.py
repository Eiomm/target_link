"""Streaming pretraining dataset over curve shards (build_curves_spark output).

Replaces the materialised corpus npz for large corpora: an IterableDataset
yields padded batches ([B, max_bins] speeds/valid + int-bucketed free labels),
holding ~one parquet batch in memory per worker — the 7-day corpus (~50M rows)
never loads whole. Shard ORDER is reshuffled per epoch (the trainer rebuilds
the DataLoader with seed=epoch); rows within a shard stay in file order (the
span mask re-randomised per step is the main augmentation anyway).

Split: PAST->FUTURE by window_id (deployment-realistic: train on earlier
windows, validate on the latest ones). A sample's curves all share its single
window_id, so the time cut keeps every sample whole — no train/val leakage at
the sample level (shards themselves are round-robin, so a shard split does
NOT have that guarantee). Falls back to a shard split only when the window
distribution cannot honour the fraction (e.g. a single-hour smoke corpus).

Labels: y/v/len buckets come from the edges json that build_curves_spark
writes next to the shards (percentile_approx over that day's samples); hour
is derived from window_id (Beijing = (w+8)%24). Batch keys match the npz
corpus path one-to-one, so the trainer's step/mask/loss code is shared.
"""
from __future__ import annotations

import glob
import json
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

ROW_COLS = ("window_id", "y", "v_sample", "n_bins", "speeds", "valid")


def load_shard_meta(shards_dir: str) -> Dict:
    """curves_meta.json.d/ (a text dir written via createDataFrame().text())."""
    hits = glob.glob(f"{shards_dir}/curves_meta.json.d/part-*.txt")
    if not hits:
        raise FileNotFoundError(f"no curves_meta under {shards_dir}")
    return json.loads(open(hits[0]).read())


def time_cutoff(files: List[str], val_fraction: float) -> Tuple[int | None, int, int]:
    """Past->future cutoff window: rows with window_id >= cut become val, aiming
    for the LAST val_fraction of rows by time. Reads only the window_id column
    (cheap). Returns (w_cut, n_train, n_val); w_cut is None when the window
    distribution cannot honour the fraction (fewer than 2 distinct windows, or
    the cut snaps to 0%/100% — e.g. a single-hour smoke corpus)."""
    wcnt: Dict[int, int] = {}
    for f in files:
        w, n = np.unique(pq.read_table(f, columns=["window_id"])
                         .column("window_id").to_numpy(), return_counts=True)
        for a, b in zip(w.tolist(), n.tolist()):
            wcnt[a] = wcnt.get(a, 0) + int(b)
    total = sum(wcnt.values())
    if len(wcnt) < 2:
        return None, total, 0
    target = (1.0 - float(val_fraction)) * total
    cum, w_cut = 0, max(wcnt)  # cut at the last window = empty val guard
    for w in sorted(wcnt):
        cum += wcnt[w]
        if cum >= target:
            w_cut = w
            break
    n_val = sum(c for w, c in wcnt.items() if w >= w_cut)
    if n_val == 0 or n_val == total:  # snapped degenerate — caller falls back
        return None, total - n_val, n_val
    return w_cut, total - n_val, n_val


class CurveShardDataset(IterableDataset):
    """Yields batch dicts of numpy arrays (keys match the npz corpus path)."""

    def __init__(self, shards_dir: str, batch_size: int = 512, max_bins: int = 40,
                 n_buckets: int = 16, seed: int = 0,
                 shard_list: List[str] | None = None,
                 window_cut: int | None = None, side: str = "all") -> None:
        super().__init__()
        if side not in ("all", "train", "val"):
            raise ValueError(f"side must be all|train|val, got {side!r}")
        if side != "all" and window_cut is None:
            raise ValueError(f"side={side!r} requires window_cut")
        self.shards: List[str] = (list(shard_list) if shard_list is not None
                                  else sorted(glob.glob(f"{shards_dir}/curves/part-*.parquet")))
        if not self.shards:
            raise FileNotFoundError(f"no curve shards under {shards_dir}/curves")
        meta = load_shard_meta(shards_dir)
        self.edges = {k: np.asarray(meta[f"{k}_edges"], dtype=np.float64)
                      for k in ("y", "v", "len")}
        self.max_bins, self.n_buckets = max_bins, n_buckets
        self.batch_size, self.seed = batch_size, seed
        self.window_cut, self.side = window_cut, side

    def _bucket(self, kind: str, values: np.ndarray) -> np.ndarray:
        e = self.edges[kind]
        return np.clip(np.digitize(values, e[1:-1]), 0, self.n_buckets - 1)

    def _rows_to_batch(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        b, n = len(df), self.max_bins
        cols = {c: df[c].to_numpy() for c in ROW_COLS}
        speeds = np.zeros((b, n), dtype=np.float32)
        valid = np.zeros((b, n), dtype=bool)
        for i, (sp, va) in enumerate(zip(cols["speeds"], cols["valid"])):
            k = min(len(sp), n)
            speeds[i, :k] = sp[:k]
            valid[i, :k] = va[:k]
        lengths = np.fromiter((min(len(sp), n) for sp in cols["speeds"]),
                              dtype=np.int64, count=b)
        hour = (cols["window_id"].astype(np.int64) % 24 + 8) % 24
        return {
            "speeds": speeds, "valid": valid, "lengths": lengths,
            "attr_y": self._bucket("y", cols["y"].astype(np.float64)),
            "attr_v": self._bucket("v", cols["v_sample"].astype(np.float64)),
            "attr_len": self._bucket("len", cols["n_bins"].astype(np.float64)),
            "attr_hour": hour.astype(np.int64),
        }

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        order = np.random.default_rng(self.seed + wid).permutation(len(self.shards))
        mine = [self.shards[i] for i in order if i % nw == wid]
        carry: List[pd.DataFrame] = []
        n_carry = 0
        for shard in mine:
            for tbl in pq.ParquetFile(shard).iter_batches(batch_size=2048):
                df = tbl.to_pandas()
                if self.window_cut is not None and self.side != "all":
                    df = df[df.window_id >= self.window_cut] if self.side == "val" \
                        else df[df.window_id < self.window_cut]
                    if df.empty:
                        continue
                carry.append(df)
                n_carry += len(carry[-1])
                while n_carry >= self.batch_size:  # flush ALL full batches — a
                    whole = pd.concat(carry, ignore_index=True)  # single if left
                    yield self._rows_to_batch(whole.iloc[:self.batch_size])  # the
                    carry = [whole.iloc[self.batch_size:]]       # tail growing
                    n_carry = len(carry[0])                      # into one giant
        if n_carry:                                              # batch (OOM)
            yield self._rows_to_batch(pd.concat(carry, ignore_index=True))


def stream_batch_to_tensors(batch: Dict[str, np.ndarray], device: str):
    """numpy batch dict -> trainer's (s, v, l, labels) on device."""
    s = torch.from_numpy(batch["speeds"]).to(device)
    v = torch.from_numpy(batch["valid"]).to(device)
    l = torch.from_numpy(batch["lengths"]).to(device)
    y = {k: torch.from_numpy(batch[f"attr_{k}"]).to(device)
         for k in ("y", "v", "len", "hour")}
    return s, v, l, y
