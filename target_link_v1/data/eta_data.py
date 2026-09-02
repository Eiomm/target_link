"""Assemble the ETA downstream dataset (spec §9) from profiles + samples.

Prediction unit = one sample (a vehicle's pass, label y_travel_s). Variants:
  A0 "speed"      x = [log L, v_bar]           production scalar
  A1 "speed-mlp"  x = [log L, lift(v_bar)]     same scalar lifted to 128-d
  A2 "ours"       x = [log L, v_bar, r_{l,t}]  + trajectory representation

The encoder-side structure is the bipartite graph built in Step 5:
  sample -> its (sub-link, window) groups (CSR samp_ptr/samp_groups)
  group  -> its profile rows (group_bounds, group-sorted as spec §8)
Batches are assembled lazily from sample rows, so every batch contains whole
groups (spec §8 mean needs all K trajectories together). Splits are BY LINK:
a link and all its trajectories live in exactly one of train/val/test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from target_link_v1.data.groups import build_group_index, sort_by_group


def _ragged(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Concatenation of ranges [starts[i], starts[i]+sizes[i])."""
    total = int(sizes.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.cumsum(sizes) - sizes
    return np.arange(total) - np.repeat(offs, sizes) + np.repeat(starts, sizes)


@dataclass
class ETAData:
    """Group-sorted profile arrays + sample table + sample->group CSR."""

    # encoder inputs (profile rows, sorted by group id)
    speeds: np.ndarray      # [P, max_bins] f32
    valid: np.ndarray       # [P, max_bins] bool
    lengths: np.ndarray     # [P] i32
    group_bounds: np.ndarray  # [G+1] row range of each (sub-link, window) group

    # sample table (S rows, aligned)
    sample_id: np.ndarray   # [S] str
    link_id: np.ndarray     # [S] str
    window_id: np.ndarray   # [S] i64
    y: np.ndarray           # [S] f32, seconds
    log_y: np.ndarray       # [S] f32
    L_m: np.ndarray         # [S] f32, raw link length in metres
    L_n: np.ndarray         # [S] f32, standardised log length
    v_n: np.ndarray         # [S] f32, v_bar / v_norm
    v_bar: np.ndarray       # [S] f32, raw link-window mean speed (m/s)
    n_trajs_lw: np.ndarray  # [S] i32, K of the sample's link-window (strata)
    split: np.ndarray       # [S] i32, 0 train / 1 val / 2 test

    # sample -> groups CSR
    samp_ptr: np.ndarray    # [S+1]
    samp_groups: np.ndarray  # [E] global group ids

    def rows_of(self, which: int) -> np.ndarray:
        return np.flatnonzero(self.split == which)

    def batch(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        """Assemble one training/eval batch from sample row positions."""
        deg = np.diff(self.samp_ptr)[rows]
        edge_sample = np.repeat(np.arange(len(rows)), deg)
        g = self.samp_groups[_ragged(self.samp_ptr[rows], deg)]  # [E] global group ids
        ug, edge_group = np.unique(g, return_inverse=True)       # local group ids
        gsize = self.group_bounds[ug + 1] - self.group_bounds[ug]
        prof_rows = _ragged(self.group_bounds[ug], gsize)
        return {
            "speeds": self.speeds[prof_rows],
            "valid": self.valid[prof_rows],
            "lengths": self.lengths[prof_rows],
            "prof_group": np.repeat(np.arange(len(ug)), gsize),
            "n_groups": len(ug),
            "edge_group": edge_group,
            "edge_sample": edge_sample,
            "n_samples": len(rows),
        }


def build_eta_data(cfg: Dict) -> ETAData:
    """Load profiles + samples + link_window and wire them into an ETAData."""
    d = np.load(cfg["profiles_npz"])
    meta = pd.DataFrame(
        {
            "sample_id": d["sample_id"].astype(str),
            "link_id": d["link_id"].astype(str),
            "window_id": d["window_id"].astype(np.int64),
            "sub_id": d["sub_id"].astype(np.int64),
        }
    )

    # group-sorted profile arrays (spec §8 ordering) + per-group row ranges
    gi = build_group_index(
        meta.link_id.to_numpy(), meta.sub_id.to_numpy(), meta.window_id.to_numpy()
    )
    order = sort_by_group(gi.group_idx)
    gsorted = gi.group_idx[order]
    group_bounds = np.concatenate(([0], np.flatnonzero(np.diff(gsorted)) + 1, [len(gsorted)]))

    # sample table with the production scalar v_bar (link-window mean of v_sample)
    samples = pd.read_parquet(cfg["samples_parquet"])
    lw = pd.read_parquet(cfg["link_window_parquet"])[
        ["target_link_id", "window_id", "mean_speed", "n_trajs"]
    ]
    samples = samples.merge(
        lw, left_on=["target_link_id", "window_id"],
        right_on=["target_link_id", "window_id"], how="left", validate="m:1",
    )
    if samples.mean_speed.isna().any():
        raise ValueError(f"{samples.mean_speed.isna().sum()} samples without link_window row")

    # split by link: permute unique links once, cut 80/10/10
    sc = cfg["split"]
    links = samples.target_link_id.unique()
    rng = np.random.default_rng(int(sc["seed"]))
    perm = rng.permutation(len(links))
    n_tr = int(round(float(sc["train"]) * len(links)))
    n_va = int(round(float(sc["val"]) * len(links)))
    link_split = np.empty(len(links), dtype=np.int8)
    link_split[perm[:n_tr]] = 0
    link_split[perm[n_tr:n_tr + n_va]] = 1
    link_split[perm[n_tr + n_va:]] = 2
    split = pd.Series(link_split, index=links).reindex(samples.target_link_id.to_numpy()).to_numpy()

    # normalisers computed on the train split only (no val/test leakage)
    log_len = np.log(samples.L_link_m.to_numpy(dtype=np.float64))
    mu, sd = log_len[split == 0].mean(), log_len[split == 0].std()
    L_n = ((log_len - mu) / sd).astype(np.float32)
    v_bar = samples.mean_speed.to_numpy(dtype=np.float32)
    y = samples.y_travel_s.to_numpy(dtype=np.float32)

    # sample -> groups CSR from the (sample, group) edges of the profile table
    sg = meta.assign(gid=gi.group_idx).drop_duplicates(["sample_id", "gid"])
    row_of = pd.Index(samples.sample_id).get_indexer(sg.sample_id)
    if (row_of < 0).any() or (np.bincount(row_of, minlength=len(samples)) == 0).any():
        raise ValueError("sample/group join left orphans")
    order2 = np.argsort(row_of, kind="stable")
    samp_groups = sg.gid.to_numpy(dtype=np.int64)[order2]
    deg = np.bincount(row_of, minlength=len(samples))
    samp_ptr = np.concatenate(([0], np.cumsum(deg))).astype(np.int64)

    return ETAData(
        speeds=d["speeds"][order], valid=d["valid"][order], lengths=d["lengths"][order],
        group_bounds=group_bounds.astype(np.int64),
        sample_id=samples.sample_id.to_numpy(dtype=object).astype("U"),
        link_id=samples.target_link_id.to_numpy(dtype=object).astype("U"),
        window_id=samples.window_id.to_numpy(dtype=np.int64),
        y=y, log_y=np.log(y).astype(np.float32),
        L_m=samples.L_link_m.to_numpy(dtype=np.float32), L_n=L_n,
        v_n=(v_bar / float(cfg["v_norm"])).astype(np.float32), v_bar=v_bar,
        n_trajs_lw=samples.n_trajs.to_numpy(dtype=np.int32),
        split=split.astype(np.int8), samp_ptr=samp_ptr, samp_groups=samp_groups,
    )
