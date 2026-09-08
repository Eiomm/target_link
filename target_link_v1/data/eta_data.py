"""Assemble the ETA downstream dataset (spec §9) from profiles + samples.

Prediction unit = one sample (a vehicle's pass, label y_travel_s). Variants:
  A0 "speed"      x = [log L, v_bar]           production scalar
  A1 "speed-mlp"  x = [log L, lift(v_bar)]     same scalar lifted to 128-d
  A2 "ours"       x = [log L, v_bar, r_{l,t}]  + trajectory representation

The encoder-side structure is the bipartite graph built in Step 5:
  sample -> its (sub-link, window) groups (CSR samp_ptr/samp_groups)
  group  -> its profile rows (group_bounds, group-sorted as spec §8)
Batches are assembled lazily from sample rows, so every batch contains whole
groups (spec §8 mean needs all K trajectories together). Splits: BY LINK
(a link and all its trajectories live in exactly one of train/val/test),
BY TIME (whole hour-windows), or a precomputed "manifest" column written by
tools/split_random.py (random shuffle at sample/link/window unit).
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
    td_s: np.ndarray        # [S] f32, target traversal time — oracle y≡td floor
    log_y: np.ndarray       # [S] f32
    z_y: np.ndarray         # [S] f32, log_y z-scored with TRAIN-split stats
    y_mu: float             # train mean of log_y (inverse transform)
    y_sd: float             # train std of log_y
    L_m: np.ndarray         # [S] f32, raw link length in metres
    L_n: np.ndarray         # [S] f32, standardised log length
    v_n: np.ndarray         # [S] f32, v_bar / v_norm
    v_bar: np.ndarray       # [S] f32, raw link-window mean speed (m/s)
    n_trajs_lw: np.ndarray  # [S] i32, K of the sample's link-window (strata)
    split: np.ndarray       # [S] i32, 0 train / 1 val / 2 test

    # sample -> groups CSR
    samp_ptr: np.ndarray    # [S+1]
    samp_groups: np.ndarray  # [E] global group ids
    split_mode: str = "link"  # "link" | "time" (links repeat by design) | "manifest" (precomputed)

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
    # profiles_npz may be one file or a list (multi-day corpora are concatenated)
    spec = cfg["profiles_npz"]
    paths = [spec] if isinstance(spec, str) else list(spec)
    parts = [np.load(p) for p in paths]
    d = {k: np.concatenate([x[k] for x in parts]) for k in parts[0].files}
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

    # split: mode "link" permutes unique links once and cuts 80/10/10 (a link
    # lives in exactly one split); mode "time" assigns whole hour-windows by
    # wall-clock — links intentionally repeat across splits (deployment setup:
    # train on earlier days, evaluate on later ones).
    sc = cfg["split"]
    split_mode = sc.get("mode", "link")
    if split_mode == "manifest":
        # fixed split precomputed & persisted by tools/split_random.py (the
        # 'split' int8 column on samples_parquet) — random/unit/seed live there
        if "split" not in samples.columns:
            raise ValueError("split.mode=manifest requires a 'split' column on samples_parquet")
        split = samples["split"].to_numpy(dtype=np.int8)
    elif split_mode == "time":
        def win_of(s: str) -> int:  # "2026-08-20 07:00" (Beijing) -> window_id
            return int(pd.Timestamp(s, tz="Asia/Shanghai").timestamp()) // 3600

        win_map = {**{win_of(s): 0 for s in sc["train"]},
                   **{win_of(s): 1 for s in sc["val"]},
                   **{win_of(s): 2 for s in sc["test"]}}
        split = samples.window_id.map(win_map).to_numpy(dtype=np.float64)
        n_drop = int(np.isnan(split).sum())
        if n_drop:
            print(f"[eta_data] time-split: dropping {n_drop:,} samples in "
                  f"unlisted windows (boundary spillover)")
            keep = ~np.isnan(split)
            samples, split = samples[keep].reset_index(drop=True), split[keep].astype(np.int8)
        else:
            split = split.astype(np.int8)
    else:
        links = samples.target_link_id.unique()
        rng = np.random.default_rng(int(sc["seed"]))
        perm = rng.permutation(len(links))
        n_tr = int(round(float(sc["train"]) * len(links)))
        n_va = int(round(float(sc["val"]) * len(links)))
        link_split = np.empty(len(links), dtype=np.int8)
        link_split[perm[:n_tr]] = 0
        link_split[perm[n_tr:n_tr + n_va]] = 1
        link_split[perm[n_tr + n_va:]] = 2
        split = pd.Series(link_split, index=links).reindex(
            samples.target_link_id.to_numpy()).to_numpy()

    # normalisers computed on the train split only (no val/test leakage)
    log_len = np.log(samples.L_link_m.to_numpy(dtype=np.float64))
    mu, sd = log_len[split == 0].mean(), log_len[split == 0].std()
    L_n = ((log_len - mu) / sd).astype(np.float32)
    v_bar = samples.mean_speed.to_numpy(dtype=np.float32)
    y = samples.y_travel_s.to_numpy(dtype=np.float32)
    # z-scored log target: makes the output layer start at the mean-prediction
    # point (raw log_y mean ~2.6 burned ~100 steps and stalled the out_dim
    # arm entirely — see tools/debug_overfit.py)
    log_y = np.log(y)
    y_mu = float(log_y[split == 0].mean())
    y_sd = float(log_y[split == 0].std())

    # sample -> groups CSR from the (sample, group) edges of the profile table
    sg = meta.assign(gid=gi.group_idx).drop_duplicates(["sample_id", "gid"])
    row_of = pd.Index(samples.sample_id).get_indexer(sg.sample_id)
    # time-split may drop boundary samples: their profiles become unreferenced
    in_set = row_of >= 0
    sg, row_of = sg[in_set], row_of[in_set]
    if (np.bincount(row_of, minlength=len(samples)) == 0).any():
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
        y=y, td_s=samples.td_target.to_numpy(dtype=np.float32),
        log_y=log_y.astype(np.float32),
        z_y=((log_y - y_mu) / y_sd).astype(np.float32), y_mu=y_mu, y_sd=y_sd,
        L_m=samples.L_link_m.to_numpy(dtype=np.float32), L_n=L_n,
        v_n=(v_bar / float(cfg["v_norm"])).astype(np.float32), v_bar=v_bar,
        n_trajs_lw=samples.n_trajs.to_numpy(dtype=np.int32),
        split=split.astype(np.int8), samp_ptr=samp_ptr, samp_groups=samp_groups,
        split_mode=split_mode,
    )
