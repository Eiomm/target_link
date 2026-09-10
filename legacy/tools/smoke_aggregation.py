"""Smoke test for vehicle-to-link mean aggregation (spec §8) on real profiles.

Checks: group-index stats on real data, scatter-mean == direct per-group mean,
K=1 identity, gradient flow encoder <- scatter_mean, and grouping-key sanity —
for single-sub links the aggregated sub-level mean speed must track the
production link_window.mean_speed.

Usage: python legacy/tools/smoke_aggregation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_link_v1.data import build_group_index, group_aligned_chunks, sort_by_group  # noqa: E402
from target_link_v1.models import LinkAggregator, TrajectoryEncoder, scatter_mean  # noqa: E402
from target_link_v1.utils import seed_everything  # noqa: E402

MAX_ROWS = 65536  # encoder forward chunk (group-aligned)


def main() -> None:
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # --- group real profiles by (link_id, sub_id, window_id) ------------------
    d = np.load("data/processed/profiles_l200.npz")
    gi = build_group_index(d["link_id"], d["sub_id"], d["window_id"])
    k = gi.keys.n_trajs.to_numpy()
    print(f"[agg] {len(gi.group_idx)} profiles -> {len(k)} (sub-link, window) groups "
          f"({gi.keys.link_id.nunique()} links, {gi.keys.window_id.nunique()} windows)")
    print(f"[agg] trajs/group: p50={np.median(k):.0f} p90={np.percentile(k, 90):.0f} "
          f"max={k.max()} | K=1 groups: {(k == 1).mean():.1%}")

    # --- sort rows by group so chunks never split a group ---------------------
    order = sort_by_group(gi.group_idx)
    gid_np = gi.group_idx[order]
    speeds = torch.from_numpy(d["speeds"][order]).to(device)
    valid = torch.from_numpy(d["valid"][order]).to(device)
    lengths = torch.from_numpy(d["lengths"][order]).to(device)
    group_idx = torch.from_numpy(gid_np).to(device)
    n_groups = len(k)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(gid_np)) + 1, [len(gid_np)]))

    # --- encode all profiles (eval, chunked), aggregate with scatter-mean ------
    enc = TrajectoryEncoder().to(device).eval()
    agg = LinkAggregator()
    with torch.no_grad():
        r = torch.cat([
            enc(speeds[rows], valid[rows], lengths[rows])
            for rows in group_aligned_chunks(gid_np, MAX_ROWS)
        ])
    r_link = agg(r, group_idx, n_groups)
    assert r_link.shape == (n_groups, 128) and torch.isfinite(r_link).all()
    print(f"[agg] forward OK: r_link {tuple(r_link.shape)}, finite, "
          f"|r_link| mean={r_link.norm(dim=-1).mean():.3f}")

    # --- scatter-mean == direct mean over member rows (independent recompute) --
    r_np = r.cpu().numpy()
    for g in rng.choice(n_groups, size=256, replace=False):
        ref = r_np[bounds[g]:bounds[g + 1]].mean(axis=0)
        assert np.allclose(r_link[g].cpu().numpy(), ref, atol=1e-4), g
    print("[agg] scatter-mean == direct group mean for 256 random groups")

    # --- K == 1 groups: aggregation must be the identity ----------------------
    for g in rng.choice(np.flatnonzero(k == 1), size=1000, replace=False):
        assert torch.allclose(r_link[g], r[bounds[g]], atol=1e-6), g
    print(f"[agg] K=1 identity OK ({(k == 1).mean():.0%} of groups)")

    # --- backward: loss on r_link reaches the encoder through scatter-mean -----
    # (train-mode attention kernels cap the batch at 65535, hence the smaller chunk)
    rows = group_aligned_chunks(gid_np, 16384)[0]
    enc.train()
    r1 = enc(speeds[rows], valid[rows], lengths[rows])
    scatter_mean(r1, group_idx[rows], int(gid_np[rows].max()) + 1).pow(2).mean().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    print(f"[agg] backward OK: {len(grads)} grad tensors finite, nonzero")

    # --- grouping-key sanity: sub-level v_bar vs production link_window --------
    # For single-sub links the sub IS the link, so the K-traj mean of per-profile
    # valid-mean bin speeds must track link_window.mean_speed (arithmetic vs
    # production L/sum(T_diff) averaging -> high but not perfect correlation).
    with torch.no_grad():
        v_traj = (speeds.sum(dim=1) / valid.sum(dim=1).clamp(min=1)).cpu().numpy()
    v_bar = scatter_mean(
        torch.from_numpy(v_traj).to(device).unsqueeze(1), group_idx, n_groups
    ).squeeze(1).cpu().numpy()
    sub_map = pd.read_parquet("data/processed/link_sub_map_l200.parquet")
    n_subs = sub_map.groupby("target_link_id").size()
    single_links = n_subs[n_subs == 1].index
    keys = gi.keys.assign(v_bar=v_bar)
    chk = keys[keys.link_id.isin(single_links)].merge(
        pd.read_parquet("data/processed/link_window.parquet")[
            ["target_link_id", "window_id", "mean_speed"]
        ],
        left_on=["link_id", "window_id"], right_on=["target_link_id", "window_id"],
        how="inner",
    )
    corr = chk.v_bar.corr(chk.mean_speed)
    rel = (chk.v_bar - chk.mean_speed).abs().median() / chk.mean_speed.median()
    print(f"[agg] single-sub links: {len(chk)} groups | corr(v_bar, mean_speed)="
          f"{corr:.3f}, median rel dev={rel:.1%}")
    assert corr > 0.9, corr

    print("[agg] all checks passed")


if __name__ == "__main__":
    main()
