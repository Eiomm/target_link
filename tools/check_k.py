"""K-distribution audit for a processed data dir (single-file vs full-hour).

Reports, for --data data/processed and/or data/processed_h3:
  - per-window sample/link counts
  - K distribution at (link, window) level (link_window.n_trajs)
  - K distribution at (sub-link, window) level (profiles npz groups)
  - link overlap across windows (fraction of links seen in >1 window)
  - sub-level K == link-level K invariance (L_sub does not split K)

Usage: python tools/check_k.py --data data/processed_h3 [--compare data/processed]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def k_stats(k: np.ndarray) -> str:
    p = np.percentile(k, [50, 90, 99]).round(1)
    return (f"p50 {p[0]}  p90 {p[1]}  p99 {p[2]}  max {k.max()}  "
            f"K=1 {(k == 1).mean() * 100:.1f}%  K>=2 {(k >= 2).mean() * 100:.1f}%  "
            f"mean {k.mean():.2f}")


def audit(tag: str, d: Path, l_sub: int = 200) -> None:
    print(f"\n===== {tag}: {d} =====")
    s = pd.read_parquet(d / "samples.parquet", columns=["sample_id", "target_link_id", "window_id"])
    lw = pd.read_parquet(d / "link_window.parquet")
    per_w = s.groupby("window_id").size()
    print(f"samples {len(s):,}  links {s.target_link_id.nunique():,}  windows {len(per_w)}")
    print("per-window samples:", dict(per_w))

    k = lw.n_trajs.to_numpy()
    print(f"[K @ (link,window)]  {k_stats(k)}")

    npz = d / f"profiles_l{l_sub}.npz"
    if npz.exists():
        z = np.load(npz)
        meta = pd.DataFrame({
            "sample_id": z["sample_id"].astype(str), "link_id": z["link_id"].astype(str),
            "window_id": z["window_id"].astype(np.int64), "sub_id": z["sub_id"].astype(np.int64),
        })
        g = meta.groupby(["link_id", "sub_id", "window_id"]).size().to_numpy()
        print(f"[K @ (sub,window) L_sub={l_sub}]  {k_stats(g)}")

        # L_sub invariance: every sample covers all subs of its link?
        nsub_s = meta.groupby(["link_id", "window_id", "sample_id"]).sub_id.nunique()
        nsub_l = meta.groupby(["link_id", "window_id"]).sub_id.nunique()
        cov = nsub_s.reset_index().merge(nsub_l.rename("n").reset_index(), on=["link_id", "window_id"])
        print(f"sample covers ALL subs of its link: {(cov.sub_id == cov.n).mean() * 100:.2f}%")

    both = lw.groupby("target_link_id").window_id.nunique()
    print(f"links in >1 window: {(both > 1).mean() * 100:.1f}%  (multi-window links: {(both > 1).sum():,})")

    # duplicate sample check (sharding should keep each sample in exactly one file)
    dup = s.sample_id.duplicated().sum()
    print(f"duplicate sample_id rows: {dup}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed_h3")
    ap.add_argument("--compare", default=None, help="second data dir to print first")
    ap.add_argument("--l-sub", type=int, default=200)
    a = ap.parse_args()
    if a.compare:
        audit("OLD (single file)", Path(a.compare), a.l_sub)
    audit("NEW", Path(a.data), a.l_sub)
