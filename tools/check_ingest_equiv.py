"""Equivalence check between two ingest outputs (e.g. pandas ref vs spark).

Compares samples / bins / link_window on the semantic keys, reporting exact
match rates for integers/strings and tolerance-based diffs for floats (Spark
sums double-order differently than pandas, so tiny drift is expected).

Usage:
  python tools/check_ingest_equiv.py --ref data/processed --new data/processed_spark_smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_samples(d: Path) -> pd.DataFrame:
    return pd.read_parquet(d / "samples.parquet").sort_values("sample_id") \
        .reset_index(drop=True)


def load_bins(d: Path) -> pd.DataFrame:
    files = sorted((d / "bins_shards").glob("*.parquet")) if (d / "bins_shards").is_dir() \
        else [d / "bins.parquet"]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["sample_id", "rel_bin_idx"]).reset_index(drop=True)


def col_diff(a: pd.Series, b: pd.Series, atol: float) -> tuple[int, float]:
    """(n_mismatch, max_abs_diff); NaN==NaN counts as equal."""
    if pd.api.types.is_float_dtype(a):
        d = np.abs(a.to_numpy(np.float64) - b.to_numpy(np.float64))
        bad = ~(np.isnan(a.to_numpy()) & np.isnan(b.to_numpy()))
        d = np.where(bad, d, 0.0)
        n = int((d > atol).sum())
        return n, float(d.max()) if len(d) else 0.0
    n = int((a.to_numpy() != b.to_numpy()).sum())
    return n, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference processed dir (pandas ingest)")
    ap.add_argument("--new", required=True, help="new processed dir (spark ingest)")
    args = ap.parse_args()
    ref, new = Path(args.ref), Path(args.new)

    print("== samples ==")
    a, b = load_samples(ref), load_samples(new)
    print(f"rows: ref {len(a):,} / new {len(b):,}")
    if len(a) != len(b):
        only_a = set(a.sample_id) - set(b.sample_id)
        only_b = set(b.sample_id) - set(a.sample_id)
        print(f"  !! sample_id only in ref: {len(only_a)}; only in new: {len(only_b)}")
        print("  sample only_ref:", list(only_a)[:3], " only_new:", list(only_b)[:3])
        return
    if not a.sample_id.tolist() == b.sample_id.tolist():
        print("  !! sample_id sets equal but ORDER differs (fine — key-based compare)")
        b = b.set_index("sample_id").loc[a.sample_id].reset_index()
    ok = True
    for c, atol in [("target_link_id", 0), ("window_id", 0), ("bin_idx_min", 0),
                    ("n_bins_target", 0), ("t_enter", 1e-6), ("td_target", 1e-4),
                    ("v_sample", 1e-6), ("observed_ratio", 1e-9), ("eff_len_ratio", 1e-4),
                    ("y_travel_s", 1e-9), ("L_link_m", 1e-9)]:
        if c not in a.columns or c not in b.columns:
            continue
        n, mx = col_diff(a[c], b[c], atol)
        flag = "" if n == 0 else f"  <-- {n} rows differ"
        print(f"  {c:16s} mismatch {n:>8,}  maxdiff {mx:.3e}{flag}")
        ok &= n == 0

    print("== bins ==")
    ba, bb = load_bins(ref), load_bins(new)
    print(f"rows: ref {len(ba):,} / new {len(bb):,}")
    if len(ba) != len(bb):
        ka = set(zip(ba.sample_id, ba.rel_bin_idx))
        kb = set(zip(bb.sample_id, bb.rel_bin_idx))
        print(f"  !! key diff: only ref {len(ka - kb):,}, only new {len(kb - ka):,}")
        return
    merged = ba.merge(bb, on=["sample_id", "rel_bin_idx"], suffixes=("_r", "_n"))
    for c, atol in [("ratio", 0), ("observed", 0), ("T_diff", 1e-4)]:
        n, mx = col_diff(merged[f"{c}_r"], merged[f"{c}_n"], atol)
        flag = "" if n == 0 else f"  <-- {n} rows differ"
        print(f"  {c:16s} mismatch {n:>8,}  maxdiff {mx:.3e}{flag}")
        ok &= n == 0

    print("== link_window ==")
    la = pd.read_parquet(ref / "link_window.parquet")
    lb = pd.read_parquet(new / "link_window.parquet")
    m = la.merge(lb, on=["target_link_id", "window_id"], suffixes=("_r", "_n"))
    print(f"rows: ref {len(la):,} / new {len(lb):,} / joined {len(m):,}")
    for c, atol in [("n_trajs", 0), ("mean_speed", 1e-6), ("y_travel_mean", 1e-9)]:
        if f"{c}_r" not in m.columns:
            continue
        n, mx = col_diff(m[f"{c}_r"], m[f"{c}_n"], atol)
        flag = "" if n == 0 else f"  <-- {n} rows differ"
        print(f"  {c:16s} mismatch {n:>8,}  maxdiff {mx:.3e}{flag}")
        ok &= n == 0

    print("EQUIVALENT ✅" if ok else "DIVERGENT ❌ (see flags above)")


if __name__ == "__main__":
    main()
