"""Streaming ingest: raw corridor parquet -> target-link tables, constant memory.

Same outputs as tools/ingest.py (samples / bins / link_window / stats) but
processes files one at a time and writes bins as sharded parquet, so peak RAM
is bounded by ONE file (~25M rows ≈ 2GB) instead of all 45 (~1.1B rows).

samples (one row per trajectory-pass) and link_window ((link,window) mean_speed)
are small enough to accumulate in memory; bins (the long 10m-bin table) are the
memory killer — each file's bins go straight to data/processed_h3/bins_shards/.

Usage: python tools/ingest_streaming.py --config configs/ingest_ts_day0821.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.utils import dump_json, load_config  # noqa: E402

RAW_COLS = [
    "sample_id", "target_link_id", "t_enter", "seg_mark", "bin_idx",
    "ratio", "T_diff", "observed",
    "y_travel_s", "L_link_m",
    "link_fc", "link_speed_class", "link_kind", "link_lane", "link_direction", "link_urban",
]
STATIC_COLS = [
    "y_travel_s", "L_link_m",
    "link_fc", "link_speed_class", "link_kind", "link_lane", "link_direction", "link_urban",
]


def process_file(path: Path, window_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter one raw file to (samples_df, bins_df). Identical to ingest.py."""
    pf = pq.ParquetFile(path)
    sample_chunks, bin_chunks = [], []
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg, columns=RAW_COLS).to_pandas()
        tgt = df[df.seg_mark == 1]
        if tgt.empty:
            continue
        tgt = tgt.assign(_td_ok=tgt.T_diff.notna() & (tgt.T_diff > 0))
        tgt = (
            tgt.sort_values(["sample_id", "bin_idx", "_td_ok", "ratio"], kind="stable")
            .drop_duplicates(["sample_id", "bin_idx"], keep="last")
            .sort_values(["sample_id", "bin_idx"], kind="stable")
            .drop(columns="_td_ok")
        )
        grp = tgt.groupby("sample_id", sort=False)
        agg = grp.agg(
            target_link_id=("target_link_id", "first"),
            td_target=("T_diff", "sum"),
            bin_idx_min=("bin_idx", "min"),
            n_bins_target=("bin_idx", "size"),
            n_observed=("observed", "sum"),
            eff_len_ratio=("ratio", "sum"),
            t_enter=("t_enter", "first"),
            **{c: (c, "first") for c in STATIC_COLS},
        )
        bins = tgt[["sample_id", "bin_idx", "ratio", "T_diff", "observed"]].copy()
        bins["rel_bin_idx"] = bins.bin_idx - bins.sample_id.map(agg.bin_idx_min).astype("int64")
        sample_chunks.append(agg)
        bin_chunks.append(bins)
    samples = pd.concat(sample_chunks, ignore_index=False)
    bins = pd.concat(bin_chunks, ignore_index=True)
    return samples.reset_index(), bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ingest_ts_day0821.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    p = cfg["params"]
    out = cfg["output"]
    out_dir = Path(cfg.get("project_root", ".")) / out["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "bins_shards"
    shard_dir.mkdir(exist_ok=True)
    # clear stale shards so reruns don't double-count
    for old in shard_dir.glob("*.parquet"):
        old.unlink()

    n_rows_raw = 0
    sample_list = []
    n_bins_total = 0
    n_bins_nan_td = 0
    n_samples_with_nan = set()
    files = cfg["input"]["files"]
    for i, rel in enumerate(files):
        path = Path(rel)
        pf = pq.ParquetFile(path)
        n_rows_raw += pf.metadata.num_rows
        s, b = process_file(path, p["window_s"])
        # track NaN-T_diff bins for stats (cheap, on this file only)
        nan_mask = b.T_diff.isna()
        n_bins_nan_td += int(nan_mask.sum())
        n_samples_with_nan.update(b.loc[nan_mask, "sample_id"].unique())
        # compact dtypes before writing shard (same as ingest.py bins_out)
        b = b[["sample_id", "rel_bin_idx", "ratio", "T_diff", "observed"]].astype(
            {"rel_bin_idx": "int32", "observed": "int8", "ratio": "float32", "T_diff": "float32"}
        )
        b.to_parquet(shard_dir / f"bins_{i:05d}.parquet", index=False)
        n_bins_total += len(b)
        sample_list.append(s)
        print(f"[ingest] {i + 1}/{len(files)} {path.name}: {len(s)} samples, {len(b)} bin rows "
              f"(cum bins {n_bins_total:,})", flush=True)
        del b  # free before next file

    # --- samples: small enough to concat in memory --------------------------
    samples = pd.concat(sample_list, ignore_index=True)
    del sample_list

    n_before = len(samples)
    keep = samples.td_target > p["min_td_target_s"]
    samples = samples[keep].copy()
    n_dropped = n_before - len(samples)

    samples["v_sample"] = samples.L_link_m / samples.td_target
    samples["observed_ratio"] = samples.n_observed / samples.n_bins_target.clip(lower=1)
    samples["window_id"] = (samples.t_enter // p["window_s"]).astype("int64")

    lw = samples.groupby(["target_link_id", "window_id"]).agg(
        mean_speed=("v_sample", "mean"),
        speed_std=("v_sample", "std"),
        n_trajs=("v_sample", "size"),
        y_travel_mean=("y_travel_s", "mean"),
    ).reset_index()

    samples_out = samples[[
        "sample_id", "target_link_id", "window_id", "t_enter", "td_target", "v_sample",
        "bin_idx_min", "n_bins_target", "observed_ratio", "eff_len_ratio",
    ] + STATIC_COLS]
    samples_out.to_parquet(out_dir / out["samples"], index=False)
    lw.to_parquet(out_dir / out["link_window"], index=False)

    q = lambda s, ps: [round(float(s.quantile(x)), 3) for x in ps]
    stats = {
        "n_files": len(files),
        "n_rows_raw": int(n_rows_raw),
        "n_rows_target_bins": int(n_bins_total),
        "n_samples": int(len(samples)),
        "n_dropped_min_td": int(n_dropped),
        "n_bins_nan_td": int(n_bins_nan_td),
        "n_samples_with_nan_td_bins": int(len(n_samples_with_nan)),
        "n_links": int(samples.target_link_id.nunique()),
        "n_windows": int(samples.window_id.nunique()),
        "window_hours_utc8": [
            str(pd.Timestamp(w * p["window_s"], unit="s", tz="Asia/Shanghai"))
            for w in sorted(samples.window_id.unique())
        ],
        "bins_per_sample_p10_p50_p90_mean": q(samples.n_bins_target, [0.1, 0.5, 0.9])
            + [round(float(samples.n_bins_target.mean()), 2)],
        "observed_ratio_p10_p50_p90": q(samples.observed_ratio, [0.1, 0.5, 0.9]),
        "v_sample_mps_p10_p50_p90": q(samples.v_sample, [0.1, 0.5, 0.9]),
        "L_link_m_p10_p50_p90": q(samples.L_link_m, [0.1, 0.5, 0.9]),
        "y_travel_s_p10_p50_p90": q(samples.y_travel_s, [0.1, 0.5, 0.9]),
        "trajs_per_link_window_p10_p50_p90": q(lw.n_trajs, [0.1, 0.5, 0.9]),
        "bins_shard_dir": str(shard_dir),
        "n_bins_shards": len(files),
        "note": "streaming ingest: bins written as per-file shards, not concatenated",
    }
    dump_json(stats, out_dir / out["stats"])
    print(f"[ingest] wrote {out_dir}/ : {len(samples)} samples, {n_bins_total} bin rows "
          f"({len(files)} shards), {len(lw)} link-windows")
    print(f"[ingest] stats -> {out_dir / out['stats']}")
    print(f"[ingest] bins shards -> {shard_dir} (read via pyarrow.dataset in build_profiles)")


if __name__ == "__main__":
    main()
