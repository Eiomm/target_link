"""Ingest raw corridor parquet -> target-link tables + link-window mean_speed.

Reads the raw HDFS-pulled corridor files, keeps only seg_mark==1 rows (target
link body), derives the per-sample target traversal speed, and writes:

  samples.parquet     one row per sample (window_id, y_travel_s, v_sample, ...)
  bins.parquet        bin-level long table (rel position, ratio, T_diff, observed)
  link_window.parquet (target_link_id, window_id) mean_speed baseline feature
  ingest_stats.json   basic stats required by spec §11

Usage: python tools/ingest.py --config configs/ingest.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.utils import dump_json, load_config  # noqa: E402

# Columns pulled from raw files (sample-level statics are taken from seg_mark==1 first).
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
    """Filter one raw file to (samples_df, bins_df)."""
    pf = pq.ParquetFile(path)
    sample_chunks, bin_chunks = [], []
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg, columns=RAW_COLS).to_pandas()
        tgt = df[df.seg_mark == 1]
        if tgt.empty:
            continue
        # dedupe (sample_id, bin_idx): overlapping corridor segments emit the
        # same bin twice; keep the row with valid T_diff and the largest ratio
        # (the segment owning most of the bin).
        tgt = tgt.assign(_td_ok=tgt.T_diff.notna() & (tgt.T_diff > 0))
        tgt = (
            tgt.sort_values(["sample_id", "bin_idx", "_td_ok", "ratio"], kind="stable")
            .drop_duplicates(["sample_id", "bin_idx"], keep="last")
            .sort_values(["sample_id", "bin_idx"], kind="stable")
            .drop(columns="_td_ok")
        )
        # per-sample aggregation over target bins
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
        # relative bin position inside the target segment (0-based)
        bins["rel_bin_idx"] = bins.bin_idx - bins.sample_id.map(agg.bin_idx_min).astype("int64")
        sample_chunks.append(agg)
        bin_chunks.append(bins)
    samples = pd.concat(sample_chunks, ignore_index=False)
    bins = pd.concat(bin_chunks, ignore_index=True)
    return samples.reset_index(), bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ingest.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    p = cfg["params"]
    out = cfg["output"]
    out_dir = Path(cfg.get("project_root", ".")) / out["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows_raw = 0
    sample_list, bin_list = [], []
    for rel in cfg["input"]["files"]:
        path = Path(rel)
        pf = pq.ParquetFile(path)
        n_rows_raw += pf.metadata.num_rows
        s, b = process_file(path, p["window_s"])
        sample_list.append(s)
        bin_list.append(b)
        print(f"[ingest] {path.name}: {len(s)} samples, {len(b)} target-bin rows")

    samples = pd.concat(sample_list, ignore_index=True)
    bins = pd.concat(bin_list, ignore_index=True)

    # quality filter: non-positive / tiny target traversal time
    n_before = len(samples)
    keep = samples.td_target > p["min_td_target_s"]
    samples = samples[keep].copy()
    keep_ids = set(samples.sample_id)
    bins = bins[bins.sample_id.isin(keep_ids)]
    n_dropped = n_before - len(samples)

    # per-sample derived columns
    samples["v_sample"] = samples.L_link_m / samples.td_target
    samples["observed_ratio"] = samples.n_observed / samples.n_bins_target.clip(lower=1)
    samples["window_id"] = (samples.t_enter // p["window_s"]).astype("int64")

    # (target_link_id, window_id) baseline feature: production mean speed
    lw = samples.groupby(["target_link_id", "window_id"]).agg(
        mean_speed=("v_sample", "mean"),
        speed_std=("v_sample", "std"),
        n_trajs=("v_sample", "size"),
        y_travel_mean=("y_travel_s", "mean"),
    ).reset_index()

    # persist (samples: sample-level table; bins: keep only needed cols, compact dtypes)
    samples_out = samples[[
        "sample_id", "target_link_id", "window_id", "t_enter", "td_target", "v_sample",
        "bin_idx_min", "n_bins_target", "observed_ratio", "eff_len_ratio",
    ] + STATIC_COLS]
    bins_out = bins[["sample_id", "rel_bin_idx", "ratio", "T_diff", "observed"]].astype(
        {"rel_bin_idx": "int32", "observed": "int8", "ratio": "float32", "T_diff": "float32"}
    )
    samples_out.to_parquet(out_dir / out["samples"], index=False)
    bins_out.to_parquet(out_dir / out["bins"], index=False)
    lw.to_parquet(out_dir / out["link_window"], index=False)

    # spec §11 basic stats
    q = lambda s, ps: [round(float(s.quantile(x)), 3) for x in ps]
    stats = {
        "n_files": len(cfg["input"]["files"]),
        "n_rows_raw": int(n_rows_raw),
        "n_rows_target_bins": int(len(bins_out)),
        "n_samples": int(len(samples)),
        "n_dropped_min_td": int(n_dropped),
        "n_bins_nan_td": int(bins_out.T_diff.isna().sum()),
        "n_samples_with_nan_td_bins": int(
            bins_out.loc[bins_out.T_diff.isna(), "sample_id"].nunique()
        ),
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
    }
    dump_json(stats, out_dir / out["stats"])
    print(f"[ingest] wrote {out_dir}/ : {len(samples)} samples, "
          f"{len(bins_out)} bin rows, {len(lw)} link-windows")
    print(f"[ingest] stats -> {out_dir / out['stats']}")


if __name__ == "__main__":
    main()
