"""Build padded spatial motion profiles + sub-link split from ingested bins.

For each (sample, sub-link) pair produces the 10m-bin speed sequence of spec §4:

  v_i = bin_size_m * ratio_i / T_diff_i      (T_diff covers the ratio share of
                                              the bin that belongs to the link
                                              — verified against v_sample)

invalid bins (NaN/<=0 T_diff, or speed above the physical cap) get m_i=0 and
speed 0 in the padded array. Sub-link split follows spec §3: links longer than
L_sub are cut at cumulative ratio distance; the last sub keeps its true length.

Outputs (per L):
  profiles_l{L}.npz          speeds [N, max_bins] f32, valid bool, lengths,
                             observed flags + per-row metadata
  link_sub_map_l{L}.parquet  original link id <-> sub-link bins/length
  profile_stats_l{L}.json    stats (spec §11)

Usage: python tools/build_profiles.py --config configs/profiles.yaml [--l-sub 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.utils import dump_json, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles.yaml")
    parser.add_argument("--l-sub", type=float, default=None,
                        help="override params.l_sub_m (Ablation 3 sweep)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    p = cfg["params"]
    if args.l_sub is not None:
        p["l_sub_m"] = args.l_sub
    L = int(p["l_sub_m"])
    max_bins = int(p["max_bins"])

    in_dir, out_dir = Path(cfg["input"]["dir"]), Path(cfg["output"]["dir"])
    samples = pd.read_parquet(in_dir / cfg["input"]["samples"])
    bins = pd.read_parquet(in_dir / cfg["input"]["bins"])

    # --- bin-level local speed and validity ---------------------------------
    bins = bins.sort_values(["sample_id", "rel_bin_idx"], kind="stable").reset_index(drop=True)
    td = bins.T_diff.to_numpy(dtype=np.float64)
    ratio = bins.ratio.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p["bin_size_m"] * ratio / td
    invalid = ~np.isfinite(v) | (v <= 0) | (v > p["v_invalid_above"])
    bins["v"] = np.where(invalid, 0.0, v).astype(np.float32)
    bins["invalid"] = invalid

    # --- sub-link assignment (spec §3) --------------------------------------
    # bin start position along the link, from cumulative ratio share
    s_start = (bins.groupby("sample_id", sort=False).ratio.cumsum() - bins.ratio) * p["bin_size_m"]
    n_subs_link = np.ceil(samples.set_index("sample_id").L_link_m / L).clip(lower=1)
    bins["sub_id"] = np.clip(
        np.floor(s_start / L).astype("int32"), 0, bins.sample_id.map(n_subs_link).to_numpy() - 1
    ).astype("int32")

    # --- group rows by (sample, sub_id) and scatter into padded arrays -------
    key = bins.sample_id + "#" + bins.sub_id.astype(str)
    grp_id, _ = pd.factorize(key)
    seq_idx = bins.groupby(key, sort=False).cumcount().to_numpy()
    lengths = np.bincount(grp_id, minlength=grp_id.max() + 1)
    if lengths.max() > max_bins:
        raise ValueError(
            f"longest sub-link profile has {lengths.max()} bins > max_bins={max_bins}; "
            "increase max_bins or lower l_sub_m"
        )
    N = len(lengths)
    speeds = np.zeros((N, max_bins), dtype=np.float32)
    valid = np.zeros((N, max_bins), dtype=bool)
    observed = np.zeros((N, max_bins), dtype=bool)
    speeds[grp_id, seq_idx] = bins.v
    valid[grp_id, seq_idx] = ~bins.invalid.to_numpy()
    observed[grp_id, seq_idx] = bins.observed.to_numpy() == 1

    # --- per-profile metadata ------------------------------------------------
    meta = bins.groupby(grp_id, sort=True)[["sample_id", "sub_id"]].first()
    meta = meta.join(
        samples.set_index("sample_id")[[
            "target_link_id", "window_id", "y_travel_s", "v_sample", "td_target", "n_bins_target"
        ]],
        on="sample_id",
    ).reset_index(drop=True)

    # --- link <-> sub-link map (one row per unique sub-link) ------------------
    per_profile = (
        bins.groupby(["sample_id", "sub_id"])
        .agg(n_bins=("v", "size"), eff_len_ratio=("ratio", "sum"))
        .reset_index()
        .merge(samples[["sample_id", "target_link_id"]], on="sample_id")
    )
    sub_map = (
        per_profile.groupby(["target_link_id", "sub_id"])
        .agg(n_bins=("n_bins", "median"), eff_len_ratio=("eff_len_ratio", "median"),
             n_profiles=("sample_id", "size"))
        .reset_index()
    )
    sub_map["n_bins"] = sub_map.n_bins.astype(int)
    sub_map["eff_len_m"] = (sub_map.eff_len_ratio * p["bin_size_m"]).round(1)
    sub_map = sub_map.drop(columns="eff_len_ratio")

    # --- persist ---------------------------------------------------------------
    out_npz = out_dir / cfg["output"]["profiles_file"].format(L=L)
    np.savez_compressed(
        out_npz,
        speeds=speeds, valid=valid, observed=observed, lengths=lengths.astype(np.int32),
        sample_id=meta.sample_id.to_numpy(dtype=object).astype("U"),
        link_id=meta.target_link_id.to_numpy(dtype=object).astype("U"),
        window_id=meta.window_id.to_numpy(dtype=np.int64),
        sub_id=meta.sub_id.to_numpy(dtype=np.int32),
        y_travel_s=meta.y_travel_s.to_numpy(dtype=np.float32),
        v_sample=meta.v_sample.to_numpy(dtype=np.float32),
        td_target=meta.td_target.to_numpy(dtype=np.float32),
        n_bins_target=meta.n_bins_target.to_numpy(dtype=np.int32),
    )
    sub_map.to_parquet(out_dir / cfg["output"]["sub_map_file"].format(L=L), index=False)

    # --- stats (spec §11) ------------------------------------------------------
    q = lambda a, ps: [round(float(np.nanpercentile(a, x)), 3) for x in ps]
    stats = {
        "l_sub_m": L,
        "max_bins": max_bins,
        "n_profiles": int(N),
        "n_samples_in": int(len(samples)),
        "n_links": int(meta.target_link_id.nunique()),
        "n_sub_links": int(len(sub_map)),
        "n_link_window_keys": int(meta.drop_duplicates(["target_link_id", "window_id"]).shape[0]),
        "bins_per_profile_p10_p50_p90_mean": q(lengths, [10, 50, 90]) + [round(float(lengths.mean()), 2)],
        "valid_ratio_p10_p50_p90": q(valid.sum(axis=1) / np.maximum(lengths, 1), [10, 50, 90]),
        "n_invalid_bins": int(bins.invalid.sum()),
        "n_invalid_bad_td": int((~(np.isfinite(td) & (td > 0))).sum()),
        "n_invalid_over_speed": int(bins.invalid.sum() - (~(np.isfinite(td) & (td > 0))).sum()),
        "profiles_per_sample_mean": round(float(N / len(samples)), 3),
        "v_bin_mps_p10_p50_p90": q(speeds[valid], [10, 50, 90]),
    }
    dump_json(stats, out_dir / cfg["output"]["stats"].format(L=L))
    print(f"[profiles] L_sub={L}m -> {N} profiles ({meta.target_link_id.nunique()} links, "
          f"{len(sub_map)} sub-links), pad={max_bins}")
    print(f"[profiles] wrote {out_npz.name}, {len(sub_map)}-row sub-map, stats json")


if __name__ == "__main__":
    main()
