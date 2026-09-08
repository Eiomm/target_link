"""Build padded spatial motion profiles + sub-link split from ingested bins.

For each (sample, sub-link) pair produces the 10m-bin speed sequence of spec §4:

  v_i = bin_size_m * ratio_i / T_diff_i      (T_diff covers the ratio share of
                                              the bin that belongs to the link
                                              — verified against v_sample)

invalid bins (NaN/<=0 T_diff, or speed above the physical cap) get m_i=0 and
speed 0 in the padded array. Sub-link split follows spec §3: links longer than
L_sub are cut at cumulative ratio distance; the last sub keeps its true length.

Streaming: bins are processed one shard at a time (trajectories are file-sharded
by ingest, so a sample's bins never span shards and shard-local grouping is
exact), and the final npz is written member-by-member straight from the shard
pieces. Peak RSS ~ single shard + accumulated outputs, well under the 30GB pod
memcg limit that killed the full-load version on day20 (378M bin rows).

Outputs (per L):
  profiles_l{L}.npz          speeds [N, max_bins] f32, valid bool, lengths,
                             observed flags + per-row metadata
  link_sub_map_l{L}.parquet  original link id <-> sub-link bins/length
  profile_stats_l{L}.json    stats (spec §11)

Usage: python tools/build_profiles.py --config configs/profiles.yaml [--l-sub 200]
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.utils import dump_json, load_config  # noqa: E402

# sample columns needed beyond the group key (joined onto every profile row)
META_COLS = ["target_link_id", "window_id", "y_travel_s", "v_sample", "td_target", "n_bins_target"]

try:  # return freed arenas to the OS so memcg RSS tracks live data (glibc only)
    import ctypes

    _libc = ctypes.CDLL("libc.so.6")

    def _trim() -> None:
        _libc.malloc_trim(0)
except OSError:  # pragma: no cover
    def _trim() -> None:
        pass


def process_shard(bins: pd.DataFrame, samples_idx: pd.DataFrame, p: dict, L: int, max_bins: int) -> dict:
    """Compute per-(sample, sub) profile pieces for one bins shard.

    Returns numpy arrays (speeds/valid/observed/lengths/eff_len), a meta
    DataFrame (one row per profile) and bin-level invalid counters. All
    formulas match the previous single-load implementation bit-for-bit.
    """
    n_rows = len(bins)
    if n_rows == 0:
        return {
            "speeds": np.zeros((0, max_bins), dtype=np.float32),
            "valid": np.zeros((0, max_bins), dtype=bool),
            "observed": np.zeros((0, max_bins), dtype=bool),
            "lengths": np.zeros(0, dtype=np.int32),
            "meta": pd.DataFrame(columns=["sample_id", "sub_id"] + META_COLS),
            "eff_len": np.zeros(0, dtype=np.float32),
            "n_invalid": 0, "n_bad_td": 0,
        }
    codes, uniq = pd.factorize(bins.sample_id, sort=False)
    rel = bins.rel_bin_idx.to_numpy()

    # ingest writes rows in (sample_id, bin_idx) order; sort only if violated
    order = np.lexsort((rel, codes))
    if not np.array_equal(order, np.arange(n_rows)):
        codes = codes[order]
        rel = rel[order]
        ratio = bins.ratio.to_numpy(dtype=np.float64)[order]
        td = bins.T_diff.to_numpy(dtype=np.float64)[order]
        observed = bins.observed.to_numpy()[order]
    else:
        ratio = bins.ratio.to_numpy(dtype=np.float64)
        td = bins.T_diff.to_numpy(dtype=np.float64)
        observed = bins.observed.to_numpy()

    # --- bin-level local speed and validity ---------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p["bin_size_m"] * ratio / td
    bad_td = ~(np.isfinite(td) & (td > 0))
    invalid = ~np.isfinite(v) | (v <= 0) | (v > p["v_invalid_above"])
    v32 = np.where(invalid, 0.0, v).astype(np.float32)

    # --- sub-link assignment (spec §3) --------------------------------------
    # bin start position along the link, from cumulative ratio share
    # (groupby cumsum keeps the exact float op order of the old version)
    cum = pd.Series(ratio).groupby(codes, sort=False).cumsum().to_numpy()
    s_start = (cum - ratio) * p["bin_size_m"]

    L_link = samples_idx.L_link_m.reindex(pd.Index(uniq)).to_numpy(dtype=np.float64)
    orphan = ~np.isfinite(L_link)
    if orphan.any():  # bins referencing samples dropped from the samples table
        row_keep = ~orphan[codes]
        print(f"[profiles]   WARNING: dropping {int((~row_keep).sum())} bin rows with "
              f"{int(orphan.sum())} orphan sample_ids", flush=True)
        codes, rel, ratio, td, observed = (a[row_keep] for a in (codes, rel, ratio, td, observed))
        cum = cum[row_keep]
        v32, invalid, bad_td = v32[row_keep], invalid[row_keep], bad_td[row_keep]
        s_start = s_start[row_keep]
        n_rows = len(codes)

    n_subs = np.maximum(np.ceil(L_link / L), 1.0)
    sub = np.clip(
        np.floor(s_start / L).astype(np.int64), 0, n_subs[codes].astype(np.int64) - 1
    ).astype(np.int32)

    # --- contiguous (sample, sub) blocks -> one profile each ----------------
    new_grp = np.empty(n_rows, dtype=bool)
    new_grp[0] = True
    np.not_equal(codes[1:], codes[:-1], out=new_grp[1:])
    new_grp[1:] |= sub[1:] != sub[:-1]  # sub is non-decreasing within a sample
    grp = np.cumsum(new_grp) - 1
    head = np.flatnonzero(new_grp)
    lengths = np.diff(np.append(head, n_rows))
    if lengths.max() > max_bins:
        raise ValueError(
            f"longest sub-link profile has {lengths.max()} bins > max_bins={max_bins}; "
            "increase max_bins or lower l_sub_m"
        )
    seq_idx = np.arange(n_rows) - np.repeat(head, lengths)

    # --- scatter into padded arrays ------------------------------------------
    N = len(lengths)
    speeds = np.zeros((N, max_bins), dtype=np.float32)
    valid = np.zeros((N, max_bins), dtype=bool)
    observed_pad = np.zeros((N, max_bins), dtype=bool)
    speeds[grp, seq_idx] = v32
    valid[grp, seq_idx] = ~invalid
    observed_pad[grp, seq_idx] = observed == 1

    # --- per-profile metadata + ratio sum (sub-map inputs) -------------------
    meta = pd.DataFrame({"sample_id": uniq[codes[head]], "sub_id": sub[head]})
    meta = meta.join(samples_idx[META_COLS], on="sample_id").reset_index(drop=True)
    eff_len = np.add.reduceat(ratio, head).astype(np.float32)

    return {
        "speeds": speeds, "valid": valid, "observed": observed_pad,
        "lengths": lengths.astype(np.int32), "meta": meta, "eff_len": eff_len,
        "n_invalid": int(invalid.sum()), "n_bad_td": int(bad_td.sum()),
    }


def _write_member_2d(zf: zipfile.ZipFile, name: str, pieces: list[np.ndarray]) -> int:
    """Write row-block pieces as one .npy zip member without concatenating."""
    n = sum(p.shape[0] for p in pieces)
    with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
        np.lib.format.write_array_header_1_0(f, {
            "descr": np.lib.format.dtype_to_descr(pieces[0].dtype),
            "fortran_order": False, "shape": (n, pieces[0].shape[1]),
        })
        for p in pieces:
            f.write(memoryview(p.data).cast("B"))  # zero-copy
    return n


def _write_member(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, arr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles.yaml")
    parser.add_argument("--l-sub", type=float, default=None,
                        help="override params.l_sub_m (Ablation 3 sweep)")
    parser.add_argument("--input-dir", default=None,
                        help="override cfg input.dir (k8s-job entrypoint)")
    parser.add_argument("--output-dir", default=None,
                        help="override cfg output.dir (k8s-job entrypoint)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.input_dir:
        cfg["input"]["dir"] = args.input_dir
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    p = cfg["params"]
    if args.l_sub is not None:
        p["l_sub_m"] = args.l_sub
    L = int(p["l_sub_m"])
    max_bins = int(p["max_bins"])

    in_dir, out_dir = Path(cfg["input"]["dir"]), Path(cfg["output"]["dir"])
    samples = pd.read_parquet(in_dir / cfg["input"]["samples"],
                              columns=["sample_id", "L_link_m"] + META_COLS)
    # bins may be a single parquet or a glob over streaming-ingest shards
    bins_spec = str(in_dir / cfg["input"]["bins"])
    bins_paths = sorted(glob.glob(bins_spec)) or ([bins_spec] if Path(bins_spec).exists() else [])
    if not bins_paths:
        raise FileNotFoundError(f"no bins found for {bins_spec}")
    n_samples_in = len(samples)
    samples_idx = samples.drop_duplicates("sample_id").set_index("sample_id")
    del samples

    # --- shard loop: accumulate profile pieces -------------------------------
    acc: dict[str, list] = {k: [] for k in ("speeds", "valid", "observed", "lengths", "meta", "eff_len")}
    per_profile: list[pd.DataFrame] = []   # (link, sub, n_bins, eff_len_ratio) rows
    lw_pairs: list[pd.DataFrame] = []      # unique (link, window) per shard
    valid_speeds: list[np.ndarray] = []    # bin-level speeds of valid bins (stats)
    valid_ratio: list[np.ndarray] = []     # per-profile valid fraction (stats)
    n_invalid = n_bad_td = 0
    for i, path in enumerate(bins_paths):
        bins = pd.read_parquet(path)
        sh = process_shard(bins, samples_idx, p, L, max_bins)
        del bins
        for k in acc:
            acc[k].append(sh[k])
        per_profile.append(pd.DataFrame({
            "target_link_id": sh["meta"].target_link_id.to_numpy(),
            "sub_id": sh["meta"].sub_id.to_numpy(),
            "n_bins": sh["lengths"],
            "eff_len_ratio": sh["eff_len"],
        }))
        lw_pairs.append(sh["meta"][["target_link_id", "window_id"]].drop_duplicates())
        valid_speeds.append(sh["speeds"][sh["valid"]])
        valid_ratio.append(sh["valid"].sum(axis=1) / np.maximum(sh["lengths"], 1))
        n_invalid += sh["n_invalid"]
        n_bad_td += sh["n_bad_td"]
        print(f"[profiles] shard {i + 1}/{len(bins_paths)}: {len(sh['meta'])} profiles "
              f"(cum {sum(len(m) for m in acc['meta']):,})", flush=True)
        del sh
        gc.collect()
        _trim()

    del samples_idx  # joined during the loop; frees ~3GB for the write phase
    gc.collect()
    _trim()

    # --- cross-shard duplicate (sample, sub) keys (trajectory split across
    # part files) would produce partial profiles for the same key — drop them.
    # Within one shard keys are unique by construction, so only cross-shard
    # collisions matter; check the two key columns only (cheap).
    keys = pd.DataFrame({
        "sample_id": np.concatenate([m.sample_id.to_numpy(object) for m in acc["meta"]]),
        "sub_id": np.concatenate([m.sub_id.to_numpy(np.int32) for m in acc["meta"]]),
    })
    dup = keys.duplicated(["sample_id", "sub_id"], keep=False).to_numpy()
    n_dup_keys = int(keys[dup].drop_duplicates().shape[0]) if dup.any() else 0
    del keys
    rebuild_lw = False
    if dup.any():
        print(f"[profiles] WARNING: {n_dup_keys} (sample, sub) keys duplicated across shards; "
              f"dropping {int(dup.sum())} partial profiles", flush=True)
        offs = np.cumsum(np.append(0, [len(m) for m in acc["meta"]]))
        for j in range(len(acc["meta"])):
            seg = dup[offs[j]:offs[j + 1]]
            if not seg.all():
                for k in ("speeds", "valid", "observed", "lengths", "eff_len"):
                    acc[k][j] = acc[k][j][seg]
                acc["meta"][j] = acc["meta"][j][seg]
                per_profile[j] = per_profile[j][seg]
        # stats pieces no longer align with profile rows — rebuild from survivors
        valid_speeds = [s[v] for s, v in zip(acc["speeds"], acc["valid"])]
        valid_ratio = [v.sum(axis=1) / np.maximum(l, 1)
                       for v, l in zip(acc["valid"], acc["lengths"])]
        rebuild_lw = True

    # --- stream the npz member-by-member, freeing each piece list after use ---
    N = int(sum(p.shape[0] for p in acc["lengths"]))
    out_npz = out_dir / cfg["output"]["profiles_file"].format(L=L)
    tmp_npz = out_dir / (cfg["output"]["profiles_file"].format(L=L) + ".tmp")
    with zipfile.ZipFile(tmp_npz, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        _write_member_2d(zf, "speeds", acc["speeds"]); del acc["speeds"]
        _write_member_2d(zf, "valid", acc["valid"]); del acc["valid"]
        _write_member_2d(zf, "observed", acc["observed"]); del acc["observed"]
        gc.collect(); _trim()
        _write_member(zf, "lengths", lengths := np.concatenate(acc["lengths"]))
        del acc["lengths"]
        # meta columns, numeric first then the heavy fixed-width strings last
        for col, dt in (("window_id", np.int64), ("sub_id", np.int32),
                        ("y_travel_s", np.float32), ("v_sample", np.float32),
                        ("td_target", np.float32), ("n_bins_target", np.int32)):
            _write_member(zf, col, np.concatenate(
                [m[col].to_numpy(dtype=dt) for m in acc["meta"]]))
        for col in ("sample_id", "link_id"):
            src = "sample_id" if col == "sample_id" else "target_link_id"
            arr = np.concatenate([m[src].to_numpy(object) for m in acc["meta"]]).astype("U")
            _write_member(zf, col, arr)
            del arr
            gc.collect(); _trim()
    os.replace(tmp_npz, out_npz)  # atomic: a crash never leaves a half npz for the skip guard

    # --- link <-> sub-link map (one row per unique sub-link) ------------------
    meta = pd.concat(acc["meta"], ignore_index=True)
    del acc["meta"]
    if rebuild_lw:
        lw_pairs = [meta[["target_link_id", "window_id"]].drop_duplicates()]
    per_profile = pd.concat(per_profile, ignore_index=True)
    per_profile["target_link_id"] = per_profile.target_link_id.astype("category")
    sub_map = (
        per_profile.groupby(["target_link_id", "sub_id"], observed=True)
        .agg(n_bins=("n_bins", "median"), eff_len_ratio=("eff_len_ratio", "median"),
             n_profiles=("n_bins", "size"))
        .reset_index()
    )
    del per_profile
    sub_map["n_bins"] = sub_map.n_bins.astype(int)
    sub_map["eff_len_m"] = (sub_map.eff_len_ratio * p["bin_size_m"]).round(1)
    sub_map = sub_map.drop(columns="eff_len_ratio")
    sub_map.to_parquet(out_dir / cfg["output"]["sub_map_file"].format(L=L), index=False)

    # --- stats (spec §11) ------------------------------------------------------
    n_link_window_keys = int(pd.concat(lw_pairs, ignore_index=True)
                             .drop_duplicates(["target_link_id", "window_id"]).shape[0])
    del lw_pairs
    q = lambda a, ps: [round(float(np.nanpercentile(a, x)), 3) for x in ps]
    stats = {
        "l_sub_m": L,
        "max_bins": max_bins,
        "n_profiles": int(N),
        "n_samples_in": int(n_samples_in),
        "n_links": int(meta.target_link_id.nunique()),
        "n_sub_links": int(len(sub_map)),
        "n_link_window_keys": n_link_window_keys,
        "bins_per_profile_p10_p50_p90_mean": q(lengths, [10, 50, 90]) + [round(float(lengths.mean()), 2)],
        "valid_ratio_p10_p50_p90": q(np.concatenate(valid_ratio), [10, 50, 90]),
        "n_invalid_bins": n_invalid,
        "n_invalid_bad_td": n_bad_td,
        "n_invalid_over_speed": n_invalid - n_bad_td,
        "profiles_per_sample_mean": round(float(N / max(n_samples_in, 1)), 3),
        "v_bin_mps_p10_p50_p90": q(np.concatenate(valid_speeds), [10, 50, 90]),
        "n_bins_shards": len(bins_paths),
    }
    del valid_ratio, valid_speeds, lengths
    dump_json(stats, out_dir / cfg["output"]["stats"].format(L=L))
    print(f"[profiles] L_sub={L}m -> {N} profiles ({meta.target_link_id.nunique()} links, "
          f"{len(sub_map)} sub-links), pad={max_bins}")
    print(f"[profiles] wrote {out_npz.name}, {len(sub_map)}-row sub-map, stats json")


if __name__ == "__main__":
    main()
