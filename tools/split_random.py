"""Random-shuffle split over multi-day processed corpora (e.g. day20 + day21).

STREAMING merge: reads the per-day samples.parquet in ~1M-row batches and
writes the merged corpus + `split` column incrementally, so peak RAM is one
batch plus small lookup tables — independent of corpus size (~48 full hours
would be ~500M samples; the node is shared, do NOT materialise that in RAM).

Outputs everything train_eta.py needs — no other code changes:
  <out>/samples.parquet      merged samples of all --inputs + `split` int8 col
  <out>/link_window.parquet  merged (link, window) mean_speed table
  <out>/split_stats.json     per-split counts / K=1 share / leakage overlaps
  <config_out>               ready-to-run yaml (split.mode=manifest)

Split units (--unit), decided BEFORE shuffling:
  sample  every vehicle pass shuffled independently (literal 随机打乱).
          WARNING: the same link — even the same (link, window) group — lands
          in train AND test, so absolute metrics are optimistic; fine for
          A0/A1/A2 relative comparison. Uses a full-row permutation (~4 GB
          transient at 500M rows).
  link    whole links shuffled (spec §17 protocol: test links never seen).
  window  whole hour-windows shuffled (random time holdout).

Usage:
  python tools/split_random.py \
      --inputs data/processed_ts/day0820 data/processed_ts/day0821 \
      --out data/processed_ts/day20_21 --unit sample --seed 0
  python tools/train_eta.py --config configs/eta_day20_21.yaml --variant ours
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.utils import dump_json, load_config, save_config  # noqa: E402

SAMPLES_REQUIRED = ("sample_id", "target_link_id", "window_id", "y_travel_s")
LW_REQUIRED = ("target_link_id", "window_id", "mean_speed", "n_trajs")
SPLIT_NAMES = ("train", "val", "test")
BATCH_ROWS = 1_000_000
DEDUP_MAX_ROWS = 120_000_000  # sample_id dup check loads one column; skip above this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", nargs="+", required=True,
                   help="processed day dirs (each with samples.parquet + link_window.parquet)")
    p.add_argument("--out", required=True, help="output dir for the merged split corpus")
    p.add_argument("--unit", choices=("sample", "link", "window"), default="sample",
                   help="shuffle unit (see module docstring for the leakage trade-off)")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-config", default="configs/eta.yaml",
                   help="config to copy model/train sections from")
    p.add_argument("--config-out", default=None,
                   help="emitted config path (default configs/eta_<out-name>.yaml)")
    return p.parse_args()


def split_of_units(n_units: int, ratios: tuple[float, float, float], seed: int) -> np.ndarray:
    """Seeded shuffle of n units, cut by ratios -> int8 split code per unit."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_units)
    n_tr = int(round(ratios[0] * n_units))
    n_va = int(round(ratios[1] * n_units))
    s = np.full(n_units, 2, dtype=np.int8)
    s[perm[:n_tr]] = 0
    s[perm[n_tr:n_tr + n_va]] = 1
    return s


def link_codes_of(col: pa.ChunkedArray | pa.Array, link_universe: pa.Array) -> np.ndarray:
    """Column -> code into link_universe, hashing only the chunk's unique values."""
    da = pc.dictionary_encode(col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col)
    uniq_code = pc.index_in(da.dictionary, value_set=link_universe).to_numpy(zero_copy_only=False)
    if (uniq_code < 0).any():
        raise ValueError("samples contain target_link_id values absent from link_window")
    return uniq_code[da.indices.to_numpy(zero_copy_only=False).astype(np.int64)].astype(np.int64)


def main() -> None:
    args = parse_args()
    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) <= 0:
        raise SystemExit(f"ratios must be positive and sum to 1, got {ratios}")
    inputs = [Path(d) for d in args.inputs]
    for d in inputs:
        if not (d / "samples.parquet").is_file() or not (d / "link_window.parquet").is_file():
            raise FileNotFoundError(f"{d} is missing samples.parquet / link_window.parquet "
                                    f"(run the ingest for that day first)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_out = Path(args.config_out) if args.config_out else \
        Path("configs") / f"eta_{out.name}.yaml"
    s_paths = [d / "samples.parquet" for d in inputs]

    # ---- link_window: small (per (link, window)); merge + dedup in RAM -------
    lw = pa.concat_tables([pq.read_table(d / "link_window.parquet") for d in inputs])
    if "split" in lw.column_names:
        lw = lw.drop_columns(["split"])
    n_trajs_lw = lw.column("n_trajs").combine_chunks().to_numpy(zero_copy_only=False)

    # ---- unit universes + split maps (all small) ------------------------------
    link_universe = pc.unique(lw.column("target_link_id"))
    n_links = len(link_universe)
    link_codes_lw = link_codes_of(lw.column("target_link_id"), link_universe)
    win_lw = lw.column("window_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    win_sorted, win_slot_lw = np.unique(win_lw, return_inverse=True)
    n_win = len(win_sorted)

    # composite int key = (link, window) — struct arrays lack count_distinct kernels
    comp_lw = link_codes_lw * n_win + win_slot_lw
    n_dup_lw = len(comp_lw) - np.unique(comp_lw).size
    if n_dup_lw:
        _, first = np.unique(comp_lw, return_index=True)  # first occurrence wins
        keep = np.zeros(len(comp_lw), dtype=bool)
        keep[first] = True
        lw = lw.filter(pa.array(keep))
        n_trajs_lw, comp_lw = n_trajs_lw[keep], comp_lw[keep]
        print(f"[split_random] WARNING: {n_dup_lw:,} duplicate (link, window) rows dropped")

    if args.unit == "link":
        split_by_unit = split_of_units(n_links, ratios, args.seed)
    elif args.unit == "window":
        split_by_unit = split_of_units(n_win, ratios, args.seed)
    else:  # sample: one permutation over ALL rows; total from parquet metadata
        n_total = sum(pq.ParquetFile(p).metadata.num_rows for p in s_paths)
        split_by_row = split_of_units(int(n_total), ratios, args.seed)
        print(f"[split_random] sample-unit permutation over {n_total:,} rows "
              f"(~{n_total * 8 / 1e9:.1f} GB transient)")
    if args.unit == "sample":
        print("[split_random] NOTE: --unit sample puts the SAME link (even the same "
              "(link, window) group) into train and test — absolute metrics are "
              "optimistic; use --unit link for the spec §17 never-seen-link protocol.")

    # ---- optional sample_id dup check (boundary-leak detector, cheap column) --
    keep_mask: np.ndarray | None = None
    n_total_check = sum(pq.ParquetFile(p).metadata.num_rows for p in s_paths)
    if n_total_check > DEDUP_MAX_ROWS:
        print(f"[split_random] WARNING: {n_total_check:,} rows > dedup limit "
              f"{DEDUP_MAX_ROWS:,}; skipping sample_id dup check")
    else:
        ids = pa.concat_arrays([
            pq.read_table(p, columns=["sample_id"]).column("sample_id").combine_chunks()
            for p in s_paths])
        n_dup = len(ids) - int(pc.count_distinct(ids))
        if n_dup:
            keep_mask = ~ids.to_pandas().duplicated().to_numpy()
            print(f"[split_random] WARNING: {n_dup:,} duplicate sample_id rows dropped "
                  f"(sharding boundary leak — same pass ingested in two inputs)")

    # ---- K lookup: composite (link_code, window) key sorted once --------------
    order = np.argsort(comp_lw, kind="stable")
    comp_sorted, k_sorted = comp_lw[order], n_trajs_lw[order]

    # ---- streaming merge + split assignment -----------------------------------
    schema0 = pq.ParquetFile(s_paths[0]).schema_arrow
    if "split" in schema0.names:
        schema0 = schema0.remove(schema0.get_field_index("split"))
    target_schema = schema0.append(pa.field("split", pa.int8()))
    for p in s_paths[1:]:
        other = pq.ParquetFile(p).schema_arrow
        if {f.name: str(f.type) for f in other} != {f.name: str(f.type) for f in schema0}:
            raise ValueError(f"schema mismatch between {s_paths[0]} and {p}")

    win_date = pd.to_datetime(win_sorted * 3600, unit="s", utc=True) \
        .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
    dates = sorted(set(win_date))
    date_idx = np.searchsorted(np.array(dates), win_date)
    seen_links = np.zeros((n_links, 3), dtype=bool)
    seen_wins = np.zeros((n_win, 3), dtype=bool)
    day_counts = np.zeros((len(dates), 3), dtype=np.int64)
    split_count = np.zeros(3, dtype=np.int64)
    k1_count = np.zeros(3, dtype=np.int64)
    n_no_lw = 0

    writer = pq.ParquetWriter(out / "samples.parquet", target_schema, compression="snappy")
    row0 = 0
    try:
        for path in s_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=BATCH_ROWS):
                lo, hi = row0, row0 + batch.num_rows
                row0 = hi
                m_np = keep_mask[lo:hi] if keep_mask is not None else None
                if m_np is not None:  # drop dup rows first, split slice follows suit
                    batch = batch.filter(pa.array(m_np))
                idx = {name: batch.schema.get_field_index(name) for name in schema0.names}
                if args.unit == "sample":
                    sp = split_by_row[lo:hi]
                    if m_np is not None:
                        sp = sp[m_np]
                else:
                    sp = None
                link_c = link_codes_of(batch.column(idx["target_link_id"]), link_universe)
                w_slot = np.searchsorted(win_sorted, np.asarray(
                    batch.column(idx["window_id"]).to_numpy(zero_copy_only=False), dtype=np.int64))
                if sp is None:
                    sp = split_by_unit[link_c] if args.unit == "link" else split_by_unit[w_slot]
                # row-level K via sorted composite lookup
                pos = np.searchsorted(comp_sorted, link_c * n_win + w_slot)
                pos = np.clip(pos, 0, len(comp_sorted) - 1)
                ok = comp_sorted[pos] == link_c * n_win + w_slot
                n_no_lw += int((~ok).sum())
                k_row = np.where(ok, k_sorted[pos], -1).astype(np.int32)

                split_count += np.bincount(sp, minlength=3)
                k1_count += np.bincount(sp[k_row == 1], minlength=3)
                for s in range(3):
                    seen_links[link_c[sp == s], s] = True
                    seen_wins[w_slot[sp == s], s] = True
                np.add.at(day_counts, (date_idx[w_slot], sp.astype(np.int64)), 1)

                arrays = [batch.column(idx[name]) for name in schema0.names]
                arrays.append(pa.array(sp, type=pa.int8()))
                writer.write_table(pa.Table.from_arrays(arrays, schema=target_schema),
                                   row_group_size=BATCH_ROWS)
    finally:
        writer.close()
    if n_no_lw:
        raise ValueError(f"{n_no_lw:,} sample rows have no link_window row — "
                         f"inputs' ingest dirs are inconsistent")

    # ---- stats -----------------------------------------------------------------
    per_split = {}
    for s, name in enumerate(SPLIT_NAMES):
        per_split[name] = {
            "n_samples": int(split_count[s]),
            "share": round(float(split_count[s] / max(split_count.sum(), 1)), 4),
            "n_links": int(seen_links[:, s].sum()),
            "n_windows": int(seen_wins[:, s].sum()),
            "k1_share": round(float(k1_count[s] / max(split_count[s], 1)), 4),
        }
    per_split["train_link_overlap"] = {
        "val": int((seen_links[:, 0] & seen_links[:, 1]).sum()),
        "test": int((seen_links[:, 0] & seen_links[:, 2]).sum()),
    }
    stats = {
        "unit": args.unit, "seed": args.seed, "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "inputs": [str(d) for d in inputs], "n_samples": int(split_count.sum()),
        "n_links": int(n_links), "n_windows": int(n_win),
        "n_dup_sample_id_dropped": int(0 if keep_mask is None else (~keep_mask).sum()),
        "n_dup_link_window_dropped": int(n_dup_lw),
        "per_split": per_split,
        "per_day_split_samples": {d: {SPLIT_NAMES[s]: int(day_counts[i, s]) for s in range(3)}
                                  for i, d in enumerate(dates)},
        "leakage_note": {
            "sample": "links and (link, window) groups repeat across splits — optimistic",
            "link": "links disjoint across splits (spec §17 protocol)",
            "window": "windows disjoint across splits; links repeat by design",
        }[args.unit],
    }
    pq.write_table(lw, out / "link_window.parquet", compression="snappy")
    dump_json(stats, out / "split_stats.json")

    # ---- emit the plug-and-play config -----------------------------------------
    profiles = [d / "profiles_l200.npz" for d in inputs]
    missing = [str(p) for p in profiles if not p.is_file()]
    if missing:
        print(f"[split_random] WARNING: profiles missing (config still emitted, build them "
              f"first): {missing}")
    base = load_config(args.base_config)
    cfg = copy.deepcopy(base)
    cfg["data"] = {
        "profiles_npz": [str(p) for p in profiles],
        "samples_parquet": str(out / "samples.parquet"),
        "link_window_parquet": str(out / "link_window.parquet"),
        "v_norm": base.get("data", {}).get("v_norm", 33.3),
        "split": {"mode": "manifest", "unit": args.unit, "seed": args.seed,
                  **dict(zip(SPLIT_NAMES, ratios))},
    }
    save_config(cfg, config_out)

    print(f"[split_random] wrote {out}/samples.parquet ({int(split_count.sum()):,} rows), "
          f"link_window.parquet, split_stats.json")
    print(f"[split_random] splits: " + ", ".join(
        f"{n} {per_split[n]['n_samples']:,} ({per_split[n]['share']:.1%})"
        for n in SPLIT_NAMES))
    print(f"[split_random] config -> {config_out}")
    print("[split_random] next:",
          f"python tools/train_eta.py --config {config_out} --variant ours")


if __name__ == "__main__":
    main()
