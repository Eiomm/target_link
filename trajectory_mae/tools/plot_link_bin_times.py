"""Isolated MLP experiment plotting fork: adaptive, spatially aligned cell plots.

The corpus is large, so the small scalar ``cells/`` index is scanned first;
only the selected cell's exact ``observations_v2/day=.../bucket=...`` partition
is then read and collected to the driver.  A cell is the modelling unit used
by the current pipeline:

    (map_version, target_link_id, seg_idx, 10-minute window)

If ``--seg-idx`` and/or ``--window`` are omitted, the densest matching cell is
chosen.  The output is a dependency-free SVG plus a JSON summary.  A browser
can open the SVG directly and preserves per-bin hover text.

Example (server):

    spark-submit trajectory_mae/tools/plot_link_bin_times.py \
      --corpus hdfs:///path/to/corpus_v1 \
      --link 123456789 --out runtime/link_123456789.svg

Example (local raw reference package, no Spark required):

    python trajectory_mae/tools/plot_link_bin_times.py \
      --raw-parquet data/example_link/samples_flat.parquet \
      --out runtime/example_link.svg

``T_diff`` is stored per piece.  Pieces at the same absolute ``bin_pos`` are
folded exactly as the training reader does: times and ratios are summed, and a
bin is valid only when every piece is valid.  The default displayed metric is
the full-10-m equivalent time ``sum(T_diff) / sum(ratio)``; this avoids making
a boundary bin look artificially fast merely because only part of it belongs
to the segment.  Use ``--metric raw`` to inspect the model's raw target.

The default coverage axis ends at the farthest recorded bin across ALL cell
rows, including bins whose time is invalid. It never reindexes trajectories.
Optional reliable road geometry clips a partial final bin. This tool performs
no interpolation and does not modify the 50-bin training grid.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_mae.tools.bin_svg import spatial_domain, _percentile, _quantiles_by_bin, _colour, _esc, _distance_ticks, _contiguous_statistics, render_svg

N_BINS = 50
BEIJING = ZoneInfo("Asia/Shanghai")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus",
                        help="corpus_v1 root (must contain cells/ and observations_v2/)")
    source.add_argument("--raw-parquet",
                        help="local samples_flat.parquet from extract_ref_link.py")
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--link", help="target_link_id; inferred for a single-link raw package")
    p.add_argument("--map-version", help="optional map_version (compared as string)")
    p.add_argument("--seg-idx", type=int,
                   help="500 m segment index; default: choose the densest cell")
    p.add_argument("--window",
                   help="10-minute window as epoch seconds or YYYY-mm-dd[ T]HH:MM")
    p.add_argument("--day", help="optional Beijing day YYYYmmdd; limits the cells/ scan")
    p.add_argument("--metric", choices=("equivalent-10m", "raw"),
                   default="equivalent-10m")
    p.add_argument("--x-range", choices=("coverage", "geometry", "grid"),
                   default="coverage",
                   help="coverage: farthest recorded bin; geometry: actual segment; grid: 500m")
    p.add_argument("--segment-length-m", type=float,
                   help="verified length of this segment in (0,500]; overrides raw L_link_m")
    p.add_argument("--max-heatmap-rows", type=int, default=300,
                   help="max rows drawn; all rows still contribute to statistics")
    p.add_argument("--lower-percentile", type=float, default=5.0)
    p.add_argument("--upper-percentile", type=float, default=95.0)
    p.add_argument("--buckets", type=int, default=128,
                   help="observations_v2 cell-hash bucket count")
    p.add_argument("--out", required=True, help="local driver path ending in .svg")
    p.add_argument("--master", help="optional Spark master; omit under spark-submit/YARN")
    return p.parse_args()


def _window_epoch(value):
    if value is None:
        return None
    try:
        epoch = int(value)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                parsed = datetime.strptime(value, fmt).replace(tzinfo=BEIJING)
                break
            except ValueError:
                pass
        if parsed is None:
            raise ValueError("--window must be epoch seconds or YYYY-mm-dd[ T]HH:MM")
        epoch = int(parsed.timestamp())
    if epoch % 600:
        raise ValueError("--window must be aligned to a 10-minute boundary")
    return epoch


def _corpus_root(corpus, obs_dir):
    root = corpus.rstrip("/")
    suffix = "/" + obs_dir
    return root[:-len(suffix)] if root.endswith(suffix) else root


def _field(row, name):
    return row[name]


def fold_observation(row, metric="equivalent-10m", n_bins=N_BINS):
    """Fold a ragged piece row to (values, valid, ratios), each shaped [50]."""
    names = ("T_diff", "ratio_pct", "valid", "bin_pos")
    cols = {name: list(_field(row, name)) for name in names}
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1:
        raise ValueError("ragged columns have different lengths")

    count = np.zeros(n_bins, dtype=np.int32)
    valid_count = np.zeros(n_bins, dtype=np.int32)
    times = np.zeros(n_bins, dtype=np.float64)
    ratios = np.zeros(n_bins, dtype=np.float64)
    for t, ratio_pct, is_valid, pos in zip(
            cols["T_diff"], cols["ratio_pct"], cols["valid"], cols["bin_pos"]):
        pos = int(pos)
        if not 0 <= pos < n_bins:
            raise ValueError("bin_pos outside 0..%d" % (n_bins - 1))
        count[pos] += 1
        ok = bool(is_valid) and t is not None and math.isfinite(float(t))
        valid_count[pos] += int(ok)
        if ok:
            times[pos] += float(t)
        ratios[pos] += float(ratio_pct) / 10.0

    valid = (count > 0) & (valid_count == count) & (ratios > 0)
    values = times.copy()
    if metric == "equivalent-10m":
        np.divide(times, ratios, out=values, where=ratios > 0)
    elif metric != "raw":
        raise ValueError("unknown metric: " + metric)
    values[~valid] = np.nan
    return values, valid, ratios










def observation_presence(row, n_bins=N_BINS):
    """A piece exists, independently of whether its passage time is known."""
    present = np.zeros(n_bins, dtype=bool)
    for pos in _field(row, "bin_pos"):
        j = int(pos)
        if not 0 <= j < n_bins:
            raise ValueError("bin_pos outside fixed grid")
        present[j] = True
    return present










def _json_number(value):
    return None if not math.isfinite(float(value)) else round(float(value), 6)


def _load_raw_cell(a, wanted_window, wanted_day):
    """Build corpus-equivalent observation rows from a small raw link package."""
    import pyarrow.parquet as pq

    columns = ["map_version", "target_link_id", "sample_id", "seg_mark", "seg_idx",
               "bin_idx", "sub_idx", "t_ref", "T_cum", "T_diff", "ratio", "observed"]
    if "L_link_m" in pq.read_schema(a.raw_parquet).names:
        columns.append("L_link_m")
    table = pq.read_table(a.raw_parquet, columns=columns)
    raw = table.to_pylist()
    links = sorted({str(r["target_link_id"]) for r in raw})
    if a.link is None:
        if len(links) != 1:
            raise SystemExit("raw package contains %d links; pass --link" % len(links))
        link = links[0]
    else:
        link = str(a.link)

    grouped = {}
    for r in raw:
        if int(r["seg_mark"]) != 1 or str(r["target_link_id"]) != link:
            continue
        if a.map_version is not None and str(r["map_version"]) != str(a.map_version):
            continue
        if a.seg_idx is not None and int(r["seg_idx"]) != a.seg_idx:
            continue
        key = (str(r["map_version"]), link, int(r["seg_idx"]), str(r["sample_id"]))
        grouped.setdefault(key, []).append(r)
    if not grouped:
        raise SystemExit("no matching marked rows in --raw-parquet")

    by_cell = {}
    for (map_version, target_link_id, seg_idx, sample_id), pieces in grouped.items():
        starts = []
        for r in pieces:
            tc, td = r["T_cum"], r["T_diff"]
            if (tc is not None and td is not None and
                    math.isfinite(float(tc)) and math.isfinite(float(td))):
                starts.append(float(tc) - float(td))
        if not starts:
            continue
        t_ref = max(float(r["t_ref"]) for r in pieces)
        window = int(math.floor((t_ref + min(starts)) / 600.0) * 600)
        day = datetime.fromtimestamp(window, BEIJING).strftime("%Y%m%d")
        if wanted_window is not None and window != wanted_window:
            continue
        if wanted_day is not None and day != wanted_day:
            continue
        pieces.sort(key=lambda r: (int(r["bin_idx"]), int(r["sub_idx"])))
        entry_time = t_ref + min(starts)
        row = {"sample_id": sample_id, "entry_time": entry_time,
               "T_diff": [], "ratio_pct": [],
               "valid": [], "bin_pos": []}
        for r in pieces:
            td = r["T_diff"]
            valid = td is not None and math.isfinite(float(td))
            pos = int(r["bin_idx"]) - 50 * seg_idx - 10
            row["T_diff"].append(float(td) if td is not None else math.nan)
            row["ratio_pct"].append(int(round(float(r["ratio"]) * 10)))
            row["valid"].append(valid)
            row["bin_pos"].append(pos)
        cell_key = (map_version, target_link_id, seg_idx, window, day)
        by_cell.setdefault(cell_key, []).append(row)

    if not by_cell:
        raise SystemExit("no raw observation matches the requested day/window")
    chosen_key, rows = min(
        by_cell.items(), key=lambda item: (-len(item[1]), item[0][3], item[0][2]))
    map_version, target_link_id, seg_idx, window, day = chosen_key
    chosen = {"map_version": map_version, "target_link_id": target_link_id,
              "seg_idx": seg_idx, "window": window, "day": day,
              "cell_id": None, "bucket": None, "K": len(rows)}
    # A length from a different link, segment or passage is not evidence for
    # this cell. Never guess road length from the longest observed trajectory.
    ids = {r["sample_id"] for r in rows}
    lengths = [r.get("L_link_m") for r in raw
               if str(r["map_version"]) == map_version
               and str(r["target_link_id"]) == target_link_id
               and int(r["seg_idx"]) == seg_idx and int(r["seg_mark"]) == 1
               and r["sample_id"] in ids]
    chosen["geometry_source"] = "unavailable"
    if lengths and all(v is not None and math.isfinite(float(v)) and float(v) > 0
                       for v in lengths):
        if max(lengths) - min(lengths) <= 0.01:
            length = min(500.0, float(np.median(lengths)) - 500 * seg_idx)
            if length > 0:
                chosen["segment_length_m"] = length
                chosen["geometry_source"] = "consistent raw L_link_m minus 500*seg_idx"
        else:
            chosen["geometry_source"] = "unavailable: inconsistent raw L_link_m"
    rows.sort(key=lambda r: r["sample_id"])
    return chosen, rows


def _load_corpus_cell(a, wanted_window, wanted_day):
    """Locate a cell via cells/ and read only its exact HDFS day/bucket."""
    if a.link is None:
        raise SystemExit("--link is required with --corpus")
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    builder = SparkSession.builder.appName("tl_plot_link_bin_times")
    if a.master:
        builder = builder.master(a.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    root = _corpus_root(a.corpus, a.obs_dir)
    cells = spark.read.parquet(root + "/cells").select(
        "cell_id", "map_version", "target_link_id", "seg_idx", "window", "K", "day")
    selected = cells.where(F.col("target_link_id").cast("string") == str(a.link))
    if wanted_day is not None:
        selected = selected.where(F.col("day").cast("string") == wanted_day)
    if a.map_version is not None:
        selected = selected.where(F.col("map_version").cast("string") == str(a.map_version))
    if a.seg_idx is not None:
        selected = selected.where(F.col("seg_idx") == int(a.seg_idx))
    if wanted_window is not None:
        selected = selected.where(F.col("window") == wanted_window)

    chosen = selected.orderBy(F.desc("K"), F.asc("window"), F.asc("seg_idx")).first()
    if chosen is None:
        spark.stop()
        raise SystemExit("no matching observation; check link/map/segment/window")
    cell_id = int(chosen["cell_id"])
    bucket = ((cell_id % a.buckets) + a.buckets) % a.buckets
    obs_partition = "%s/%s/day=%s/bucket=%d" % (
        root, a.obs_dir, chosen["day"], bucket)
    cell = (spark.read.parquet(obs_partition)
            .select("cell_id", "sample_id", "dt", "T_diff", "ratio_pct", "valid", "bin_pos")
            .where(F.col("cell_id") == cell_id))
    rows = cell.orderBy("sample_id").collect()
    spark.stop()
    if not rows:
        raise SystemExit("selected cell is absent from %s; check --buckets/--obs-dir" %
                         obs_partition)
    if len(rows) != int(chosen["K"]):
        raise SystemExit("cells/ says K=%d but observations partition returned %d rows" %
                         (int(chosen["K"]), len(rows)))
    meta = chosen.asDict()
    meta["bucket"] = bucket
    return meta, rows


def main():
    a = parse_args()
    if not a.out.lower().endswith(".svg"):
        raise SystemExit("--out must end in .svg")
    if a.max_heatmap_rows <= 0:
        raise SystemExit("--max-heatmap-rows must be positive")
    if a.buckets <= 0:
        raise SystemExit("--buckets must be positive")
    if a.segment_length_m is not None and (not math.isfinite(a.segment_length_m)
                                            or not 0 < a.segment_length_m <= 500):
        raise SystemExit("--segment-length-m must be finite and in (0,500]")
    if not 0 <= a.lower_percentile < a.upper_percentile <= 100:
        raise SystemExit("colour percentiles must satisfy 0 <= lower < upper <= 100")
    try:
        wanted_window = _window_epoch(a.window)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if a.day is not None and (len(a.day) != 8 or not a.day.isdigit()):
        raise SystemExit("--day must be YYYYmmdd")
    window_day = (datetime.fromtimestamp(wanted_window, BEIJING).strftime("%Y%m%d")
                  if wanted_window is not None else None)
    if a.day is not None and window_day is not None and a.day != window_day:
        raise SystemExit("--day and --window refer to different Beijing days")
    wanted_day = a.day or window_day

    if a.raw_parquet:
        chosen, rows = _load_raw_cell(a, wanted_window, wanted_day)
    else:
        chosen, rows = _load_corpus_cell(a, wanted_window, wanted_day)

    values, sample_ids, entry_times, presence = [], [], [], []
    for row in rows:
        folded, _, _ = fold_observation(row, metric=a.metric)
        values.append(folded)
        presence.append(observation_presence(row))
        sample_ids.append(row["sample_id"])
        entry_times.append(float(row["entry_time"]) if isinstance(row, dict)
                           else int(chosen["window"]) + float(row["dt"]))
    matrix = np.stack(values)
    finite = matrix[np.isfinite(matrix)]
    per_bin = _quantiles_by_bin(matrix)
    epoch = int(chosen["window"])
    cell_id = chosen.get("cell_id")
    bucket = chosen.get("bucket")
    meta = {
        "map_version": str(chosen["map_version"]),
        "target_link_id": str(chosen["target_link_id"]),
        "seg_idx": int(chosen["seg_idx"]),
        "window": epoch,
        "cell_id": cell_id,
        "day": str(chosen["day"]),
        "bucket": bucket,
        "window_local": datetime.fromtimestamp(epoch, BEIJING).strftime("%Y-%m-%d %H:%M CST"),
        "n_trajectories": len(rows),
        "n_valid_bin_observations": int(finite.size),
        "metric": a.metric,
        "x_range_mode": a.x_range,
        "segment_length_m": (a.segment_length_m if a.segment_length_m is not None
                             else chosen.get("segment_length_m")),
        "geometry_source": ("explicit --segment-length-m" if a.segment_length_m is not None
                            else chosen.get("geometry_source", "unavailable")),
        "lower_percentile": a.lower_percentile,
        "upper_percentile": a.upper_percentile,
        "overall_s": {"p05": _json_number(_percentile(finite, 5)),
                      "p25": _json_number(_percentile(finite, 25)),
                      "p50": _json_number(_percentile(finite, 50)),
                      "p75": _json_number(_percentile(finite, 75)),
                      "p95": _json_number(_percentile(finite, 95))},
    }
    draw = render_svg(matrix, sample_ids, entry_times, meta, per_bin,
                      a.out, a.max_heatmap_rows, present=np.stack(presence))
    meta.update(draw)
    meta["per_bin"] = [{k: (_json_number(v) if k.endswith("_s") else v)
                        for k, v in d.items()} for d in per_bin]
    summary_path = os.path.splitext(os.path.abspath(a.out))[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({"svg": draw["svg"], "summary": summary_path,
                      "cell": {k: meta[k] for k in
                               ("map_version", "target_link_id", "seg_idx", "window_local")},
                      "x_range_m": [meta["x_min_m"], meta["x_max_m"]],
                      "x_range_mode": meta["x_range_mode"],
                      "K": len(rows), "overall_s": meta["overall_s"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
