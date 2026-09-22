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

    spark-submit experiments/trajectory_mlp_v1/tools/plot_link_bin_times.py \
      --corpus hdfs:///path/to/corpus_v1 \
      --link 123456789 --out runtime/link_123456789.svg

Example (local raw reference package, no Spark required):

    python experiments/trajectory_mlp_v1/tools/plot_link_bin_times.py \
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
    return row[name] if isinstance(row, dict) else getattr(row, name)


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


def _percentile(values, q):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else math.nan


def _quantiles_by_bin(matrix):
    out = []
    for j in range(matrix.shape[1]):
        a = matrix[:, j]
        a = a[np.isfinite(a)]
        out.append({
            "bin_pos": j,
            "n": int(a.size),
            "p25_s": _percentile(a, 25),
            "p50_s": _percentile(a, 50),
            "p75_s": _percentile(a, 75),
        })
    return out


def _colour(value, lo, hi):
    if not math.isfinite(value):
        return "#d9dde3"
    x = min(1.0, max(0.0, (value - lo) / max(hi - lo, 1e-12)))
    stops = ((44, 162, 95), (254, 224, 139), (215, 48, 39))
    if x <= 0.5:
        a, b, u = stops[0], stops[1], x * 2
    else:
        a, b, u = stops[1], stops[2], (x - 0.5) * 2
    rgb = tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))
    return "#%02x%02x%02x" % rgb


def _esc(value):
    return html.escape(str(value), quote=True)


def observation_presence(row, n_bins=N_BINS):
    """A piece exists, independently of whether its passage time is known."""
    present = np.zeros(n_bins, dtype=bool)
    for pos in _field(row, "bin_pos"):
        j = int(pos)
        if not 0 <= j < n_bins:
            raise ValueError("bin_pos outside fixed grid")
        present[j] = True
    return present


def spatial_domain(matrix, present, mode="coverage", segment_length_m=None):
    """Use a common origin and all rows, before display subsampling.

    A recorded bin provides a 10m-grid upper bound, not its exact piece end.
    Geometry, when verified, gives the physical end of the final partial bin.
    """
    if matrix.ndim != 2 or matrix.shape[1] != N_BINS or not len(matrix):
        raise ValueError("expected a nonempty [trajectories,50] matrix")
    if present.shape != matrix.shape:
        raise ValueError("presence mask must have the same shape as matrix")
    if np.any(np.isfinite(matrix) & ~present):
        raise ValueError("finite time without a recorded bin")
    cols = np.flatnonzero(present.any(axis=0))
    if not len(cols):
        raise ValueError("no recorded bins")
    if segment_length_m is not None:
        segment_length_m = float(segment_length_m)
        if not math.isfinite(segment_length_m) or not 0 < segment_length_m <= 500:
            raise ValueError("segment length must be finite and in (0,500]")
        if cols[-1] * 10 >= segment_length_m:
            raise ValueError("recorded bin lies beyond the supplied road geometry")
    bound = float((cols[-1] + 1) * 10)
    coverage = min(bound, segment_length_m) if segment_length_m is not None else bound
    if mode == "coverage":
        end = coverage
    elif mode == "geometry":
        if segment_length_m is None:
            raise ValueError("geometry range needs --segment-length-m or consistent raw L_link_m")
        end = segment_length_m
    elif mode == "grid":
        end = 500.0
    else:
        raise ValueError("unknown x range: " + str(mode))
    return {"x_range_mode": mode, "x_min_m": 0.0, "x_max_m": end,
            "segment_length_m": segment_length_m,
            "coverage_upper_bound_m": coverage,
            "farthest_recorded_bin": int(cols[-1]),
            "coverage_resolution": "10m bin bounds; clipped by geometry when known"}


def _distance_ticks(end):
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if s >= end / 8)
    ticks = list(np.arange(0, end, step, dtype=float))
    if len(ticks) > 1 and end - ticks[-1] < end * 0.06:
        ticks.pop()
    return ticks + [float(end)]


def _contiguous_statistics(per_bin, end):
    """Never bridge an empty spatial column with a line or uncertainty band."""
    runs, run = [], []
    for d in sorted(per_bin, key=lambda item: item["bin_pos"]):
        j = d["bin_pos"]
        known = j * 10 < end and all(math.isfinite(d[k]) for k in ("p25_s", "p50_s", "p75_s"))
        if not known or (run and j != run[-1]["bin_pos"] + 1):
            if run:
                runs.append(run)
                run = []
        if known:
            run.append(d)
    if run:
        runs.append(run)
    return runs


def render_svg(matrix, sample_ids, entry_times, meta, per_bin, out_path, max_rows=300,
               present=None):
    matrix = np.asarray(matrix, dtype=np.float64)
    presence_source = "explicit_piece_records" if present is not None else "finite_values_only"
    present = np.isfinite(matrix) if present is None else np.asarray(present, dtype=bool)
    domain = spatial_domain(matrix, present, meta.get("x_range_mode", "coverage"),
                            meta.get("segment_length_m"))
    if max_rows <= 0 or len(sample_ids) != len(matrix) or len(entry_times) != len(matrix):
        raise ValueError("invalid row count or display cap")
    end = domain["x_max_m"]
    road_end = domain["segment_length_m"]
    n_draw = int(math.ceil(end / 10))
    ticks = _distance_ticks(end)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("the selected cell has no valid bin passage times")
    lo = _percentile(finite, meta["lower_percentile"])
    hi = _percentile(finite, meta["upper_percentile"])
    if not hi > lo:
        hi = lo + max(abs(lo) * 0.01, 1e-6)

    # SVG rows run top to bottom, so descending entry time places the earliest
    # passage at the bottom and later passages progressively above it.
    entry_times = np.asarray(entry_times, dtype=np.float64)
    order = np.argsort(entry_times, kind="stable")[::-1]
    if len(order) > max_rows:
        keep = np.linspace(0, len(order) - 1, max_rows).round().astype(int)
        order = order[keep]
    shown = matrix[order]
    shown_ids = [sample_ids[int(i)] for i in order]
    shown_entry = entry_times[order]
    shown_present = present[order]

    width, left, right = 1200, 124, 54
    plot_w = width - left - right
    heat_top = 148
    heat_h = min(760, max(260, len(shown) * 3.2))
    dist_top, dist_h = heat_top + heat_h + 92, 230
    height = int(dist_top + dist_h + 82)
    rh = heat_h / len(shown)
    x_of = lambda meters: left + meters / end * plot_w
    bin_end = lambda j: min((j + 1) * 10.0, end,
                            road_end if road_end is not None and j * 10 < road_end else end)
    center = lambda j: x_of((j * 10 + bin_end(j)) / 2)
    title = ("Link %s · seg %s · %s · K=%s" %
             (meta["target_link_id"], meta["seg_idx"], meta["window_local"],
              meta["n_trajectories"]))
    metric_label = ("Equivalent full-10m passage time (s)" if
                    meta["metric"] == "equivalent-10m" else "Raw bin T_diff (s)")

    s = []
    add = s.append
    add('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d" role="img">' % (width, height, width, height))
    add('<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' %
        (width, height))
    add('<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,'
        'sans-serif;fill:#20242a}.title{font-size:22px;font-weight:650}.sub{font-size:13px;'
        'fill:#59616d}.axis{font-size:12px;fill:#59616d}.label{font-size:14px;font-weight:600}'
        '.grid{stroke:#e4e7eb;stroke-width:1}.frame{fill:none;stroke:#aeb5bf;stroke-width:1}</style>')
    add('<text class="title" x="%d" y="34">%s</text>' % (left, _esc(title)))
    add('<text class="sub" x="%d" y="58">%s · colour P%.0f–P%.0f = %.3g–%.3g s</text>' %
        (left, _esc(metric_label), meta["lower_percentile"], meta["upper_percentile"], lo, hi))
    add('<defs><pattern id="invalid-time" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<rect width="6" height="6" fill="#f6ddb0"/><path d="M0,6 L6,0" stroke="#b48336" stroke-width="1"/>'
        '</pattern></defs>')
    add('<text class="sub" x="%d" y="80">Range: 0–%g m (%s); road length: %s; original bin positions retained</text>' %
        (left, end, _esc(domain["x_range_mode"]),
         "%g m" % road_end if road_end is not None else "unknown"))
    for lx0, fill, label in ((left, "#d9dde3", "No piece record (coverage unknown)"),
                             (left + 325, "url(#invalid-time)", "Recorded, time invalid"),
                             (left + 575, "#aeb5bf", "Outside known road")):
        add('<rect x="%g" y="94" width="12" height="12" fill="%s"/>' % (lx0, fill))
        add('<text class="sub" x="%g" y="105">%s</text>' % (lx0 + 18, label))

    # Compact continuous legend.
    lx, ly, lw = width - right - 250, 118, 250
    for i in range(100):
        v = lo + (hi - lo) * i / 99
        add('<rect x="%.2f" y="%d" width="%.2f" height="10" fill="%s"/>' %
            (lx + lw * i / 100, ly, lw / 100 + 0.2, _colour(v, lo, hi)))
    add('<text class="axis" x="%.1f" y="141" text-anchor="start">%.3g s</text>' % (lx, lo))
    add('<text class="axis" x="%.1f" y="141" text-anchor="end">%.3g s</text>' % (lx + lw, hi))

    # Heatmap.
    for i, row in enumerate(shown):
        for j, value in enumerate(row[:n_draw]):
            y = heat_top + i * rh
            if road_end is not None and j * 10 >= road_end:
                status, fill = "outside_geometry", "#aeb5bf"
            elif math.isfinite(value):
                status, fill = "valid", _colour(float(value), lo, hi)
            elif shown_present[i, j]:
                status, fill = "invalid_time", "url(#invalid-time)"
            else:
                status, fill = "no_record", "#d9dde3"
            add('<rect class="bin" data-bin="%d" data-status="%s" x="%.3f" y="%.3f" width="%.3f" height="%.3f" fill="%s">'
                '<title>%s · bin %d (%g–%gm) · %s</title></rect>' %
                (j, status, x_of(j * 10), y, x_of(bin_end(j)) - x_of(j * 10), rh, fill,
                 _esc(shown_ids[i]) + " · " +
                 datetime.fromtimestamp(float(shown_entry[i]), BEIJING).strftime("%H:%M:%S"),
                 j, j * 10, bin_end(j),
                 status if not math.isfinite(float(value)) else "%.4g s" % value))
            # A grid view may display the non-road remainder of a partial bin.
            if road_end is not None and j * 10 < road_end < min((j + 1) * 10, end):
                add('<rect data-status="outside_geometry" x="%.3f" y="%.3f" width="%.3f" height="%.3f" fill="#aeb5bf"/>' %
                    (x_of(road_end), y, x_of(min((j + 1) * 10, end)) - x_of(road_end), rh))
    add('<rect class="frame" x="%d" y="%d" width="%.1f" height="%.1f"/>' %
        (left, heat_top, plot_w, heat_h))
    add('<text class="label" transform="translate(24 %.1f) rotate(-90)" text-anchor="middle">'
        'Trajectory entry time (early bottom → late top; showing %d/%d)</text>' %
        (heat_top + heat_h / 2, len(shown), matrix.shape[0]))
    # Rows are equally spaced trajectories (not equally spaced seconds). Show
    # the actual entry time of representative rows so ordering is inspectable.
    tick_rows = np.unique(np.linspace(0, len(shown) - 1, min(6, len(shown))).round().astype(int))
    for i in tick_rows:
        y = heat_top + (float(i) + 0.5) * rh
        label = datetime.fromtimestamp(float(shown_entry[i]), BEIJING).strftime("%H:%M:%S")
        add('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#7c8795" stroke-width="1"/>' %
            (left - 5, y, left, y))
        add('<text class="axis" x="%d" y="%.2f" text-anchor="end">%s</text>' %
            (left - 9, y + 4, label))
    for distance in ticks:
        x = x_of(distance)
        add('<line class="grid" x1="%.2f" y1="%d" x2="%.2f" y2="%.1f"/>' %
            (x, heat_top, x, heat_top + heat_h))
        add('<text class="axis" x="%.2f" y="%.1f" text-anchor="middle">%g</text>' %
            (x, heat_top + heat_h + 19, distance))
    add('<text class="label" x="%.1f" y="%.1f" text-anchor="middle">Distance in segment (m)</text>' %
        (left + plot_w / 2, heat_top + heat_h + 47))

    # Per-bin IQR and median.  Empty bins remain gaps.
    dist_values = [d[key] for d in per_bin for key in ("p25_s", "p75_s")
                   if math.isfinite(d[key])]
    dist_lo, dist_hi = min(dist_values), max(dist_values)
    dist_span = max(dist_hi - dist_lo, abs(dist_hi) * 0.05, 1e-3)
    dist_lo = max(0.0, dist_lo - 0.08 * dist_span)
    dist_hi = dist_hi + 0.08 * dist_span
    y_of = lambda v: (dist_top + dist_h -
                      (min(max(v, dist_lo), dist_hi) - dist_lo) /
                      (dist_hi - dist_lo) * dist_h)
    for tick in np.linspace(dist_lo, dist_hi, 5):
        y = y_of(float(tick))
        add('<line class="grid" x1="%d" y1="%.2f" x2="%d" y2="%.2f"/>' %
            (left, y, width - right, y))
        add('<text class="axis" x="%d" y="%.2f" text-anchor="end">%.2g</text>' %
            (left - 9, y + 4, tick))
    for run in _contiguous_statistics(per_bin, end):
        upper = [(center(d["bin_pos"]), y_of(d["p75_s"])) for d in run]
        lower = [(center(d["bin_pos"]), y_of(d["p25_s"])) for d in reversed(run)]
        if len(run) >= 2:
            pts = " ".join("%.2f,%.2f" % p for p in upper + lower)
            add('<polygon class="iqr" points="%s" fill="#8ab4d6" fill-opacity="0.35"/>' % pts)
            med = [(center(d["bin_pos"]), y_of(d["p50_s"])) for d in run]
            add('<polyline class="median" points="%s" fill="none" stroke="#315f86" stroke-width="2.2"/>' %
                " ".join("%.2f,%.2f" % p for p in med))
        else:
            add('<line class="iqr" x1="%.2f" x2="%.2f" y1="%.2f" y2="%.2f" stroke="#8ab4d6" stroke-width="3"/>' %
                (upper[0][0], lower[0][0], upper[0][1], lower[0][1]))
    for d in per_bin:
        if d["bin_pos"] * 10 < end and math.isfinite(d["p50_s"]):
            x, y = center(d["bin_pos"]), y_of(d["p50_s"])
            add('<circle cx="%.2f" cy="%.2f" r="2.6" fill="#315f86"><title>'
                'bin %d · n=%d · P25/P50/P75=%.3g/%.3g/%.3g s</title></circle>' %
                (x, y, d["bin_pos"], d["n"], d["p25_s"], d["p50_s"], d["p75_s"]))
    add('<rect class="frame" x="%d" y="%.1f" width="%.1f" height="%.1f"/>' %
        (left, dist_top, plot_w, dist_h))
    add('<text class="label" x="%d" y="%.1f">Per-bin distribution</text>' % (left, dist_top - 17))
    add('<text class="sub" x="%d" y="%.1f">median line · shaded P25–P75 · zoomed y-scale</text>' %
        (left + 164, dist_top - 17))
    y_label = "Seconds / 10m" if meta["metric"] == "equivalent-10m" else "Raw T_diff (s)"
    add('<text class="label" transform="translate(28 %.1f) rotate(-90)" text-anchor="middle">%s</text>' %
        (dist_top + dist_h / 2, y_label))
    for distance in ticks:
        x = x_of(distance)
        add('<text class="axis" x="%.2f" y="%.1f" text-anchor="middle">%g</text>' %
            (x, dist_top + dist_h + 20, distance))
    add('<text class="label" x="%.1f" y="%.1f" text-anchor="middle">Distance in segment (m)</text>' %
        (left + plot_w / 2, dist_top + dist_h + 48))
    add('</svg>')

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s))
    return {**domain, "presence_source": presence_source,
            "n_recorded_bin_observations": int(present.sum()),
            "n_recorded_invalid_times": int((present & ~np.isfinite(matrix)).sum()),
            "colour_min_s": lo, "colour_max_s": hi,
            "heatmap_rows": int(len(shown)), "svg": os.path.abspath(out_path)}


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
