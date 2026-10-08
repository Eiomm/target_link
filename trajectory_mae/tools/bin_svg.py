"""SVG rendering for per-bin trajectory plots."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import math
import numpy as np
from pathlib import Path

N_BINS = 50
BEIJING = ZoneInfo("Asia/Shanghai")

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
