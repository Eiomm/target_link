"""Create dependency-light summary charts for census_biz_4part.csv.

The four ``n_YYYYMMDDHH`` columns count raw source rows, not distinct
trajectories.  Every chart therefore labels the measure as raw rows.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os

import numpy as np
import pyarrow.csv as pacsv


PERIODS = (
    ("n_2026081907", "08-19 07"),
    ("n_2026081912", "08-19 12"),
    ("n_2026081917", "08-19 17"),
    ("n_2026082003", "08-20 03"),
)
BLUE = "#2474b5"
ORANGE = "#e28b2d"
INK = "#20242a"
MUTED = "#66717e"
GRID = "#e4e8ed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default="data/census_biz_4part.csv")
    p.add_argument("--out-dir", default="runtime/census_biz_4part_charts")
    p.add_argument("--scatter-points", type=int, default=30000)
    p.add_argument("--top-links", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260911)
    return p.parse_args()


def esc(x):
    return html.escape(str(x), quote=True)


def svg_start(width, height, title, subtitle):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img">' %
        (width, height, width, height),
        '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:%s}'
        '.title{font-size:24px;font-weight:650}.sub{font-size:13px;fill:%s}.axis{font-size:12px;fill:%s}'
        '.label{font-size:14px;font-weight:600}.grid{stroke:%s;stroke-width:1}.frame{fill:none;stroke:#aeb6c0}</style>' %
        (INK, MUTED, MUTED, GRID),
        '<text class="title" x="80" y="38">%s</text>' % esc(title),
        '<text class="sub" x="80" y="63">%s</text>' % esc(subtitle),
    ]


def save_svg(parts, path):
    parts.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def nice_ticks(lo, hi, count=5):
    if hi <= lo:
        return np.array([lo])
    return np.linspace(lo, hi, count)


def heat_colour(value, lo, hi):
    x = min(1.0, max(0.0, (value - lo) / max(hi - lo, 1e-12)))
    stops = ((245, 248, 251), (95, 166, 210), (8, 81, 156))
    if x <= 0.5:
        a, b, u = stops[0], stops[1], x * 2
    else:
        a, b, u = stops[1], stops[2], (x - 0.5) * 2
    rgb = tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))
    return "#%02x%02x%02x" % rgb


def draw_histogram(values, path):
    logv = np.log10(values[values > 0])
    bins = np.linspace(math.floor(logv.min() * 4) / 4,
                       math.ceil(logv.max() * 4) / 4, 37)
    counts, edges = np.histogram(logv, bins=bins)
    width, height = 1200, 760
    left, right, top, bottom = 110, 55, 105, 105
    pw, ph = width - left - right, height - top - bottom
    maxc = max(counts.max(), 1)
    s = svg_start(width, height, "Link activity distribution",
                  "Activity = raw source rows summed across four sampled periods; x-axis is logarithmic")
    for t in nice_ticks(0, maxc, 6):
        y = top + ph * (1 - t / maxc)
        s.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (left, y, left + pw, y))
        s.append('<text class="axis" x="%d" y="%.1f" text-anchor="end">%s</text>' %
                 (left - 10, y + 4, format(int(t), ",")))
    bw = pw / len(counts)
    for i, c in enumerate(counts):
        h = ph * c / maxc
        x, y = left + i * bw + 1, top + ph - h
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s">'
                 '<title>raw rows %.0f–%.0f: %s links</title></rect>' %
                 (x, y, max(1, bw - 2), h, BLUE, 10 ** edges[i], 10 ** edges[i + 1], format(int(c), ",")))
    xmin, xmax = edges[0], edges[-1]
    powers = np.arange(math.ceil(xmin), math.floor(xmax) + 1)
    for p in powers:
        x = left + pw * (p - xmin) / (xmax - xmin)
        s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#7b8794"/>' % (x, top + ph, x, top + ph + 6))
        s.append('<text class="axis" x="%.1f" y="%d" text-anchor="middle">10<tspan baseline-shift="super" font-size="9">%d</tspan></text>' %
                 (x, top + ph + 25, p))
    med, p90, p99 = np.percentile(values, [50, 90, 99])
    s.append('<text class="sub" x="%d" y="91">Median %s · P90 %s · P99 %s raw rows</text>' %
             (left, format(int(med), ","), format(int(p90), ","), format(int(p99), ",")))
    s.append('<rect class="frame" x="%d" y="%d" width="%d" height="%d"/>' % (left, top, pw, ph))
    s.append('<text class="label" x="%.1f" y="%d" text-anchor="middle">Raw rows per link (log scale)</text>' %
             (left + pw / 2, height - 31))
    s.append('<text class="label" transform="translate(28 %.1f) rotate(-90)" text-anchor="middle">Number of links</text>' %
             (top + ph / 2))
    save_svg(s, path)


def draw_scatter(lengths, totals, parts, path, max_points, seed):
    valid = np.flatnonzero((lengths > 0) & (totals > 0))
    if len(valid) > max_points:
        valid = np.random.default_rng(seed).choice(valid, max_points, replace=False)
    x, y, p = np.log10(lengths[valid]), np.log10(totals[valid]), parts[valid]
    width, height = 1200, 820
    left, right, top, bottom = 110, 60, 110, 105
    pw, ph = width - left - right, height - top - bottom
    xmin, xmax = np.percentile(x, [0.2, 99.8])
    ymin, ymax = np.percentile(y, [0.2, 99.8])
    colors = {1: "#9ba4ae", 2: "#7a9fc2", 3: "#e2a44f", 4: "#bd4d48"}
    s = svg_start(width, height, "Road length vs. activity",
                  "Deterministic sample of %s links · activity is raw rows · both axes logarithmic" % format(len(valid), ","))
    for val in range(math.ceil(xmin), math.floor(xmax) + 1):
        xx = left + pw * (val - xmin) / (xmax - xmin)
        s.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (xx, top, xx, top + ph))
        s.append('<text class="axis" x="%.1f" y="%d" text-anchor="middle">10<tspan baseline-shift="super" font-size="9">%d</tspan></text>' % (xx, top + ph + 25, val))
    for val in range(math.ceil(ymin), math.floor(ymax) + 1):
        yy = top + ph * (1 - (val - ymin) / (ymax - ymin))
        s.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (left, yy, left + pw, yy))
        s.append('<text class="axis" x="%d" y="%.1f" text-anchor="end">10<tspan baseline-shift="super" font-size="9">%d</tspan></text>' % (left - 10, yy + 4, val))
    # Draw broad-coverage links last so they remain visible.
    for group in (1, 2, 3, 4):
        idx = np.flatnonzero(p == group)
        for i in idx:
            if not (xmin <= x[i] <= xmax and ymin <= y[i] <= ymax):
                continue
            xx = left + pw * (x[i] - xmin) / (xmax - xmin)
            yy = top + ph * (1 - (y[i] - ymin) / (ymax - ymin))
            s.append('<circle cx="%.2f" cy="%.2f" r="2" fill="%s" fill-opacity="0.34"/>' %
                     (xx, yy, colors[group]))
    lx = left + 15
    for i, group in enumerate((1, 2, 3, 4)):
        xx = lx + i * 145
        s.append('<circle cx="%d" cy="91" r="5" fill="%s"/><text class="sub" x="%d" y="95">%d/4 periods</text>' %
                 (xx, colors[group], xx + 11, group))
    s.append('<rect class="frame" x="%d" y="%d" width="%d" height="%d"/>' % (left, top, pw, ph))
    s.append('<text class="label" x="%.1f" y="%d" text-anchor="middle">Link length (m, log scale)</text>' %
             (left + pw / 2, height - 31))
    s.append('<text class="label" transform="translate(28 %.1f) rotate(-90)" text-anchor="middle">Raw rows per link (log scale)</text>' %
             (top + ph / 2))
    save_svg(s, path)


def draw_heatmap(link_ids, matrix, totals, path, top_n):
    order = np.argsort(totals, kind="stable")[-top_n:][::-1]
    data = matrix[order]
    labels = link_ids[order]
    logs = np.log1p(data.astype(np.float64))
    lo, hi = 0.0, float(logs.max())
    width, height = 1200, max(850, 135 + top_n * 17)
    left, right, top, bottom = 260, 80, 125, 55
    pw, ph = width - left - right, height - top - bottom
    cw, rh = pw / len(PERIODS), ph / top_n
    s = svg_start(width, height, "Top-link activity heatmap",
                  "Top %d links by four-period total · colour = log1p(raw rows), shared across all cells" % top_n)
    for j, (_, label) in enumerate(PERIODS):
        s.append('<text class="label" x="%.1f" y="106" text-anchor="middle">%s</text>' %
                 (left + (j + .5) * cw, label))
    for i in range(top_n):
        y = top + i * rh
        s.append('<text class="axis" x="%d" y="%.1f" text-anchor="end">%s</text>' %
                 (left - 12, y + rh * .67, esc(labels[i])))
        for j in range(len(PERIODS)):
            value = int(data[i, j])
            color = heat_colour(float(logs[i, j]), lo, hi)
            text_color = "#fff" if logs[i, j] > hi * .62 else INK
            s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s">'
                     '<title>link %s · %s · %s raw rows</title></rect>' %
                     (left + j * cw, y, cw + .3, rh + .3, color,
                      esc(labels[i]), PERIODS[j][1], format(value, ",")))
            if rh >= 15:
                s.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="10" fill="%s">%s</text>' %
                         (left + (j + .5) * cw, y + rh * .69, text_color, format(value, ",")))
    s.append('<rect class="frame" x="%d" y="%d" width="%d" height="%d"/>' % (left, top, pw, ph))
    s.append('<text class="label" transform="translate(28 %.1f) rotate(-90)" text-anchor="middle">target_link_id (descending total activity)</text>' %
             (top + ph / 2))
    save_svg(s, path)


def draw_coverage(parts, path):
    counts = np.array([(parts == i).sum() for i in range(1, 5)], dtype=np.int64)
    total = int(counts.sum())
    width, height = 1100, 730
    left, right, top, bottom = 115, 65, 115, 100
    pw, ph = width - left - right, height - top - bottom
    maxc = max(int(counts.max()), 1)
    colors = ("#9ba4ae", "#7a9fc2", "#e2a44f", "#bd4d48")
    s = svg_start(width, height, "Temporal coverage of links",
                  "How many of the four sampled periods contain at least one raw row for each link")
    for t in nice_ticks(0, maxc, 6):
        y = top + ph * (1 - t / maxc)
        s.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (left, y, left + pw, y))
        s.append('<text class="axis" x="%d" y="%.1f" text-anchor="end">%s</text>' %
                 (left - 10, y + 4, format(int(t), ",")))
    slot = pw / 4
    for i, c in enumerate(counts):
        bw = slot * .58
        x = left + i * slot + (slot - bw) / 2
        h = ph * c / maxc
        y = top + ph - h
        pct = c / total * 100 if total else 0
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" fill="%s"/>' %
                 (x, y, bw, h, colors[i]))
        s.append('<text class="label" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' %
                 (x + bw / 2, y - 23, format(int(c), ",")))
        s.append('<text class="sub" x="%.1f" y="%.1f" text-anchor="middle">%.1f%%</text>' %
                 (x + bw / 2, y - 7, pct))
        s.append('<text class="axis" x="%.1f" y="%d" text-anchor="middle">%d of 4</text>' %
                 (x + bw / 2, top + ph + 25, i + 1))
    s.append('<rect class="frame" x="%d" y="%d" width="%d" height="%d"/>' % (left, top, pw, ph))
    s.append('<text class="label" x="%.1f" y="%d" text-anchor="middle">Periods present</text>' %
             (left + pw / 2, height - 28))
    s.append('<text class="label" transform="translate(28 %.1f) rotate(-90)" text-anchor="middle">Number of links</text>' %
             (top + ph / 2))
    save_svg(s, path)


def main():
    a = parse_args()
    table = pacsv.read_csv(a.input)
    required = ["target_link_id", "L_link_m", "n_rows_total", "n_parts_present"] + [x[0] for x in PERIODS]
    missing = [x for x in required if x not in table.column_names]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))

    def col(name, dtype):
        return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)

    link_ids = col("target_link_id", str)
    lengths = col("L_link_m", np.float64)
    totals = col("n_rows_total", np.int64)
    parts = col("n_parts_present", np.int8)
    matrix = np.column_stack([col(name, np.int64) for name, _ in PERIODS])
    sum_mismatch = int(np.count_nonzero(matrix.sum(axis=1) != totals))
    parts_mismatch = int(np.count_nonzero((matrix > 0).sum(axis=1) != parts))
    if sum_mismatch or parts_mismatch:
        raise ValueError("CSV consistency check failed: total=%d, parts=%d mismatches" %
                         (sum_mismatch, parts_mismatch))

    outputs = {
        "activity_distribution": os.path.join(a.out_dir, "01_activity_distribution", "activity_distribution.svg"),
        "length_activity": os.path.join(a.out_dir, "02_length_activity", "length_activity.svg"),
        "top_links_heatmap": os.path.join(a.out_dir, "03_top_links_heatmap", "top_links_heatmap.svg"),
        "temporal_coverage": os.path.join(a.out_dir, "04_temporal_coverage", "temporal_coverage.svg"),
    }
    draw_histogram(totals, outputs["activity_distribution"])
    draw_scatter(lengths, totals, parts, outputs["length_activity"], a.scatter_points, a.seed)
    draw_heatmap(link_ids, matrix, totals, outputs["top_links_heatmap"], min(a.top_links, len(totals)))
    draw_coverage(parts, outputs["temporal_coverage"])

    summary = {
        "input": os.path.abspath(a.input),
        "measure": "raw source rows (not distinct trajectories)",
        "n_links": int(len(totals)),
        "periods": [label for _, label in PERIODS],
        "total_raw_rows": int(totals.sum()),
        "activity_raw_rows": {"min": int(totals.min()), "p50": float(np.percentile(totals, 50)),
                              "p90": float(np.percentile(totals, 90)), "p99": float(np.percentile(totals, 99)),
                              "max": int(totals.max())},
        "length_m": {"p50": float(np.percentile(lengths, 50)), "p90": float(np.percentile(lengths, 90)),
                     "max": float(lengths.max())},
        "links_by_parts_present": {str(i): int((parts == i).sum()) for i in range(1, 5)},
        "consistency_checks": {"total_mismatches": sum_mismatch, "parts_mismatches": parts_mismatch},
        "outputs": outputs,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
