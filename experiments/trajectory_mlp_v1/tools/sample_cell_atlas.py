"""Reproducible two-stage sample of training cells, plus a descriptive plot atlas.

Only writes under --out. No model fitting, interpolation, or test-day reads.
Sampling unit is a COMPLETE cell with >=3 observation rows, not a training group.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from plot_link_bin_times import fold_observation, observation_presence, _quantiles_by_bin, render_svg

TZ = ZoneInfo("Asia/Shanghai")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_counts(present, valid, observed):
    """Interior means strictly between first/last recorded bin, NOT all 50 slots."""
    indices = np.flatnonzero(present)
    interior = np.zeros(50, dtype=bool)
    if len(indices):
        interior[indices[0]:indices[-1] + 1] = True
    gps_indices = np.flatnonzero(observed)
    gps_interior = np.zeros(50, dtype=bool)
    if len(gps_indices):
        gps_interior[gps_indices[0]:gps_indices[-1] + 1] = True
    return dict(n_present=int(present.sum()), n_valid=int(valid.sum()),
                n_no_gps_valid=int((valid & ~observed).sum()),
                n_invalid_recorded=int((present & ~valid).sum()),
                n_interior_no_record=int((interior & ~present).sum()),
                n_internal_no_gps_valid=int((gps_interior & valid & ~observed).sum()),
                start_m=int(indices[0] * 10) if len(indices) else 0,
                end_upper_m=int((indices[-1] + 1) * 10) if len(indices) else 0)


def choose_cells(ids, rng, n):
    cells, counts = np.unique(ids, return_counts=True)
    eligible = cells[counts >= 3]
    chosen = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    return chosen, dict(total_cells=len(cells), eligible_cells=len(eligible),
                        observation_rows=len(ids), selected_cells=len(chosen))


def sample(args):
    out = args.out
    rng = np.random.default_rng(args.seed)
    tables, strata = [], []
    for day in args.days:
        base = args.corpus / "observations_v2" / ("day=" + day)
        buckets = sorted(int(p.name.split("=")[1]) for p in base.glob("bucket=*"))
        if buckets != list(range(128)):
            raise ValueError(f"Expected complete 128-bucket frame for {day}")
        selected = sorted(rng.choice(buckets, args.buckets_per_day, replace=False).tolist())
        for bucket in selected:
            files = sorted((base / f"bucket={bucket}").glob("*.parquet"))
            scalar = [pq.ParquetFile(p).read(columns=["cell_id"]).column(0).to_numpy() for p in files]
            ids = np.concatenate(scalar)
            chosen, counts = choose_cells(ids, rng, args.cells_per_bucket)
            if not len(chosen):
                raise ValueError("Empty eligible sampling stratum")
            selected_rows = []
            values = pa.array(chosen, type=pa.int64())
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                    kept = batch.filter(pc.is_in(batch.column("cell_id"), value_set=values))
                    if kept.num_rows:
                        selected_rows.append(pa.Table.from_batches([kept]))
            table = pa.concat_tables(selected_rows)
            table = table.append_column("sample_day", pa.array([day] * len(table)))
            table = table.append_column("sample_bucket", pa.array([bucket] * len(table), type=pa.int32()))
            tables.append(table)
            strata.append(dict(day=day, bucket=bucket, **counts,
                               inclusion_probability=args.buckets_per_day / 128 * len(chosen) / counts["eligible_cells"],
                               cell_weight=128 / args.buckets_per_day * counts["eligible_cells"] / len(chosen),
                               selected_cell_ids=[str(int(c)) for c in chosen],
                               input_files=[dict(path=str(p.resolve()), bytes=p.stat().st_size,
                                                 mtime_ns=p.stat().st_mtime_ns) for p in files]))
            print(f"sample {day}/{bucket}: {counts}, extracted {len(table)} rows", flush=True)
    sampled = pa.concat_tables(tables)
    pq.write_table(sampled, out / "sampled_observations.parquet", compression="zstd")
    write_json(out / "sampling_manifest.json", dict(
        seed=args.seed, days=args.days, corpus=str(args.corpus.resolve()),
        buckets_per_day=args.buckets_per_day, cells_per_bucket=args.cells_per_bucket,
        method="Within each day: SRSWOR buckets, then SRSWOR cells with >=3 rows; retain ALL rows of each selected cell.",
        scope="Descriptive sample of local training corpus only. No inference to unseen dates/cities, no model significance test.",
        sample_sha256=hashlib.sha256((out / "sampled_observations.parquet").read_bytes()).hexdigest(), strata=strata))


def weighted_quantile(values, weights, qs):
    order = np.argsort(values)
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = np.cumsum(w) / np.sum(w)
    return [float(x[min(np.searchsorted(cdf, q), len(x) - 1)]) for q in qs]


def ratio_ci(cells, numerator, denominator, seed, n_boot=2000):
    """Approximate stratified two-stage bootstrap; buckets, then cells within bucket.

    Uses inverse inclusion weights, retaining the day strata. Percentile intervals
    are descriptive design approximations (four sampled PSUs/day by default).
    """
    groups = defaultdict(list)
    for c in cells:
        groups[c["day"]].append(c)
    arrays = []
    for day in sorted(groups):
        by_bucket = defaultdict(list)
        for c in groups[day]:
            w = c["weight"]
            by_bucket[c["bucket"]].append([w * c[numerator], w * c[denominator]])
        arrays.append([np.array(v) for _, v in sorted(by_bucket.items())])
    totals = np.sum([a.sum(0) for day in arrays for a in day], axis=0)
    if totals[0] == 0:
        return dict(estimate=0.0, ci95=None,
                    note="No sampled events: ordinary bootstrap degenerates; no informative population upper bound from this bootstrap.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        accum = np.zeros(2)
        for day in arrays:
            for j in rng.integers(0, len(day), size=len(day)):
                a = day[j]
                accum += a[rng.integers(0, len(a), size=len(a))].sum(0)
        if accum[1]:
            estimates.append(accum[0] / accum[1])
    return dict(estimate=float(totals[0] / totals[1]),
                ci95=[float(v) for v in np.quantile(estimates, [.025, .975])])


def prepare(out):
    manifest = json.loads((out / "sampling_manifest.json").read_text())
    rows = pq.ParquetFile(out / "sampled_observations.parquet").read().to_pylist()
    lookup = {(s["day"], s["bucket"]): s for s in manifest["strata"]}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample_day"], row["sample_bucket"], row["cell_id"])].append(row)
    cells, trajectories, data = [], [], {}
    for (day, bucket, cid), group in sorted(grouped.items()):
        if len({r["sample_id"] for r in group}) != len(group):
            raise ValueError("Duplicated sample within sampled cell")
        if len({(r["map_version"], r["target_link_id"], r["seg_idx"], r["window"]) for r in group}) != 1:
            raise ValueError("Cell id collision or inconsistent metadata")
        weight = lookup[day, bucket]["cell_weight"]
        group.sort(key=lambda r: (r["dt"], r["sample_id"]))
        values, presence, gps, traj = [], [], [], []
        for row in group:
            value, valid, ratio = fold_observation(row, metric="raw")
            present = observation_presence(row)
            observed = np.zeros(50, dtype=bool)
            for pos, flag in zip(row["bin_pos"], row["observed"]):
                observed[pos] |= bool(flag)
            # Fail explicitly rather than silently changing training-reader semantics.
            original_valid = np.zeros(50, dtype=bool)
            for pos in np.flatnonzero(present):
                original_valid[pos] = all(v for p, v in zip(row["bin_pos"], row["valid"]) if p == pos)
            if not np.array_equal(original_valid, valid) or np.any(value[valid] < 0):
                raise ValueError("Plot/training validity mismatch or negative time: requires explicit QC")
            stats = gap_counts(present, valid, observed)
            tr = dict(day=day, bucket=bucket, cell_id=str(cid), sample_id=row["sample_id"],
                      weight=weight, **stats)
            trajectories.append(tr)
            traj.append(tr)
            values.append(value)
            presence.append(present)
            gps.append(observed)
        matrix, present, observed = map(np.array, (values, presence, gps))
        first = group[0]
        end = int((np.flatnonzero(present.any(0))[-1] + 1) * 10)
        record = dict(day=day, bucket=bucket, cell_id=str(cid), weight=weight,
                      target_link_id=first["target_link_id"], map_version=first["map_version"],
                      seg_idx=first["seg_idx"], window=first["window"],
                      hour=datetime.fromtimestamp(first["window"], TZ).hour,
                      K=len(group), one=1, end_upper_m=end,
                      median_valid_bins=float(np.median([t["n_valid"] for t in traj])),
                      n_gap_traj=sum(t["n_interior_no_record"] > 0 for t in traj),
                      n_internal_no_gps_traj=sum(t["n_internal_no_gps_valid"] > 0 for t in traj),
                      n_invalid_traj=sum(t["n_invalid_recorded"] > 0 for t in traj),
                      n_zero_valid_traj=sum(t["n_valid"] == 0 for t in traj),
                      n_present=int(present.sum()), n_valid=int(np.isfinite(matrix).sum()),
                      n_no_gps_valid=int((np.isfinite(matrix) & ~observed).sum()),
                      n_interior_no_record=sum(t["n_interior_no_record"] for t in traj),
                      raw_time_p50_s=float(np.nanmedian(matrix)))
        cells.append(record)
        data[str(cid)] = (matrix, present, observed, group)
    write_csv(out / "cell_statistics.csv", cells)
    write_csv(out / "trajectory_statistics.csv", trajectories)
    summary = dict(sampled_cells=len(cells), sampled_trajectories=len(trajectories),
                   sampled_unique_links=len({c["target_link_id"] for c in cells}),
                   sampled_unique_map_link_segments=len({(c["map_version"], c["target_link_id"], c["seg_idx"]) for c in cells}),
                   scanned_cells=sum(s["total_cells"] for s in manifest["strata"]),
                   scanned_eligible_cells=sum(s["eligible_cells"] for s in manifest["strata"]),
                   scanned_observation_rows=sum(s["observation_rows"] for s in manifest["strata"]),
                   valid_bins=sum(c["n_valid"] for c in cells),
                   weighting="Cell inverse inclusion weights. Trajectory/bin ratios are ratio-of-weighted-totals, not mean of cell percentages.",
                   ci_method="2000 stratified two-stage percentile bootstrap replicates: resample buckets within day, then cells; approximate, not a performance test.")
    for name, numerator, denominator in [
        ("trajectory_internal_no_record_fraction", "n_gap_traj", "K"),
        ("trajectory_internal_no_gps_valid_fraction", "n_internal_no_gps_traj", "K"),
        ("trajectory_recorded_invalid_fraction", "n_invalid_traj", "K"),
        ("valid_bin_no_direct_gps_fraction", "n_no_gps_valid", "n_valid"),
        ("mean_valid_bins_per_trajectory", "n_valid", "K"),
        ("valid_fraction_of_recorded_bins", "n_valid", "n_present"),
    ]:
        summary[name] = ratio_ci(cells, numerator, denominator, manifest["seed"])
    summary["cell_K_weighted_p10_p50_p90"] = weighted_quantile(
        [c["K"] for c in cells], [c["weight"] for c in cells], [.1, .5, .9])
    summary["cell_end_upper_m_weighted_p10_p50_p90"] = weighted_quantile(
        [c["end_upper_m"] for c in cells], [c["weight"] for c in cells], [.1, .5, .9])
    write_json(out / "statistics.json", summary)
    return manifest, cells, trajectories, data, summary


def make_plots(out, manifest, cells, trajectories, data, summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/trajectory_mlp_v1_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
    from matplotlib.patches import Patch

    cmap = LinearSegmentedColormap.from_list("travel", ["#2ca25f", "#fee08b", "#d73027"])
    cmap.set_bad("#e2e5e9")
    root = out / "cells"
    root.mkdir(exist_ok=True)
    # All selected cells get equal chance of appearing in the random gallery,
    # independent of their plotted values: the first two randomly drawn IDs/stratum.
    chosen = [(cid, "random") for s in manifest["strata"] for cid in s["selected_cell_ids"][:2]]
    already = {c for c, _ in chosen}
    categories = [
        ("internal_no_record", lambda c: c["n_interior_no_record"] > 0, lambda c: -c["n_interior_no_record"]),
        ("internal_no_gps", lambda c: c["n_internal_no_gps_traj"] > 0, lambda c: -c["n_internal_no_gps_traj"]),
        ("recorded_invalid", lambda c: c["n_invalid_traj"] > 0, lambda c: -c["n_invalid_traj"]),
        ("short_coverage", lambda c: c["end_upper_m"] <= 50, lambda c: -c["K"]),
        ("long_coverage", lambda c: c["end_upper_m"] >= 400, lambda c: -c["K"]),
    ]
    for name, predicate, key in categories:
        for c in sorted((c for c in cells if predicate(c) and c["cell_id"] not in already), key=key)[:3]:
            chosen.append((c["cell_id"], name))
            already.add(c["cell_id"])
    by_id = {c["cell_id"]: c for c in cells}
    gallery = []
    common_norm = Normalize(0, 5, clip=True)
    state_colors = ["#e2e5e9", "#aa88bb", "#9bc7ec", "#177f70"]

    def heat(ax, cid, compact=False):
        c = by_id[cid]
        matrix, present, gps, rows = data[cid]
        order = np.arange(len(rows))
        if len(order) > 300:
            order = order[np.linspace(0, len(order) - 1, 300).round().astype(int)]
        end = c["end_upper_m"]
        n = end // 10
        shown = matrix[order, :n]
        ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=common_norm,
                  extent=(0, end, 0, len(order)), aspect="auto", origin="lower", interpolation="nearest")
        invalid_y, invalid_x = np.where(present[order, :n] & ~np.isfinite(shown))
        ax.scatter((invalid_x + .5) * 10, invalid_y + .5, marker="x", color="#6c5b9c", s=12, linewidths=.7)
        ax.set_xlim(0, end)
        ax.set_xlabel("Distance from segment origin (m)")
        ax.set_ylabel(f"Trajectory rank (K={c['K']})")
        when = datetime.fromtimestamp(c["window"], TZ).strftime("%m-%d %H:%M")
        ax.set_title(f"Link {c['target_link_id']} / seg {c['seg_idx']}\n{when} | endpoint <= {end} m", fontsize=9 if compact else 11)
        return matrix, present, gps, rows

    for number, (cid, category) in enumerate(chosen, 1):
        c = by_id[cid]
        matrix, present, gps, rows = data[cid]
        if not np.isfinite(matrix).any():
            raise ValueError("Random gallery contains unplottable cell: do not silently replace it")
        stem = f"{number:02d}_{category}_{cid}"
        meta = dict(target_link_id=c["target_link_id"], seg_idx=c["seg_idx"],
                    window_local=datetime.fromtimestamp(c["window"], TZ).strftime("%Y-%m-%d %H:%M"),
                    n_trajectories=c["K"], metric="raw", lower_percentile=5, upper_percentile=95,
                    x_range_mode="coverage", segment_length_m=None)
        stats = _quantiles_by_bin(matrix)
        svg_summary = render_svg(matrix, [r["sample_id"] for r in rows],
                                 [r["window"] + r["dt"] for r in rows], meta, stats,
                                 str(root / (stem + ".svg")), present=present)
        write_json(root / (stem + ".json"), dict(**c, selection=category, plot=svg_summary))
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 2, 1]}, layout="constrained")
        heat(axes[0], cid)
        state = np.zeros(matrix.shape, dtype=int)
        state[present & ~np.isfinite(matrix)] = 1
        state[np.isfinite(matrix) & ~gps] = 2
        state[np.isfinite(matrix) & gps] = 3
        order = np.arange(len(rows))
        if len(order) > 300:
            order = order[np.linspace(0, len(order) - 1, 300).round().astype(int)]
        axes[1].imshow(state[order, :c["end_upper_m"] // 10], cmap=ListedColormap(state_colors), vmin=-.5, vmax=3.5,
                       extent=(0, c["end_upper_m"], 0, len(order)), origin="lower", aspect="auto", interpolation="nearest")
        axes[1].set(title="Observation status: same rows and spatial bins", xlabel="Distance (m)", ylabel="Trajectory rank")
        axes[1].legend(handles=[Patch(color=color, label=label) for color, label in zip(state_colors,
                       ["No piece record", "Recorded, invalid time", "Valid time, no direct GPS", "Valid time, direct GPS"])],
                       loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=8)
        med = np.array([s["p50_s"] for s in stats])
        p25, p75 = [np.array([s[f"p{p}_s"] for s in stats]) for p in (25, 75)]
        x = np.arange(50) * 10 + 5
        axes[2].plot(x, med, color="#225588", label="Median")
        axes[2].fill_between(x, p25, p75, color="#92b5d0", alpha=.5, label="25-75%")
        axes[2].set(xlim=(0, c["end_upper_m"]), ylabel="Raw bin time (s)", xlabel="Distance (m)")
        axes[2].legend(loc="upper right", fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(norm=common_norm, cmap=cmap), ax=axes[0], label="Raw bin time (s); colors clipped at 5 s", extend="max")
        fig.suptitle(f"{category} | gray: no piece record; purple x: recorded, invalid time\nRows sorted by entry time, earliest at bottom. Geometry unknown; no interpolation.", fontsize=10)
        fig.savefig(root / (stem + ".png"), dpi=130)
        plt.close(fig)
        gallery.append(dict(number=number, cell_id=cid, selection=category, stem=stem,
                            target_link_id=c["target_link_id"], K=c["K"], end_upper_m=c["end_upper_m"]))
    write_json(out / "gallery_manifest.json", gallery)
    for page, start in enumerate(range(0, len(gallery), 8), 1):
        fig, axs = plt.subplots(4, 2, figsize=(14, 14), layout="constrained")
        for ax, item in zip(axs.flat, gallery[start:start + 8]):
            heat(ax, item["cell_id"], compact=True)
            ax.text(.99, .02, f"#{item['number']} {item['selection']}", transform=ax.transAxes, ha="right", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        for ax in list(axs.flat)[len(gallery[start:start + 8]):]:
            ax.set_visible(False)
        fig.colorbar(plt.cm.ScalarMappable(norm=common_norm, cmap=cmap), ax=list(axs.flat), shrink=.6,
                     label="Raw time (s), common 0-5 s scale; values >5 s saturated", extend="max")
        fig.suptitle(f"Actual training cells | page {page} | gray = no record, x = invalid time", fontsize=14)
        fig.savefig(out / f"contact_sheet_{page:02d}.png", dpi=130)
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    weights = np.array([c["weight"] for c in cells])
    weights = weights / weights.sum() * 100
    axs[0, 0].hist([c["end_upper_m"] for c in cells], bins=np.arange(0, 501, 50), weights=weights, color="#427f9e", edgecolor="white")
    axs[0, 0].set(title="Farthest recorded endpoint per cell (not road length)", xlabel="10 m grid upper bound (m)", ylabel="Estimated eligible cells (%)")
    edges = [3, 5, 10, 20, 50, 100, 1000, float("inf")]
    counts, _ = np.histogram([c["K"] for c in cells], bins=edges, weights=weights)
    axs[0, 1].bar(["3-4", "5-9", "10-19", "20-49", "50-99", "100-999", ">=1000"], counts, color="#578c6f")
    axs[0, 1].set(title="Complete cell size (before splitting into training groups)", xlabel="Trajectories per cell", ylabel="Estimated eligible cells (%)")
    tw = np.array([t["weight"] for t in trajectories]); tw = tw / tw.sum() * 100
    axs[1, 0].hist([t["n_valid"] for t in trajectories], bins=np.arange(-.5, 51.5, 1), weights=tw, color="#ba8d50")
    axs[1, 0].set(title="Real valid bins per trajectory", xlabel="Valid bins out of 50 (unfilled)", ylabel="Estimated trajectories (%)")
    keys = ["trajectory_internal_no_record_fraction", "trajectory_recorded_invalid_fraction", "valid_bin_no_direct_gps_fraction"]
    vals = np.array([summary[k]["estimate"] for k in keys]) * 100
    cis = np.array([summary[k]["ci95"] if summary[k]["ci95"] is not None
                    else [summary[k]["estimate"]] * 2 for k in keys]) * 100
    axs[1, 1].barh(["Traj: interior no-record gap", "Traj: recorded invalid time", "Valid bins: no direct GPS"], vals,
                   xerr=np.maximum(0, np.stack([vals - cis[:, 0], cis[:, 1] - vals])), color=["#8e9ead", "#9b83b5", "#628ca3"], capsize=4)
    axs[1, 1].set(title="Different concepts; different stated denominators", xlabel="Weighted proportion (%) with approximate 95% CI", xlim=(0, 100))
    axs[1, 1].text(.03, .97, "No interior no-record gaps in sample;\n0 events does not imply population rate = 0.",
                    transform=axs[1, 1].transAxes, va="top", fontsize=8)
    fig.suptitle(f"{len(cells):,} randomly sampled cells | {len(trajectories):,} trajectories | {len(manifest['days'])} training days\nInverse-probability weighting; stratified two-stage bootstrap; descriptive statistics", fontsize=14)
    fig.savefig(out / "statistical_overview.png", dpi=150)
    plt.close(fig)
    cards = []
    for item in gallery:
        stem = html.escape(item["stem"])
        cards.append(f'<article><h3>#{item["number"]} {item["selection"]} · K={item["K"]}</h3><a href="cells/{stem}.svg"><img loading="lazy" src="cells/{stem}.png"></a><p><a href="cells/{stem}.svg">交互SVG</a> · <a href="cells/{stem}.json">来源与统计</a></p></article>')
    (out / "index.html").write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实轨迹数据图册</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#fafbfc;color:#243448}img{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}article{background:white;padding:16px;border:1px solid #ddd;border-radius:10px}p{line-height:1.7}</style>
<h1>真实轨迹数据图册</h1><p>随机样本用于描述训练语料；定向案例仅用于理解缺口。先看 <a href="统计说明.md">中文统计说明</a>。所有样本均来自实际数据，无插值、无训练。</p>
<p>PNG统一0–5秒色标，超过5秒颜色饱和但原值保留。SVG采用每张图自己的P5–P95色标，可以悬停查看值，颜色不能直接跨SVG比较。灰色=无piece记录；紫叉=有记录但耗时无效。纵轴按进入时间排序、早下晚上，行间距不是时间间距。横轴沿用同一segment坐标，终点是有记录bin的10m网格上界，不等于已知道路长度。</p>
<img src="statistical_overview.png"><div class="grid">''' + "\n".join(cards) + "</div></html>\n")
    print(f"rendered {len(gallery)} real cells, {len(range(0, len(gallery), 8))} contact sheets", flush=True)


def write_report(out):
    manifest = json.loads((out / "sampling_manifest.json").read_text())
    stats = json.loads((out / "statistics.json").read_text())
    gallery = json.loads((out / "gallery_manifest.json").read_text())
    with (out / "cell_statistics.csv").open() as f:
        cells = list(csv.DictReader(f))
    days = []
    for day in manifest["days"]:
        strata = [s for s in manifest["strata"] if s["day"] == day]
        group = [c for c in cells if c["day"] == day]
        days.append(f"| {day} | {', '.join(str(s['bucket']) for s in strata)} | {sum(s['eligible_cells'] for s in strata):,} | {len(group)} | {sum(int(c['K']) for c in group):,} |")
    rows = []
    measures = [
        ("每条轨迹平均有效 bin 数", "mean_valid_bins_per_trajectory", False),
        ("有效 bin 中无直接 GPS 的比例", "valid_bin_no_direct_gps_fraction", True),
        ("有首尾 GPS、中间至少一个 bin 无 GPS 但耗时有效的轨迹比例", "trajectory_internal_no_gps_valid_fraction", True),
        ("至少有一个‘有记录但耗时无效’bin 的轨迹比例", "trajectory_recorded_invalid_fraction", True),
        ("有 piece 记录的 bin 中，耗时有效的比例", "valid_fraction_of_recorded_bins", True),
        ("首尾 piece 之间存在整 bin 无记录缺口的轨迹比例", "trajectory_internal_no_record_fraction", True),
    ]
    for label, key, percent in measures:
        s = stats[key]
        scale, suffix = (100, "%") if percent else (1, " 个")
        interval = "样本 0 例；不报告退化的 [0,0] 区间" if s["ci95"] is None else f"{s['ci95'][0]*scale:.2f}–{s['ci95'][1]*scale:.2f}{suffix}"
        rows.append(f"| {label} | {s['estimate']*scale:.2f}{suffix} | {interval} |")
    n_random = sum(g["selection"] == "random" for g in gallery)
    examples = []
    titles = {"internal_no_gps": "中间无 GPS，但已有有效耗时", "recorded_invalid": "有记录，但部分耗时无效", "short_coverage": "短覆盖范围", "long_coverage": "较长覆盖范围"}
    for key, label in titles.items():
        item = next((g for g in gallery if g["selection"] == key), None)
        if item:
            examples.append(f"- [{label}：图 {item['number']}，link {item['target_link_id']}，K={item['K']}](cells/{item['stem']}.png)")
    text = f"""# 真实轨迹数据图册与统计说明

这次实际扫描了 **{stats['scanned_observation_rows']:,} 条 observation 记录**，得到已选分桶中的 {stats['scanned_cells']:,} 个 cell，其中 {stats['scanned_eligible_cells']:,} 个 cell 至少有 3 条轨迹。按预先固定的随机规则抽取 **{stats['sampled_cells']:,} 个完整 cell、{stats['sampled_unique_links']:,} 条不同 link、{stats['sampled_trajectories']:,} 条轨迹记录和 {stats['valid_bins']:,} 个有效 bin**。同一物理行程可能出现在不同 cell，轨迹记录数不等于独立车辆数。

本页的‘统计意义’是：明确抽样总体、随机抽样、保留来源、纠正抽样概率差异并给出不确定性。**没有训练或比较 Ours；不是模型提升的显著性检验。**

## 先看哪里

1. [全部 {len(gallery)} 张真实 cell 图的图册](index.html)：可查看 PNG，点击进入有悬停数值的 SVG。
2. [第一组 8 张随机样本概览](contact_sheet_01.png)：先直观看不同道路与覆盖范围。
3. 下面的总体统计图：统计使用全部 {stats['sampled_cells']:,} 个随机 cell，不只使用展示的图。

![随机样本总体统计](statistical_overview.png)

四幅图依次为：cell 最远记录位置、cell 完整轨迹数、每条轨迹有效 bin 数、三类不同口径的缺失/观测状态。图中 endpoint 是相对 segment 原点的最远 bin 上界，**不是道路长度，也不是最长轨迹长度**。

## 1. 怎样抽样，代表谁

- 数据只来自本地 train corpus 的 {', '.join(manifest['days'])}。本轮没有读取 22 日验证集或 23 日拟保留测试集。
- 每天在完整的 128 个分桶中，不放回随机抽取 {manifest['buckets_per_day']} 个分桶。
- 每个已选分桶内，枚举全部 cell，从 **至少有 3 条 observation 记录**的 cell 中，不放回随机抽取 {manifest['cells_per_bucket']} 个；不根据耗时、长度、缺口或画图效果筛选。
- 对每个已选 cell 提取**全部轨迹**，不截到 16 条。训练 group 是后续对 cell 组织的模型样本，不等于本次统计的完整 cell。
- 随机种子固定为 `{manifest['seed']}`。原始抽样顺序、cell_id、来源文件路径/大小/修改时间、抽取数据 SHA256 都在 [sampling_manifest.json](sampling_manifest.json)。

| 日期 | 随机分桶 | 这些分桶内符合 K≥3 的 cell 数 | 抽中 cell | 抽中轨迹记录 |
|---|---|---:|---:|---:|
{chr(10).join(days)}

统计对应上述 5 天本地语料中 **K≥3 的完整 cell 及其中轨迹/bin**；不代表 K=1/2 的稀疏 cell，不代表所有原始 GPS 数据、其他日期或其他城市，也不是对 link 等概率抽样。这里覆盖许多不同 link，但 link 不是抽样单位。

## 2. 为什么不能直接对每桶百分比求平均

每桶固定抽 {manifest['cells_per_bucket']} 个 cell，但各桶符合条件的 cell 总数不同。设第 d 天第 b 桶有 N 个符合条件的 cell，抽 n 个，该天共 128 桶、抽 k 桶，则：

```text
某 cell 被抽中的概率 = (k / 128) × (n / N)
该 cell 的统计权重   = 1 / 这个概率

轨迹缺口比例 = Σ(权重 × 该 cell 中有缺口轨迹数)
             / Σ(权重 × 该 cell 中全部轨迹数)
```

所以报告采用‘加权总数之比’，而不是先算每个 cell 的百分比再平均。bin 比例同理，分母改为相应 bin 数。百分比的分母已逐项写明，不能把‘轨迹比例’与‘bin 比例’相减。

区间采用 **2000 次、按日期分层的两阶段 bootstrap**：每天重抽分桶，再在抽到的桶内重抽 cell，始终使用抽样权重。只有 {len(manifest['strata'])} 个被抽中的一级分桶，因此 95% 区间应视为近似的描述性抽样不确定性；不应把 {stats['valid_bins']:,} 个 bin 当成同等数量的独立样本。它不覆盖原始数据制作偏差、时间标签误差或跨日期泛化误差。

## 3. 这次实际看到什么

| 统计量（加权估计） | 估计值 | 近似 95% 区间 |
|---|---:|---|
{chr(10).join(rows)}

cell 轨迹数的加权 P10/P50/P90 为 **{' / '.join(str(int(v)) for v in stats['cell_K_weighted_p10_p50_p90'])} 条**；cell 最远记录 bin 上界的加权 P10/P50/P90 为 **{' / '.join(str(int(v)) for v in stats['cell_end_upper_m_weighted_p10_p50_p90'])} m**。这些是统计量点估计，没有另算分位数置信区间。

最关键的区别用一个 5-bin 示意说明（仅为解释，非实测行）：

```text
bin                 0       1       2       3       4
有 piece 记录       有      有      有      有      有
observed / GPS       1       0       0       0       1
valid / 耗时可用      1       1       1       1       1
T_diff              1.0     1.1     1.2     1.0     1.1

中间没有直接 GPS，但这 3 个 bin 在语料中已有有效耗时。
这不等于 [1.0, ?, ?, ?, 1.1]，不能仅凭 observed=0 再插值。
```

‘首尾 piece 之间缺失整 bin’与‘首尾 GPS 之间没有直接 GPS’是两种统计。前者本样本为 0 例，后者很常见。0 例只表示本样本未遇到，不证明原始数据或其他 cell 中没有；普通 bootstrap 会重复 0，故不输出虚假的精确区间 [0,0]。

**对实验草稿的影响：** 先尊重 `valid` 与 `observed` 的分工。保留 `valid=1, observed=0` 的已有耗时；对真正 `valid=0` 的 bin 保留无效标记，不制造监督标签；短轨迹尾部的未知覆盖不要补成 50 个有效值。本统计并未验证已有 `T_diff` 的上游生成过程是否正确或是否使用未来信息，仍需单独审计，不能据此宣布无泄漏。

## 4. 每张图怎样看

- **上图**：原始 `T_diff` 秒数热力图。PNG 全部使用统一 0–5 秒色标；大于 5 秒只在颜色上饱和，原始数值仍保留。下方统计曲线和 SVG 悬停可查看真实范围。部分覆盖 bin 的 raw 耗时不是完整 10m 的等效耗时，不能仅凭颜色判断速度。
- **中图**：完全相同的轨迹和空间列，显示状态。深绿=有直接 GPS 且耗时有效；浅蓝=无直接 GPS 但耗时有效；紫色=有记录但耗时无效；灰色=没有 piece 记录。
- **下图**：同一空间 bin 的耗时中位数及 25%–75% 分位带。无有效值的位置断开，不跨缺口连线。
- 横轴统一从当前 segment 的原点 0 开始，到该 cell 所有轨迹最远有记录 bin 的上界；每条轨迹不会左移、拉伸。**本 corpus 没有可靠道路长度字段，因此 50m 覆盖不自动等于 50m 短道路。**
- 纵向按进入时间排列，最早在下方；行间等距是轨迹排名，不是连续时间间距。超过 300 条只对显示行做等间距抽取，统计和横轴范围仍用全部轨迹。
- SVG 使用每张图自身 P5–P95 色标，方便看该 cell 内部变化；不同 SVG 不可直接按颜色深浅比较绝对耗时。SVG 顶部写明自身范围，PNG/拼图才使用共同色标。

## 5. 随机展示与诊断案例分开

前 **{n_random} 张**来自每个已选分桶随机抽样序列的前两个 cell，顺序在看图前已确定；余下 **{len(gallery)-n_random} 张**从同一随机样本里定向选取，分别展示‘中间无 GPS 但有效’、‘有记录但无效’、短/长覆盖，并显式标注类型。没有找到整 bin 无记录缺口，因此没有虚构这种真实案例。

定向案例可以解释现象，不能用它们的比例估计总体频率。原 examples 目录中的 SYNTHETIC 图仍只是合成示意，不计入本报告。

{chr(10).join(examples)}

## 6. 文件与复现

- [cell_statistics.csv](cell_statistics.csv)：每个随机 cell 的权重、K、空间范围、有效/观测/缺口计数。
- [trajectory_statistics.csv](trajectory_statistics.csv)：每条轨迹的对应计数，可复核每个分母。
- [statistics.json](statistics.json)：机器可读的估计值与区间。
- [gallery_manifest.json](gallery_manifest.json)：展示图片和真实 cell 的对应关系。
- `sampled_observations.parquet`：提取后的原始 ragged 列，用于重新画图；未插值、未修改源数据。

在仓库根目录、已有 numpy/pyarrow/matplotlib 环境运行：

```bash
# 完整抽样与绘图；只读训练语料
python experiments/trajectory_mlp_v1/tools/sample_cell_atlas.py

# 已有抽样数据时，仅重新计算统计、画图与生成本说明
python experiments/trajectory_mlp_v1/tools/sample_cell_atlas.py --stage report
```

所有本轮产物位于独立实验目录，原模型与训练代码未改变。新方法的学习过程请从 [实验设计与技术协议](../docs/design.md) 阅读。
"""
    (out / "统计说明.md").write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path("runtime/cell_weekend_tune_20260817_22/train"))
    p.add_argument("--out", type=Path, default=Path("experiments/trajectory_mlp_v1/data_atlas"))
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--days", nargs="+", default=[f"202608{d}" for d in range(17, 22)])
    p.add_argument("--buckets-per-day", type=int, default=4)
    p.add_argument("--cells-per-bucket", type=int, default=64)
    p.add_argument("--stage", choices=["all", "sample", "report"], default="all")
    args = p.parse_args()
    if not 1 <= args.buckets_per_day <= 128 or args.cells_per_bucket < 1:
        p.error("invalid sampling size")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "sample"):
        sample(args)
    if args.stage in ("all", "report"):
        make_plots(args.out, *prepare(args.out))
        write_report(args.out)


if __name__ == "__main__":
    main()
