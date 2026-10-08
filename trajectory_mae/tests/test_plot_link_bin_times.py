"""Spatial correctness checks for the isolated plotting fork; no training data."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "plot_link_bin_times.py"
spec = importlib.util.spec_from_file_location("mlp_plot", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
NS = {"s": "http://www.w3.org/2000/svg"}


def render(tmp_path, matrix, present, **options):
    meta = dict(target_link_id="example", seg_idx=0, window_local="2026-08-22 08:00",
                n_trajectories=len(matrix), metric="raw", lower_percentile=5,
                upper_percentile=95)
    cap = options.pop("max_rows", 300)
    meta.update(options)
    out = tmp_path / "plot.svg"
    result = plot.render_svg(matrix, [f"row-{i}" for i in range(len(matrix))],
                             np.arange(len(matrix)), meta, plot._quantiles_by_bin(matrix),
                             str(out), max_rows=cap, present=present)
    return result, ET.parse(out).getroot()


def test_farthest_endpoint_is_not_longest_trajectory():
    values = np.full((2, 50), np.nan)
    values[0, :8] = 1.0             # 80m long, ends at 80m
    values[1, 12:17] = 2.0          # 50m long, ends at 170m
    domain = plot.spatial_domain(values, np.isfinite(values))
    assert domain["x_min_m"] == 0
    assert domain["x_max_m"] == 170


def test_unknown_time_at_recorded_endpoint_is_not_cropped(tmp_path):
    values = np.full((1, 50), np.nan)
    values[0, 0] = 1
    present = np.isfinite(values)
    present[0, 9] = True
    result, svg = render(tmp_path, values, present)
    assert result["x_max_m"] == 100
    assert result["n_recorded_invalid_times"] == 1
    bins = svg.findall("s:rect[@class='bin']", NS)
    assert bins[-1].get("data-status") == "invalid_time"
    assert bins[1].get("data-status") == "no_record"


def test_display_subsampling_does_not_change_spatial_range(tmp_path):
    values = np.full((3, 50), np.nan)
    values[:, 0] = 1
    values[0, 16] = 2  # earliest row; cap=1 draws only the latest row
    result, _ = render(tmp_path, values, np.isfinite(values), max_rows=1)
    assert result["heatmap_rows"] == 1
    assert result["x_max_m"] == 170


def test_short_link_and_partial_final_bin_keep_metric_coordinates(tmp_path):
    values = np.full((1, 50), np.nan)
    values[0, :3] = [1, 2, 0.5]
    result, svg = render(tmp_path, values, np.isfinite(values), segment_length_m=25)
    bins = svg.findall("s:rect[@class='bin']", NS)
    assert len(bins) == 3
    assert result["x_max_m"] == 25
    assert float(bins[2].get("width")) == pytest.approx(float(bins[0].get("width")) / 2)
    assert "20–25m" in bins[2].find("s:title", NS).text


def test_delayed_start_keeps_empty_leading_columns(tmp_path):
    values = np.full((1, 50), np.nan)
    values[0, 12:17] = 1
    result, svg = render(tmp_path, values, np.isfinite(values))
    bins = svg.findall("s:rect[@class='bin']", NS)
    assert result["x_max_m"] == 170
    assert all(b.get("data-status") == "no_record" for b in bins[:12])
    assert bins[12].get("data-bin") == "12"


def test_geometry_mode_exposes_unobserved_tail(tmp_path):
    values = np.full((1, 50), np.nan)
    values[0, :3] = 1
    result, svg = render(tmp_path, values, np.isfinite(values),
                         x_range_mode="geometry", segment_length_m=50)
    assert result["x_max_m"] == 50
    assert result["coverage_upper_bound_m"] == 30
    assert len(svg.findall("s:rect[@class='bin']", NS)) == 5


def test_grid_mode_marks_outside_road_including_partial_bin(tmp_path):
    values = np.full((1, 50), np.nan)
    values[0, :3] = 1
    result, svg = render(tmp_path, values, np.isfinite(values),
                         x_range_mode="grid", segment_length_m=25)
    assert result["x_max_m"] == 500
    assert len(svg.findall("s:rect[@data-status='outside_geometry']", NS)) == 48


def test_median_and_iqr_do_not_bridge_unknown_columns(tmp_path):
    values = np.full((2, 50), np.nan)
    values[:, [0, 1, 4, 5]] = [[1, 2, 3, 4], [2, 3, 4, 5]]
    _, svg = render(tmp_path, values, np.isfinite(values))
    curves = svg.findall("s:polyline[@class='median']", NS)
    bands = svg.findall("s:polygon[@class='iqr']", NS)
    assert len(curves) == len(bands) == 2
    assert all(len(c.get("points").split()) == 2 for c in curves)
    first_end = float(curves[0].get("points").split()[-1].split(",")[0])
    second_start = float(curves[1].get("points").split()[0].split(",")[0])
    assert second_start > first_end


def test_presence_is_independent_of_finite_time_and_ratio():
    row = dict(bin_pos=[1, 9, 9], T_diff=[1, np.nan, np.nan],
               ratio_pct=[10, 0, 0], valid=[True, False, False])
    present = plot.observation_presence(row)
    values, _, _ = plot.fold_observation(row, metric="raw")
    assert present[9] and np.isnan(values[9])
    assert plot.spatial_domain(values[None], present[None])["x_max_m"] == 100


@pytest.mark.parametrize("length", [0, -10, 501, float("inf"), float("nan")])
def test_invalid_geometry_is_rejected(length):
    values = np.full((1, 50), np.nan)
    values[0, 0] = 1
    with pytest.raises(ValueError, match="segment length"):
        plot.spatial_domain(values, np.isfinite(values), segment_length_m=length)


def test_contradictory_geometry_cannot_silently_hide_records():
    values = np.full((1, 50), np.nan)
    values[0, 6] = 1
    with pytest.raises(ValueError, match="beyond"):
        plot.spatial_domain(values, np.isfinite(values), segment_length_m=50)
    with pytest.raises(ValueError, match="geometry range needs"):
        plot.spatial_domain(values, np.isfinite(values), mode="geometry")


def write_raw(path, length=50, seg_idx=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    for sample in ("A", "B"):
        for j in (0, 1, 4):
            rows.append(dict(map_version="map", target_link_id="short-link",
                sample_id=sample, seg_mark=1, seg_idx=seg_idx,
                bin_idx=10 + 50 * seg_idx + j, sub_idx=0, t_ref=1787000400.0,
                T_cum=float(j+1) if j != 4 else float("nan"),
                T_diff=1.0 if j != 4 else float("nan"), ratio=1.0, observed=1,
                L_link_m=float(length)))
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_raw_cli_infers_short_link_and_retains_invalid_endpoint(tmp_path):
    raw, out = tmp_path / "raw.parquet", tmp_path / "short.svg"
    write_raw(raw)
    done = subprocess.run([sys.executable, str(SCRIPT), "--raw-parquet", str(raw),
                           "--metric", "raw", "--out", str(out)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["x_max_m"] == summary["segment_length_m"] == 50
    assert summary["presence_source"] == "explicit_piece_records"
    assert summary["n_recorded_invalid_times"] == 2
    assert len(summary["per_bin"]) == 50  # statistics retain the training grid
    ET.parse(out)


def test_raw_final_segment_uses_remaining_link_length(tmp_path):
    raw = tmp_path / "raw.parquet"
    write_raw(raw, length=550, seg_idx=1)
    args = SimpleNamespace(raw_parquet=str(raw), link=None, map_version=None, seg_idx=1)
    chosen, _ = plot._load_raw_cell(args, None, None)
    assert chosen["segment_length_m"] == 50


def test_raw_cli_without_geometry_uses_coverage_and_geometry_mode_fails(tmp_path):
    import pyarrow.parquet as pq
    raw, out = tmp_path / "raw.parquet", tmp_path / "coverage.svg"
    write_raw(raw)
    table = pq.read_table(raw).drop(["L_link_m"])
    pq.write_table(table, raw)
    cmd = [sys.executable, str(SCRIPT), "--raw-parquet", str(raw), "--out", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["segment_length_m"] is None and summary["x_max_m"] == 50
    done = subprocess.run(cmd + ["--x-range", "geometry"], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode != 0 and "geometry range needs" in done.stderr
