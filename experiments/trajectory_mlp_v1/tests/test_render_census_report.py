import csv
import json

from experiments.trajectory_mlp_v1.tools import render_census_report as report


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_without_optional_plot(out, data):
    original = report._plot
    report._plot = lambda *_: None
    try:
        return report._render(out, data)
    finally:
        report._plot = original


def _full_fixture(out, geometry_status="static_same_version_subset"):
    parts = out / "parts"
    parts.mkdir(parents=True)
    (out / "aggregation_complete.json").write_text(json.dumps({
        "status": "complete", "partitions": 896, "seed": 20260921, "epoch": 0,
        "m_max": 64, "geometry_status": geometry_status,
    }))
    (out / "input_manifest.json").write_text(json.dumps({"partitions": 896, "days": report.DAYS}))
    receipt = dict(rows=1, data_seed=20260921, retained_observations=0,
                   dropped_no_valid=0, dropped_tail=1, groups=0, hidden_trajectories=0,
                   supervised_bins=0, supported_supervised_bins=0,
                   n_present_hist=[0, 1] + [0] * 49, n_valid_hist=[0, 1] + [0] * 49)
    for day in report.DAYS:
        for bucket in range(128):
            value = dict(receipt, day=day, bucket=str(bucket))
            (parts / f"{day}_{bucket:03d}.json").write_text(json.dumps(value))
    # This audit sidecar must not be included in receipt enumeration.
    (parts / "20260817_000.identity.json").write_text("{}")
    (out / "identity_audit.json").write_text(json.dumps({"status": "passed", "partitions": 896, "rows": 896}))

    daily_fields = ["day", "observations", "pieces", "recorded_bins", "valid_bins",
                    "internal_gap_observations", "usable_observations", "covered_m", "cells",
                    "links", "versioned_links", "distinct_traj_ids"]
    daily = [dict(day=day, observations=128, pieces=128, recorded_bins=128, valid_bins=128,
                  internal_gap_observations=0, usable_observations=128, covered_m=128,
                  cells=128, links=128, versioned_links=128, distinct_traj_ids=128)
             for day in report.DAYS]
    _write_csv(out / "daily_totals.csv", daily_fields, daily)
    total = dict(daily[0], observations=896, pieces=896, recorded_bins=896, valid_bins=896,
                 usable_observations=896, covered_m=896, cells=896, links=896,
                 versioned_links=896, distinct_traj_ids=896)
    total.pop("day")
    _write_csv(out / "seven_day_totals.csv", daily_fields[1:], [total])

    _write_csv(out / "window_10min.csv", ["window_start", "local_time", "observations",
               "distinct_traj_ids", "links", "cells"], [
        dict(window_start=window, local_time="x", observations=1 if i < 896 else 0,
             distinct_traj_ids=1 if i < 896 else 0, links=1 if i < 896 else 0,
             cells=1 if i < 896 else 0)
        for i, window in enumerate(report._expected_windows())
    ])
    _write_csv(out / "integrity.csv", ["observations", "partition_day_mismatch",
               "malformed_sample_ids", "outside_days", "misaligned_windows"], [
        dict(observations=896, partition_day_mismatch=0, malformed_sample_ids=0,
             outside_days=0, misaligned_windows=0)
    ])
    _write_csv(out / "coverage_bins_distribution.csv", ["day", "n_present", "n_valid", "observations"], [
        dict(day=day, n_present=1, n_valid=1, observations=128) for day in report.DAYS
    ])
    _write_csv(out / "daily_training_retention.csv", ["day", "cells", "trainable_cells",
               "observations", "dropped_no_valid", "dropped_small_tail", "retained_observations",
               "training_groups"], [
        dict(day=day, cells=128, trainable_cells=0, observations=128, dropped_no_valid=0,
             dropped_small_tail=128, retained_observations=0, training_groups=0) for day in report.DAYS
    ])
    _write_csv(out / "daily_mask_reference.csv", ["day", "rows", "retained_observations",
               "dropped_no_valid", "dropped_tail", "groups", "hidden_trajectories",
               "supervised_bins", "supported_supervised_bins"], [
        dict(day=day, rows=128, retained_observations=0, dropped_no_valid=0, dropped_tail=128,
             groups=0, hidden_trajectories=0, supervised_bins=0, supported_supervised_bins=0)
        for day in report.DAYS
    ])
    _write_csv(out / "cell_k_distribution.csv", ["day", "kind", "k", "cells"], [
        item for day in report.DAYS for item in (
            dict(day=day, kind="raw", k=1, cells=128),
            dict(day=day, kind="usable", k=0, cells=128),
        )
    ])


def test_validated_full_artifacts_render_report_and_ignore_identity_sidecars(tmp_path):
    _full_fixture(tmp_path)
    data = report._validate(tmp_path)
    body = _render_without_optional_plot(tmp_path, data)
    assert "七天 observations_v2 P0 普查报告" in body
    assert "1008 个窗口" in body
    assert "不是订单数" in body


def test_partial_aggregate_is_rejected(tmp_path):
    (tmp_path / "aggregation_complete.json").write_text(json.dumps({"status": "partial_smoke"}))
    try:
        report._validate(tmp_path)
    except report.ValidationError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("partial aggregate unexpectedly accepted")


def test_dynamic_geometry_uses_source_accounting_and_dynamic_title(tmp_path):
    _full_fixture(tmp_path, geometry_status="dynamic_same_version")
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    (geometry / "manifest.json").write_text(json.dumps({"map_version": "2026081412", "unique_links": 2976427}))
    _write_csv(tmp_path / "geometry_match.csv", ["day", "match_status", "observations", "versioned_links"], [
        dict(day=day, match_status="exact_version", observations=128, versioned_links=128) for day in report.DAYS
    ])
    _write_csv(tmp_path / "exact_geometry_coverage.csv", ["day", "coverage_band", "observations"], [
        dict(day=day, coverage_band="05_95-105pct", observations=128) for day in report.DAYS
    ])
    _write_csv(tmp_path / "geometry_sources.csv", ["day", "geometry_source", "observations"], [
        dict(day=day, geometry_source="raw_dynamic_same_version", observations=128) for day in report.DAYS
    ])
    _write_csv(tmp_path / "geometry_segment_types.csv", ["day", "segment_type", "observations", "versioned_segments"], [
        dict(day=day, segment_type="full_500m_segment", observations=128, versioned_segments=128)
        for day in report.DAYS
    ])
    _write_csv(tmp_path / "exact_segment_lengths.csv", ["length_m", "segments"], [
        dict(length_m=500, segments=128)
    ])
    data = report._validate(tmp_path)
    body = _render_without_optional_plot(tmp_path, data)
    assert "同版本原始路长参考覆盖比" in body
    assert "动态同版本原始路长已参与" in body
    assert "full_500m_segment" in body and "跨七天去重的版本化 segment 长度" in body
