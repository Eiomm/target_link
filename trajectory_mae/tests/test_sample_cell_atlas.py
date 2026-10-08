"""Check sampling/statistical semantics that affect the report's conclusions."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("cell_atlas", TOOLS / "sample_cell_atlas.py")
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


def test_internal_gap_excludes_unknown_leading_and_trailing_extent():
    present = np.zeros(50, dtype=bool)
    present[[12, 13, 16]] = True
    valid = present.copy()
    valid[13] = False
    observed = np.zeros(50, dtype=bool)
    observed[12] = True
    result = atlas.gap_counts(present, valid, observed)
    assert result == dict(n_present=3, n_valid=2, n_no_gps_valid=1,
                          n_invalid_recorded=1, n_interior_no_record=2,
                          n_internal_no_gps_valid=0,
                          start_m=120, end_upper_m=170)


def test_sampling_is_seeded_unique_and_does_not_exclude_small_eligible_cells():
    ids = np.repeat(np.arange(7), [1, 2, 3, 4, 5, 6, 7])
    chosen, frame = atlas.choose_cells(ids, np.random.default_rng(11), 3)
    again, _ = atlas.choose_cells(ids, np.random.default_rng(11), 3)
    np.testing.assert_array_equal(chosen, again)
    assert len(set(chosen)) == 3
    assert set(chosen).issubset({2, 3, 4, 5, 6})
    assert frame == dict(total_cells=7, eligible_cells=5, observation_rows=28, selected_cells=3)
    all_cells, _ = atlas.choose_cells(ids, np.random.default_rng(11), 100)
    assert set(all_cells) == {2, 3, 4, 5, 6}


def test_ratio_estimate_uses_weighted_totals_not_average_percentages():
    cells = [dict(day="a", bucket=0, weight=2, num=1, den=2),
             dict(day="a", bucket=1, weight=3, num=9, den=10)]
    stats = atlas.ratio_ci(cells, "num", "den", seed=42, n_boot=100)
    assert stats["estimate"] == pytest.approx(29 / 34)
    assert stats["estimate"] != pytest.approx((.5 + .9) / 2)
    assert 0 <= stats["ci95"][0] <= stats["ci95"][1] <= 1


def test_weighted_quantile_respects_inclusion_weights():
    assert atlas.weighted_quantile([10, 20, 30], [1, 100, 1], [.1, .5, .9]) == [20, 20, 20]


def test_no_events_does_not_produce_false_zero_width_confidence_interval():
    stats = atlas.ratio_ci([dict(day="a", bucket=0, weight=2, num=0, den=100)],
                           "num", "den", seed=42)
    assert stats["estimate"] == 0
    assert stats["ci95"] is None


def test_gps_gap_can_have_valid_time_and_no_missing_piece():
    present = np.zeros(50, dtype=bool)
    present[:5] = True
    observed = np.zeros(50, dtype=bool)
    observed[[0, 4]] = True
    stats = atlas.gap_counts(present, present, observed)
    assert stats["n_internal_no_gps_valid"] == 3
    assert stats["n_interior_no_record"] == 0
