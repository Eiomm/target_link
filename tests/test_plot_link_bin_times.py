import importlib.util
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_link_bin_times", ROOT / "tools" / "plot_link_bin_times.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def test_fold_observation_matches_piece_contract():
    row = {
        "T_diff": [0.4, 0.6, 2.0, np.nan],
        "ratio_pct": [4, 6, 5, 5],
        "valid": [True, True, True, False],
        "bin_pos": [3, 3, 8, 8],
    }
    eq, valid, ratio = plot.fold_observation(row, "equivalent-10m")
    raw, _, _ = plot.fold_observation(row, "raw")

    assert valid[3]
    assert ratio[3] == 1.0
    assert eq[3] == 1.0
    assert raw[3] == 1.0
    assert not valid[8]                 # all pieces must be valid
    assert np.isnan(eq[8])
    assert np.isnan(eq[0])              # absent bin is not a zero-second bin


def test_partial_bin_is_normalised_to_full_10m():
    row = {"T_diff": [0.8], "ratio_pct": [4], "valid": [True], "bin_pos": [0]}
    eq, _, _ = plot.fold_observation(row, "equivalent-10m")
    raw, _, _ = plot.fold_observation(row, "raw")
    assert eq[0] == 2.0
    assert raw[0] == 0.8


def test_corpus_root_accepts_root_or_observation_path():
    root = "hdfs://cluster/path/corpus_v1"
    assert plot._corpus_root(root, "observations_v2") == root
    assert plot._corpus_root(root + "/observations_v2/", "observations_v2") == root


def test_svg_and_summary_inputs(tmp_path):
    matrix = np.full((3, 50), np.nan)
    matrix[0, :4] = [1, 2, 3, 4]
    matrix[1, :4] = [2, 3, 4, 5]
    matrix[2, :4] = [3, 4, 5, 6]
    per_bin = plot._quantiles_by_bin(matrix)
    meta = {"target_link_id": "L1", "seg_idx": 0,
            "window_local": "2026-08-17 08:00 CST", "n_trajectories": 3,
            "metric": "equivalent-10m", "lower_percentile": 5.0,
            "upper_percentile": 95.0}
    out = tmp_path / "link.svg"
    result = plot.render_svg(matrix, ["a", "b", "c"], [100, 200, 300],
                             meta, per_bin, out)
    text = out.read_text()
    assert text.startswith("<svg")
    assert "Link L1" in text
    assert "grey = invalid/missing" in text
    assert "early bottom" in text
    assert "Trajectory entry time" in text
    assert "zoomed y-scale" in text
    assert result["heatmap_rows"] == 3
    ElementTree.parse(out)
