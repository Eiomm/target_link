import numpy as np
import pyarrow as pa
import pytest

from experiments.trajectory_mlp_v1.tools.census_partition_metrics import summarize_batch


def _rows(*pieces):
    """Build rows as (T_diff, ratio_pct, valid, bin_pos) tuples."""
    values = []
    for index, (times, ratios, valid, bins) in enumerate(pieces):
        values.append(dict(cell_id=100 + index, sample_id=f"sample-{index}", window="w0",
                           target_link_id=200 + index, map_version="v1", seg_idx=index,
                           dt=np.float32(10 + index), T_diff=times, ratio_pct=ratios,
                           valid=valid, bin_pos=bins))
    return pa.Table.from_pylist(values)


def test_p0_fold_handles_multi_piece_invalid_short_cover_and_internal_gap():
    source = _rows(
        # bin 3 is split; one bad piece invalidates exactly that bin.
        ([1.0, np.nan, 2.0, 4.0], [10, 20, 30, 40], [True, False, True, True], [3, 3, 5, 7]),
        # A short observed range has no internal gap.
        ([1.0, 2.0], [8, 12], [True, True], [20, 21]),
        # Missing bin 11 between present bins 10 and 12 is an internal gap.
        ([1.0, 3.0], [5, 6], [True, True], [10, 12]),
    )

    out = summarize_batch(source)

    assert out.column_names == [
        "cell_id", "sample_id", "window", "target_link_id", "map_version", "seg_idx", "dt",
        "n_present", "n_valid", "has_internal_gap", "covered_m", "present_bits", "valid_bits",
        "usable", "n_pieces",
    ]
    got = out.to_pydict()
    assert got["n_present"] == [3, 2, 2]
    assert got["n_valid"] == [2, 2, 2]
    assert got["has_internal_gap"] == [True, False, True]
    assert got["covered_m"] == [100.0, 20.0, 11.0]
    assert got["n_pieces"] == [4, 2, 2]
    assert got["usable"] == [True, True, True]
    assert got["present_bits"][0] == (1 << 3) | (1 << 5) | (1 << 7)
    assert got["valid_bits"][0] == (1 << 5) | (1 << 7)


def test_p0_validity_matches_training_reader_all_piece_rule():
    source = _rows(([np.nan, 1.5], [3, 7], [False, True], [4, 4]))
    out = summarize_batch(source).to_pydict()
    # CellDataset._scatter uses count == valid-count for the same criterion.
    assert out["n_present"] == [1]
    assert out["n_valid"] == [0]
    assert out["valid_bits"] == [0]
    assert out["usable"] == [False]


@pytest.mark.parametrize(
    "source, pattern",
    [
        (_rows(([1.0], [1], [True], [0])), "missing required column 'window'"),
        (_rows(([1.0], [1], [True], [50])), "bin_pos outside 0..49"),
        (_rows(([np.nan], [1], [True], [0])), "valid T_diff must be finite"),
        (_rows(([1.0], [-1], [True], [0])), "ratio_pct must be finite"),
    ],
)
def test_contract_rejects_bad_data(source, pattern):
    if "missing required" in pattern:
        source = source.drop(["window"])
    with pytest.raises(ValueError, match=pattern):
        summarize_batch(source)


def test_contract_rejects_mismatched_piece_lengths_and_overlarge_batch():
    bad = pa.table({
        "cell_id": [1], "sample_id": ["a"], "window": ["w"], "target_link_id": [2],
        "map_version": ["v"], "seg_idx": [0], "dt": [1.0], "T_diff": [[1.0, 2.0]],
        "ratio_pct": [[1.0]], "valid": [[True, True]], "bin_pos": [[0, 1]],
    })
    with pytest.raises(ValueError, match="different list lengths"):
        summarize_batch(bad)

    one = _rows(([1.0], [1], [True], [0]))
    oversized = pa.concat_tables([one] * 65_537)
    with pytest.raises(ValueError, match="maximum is 65536"):
        summarize_batch(oversized)


def test_empty_lists_and_slices_preserve_offsets_without_piece_rows():
    source = _rows(
        ([], [], [], []),
        ([1.0, 2.0], [4, 6], [True, True], [48, 49]),
        ([1.0, 2.0], [4, 6], [True, True], [0, 49]),
    )
    full = summarize_batch(source).to_pydict()
    assert full["n_pieces"] == [0, 2, 2]
    assert full["n_present"] == [0, 2, 2]
    assert full["has_internal_gap"] == [False, False, True]

    empty = summarize_batch(source.slice(0, 0))
    assert empty.num_rows == 0 and empty.column_names == summarize_batch(source).column_names
    assert summarize_batch(source.slice(1, 1)).to_pydict() == {
        key: [values[1]] for key, values in full.items()
    }

    bad_after_empty = _rows(([], [], [], []), ([1.0], [-1], [True], [0]))
    with pytest.raises(ValueError, match="row 1: ratio_pct"):
        summarize_batch(bad_after_empty)
