"""Focused contracts for the independent trajectory-MLP corpus reader."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells


def _write(root: Path, day: str, bucket: int, counts, *, invalid=()):
    rows = []
    for cell, count in counts.items():
        for i in range(count):
            bad = (cell, i) in invalid
            rows.append({"cell_id": cell, "sample_id": f"{cell:04d}-{i:04d}",
                         "dt": np.float32(i % 600),
                         "T_diff": [np.float32(np.nan if bad else i + 1)],
                         "ratio_pct": [np.int8(10)], "observed": [False],
                         "valid": [not bad], "bin_pos": [np.int8(i % 50)]})
    part = root / "observations_v2" / f"day={day}" / f"bucket={bucket}"
    part.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), part / "part-0.parquet")


def _items(root, *, m_max=64, epoch=0):
    return list(CellDataset([str(root)], ["20260817"], m_max=m_max,
                            seed=99, epoch=epoch))


def test_fixed_chunk_tail_rules_and_persistent_membership(tmp_path):
    _write(tmp_path, "20260817", 0, {10: 130, 20: 65, 30: 67, 40: 4},
           invalid={(40, i) for i in range(4)})
    first = _items(tmp_path)
    by_cell = {}
    for item in first:
        by_cell.setdefault(item["cell_id"], []).append(item)
    assert sorted(x["group_size"] for x in by_cell[10]) == [64, 64]
    assert sorted(x["group_size"] for x in by_cell[20]) == [64]
    assert sorted(x["group_size"] for x in by_cell[30]) == [3, 64]
    assert all(x["K"] == 130 and x["K_raw"] == 130 for x in by_cell[10])
    assert all(x["dropped_tail"] == 2 for x in by_cell[10])
    stats = first[0]["partition_stats"]
    assert stats["raw_rows"] == 266 and stats["usable_rows"] == 262
    assert stats["dropped_no_valid"] == 4 and stats["dropped_tail"] == 3
    assert stats["groups"] == 5 and stats["full_groups"] == 4 and stats["tail_groups"] == 1
    members = {x["group_id"]: x["sample_ids"] for x in first}
    assert members == {x["group_id"]: x["sample_ids"] for x in _items(tmp_path, epoch=9)}


def test_floor_half_mask_seed_and_eval_are_reproducible(tmp_path):
    _write(tmp_path, "20260817", 0, {10: 3, 20: 5, 30: 64})
    items = _items(tmp_path)
    a = collate_cells(items, epoch=1)
    torch.testing.assert_close(a["mae_mask"], collate_cells(items, epoch=1)["mae_mask"])
    assert sorted(int(a["mae_mask"][i].sum()) for i in range(3)) == [1, 2, 32]
    assert any(not torch.equal(a["mae_mask"], collate_cells(items, epoch=e)["mae_mask"])
               for e in range(2, 10))


def test_piece_fold_does_not_use_observed_and_discards_zero_valid(tmp_path):
    _write(tmp_path, "20260817", 0, {10: 4}, invalid={(10, 0)})
    item = _items(tmp_path)[0]
    assert item["K_raw"] == 4 and item["K"] == 3 and item["dropped_no_valid"] == 1
    assert item["x"].shape == (3, 50, 3)
    assert not item["x"][..., 2].any()  # observed=False did not erase valid labels


def test_variable_batch_padding_and_duplicate_partition_guard(tmp_path):
    _write(tmp_path / "a", "20260817", 0, {10: 3, 20: 5})
    batch = collate_cells(_items(tmp_path / "a"), m_max=8, epoch=4)
    assert batch["x"].shape == (2, 8, 50, 3)
    assert batch["day"] == ["20260817", "20260817"] and batch["bucket"] == ["0", "0"]
    assert not batch["traj_valid"][0, 3:].any() and not batch["mae_mask"][0, 3:].any()
    _write(tmp_path / "b", "20260817", 0, {30: 3})
    with pytest.raises(ValueError, match="duplicate observations partition"):
        CellDataset([str(tmp_path / "a"), str(tmp_path / "b")], ["20260817"])
