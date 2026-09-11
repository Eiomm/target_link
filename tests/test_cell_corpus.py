from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from target_link_v1.data.cell_corpus import CellCorpusDataset, collate_cells
from target_link_v1.models.cell_mae import CellMAE, masked_reconstruction_loss


def _write_partition(root: Path, name: str, table: pa.Table) -> None:
    part = root / name / "day=20260821" / "bucket=0"
    part.mkdir(parents=True)
    pq.write_table(table, part / "part-00000.parquet")


def _synthetic_corpus(root: Path) -> None:
    rows = []
    for cell, size in ((101, 4), (202, 5)):
        for j in range(size):
            # bin 3 is split across links. For the first trajectory one piece
            # is invalid, while bin 17 keeps every trajectory valid overall.
            bad = cell == 101 and j == 0
            rows.append(dict(
                cell_id=cell,
                sample_id=f"s{cell}-{j}",
                dt=np.float32(10 + j),
                n_pieces=np.int16(3),
                T_diff=[np.float32(0.7), np.float32(np.nan if bad else 0.3),
                        np.float32(2.0 + j)],
                ratio_pct=[np.int8(7), np.int8(3), np.int8(5)],
                observed=[False, True, True],
                valid=[True, not bad, True],
                bin_pos=[np.int8(3), np.int8(3), np.int8(17)],
                map_version="m", target_link_id=f"L{cell}",
                seg_idx=np.int32(3), window=np.int64(1000)))
    obs = pa.Table.from_pylist(rows)
    _write_partition(root, "observations", obs)

    groups = []
    for cell, size in ((101, 4), (202, 5)):
        groups.append(dict(group_id=f"g{cell}", cell_id=cell, K=size,
                           group_size=size,
                           sample_ids=[f"s{cell}-{j}" for j in range(size)],
                           window=np.int64(1000)))
    _write_partition(root, "training_groups", pa.Table.from_pylist(groups))


def test_dataset_piece_folding_and_absolute_bin_position(tmp_path):
    _synthetic_corpus(tmp_path)
    ds = CellCorpusDataset(tmp_path, obs_dir="observations", groups_dir="training_groups",
                           shuffle_groups=False)
    items = list(ds)
    assert [it["x"].shape for it in items] == [(4, 50, 3), (5, 50, 3)]

    first = items[0]
    # No per-trajectory renumbering: pieces remain at absolute positions 3/17.
    assert np.flatnonzero(first["x"][0, :, 1]).tolist() == [3, 17]
    # Invalid multi-piece bin: all-piece ratio/observed survive, time is zero.
    np.testing.assert_allclose(first["x"][0, 3], [0.0, 1.0, 1.0])
    assert not first["bin_valid"][0, 3]
    # Fully valid split bin sums each piece exactly once (no T_diff * ratio).
    np.testing.assert_allclose(first["x"][1, 3], [1.0, 1.0, 1.0])
    assert first["bin_valid"][1, 3]
    assert first["traj_valid"].all()


def test_collate_padding_and_reproducible_epoch_mask(tmp_path):
    _synthetic_corpus(tmp_path)
    ds = CellCorpusDataset(tmp_path, obs_dir="observations", groups_dir="training_groups",
                           shuffle_groups=False, epoch=7)
    items = list(ds)
    batch = collate_cells(items)

    assert batch["x"].shape == (2, 16, 50, 3)
    assert batch["bin_valid"].shape == (2, 16, 50)
    assert batch["traj_valid"].shape == batch["mae_mask"].shape == (2, 16)
    assert batch["delta_t"].shape == (2, 16)
    assert batch["x"].dtype == batch["delta_t"].dtype == torch.float32
    assert batch["bin_valid"].dtype == batch["traj_valid"].dtype == torch.bool
    assert batch["mae_mask"].dtype == torch.bool
    for i, m in enumerate((4, 5)):
        assert not batch["x"][i, m:].any()
        assert not batch["bin_valid"][i, m:].any()
        assert not batch["traj_valid"][i, m:].any()
        assert not batch["delta_t"][i, m:].any()
        assert not batch["mae_mask"][i, m:].any()
        assert int(batch["mae_mask"][i].sum()) == round(0.5 * m)
        assert int((batch["traj_valid"][i] & ~batch["mae_mask"][i]).sum()) >= 2

    torch.testing.assert_close(batch["mae_mask"], collate_cells(items)["mae_mask"])
    changed = [collate_cells(items, epoch=e)["mae_mask"] for e in range(8, 20)]
    assert any(not torch.equal(batch["mae_mask"], mask) for mask in changed)


def test_cell_mae_visible_only_forward_backward(tmp_path):
    _synthetic_corpus(tmp_path)
    items = list(CellCorpusDataset(
        tmp_path, obs_dir="observations", groups_dir="training_groups",
        shuffle_groups=False, epoch=3))
    batch = collate_cells(items, epoch=3)
    torch.manual_seed(0)
    model = CellMAE(d_model=16, heads=2, traj_layers=1,
                    level2_layers=1, dropout=0).eval()
    output = model(batch)
    assert output["representation"].shape == (2, 16)
    assert output["prediction"].shape == output["target"].shape == (2, 16, 50)
    loss = masked_reconstruction_loss(output, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    changed = dict(batch, x=batch["x"].clone())
    changed["x"][batch["mae_mask"]] += 1000.0
    altered = model(changed)
    torch.testing.assert_close(output["representation"], altered["representation"])
    torch.testing.assert_close(output["prediction"], altered["prediction"])


def test_groups_per_partition_caps_each_partition(tmp_path):
    _synthetic_corpus(tmp_path)
    ds = CellCorpusDataset(
        tmp_path, obs_dir="observations", groups_dir="training_groups",
        shuffle_groups=False, groups_per_partition=1)
    assert [item["group_id"] for item in ds] == ["g101"]
    with pytest.raises(ValueError, match="groups_per_partition"):
        CellCorpusDataset(
            tmp_path, obs_dir="observations", groups_dir="training_groups",
            groups_per_partition=0)


def test_reader_sort_guard_does_not_overflow_at_int64_boundary(tmp_path):
    """A max-positive -> min-negative descent was invisible to np.diff(int64)."""
    _synthetic_corpus(tmp_path)
    path = next((tmp_path / "observations").rglob("*.parquet"))
    # Read the physical file directly; pq.read_table(path) also discovers the
    # hive day/bucket directory columns and would persist them into the file.
    table = pq.ParquetFile(path).read()
    cell_id = np.array([np.iinfo(np.int64).max] * 4 +
                       [np.iinfo(np.int64).min] * 5, dtype=np.int64)
    table = table.set_column(table.schema.get_field_index("cell_id"), "cell_id",
                             pa.array(cell_id))
    pq.write_table(table, path)

    ds = CellCorpusDataset(tmp_path, obs_dir="observations",
                           groups_dir="training_groups", shuffle_groups=False)
    with pytest.raises(ValueError, match="not sorted by cell_id"):
        list(ds)
