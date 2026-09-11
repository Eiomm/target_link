from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.normalize_cell_partition import MARKER, normalize_partition


def _write(directory: Path, name: str, rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / name)


def _obs(cell, sample, value):
    return dict(
        cell_id=np.int64(cell), sample_id=sample, dt=np.float32(value),
        n_pieces=np.int16(2), T_diff=[np.float32(value), np.float32(value + 0.5)],
        ratio_pct=[np.int8(4), np.int8(6)], observed=[True, False],
        valid=[True, True], bin_pos=[np.int8(2), np.int8(2)],
        map_version="m", target_link_id=f"L{cell}", seg_idx=np.int32(0),
        window=np.int64(1000))


def _fixture(tmp_path):
    obs_dir, groups_dir = tmp_path / "obs", tmp_path / "groups"
    # Deliberately disorder cells across two physical files and sample IDs
    # within a cell. Values make whole-row alignment easy to verify.
    _write(obs_dir, "part-00000.parquet", [
        _obs(20, "s20-b", 20.2), _obs(10, "s10-c", 10.3)])
    _write(obs_dir, "part-00001.parquet", [
        _obs(10, "s10-a", 10.1), _obs(20, "s20-a", 20.1),
        _obs(10, "s10-b", 10.2), _obs(20, "s20-c", 20.3)])
    _write(groups_dir, "part-00000.parquet", [
        dict(group_id="g10", cell_id=np.int64(10), K=np.int32(3),
             group_size=np.int64(3), sample_ids=["s10-c", "s10-a", "s10-b"],
             window=np.int64(1000)),
        dict(group_id="g20", cell_id=np.int64(20), K=np.int32(3),
             group_size=np.int64(3), sample_ids=["s20-b", "s20-a", "s20-c"],
             window=np.int64(1000)),
    ])
    return obs_dir, groups_dir


def test_normalize_sorts_complete_rows_and_validates_membership(tmp_path):
    obs_dir, groups_dir = _fixture(tmp_path)
    out = tmp_path / "out" / "bucket=0"
    result = normalize_partition(obs_dir, groups_dir, out, row_group_size=2)

    table = pq.ParquetFile(out / "part-00000.parquet").read()
    assert table["cell_id"].to_pylist() == [10, 10, 10, 20, 20, 20]
    assert table["sample_id"].to_pylist() == [
        "s10-a", "s10-b", "s10-c", "s20-a", "s20-b", "s20-c"]
    assert table["dt"].to_pylist() == pytest.approx([10.1, 10.2, 10.3, 20.1, 20.2, 20.3])
    assert table["T_diff"].to_pylist()[0] == pytest.approx([10.1, 10.6])
    assert result["rows"] == 6
    assert result["group_members"] == 6
    assert (out / MARKER).is_file()


def test_normalize_rejects_wrong_group_membership_without_publishing(tmp_path):
    obs_dir, groups_dir = _fixture(tmp_path)
    path = groups_dir / "part-00000.parquet"
    groups = pq.ParquetFile(path).read().to_pylist()
    groups[0]["sample_ids"][0] = "not-an-observation"
    pq.write_table(pa.Table.from_pylist(groups), path)
    out = tmp_path / "out" / "bucket=0"

    with pytest.raises(ValueError, match="membership"):
        normalize_partition(obs_dir, groups_dir, out)
    assert not out.exists()
