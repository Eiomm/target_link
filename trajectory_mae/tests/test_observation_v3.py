"""Contracts for the compact, published ``observations_v3`` corpus."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trajectory_mae import data as data_module
from trajectory_mae.data import CellDataset


DAY = "20260817"
M_MAX = 4
SEED = 99
FORMAT = "trajectory_mlp_observation_v3"


def _row(cell_id, sample_id, *, dt=1, times=(1,), ratios=None, valid=None, bins=(0,)):
    if ratios is None:
        ratios = (10,) * len(times)
    if valid is None:
        valid = (True,) * len(times)
    return dict(cell_id=cell_id, sample_id=sample_id, dt=np.float32(dt),
                T_diff=[np.float32(x) for x in times],
                ratio_pct=[np.int8(x) for x in ratios], valid=list(valid),
                bin_pos=[np.int8(x) for x in bins])


def _write_source(root):
    """Two shuffled files, including incomplete duplicate bins and empty bucket 1."""
    rows = [
        _row(0, "zeta", dt=9, times=(2, 3), bins=(0, 3)),
        _row(0, "alpha", dt=1, times=(4,), bins=(1,)),
        # It has a true value, so it is a candidate, but bin 2 is incomplete.
        _row(0, "mixed", dt=2, times=(5, np.nan), valid=(True, False), bins=(2, 2)),
        _row(0, "none", dt=3, times=(np.nan,), valid=(False,), bins=(4,)),
        _row(0, "beta", dt=4, times=(6,), bins=(5,)),
        _row(0, "gamma", dt=5, times=(7,), bins=(6,)),
        _row(0, "tail", dt=6, times=(8,), bins=(7,)),
        # Stored despite being too small to form a group.
        _row(128, "small-b", dt=7, times=(9,), bins=(8,)),
        _row(128, "small-a", dt=8, times=(10,), bins=(9,)),
        _row(256, "also-mixed", dt=10, times=(11, np.nan), valid=(True, False), bins=(10, 10)),
        _row(256, "all-invalid", dt=11, times=(np.nan,), valid=(False,), bins=(11,)),
    ]
    rng = np.random.default_rng(812)
    rng.shuffle(rows)
    part = root / "observations_v2" / f"day={DAY}" / "bucket=0"
    part.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows[:5]), part / "z-last.parquet")
    pq.write_table(pa.Table.from_pylist(rows[5:]), part / "a-first.parquet")

    empty = root / "observations_v2" / f"day={DAY}" / "bucket=1"
    empty.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        _row(1, "invalid-1", times=(np.nan,), valid=(False,), bins=(0,)),
        _row(1, "invalid-2", times=(np.nan,), valid=(False,), bins=(1,)),
    ]), empty / "part.parquet")


def _build(source, destination, bucket="0"):
    # Keep collection independent from the in-progress builder implementation.
    from trajectory_mae.tools.prepare_observation_v3 import build_partition
    return build_partition((str(source), str(destination), DAY, bucket, M_MAX, SEED))


def _publish(destination, receipts):
    marker = dict(format=FORMAT, m_max=M_MAX, data_seed=SEED,
                  partitions={f"{day}/{bucket}": receipt for day, bucket, receipt in receipts})
    (destination / "_OBSERVATION_V3_SUCCESS.json").write_text(json.dumps(marker))


def _items(root, epoch=0):
    return list(CellDataset([str(root)], [DAY], m_max=M_MAX, seed=SEED, epoch=epoch))


def _by_group(items):
    return {item["group_id"]: item for item in items}


def test_v3_compacts_sorted_rows_and_matches_legacy_members_and_features(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "v3"
    _write_source(source)
    receipt0 = _build(source, destination, "0")
    receipt1 = _build(source, destination, "1")
    _publish(destination, (receipt0, receipt1))

    assert receipt0[2]["stored_rows"] == 7
    stats = receipt0[2]["stats"]
    assert {key: stats[key] for key in ("raw_rows", "candidate_rows", "usable_rows",
                                        "dropped_no_valid", "dropped_tail", "groups")} == {
        "raw_rows": 11, "candidate_rows": 9, "usable_rows": 7,
        "dropped_no_valid": 4, "dropped_tail": 3, "groups": 1,
    }
    assert receipt1[2]["stored_rows"] == 0
    assert receipt1[2]["stats"]["groups"] == 0

    observations = destination / receipt0[2]["observations"]["path"]
    table = pq.ParquetFile(observations).read()
    assert table.column_names == ["cell_id", "sample_id", "dt", "T_diff", "ratio_pct", "valid", "bin_pos"]
    identities = list(zip(table["cell_id"].to_pylist(), table["sample_id"].to_pylist()))
    assert identities == sorted(identities)
    assert identities == [(0, "alpha"), (0, "beta"), (0, "gamma"), (0, "tail"),
                          (0, "zeta"), (128, "small-a"), (128, "small-b")]

    with np.load(destination / receipt0[2]["index"]["path"], allow_pickle=False) as index:
        assert set(index.files) >= {"ptr", "rows", "cell_id", "K", "K_raw", "group_index"}
        np.testing.assert_array_equal(index["ptr"], [0, 4])
        assert index["rows"].min() >= 0 and index["rows"].max() < len(table)
        indexed_samples = set(table["sample_id"].take(pa.array(index["rows"])).to_pylist())
        assert len(indexed_samples) == 4
        assert indexed_samples <= {"alpha", "beta", "gamma", "tail", "zeta"}
        np.testing.assert_array_equal(index["cell_id"], [0])
        np.testing.assert_array_equal(index["K"], [5])
        np.testing.assert_array_equal(index["K_raw"], [7])
        np.testing.assert_array_equal(index["group_index"], [0])

    legacy = _by_group(_items(source))
    v3_dataset = CellDataset([str(destination)], [DAY], m_max=M_MAX, seed=SEED)

    def forbidden(*args, **kwargs):
        raise AssertionError("v3 reader must consume its saved filtered rows and group index")

    monkeypatch.setattr(v3_dataset, "_group_specs", forbidden)
    monkeypatch.setattr(CellDataset, "_cells", staticmethod(forbidden))
    monkeypatch.setattr(CellDataset, "_has_valid_bin", staticmethod(forbidden))
    monkeypatch.setattr(data_module, "_row_any", forbidden)
    compact = _by_group(list(v3_dataset))
    assert compact.keys() == legacy.keys()
    for group_id, before in legacy.items():
        after = compact[group_id]
        for field in ("cell_id", "sample_ids", "K", "K_raw", "group_size", "partition_stats"):
            assert after[field] == before[field]
        for field in ("x", "bin_valid", "delta_t"):
            np.testing.assert_array_equal(after[field], before[field])

    from trajectory_mae import run
    fingerprint = run.file_manifest([str(destination)], [DAY])
    assert fingerprint["partitions"] == 2
    assert len(fingerprint["files"]) == 4  # parquet and group index for each bucket
    assert fingerprint["prepared_artifacts"][0]["format"] == FORMAT


def test_v3_protocol_publication_and_index_integrity_guards(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "v3"
    _write_source(source)
    receipt = _build(source, destination)
    _publish(destination, (receipt,))

    with pytest.raises(ValueError, match="protocol mismatch"):
        CellDataset([str(destination)], [DAY], m_max=8, seed=SEED)
    with pytest.raises(ValueError, match="protocol mismatch"):
        CellDataset([str(destination)], [DAY], m_max=M_MAX, seed=3)

    (destination / "_BUILDING").touch()
    with pytest.raises(ValueError, match="not published"):
        CellDataset([str(destination)], [DAY], m_max=M_MAX, seed=SEED)
    (destination / "_BUILDING").unlink()

    index = destination / receipt[2]["index"]["path"]
    with index.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="artifact changed|checksum"):
        _items(destination)


def test_v3_cli_recycles_worker_and_publishes_all_partitions(tmp_path):
    """One worker must survive more than its four-job recycle allowance."""
    train, val, output = tmp_path / "train", tmp_path / "val", tmp_path / "output"
    for root, day in ((train, DAY), (val, "20260823")):
        for bucket in range(5):
            partition = root / "observations_v2" / f"day={day}" / f"bucket={bucket}"
            partition.mkdir(parents=True)
            rows = [_row(bucket, f"{day}-{bucket}-{i}", dt=i, times=(i + 1,), bins=(i,))
                    for i in range(3)]
            pq.write_table(pa.Table.from_pylist(rows), partition / "part.parquet")

    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "trajectory_mae.tools.prepare_observation_v3",
               "--train", str(train), "--val", str(val), "--out", str(output),
               "--train-days", DAY, "--val-days", "20260823", "--buckets", "0", "1", "2", "3", "4",
               "--workers", "1", "--m-max", str(M_MAX), "--seed", str(SEED)]
    completed = subprocess.run(command, cwd=Path(__file__).parents[2], env=environment,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    success = json.loads((output / "_SUCCESS.json").read_text())
    assert success["status"] == "passed" and success["partitions"] == 10
    for side, day in (("train", DAY), ("val", "20260823")):
        marker = json.loads((output / side / "_OBSERVATION_V3_SUCCESS.json").read_text())
        assert len(marker["partitions"]) == 5 and marker["total_groups"] == 5
        assert all(receipt["stored_rows"] == 3 and receipt["stats"]["groups"] == 1
                   for receipt in marker["partitions"].values())
        items = list(CellDataset([str(output / side)], [day], m_max=M_MAX, seed=SEED))
        assert len(items) == 5 and all(item["group_size"] == 3 for item in items)
