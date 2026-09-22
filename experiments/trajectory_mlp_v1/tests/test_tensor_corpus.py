"""End-to-end contracts for the published, padded tensor corpus."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from experiments.trajectory_mlp_v1.data import CellDataset, _Store, collate_cells
from experiments.trajectory_mlp_v1 import run
from experiments.trajectory_mlp_v1.tensor_corpus import FORMAT, verify


DAY = "20260817"
SEED = 20260921


def _write_source(root):
    """Write deliberately shuffled data, including one entirely invalid bucket."""
    rows = []
    # Counts exercise full chunks, retained 3-member tails, and discarded tails.
    for cell, count in [(0, 130), (128, 67), (256, 5)]:
        for i in range(count):
            rows.append(dict(cell_id=cell, sample_id=f"{cell}-{i:03d}", dt=np.float32(i),
                             T_diff=[np.float32(i + 1), np.float32(i + 2)],
                             ratio_pct=[np.float32(10), np.float32(7)], valid=[True, True],
                             bin_pos=[np.int8(i % 50), np.int8((i + 3) % 50)]))
    np.random.default_rng(81).shuffle(rows)
    bucket = root / "observations_v2" / f"day={DAY}" / "bucket=0"
    bucket.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), bucket / "part.parquet")

    invalid = [dict(cell_id=1, sample_id=f"bad-{i}", dt=np.float32(i),
                    T_diff=[np.float32(np.nan)], ratio_pct=[np.float32(10)], valid=[False],
                    bin_pos=[np.int8(0)]) for i in range(4)]
    bad_bucket = root / "observations_v2" / f"day={DAY}" / "bucket=1"
    bad_bucket.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(invalid), bad_bucket / "part.parquet")


def _build(source, output, *, days=(DAY,), m_max=64, seed=SEED):
    # Import lazily so this test file can be collected while the builder is added.
    from experiments.trajectory_mlp_v1.tools.prepare_tensors import build
    result = build(source, output, list(days), m_max=m_max, seed=seed)
    assert result["format"] == FORMAT
    return result


def _items(root, epoch, *, m_max=64, seed=SEED):
    return list(CellDataset([str(root)], [DAY], m_max=m_max, seed=seed, epoch=epoch))


def _assert_item_equal(old, new):
    for key in ("group_id", "sample_ids", "cell_id", "K", "K_raw", "group_size", "day", "bucket"):
        assert new[key] == old[key]
    assert new["tensor_ready"] is True
    assert new["x"].shape == (64, 50, 3)
    assert new["bin_valid"].shape == (64, 50)
    assert new["traj_valid"].shape == (64,)
    assert new["delta_t"].shape == (64,)
    size = old["group_size"]
    for key in ("x", "bin_valid", "traj_valid", "delta_t"):
        np.testing.assert_array_equal(new[key][:size], old[key])
        assert not new[key][size:].any()
    assert new["partition_stats"] == old["partition_stats"]


def test_tensor_build_matches_legacy_multiple_epochs_and_never_rebuilds(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "tensors"
    _write_source(source)
    ready = _build(source, output)
    assert (output / "_TENSORS_SUCCESS.json").exists()
    assert not (output / "_TENSORS_BUILDING").exists()
    assert set(ready["partitions"]) == {f"{DAY}/0", f"{DAY}/1"}
    assert ready["partitions"][f"{DAY}/1"]["stats"]["groups"] == 0

    old_by_epoch = {epoch: _items(source, epoch) for epoch in (0, 1, 1_000_000)}
    for epoch, old in old_by_epoch.items():
        new_ds = CellDataset([str(output)], [DAY], seed=SEED, epoch=epoch)
        def forbidden(*args, **kwargs):
            raise AssertionError("tensor reader must not read parquet or rebuild groups")
        monkeypatch.setattr(_Store, "read", forbidden)
        monkeypatch.setattr(new_ds, "_scatter", forbidden)
        monkeypatch.setattr(new_ds, "_group_specs", forbidden)
        new = list(new_ds)
        assert [item["group_id"] for item in new] == [item["group_id"] for item in old]
        for before, after in zip(old, new):
            _assert_item_equal(before, after)
        old_batch, new_batch = collate_cells(old, epoch=epoch), collate_cells(new, epoch=epoch)
        assert old_batch.keys() == new_batch.keys()
        for key in ("x", "bin_valid", "traj_valid", "delta_t", "mae_mask", "cell_id", "K", "K_raw", "group_size"):
            torch.testing.assert_close(old_batch[key], new_batch[key])
        for key in ("group_id", "sample_ids", "day", "bucket", "partition_stats", "m_max", "n_bins"):
            assert old_batch[key] == new_batch[key]


def test_tensor_protocol_publication_and_size_guards(tmp_path):
    source, output = tmp_path / "source", tmp_path / "tensors"
    _write_source(source)
    _build(source, output)
    with pytest.raises(ValueError, match="protocol mismatch"):
        CellDataset([str(output)], [DAY], m_max=16, seed=SEED)
    with pytest.raises(ValueError, match="protocol mismatch"):
        CellDataset([str(output)], [DAY], seed=3)

    (output / "_TENSORS_BUILDING").touch()
    with pytest.raises(ValueError, match="not published"):
        CellDataset([str(output)], [DAY])
    (output / "_TENSORS_BUILDING").unlink()

    manifest = json.loads((output / "_TENSORS_SUCCESS.json").read_text())
    rec = next(iter(manifest["partitions"][f"{DAY}/0"]["arrays"].values()))
    with (output / rec["path"]).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        list(CellDataset([str(output)], [DAY]))


def test_payload_checksum_and_resume_are_checked_before_republishing(tmp_path):
    source, output = tmp_path / "source", tmp_path / "tensors"
    _write_source(source)
    first = _build(source, output)
    # Resume after a crash after all receipts were committed but before publication.
    (output / "_TENSORS_SUCCESS.json").unlink()
    resumed = _build(source, output)
    assert resumed["partitions"] == first["partitions"]
    # A completed repeat checks the immutable source as well as all tensor hashes.
    assert _build(source, output) == json.loads((output / "_TENSORS_SUCCESS.json").read_text())

    payload = output / resumed["partitions"][f"{DAY}/0"]["payload"]["path"]
    with payload.open("r+b") as f:
        f.seek(0)
        byte = f.read(1)
        f.seek(0)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(output)

    # Restore a separate published corpus, then prove repeat-build rejects a changed source.
    clean = tmp_path / "clean"
    _build(source, clean)
    part = source / "observations_v2" / f"day={DAY}" / "bucket=0" / "part.parquet"
    with part.open("ab") as f:
        f.write(b"source changed")
    with pytest.raises(ValueError, match="Source changed since tensor publication"):
        _build(source, clean)


def test_tensor_and_legacy_roots_can_be_mixed_and_run_manifest_records_tensor(tmp_path):
    source, output = tmp_path / "source", tmp_path / "tensors"
    _write_source(source)
    _build(source, output)
    # A tensor root cannot overlap an old root's day/bucket, but distinct days can mix.
    legacy = tmp_path / "legacy"
    rows = [dict(cell_id=0, sample_id=f"second-day-{i}", dt=np.float32(1), T_diff=[np.float32(1)],
                 ratio_pct=[np.float32(10)], valid=[True], bin_pos=[np.int8(0)]) for i in range(3)]
    target = legacy / "observations_v2" / "day=20260818" / "bucket=0"
    target.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), target / "part.parquet")
    mixed = list(CellDataset([str(output), str(legacy)], [DAY, "20260818"], seed=SEED))
    assert {item["day"] for item in mixed} == {DAY, "20260818"}

    fingerprint = run.file_manifest([str(output)], [DAY])
    assert fingerprint["partitions"] == 2
    assert fingerprint["prepared_artifacts"][0]["format"] == FORMAT
    assert len(fingerprint["files"]) == sum(len(p["arrays"]) + 1 for p in
                                          json.loads((output / "_TENSORS_SUCCESS.json").read_text())["partitions"].values())


def test_tensor_corpus_runs_cpu_train_and_evaluate(tmp_path):
    """The compressed format is a complete training input, not only a reader fixture."""
    from experiments.trajectory_mlp_v1.tests.test_run import corpus

    source, tensors = tmp_path / "source", tmp_path / "tensors"
    corpus(source)
    _build(source, tensors, days=("20260817", "20260823"))
    common = ["--data", str(tensors), "--train-days", "20260817", "--val-days", "20260823",
              "--device", "cpu", "--threads", "1", "--batch-size", "2", "--bootstrap", "2",
              "--d-model", "16", "--heads", "2", "--layers", "1", "--dropout", "0",
              "--plot-every", "0"]
    fit = tmp_path / "fit"
    run.main(["train", *common, "--out", str(fit), "--epochs", "1"])
    run.main(["evaluate", *common, "--out", str(tmp_path / "evaluation"),
              "--checkpoint", str(fit / "best.pt")])
    assert (fit / "best_val_metrics.json").exists()
    result = json.loads((tmp_path / "evaluation" / "metrics.json").read_text())
    assert result["ours"]["bin_mae_seconds"] >= 0
