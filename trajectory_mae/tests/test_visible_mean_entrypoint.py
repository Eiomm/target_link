"""Fair comparison and bounded CPU checks; never run the real corpus."""
import copy
import json

import pytest

from trajectory_mae import run
from trajectory_mae.tests.test_run import corpus
from trajectory_mae.tests.test_evaluation import batch
from trajectory_mae.evaluation import visible_mean
from trajectory_mae.tools import evaluate_visible_mean as baseline


def test_baseline_compares_saved_model_without_model_or_prediction_export(tmp_path, monkeypatch):
    corpus(tmp_path / "data")
    root = tmp_path / "model"
    run.main(["train", "--data", str(tmp_path / "data"), "--train-days", "20260817",
              "--val-days", "20260823", "--out", str(root / "artifacts"),
              "--device", "cpu", "--threads", "1", "--workers", "0",
              "--batch-size", "2", "--epochs", "1", "--d-model", "16",
              "--heads", "2", "--layers", "1", "--decoder-layers", "1",
              "--dropout", "0", "--plot-every", "0"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not construct/load the model")

    monkeypatch.setattr(run, "TrajectoryMLPMAE", forbidden)
    monkeypatch.setattr(run, "restore", forbidden)
    out = tmp_path / "baseline"
    result = baseline.main(["--model-run", str(root), "--out", str(out),
                            "--batch-size", "1", "--workers", "0", "--threads", "1"])
    ours = json.loads((root / "artifacts/best_val_metrics.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["eval_mask_id"] == ours["eval_mask_id"]
    assert metrics["bins"] == ours["bins"] == 8
    assert metrics["ours"] is None
    assert result["epochs"][0]["epoch"] == 1
    assert result["epochs"][0]["metrics"]["bin_mae_seconds"]["model"] == ours["ours"]["bin_mae_seconds"]
    assert (out / "comparison.md").exists()
    assert not list(out.rglob("*.csv"))
    assert not list(out.rglob("*.pt"))
    # Reject changed source metadata before spending time on a second pass.
    manifest_path = root / "artifacts/data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["validation"]["metadata_sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="验证数据"):
        baseline.main(["--model-run", str(root), "--out", str(tmp_path / "bad")])
    assert not (tmp_path / "bad").exists()


def test_comparison_requires_paired_masks_and_uses_correct_improvement_sign():
    values = {key: 2.0 for key in baseline.METRICS}
    metrics = dict(eval_mask_id="same", bins=12, trajectories=3, groups=2, groups_total=2,
                   baseline=values)
    model = {**metrics, "ours": {key: 1.0 for key in baseline.METRICS}}
    report = baseline.compare(metrics, [(1, model)])
    assert report["epochs"][0]["metrics"]["bin_mae_seconds"]["improvement_percent"] == 50
    for key in ("eval_mask_id", "bins", "trajectories", "groups", "groups_total"):
        changed = copy.deepcopy(model)
        changed[key] = "different"
        with pytest.raises(ValueError, match="不匹配"):
            baseline.compare(metrics, [(1, changed)])


def test_reference_epoch_snapshot_ignores_incomplete_append(tmp_path):
    record = dict(epoch=0, val={"ours": {"bin_mae_seconds": .5}})
    (tmp_path / "metrics.jsonl").write_text(json.dumps(record) + '\n{"epoch":1')
    assert baseline.reference_epochs(tmp_path) == [(1, record["val"])]


def test_fill_uses_visible_same_bin_mean_and_fallback_without_hidden_targets():
    data = batch([[2., 4.], [6., 8.], [20., 30., 40.]], hidden=[False, False, True])
    prediction = visible_mean(data)["prediction_seconds"][0, 2, :3]
    # Bin 2 has no visible support: use mean(2, 4, 6, 8) = 5.
    assert prediction.tolist() == [4., 6., 5.]
    data["x"][0, 2, :, 0] = 1000.
    assert visible_mean(data)["prediction_seconds"][0, 2, :3].tolist() == [4., 6., 5.]
