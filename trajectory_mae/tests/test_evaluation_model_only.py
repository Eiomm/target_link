"""Regression coverage for direct model-only trajectory evaluation."""
import csv

import pytest
import torch

from trajectory_mae import evaluation


def _batch():
    # Two groups exercise mask identity, strata, and a fallback same-bin
    # diagnostic.  Each group has one hidden trajectory.
    x = torch.zeros(2, 3, 50, 3)
    valid = torch.zeros(2, 3, 50, dtype=torch.bool)
    valid[:, :, :2] = True
    # Group a has no visible value in bin 1, but its hidden trajectory does.
    valid[0, :2, 1] = False
    x[0, :, :2, 0] = torch.tensor([[2., 6.], [4., 8.], [9., 11.]])
    x[1, :, :2, 0] = torch.tensor([[3., 7.], [5., 9.], [10., 12.]])
    x[..., 1] = valid
    hidden = torch.tensor([[False, False, True], [False, True, False]])
    return dict(x=x, bin_valid=valid, traj_valid=valid.any(-1), mae_mask=hidden,
                cell_id=torch.tensor([17, 23]), K=torch.tensor([3, 3]),
                group_id=["a", "b"], sample_ids=[["a0", "a1", "a2"], ["b0", "b1", "b2"]])


def test_model_only_skips_mean_and_bootstrap_but_preserves_ours_audit(tmp_path, monkeypatch):
    batch = _batch()
    prediction = batch["x"][..., 0].clone()
    prediction[0, 2, :2] += torch.tensor([1., 2.])
    prediction[1, 1, :2] += torch.tensor([3., 4.])

    default = evaluation.Evaluator(tmp_path / "default").update(prediction, batch).finalize(bootstrap=7)

    def forbidden(*args, **kwargs):
        raise AssertionError("model-only evaluation must not compute a baseline or bootstrap")

    monkeypatch.setattr(evaluation, "visible_mean", forbidden)
    monkeypatch.setattr(evaluation.Evaluator, "_bootstrap", forbidden)
    direct = evaluation.Evaluator(tmp_path / "direct", include_baseline=False).update(prediction, batch).finalize(bootstrap=7)

    assert direct["baseline"] is None
    assert direct["paired_ci"] is None
    assert direct["ours"] == pytest.approx(default["ours"])
    assert direct["eval_mask_id"] == default["eval_mask_id"]
    for key in ("groups", "groups_total", "no_supervised_groups", "bins", "trajectories",
                "same_bin_coverage", "fallback_count", "fallback_rate", "unscorable_groups"):
        assert direct[key] == pytest.approx(default[key])
    for category, values in default["stratified"].items():
        for value, expected in values.items():
            assert direct["stratified"][category][value]["baseline"] is None
            assert direct["stratified"][category][value]["ours"] == pytest.approx(expected["ours"])

    with open(tmp_path / "direct" / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == direct["bins"]
    assert all(row["baseline_seconds"] == "" for row in rows)
    assert {row["same_bin_support"] for row in rows} == {"0", "1"}
    assert (tmp_path / "direct" / "eval_mask_identity.jsonl").read_bytes() == (tmp_path / "default" / "eval_mask_identity.jsonl").read_bytes()


def test_model_only_requires_prediction_and_handles_no_supervision(monkeypatch):
    batch = _batch()
    with pytest.raises(ValueError, match="prediction_seconds is required"):
        evaluation.Evaluator(include_baseline=False).update(None, batch)

    batch["mae_mask"][:] = True  # hidden labels without any visible context
    prediction = torch.zeros_like(batch["x"][..., 0])
    monkeypatch.setattr(evaluation, "visible_mean", lambda *_: (_ for _ in ()).throw(AssertionError("called")))
    result = evaluation.Evaluator(include_baseline=False).update(prediction, batch).finalize(bootstrap=99)
    assert result["baseline"] is None
    assert result["ours"] is None
    assert result["paired_ci"] is None
    assert result["bins"] == 0
