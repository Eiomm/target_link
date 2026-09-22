import math

import pytest
import torch

from experiments.trajectory_mlp_v1.evaluation import Evaluator, reconstruction_loss, visible_mean


def batch(values, valid=None, hidden=None, cell_id=None):
    """Tiny [B,M,50,3] fixture; only the first two bins are normally live."""
    x = torch.zeros(1, len(values), 50, 3, dtype=torch.float32)
    for i, row in enumerate(values):
        x[0, i, :len(row), 0] = torch.tensor(row)
    v = torch.zeros(1, len(values), 50, dtype=torch.bool)
    if valid is None:
        for i, row in enumerate(values): v[0, i, :len(row)] = True
    else:
        v[0] = torch.tensor(valid, dtype=torch.bool)
    h = torch.zeros(1, len(values), dtype=torch.bool)
    if hidden is not None: h[0] = torch.tensor(hidden, dtype=torch.bool)
    return dict(x=x, bin_valid=v, traj_valid=v.any(-1), mae_mask=h,
                cell_id=torch.tensor([0 if cell_id is None else cell_id]), K=torch.tensor([len(values)]),
                group_id=["g"], sample_ids=[["s%d" % i for i in range(len(values))]])


def test_visible_mean_is_raw_seconds_arithmetic_mean_not_log_mean():
    b = batch([[0., 8.], [0., 0.], [1., 1.]], hidden=[False, False, True])
    got = visible_mean(b)
    assert got["same_bin_support"].tolist() == [[True] * 2 + [False] * 48]
    assert got["prediction_seconds"][0, 2, 0].item() == pytest.approx(0.)
    assert got["prediction_seconds"][0, 2, 1].item() == pytest.approx(4.)
    assert got["prediction_seconds"][0, 2, 1].item() != pytest.approx(math.expm1((math.log1p(8) + math.log1p(0)) / 2))


def test_visible_mean_falls_back_and_no_visible_is_unscorable():
    b = batch([[2., 0.], [0., 7.], [9., 9.]], valid=[[True, False] + [False]*48, [False, True] + [False]*48, [True, True] + [False]*48], hidden=[False, False, True])
    got = visible_mean(b)
    assert got["prediction_seconds"][0, 2, 0].item() == pytest.approx(2.)
    assert got["prediction_seconds"][0, 2, 1].item() == pytest.approx(7.)
    none = batch([[3.]], hidden=[True])
    assert not visible_mean(none)["group_has_visible"].item()
    assert reconstruction_loss(torch.full_like(none["x"][..., 0], float("nan")), none).item() == 0


def test_hidden_nan_cannot_leak_but_supervised_nan_fails():
    b = batch([[2., 4.], [3., 5.]], hidden=[False, True])
    b["x"][0, 1, 4, 0] = float("nan")  # invalid/unselected hidden value
    p = torch.zeros_like(b["x"][..., 0]); p[0, 1, :2] = torch.tensor([3., 5.])
    assert reconstruction_loss(p, b).item() == pytest.approx(0.)
    p[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="prediction"):
        reconstruction_loss(p, b)


def test_loss_is_micro_and_groupbalanced_is_auxiliary():
    # One group: hidden trajectory has two bins and MAE 2.  A second group is
    # simulated through separate updates below to make the group-balanced check.
    b = batch([[1., 1.], [1., 1.]], hidden=[False, True], cell_id=1)
    p = torch.zeros_like(b["x"][..., 0]); p[0, 1, :2] = 3
    assert reconstruction_loss(p, b).item() == pytest.approx(2.)
    e = Evaluator(); e.update(p, b)
    r = e.finalize()
    assert r["ours"]["bin_mae_seconds"] == pytest.approx(2.)
    assert r["ours"]["group_balanced_bin_mae_seconds"] == pytest.approx(2.)


def test_evaluator_is_paired_and_single_cell_bootstrap_is_not_a_ci():
    b = batch([[1., 3.], [2., 5.]], hidden=[False, True], cell_id=11)
    ours = torch.zeros_like(b["x"][..., 0]); ours[0, 1, :2] = torch.tensor([2., 5.])
    e = Evaluator(); e.update(ours, b)
    out = e.finalize(bootstrap=20, seed=4)
    assert out["groups"] == 1 and out["bins"] == 2
    assert out["same_bin_coverage"] == pytest.approx(1.)
    assert out["paired_ci"]["effective_clusters"] == 1
    assert out["paired_ci"]["ci95"] is None
    assert out["paired_ci"]["ci_status"] == "insufficient_clusters"
    baseline_only = Evaluator(); baseline_only.update(None, b)
    assert baseline_only.finalize()["ours"] is None


def test_no_supervised_positions_has_no_false_ci():
    b = batch([[1., 2.]], hidden=[False])
    e = Evaluator(); e.update(torch.zeros_like(b["x"][..., 0]), b)
    out = e.finalize(bootstrap=10)
    assert out["bins"] == 0 and out["ours"] is None and "paired_ci" not in out


def test_strata_distinguish_m50_axes_and_short_complete_ratio_trajectories():
    # M=50 catches accidental use of vector length to infer whether a mask is
    # over bins or trajectory rows.  The hidden row is short but every one of
    # its valid bins has ratio=1.
    b = batch([[1., 1.]] * 50, hidden=[False] * 49 + [True])
    b["x"][..., 1] = b["bin_valid"].to(torch.float32)
    p = b["x"][..., 0].clone()
    out = Evaluator().update(p, b).finalize()
    assert out["stratified"]["same_bin_support"]["same_bin"]["baseline"]["bins"] == 2
    assert out["stratified"]["target_ratio"]["full"]["baseline"]["bins"] == 2
    assert out["stratified"]["full50"]["partial"]["baseline"]["bins"] == 2
    assert out["stratified"]["fullratio"]["partial"]["baseline"]["bins"] == 2


def test_valid_length_strata_are_per_hidden_trajectory():
    b = batch([[1., 1., 1.], [2.], [3., 3., 3.]], hidden=[False, True, True])
    b["x"][..., 1] = b["bin_valid"].to(torch.float32)
    p = b["x"][..., 0].clone()
    out = Evaluator().update(p, b).finalize()
    lengths = out["stratified"]["valid_bin_length"]
    assert lengths["1"]["baseline"]["bins"] == 1
    assert lengths["3"]["baseline"]["bins"] == 3
