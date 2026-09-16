"""CPU contract tests for CellMAE — synthetic tensors, no corpus, no GPU.

These pin the parts of the model that a corpus-blind local edit can silently
break, and they run in a second or two on a laptop. They are NOT a substitute
for the real-batch acceptance run (tools/check_cell_batch.py on the server): the
corpus reader, the group policy and the actual NaN layout are only exercised
there.

Run: PYTHONPATH=<repo> pytest -q tests/test_cell_mae.py
"""
from __future__ import annotations

import pytest
import torch

from target_link_v1.models.cell_mae import (
    CellMAE,
    masked_reconstruction_loss,
    reconstruction_error_sums,
    reconstruction_loss_by_group,
    reconstruction_mask,
)

B, M, N = 2, 4, 50
HOLE, UNKNOWN = 20, 9        # bin 20: nothing present; bin 9: piece, no time


def make_batch():
    """A group sane by construction, with one trajectory per special case:

    slot 0  MASKED, and it also carries a hole at bin 20
    slot 1  visible, carries an invalid bin at 9 (ratio/observed present,
            T_diff unknown -- the corpus's NaN-as-non-null convention)
    slot 2  visible, carries a hole at bin 20
    slot 3  visible, clean
    """
    g = torch.Generator().manual_seed(0)
    x = torch.rand(B, M, N, 3, generator=g)
    x[..., 2] = (x[..., 2] > 0.5).float()          # observed is 0/1
    x[..., 0] = x[..., 0] * 40.0                   # T_diff seconds
    bin_valid = torch.ones(B, M, N, dtype=torch.bool)
    for b in range(B):
        for slot in (0, 2):                        # holes
            x[b, slot, HOLE] = 0.0
            bin_valid[b, slot, HOLE] = False
        x[b, 1, UNKNOWN] = torch.tensor([0.0, 1.0, 1.0])
        bin_valid[b, 1, UNKNOWN] = False
    mae_mask = torch.zeros(B, M, dtype=torch.bool)
    mae_mask[:, 0] = True
    return dict(x=x, bin_valid=bin_valid,
                traj_valid=torch.ones(B, M, dtype=torch.bool),
                delta_t=torch.arange(B * M).float().reshape(B, M) * 30.0,
                mae_mask=mae_mask)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = CellMAE(d_model=32, heads=4, traj_layers=1, level2_layers=1, dropout=0.0)
    m.eval()
    return m


def test_shapes(model):
    out = model(make_batch())
    assert out["representation"].shape == (B, 32)
    assert out["trajectory_state"].shape == (B, M, 32)
    assert out["group_bin_state"].shape == (B, N, 32)
    assert out["group_bin_valid"].shape == (B, N)
    assert out["prediction"].shape == (B, M, N)
    assert out["target"].shape == (B, M, N)
    assert torch.isfinite(out["prediction"]).all()


def test_masked_trajectory_is_invisible_everywhere(model):
    """Whole-trajectory MAE: a masked trajectory is out of the level-2 KV set,
    so it cannot leak into anyone's prediction -- least of all its own. An edit
    that lets it back in (skip connection, keeping it in `keep`) breaks here."""
    a, b = make_batch(), make_batch()
    b["x"][:, 0] = torch.rand_like(b["x"][:, 0]) * 99.0 + 1.0
    b["bin_valid"][:, 0] = True
    oa, ob = model(a), model(b)
    assert torch.allclose(oa["prediction"][:, 0], ob["prediction"][:, 0])
    assert torch.allclose(oa["prediction"], ob["prediction"])
    assert torch.allclose(oa["representation"], ob["representation"])
    assert torch.allclose(oa["group_bin_state"], ob["group_bin_state"])
    assert torch.equal(oa["group_bin_valid"], ob["group_bin_valid"])


def test_visible_trajectory_moves_the_aggregate(model):
    """The complement: editing a visible trajectory must move the group state
    and therefore the masked trajectory's reconstruction target."""
    a, b = make_batch(), make_batch()
    b["x"][:, 3] = torch.rand_like(b["x"][:, 3]) * 40.0
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["representation"], ob["representation"])
    assert not torch.allclose(oa["group_bin_state"], ob["group_bin_state"])
    assert not torch.allclose(oa["prediction"][:, 0], ob["prediction"][:, 0])


def test_padding_is_inert(model):
    """traj_valid=0 slots are all-zero and must contribute nothing, whatever
    garbage a caller leaves in them."""
    a, b = make_batch(), make_batch()
    for t in (a, b):
        t["traj_valid"][:, 3] = False
    b["x"][:, 3] = torch.randn_like(b["x"][:, 3]).abs() * 50.0
    b["delta_t"][:, 3] = 1e6
    oa, ob = model(a), model(b)
    assert torch.allclose(oa["prediction"][:, :3], ob["prediction"][:, :3])
    assert torch.allclose(oa["representation"], ob["representation"])


def test_invalid_bin_is_excluded_by_the_frozen_mainline(model):
    """The current three-feature model uses bin_valid as its attention and
    pooling mask. Retained ratio/observed values at an invalid bin are inert."""
    a, b = make_batch(), make_batch()
    b["x"][:, 1, UNKNOWN, 1] = 0.4                 # ratio of the invalid bin
    oa, ob = model(a), model(b)
    assert torch.allclose(oa["trajectory_state"][:, 1],
                          ob["trajectory_state"][:, 1])
    assert torch.allclose(oa["group_bin_state"], ob["group_bin_state"])


def test_hole_is_invisible_until_bin_valid_changes(model):
    """Changing features cannot activate a masked bin; bin_valid is the frozen
    mainline's only attention/pooling switch."""
    a, b = make_batch(), make_batch()
    b["x"][:, 2, HOLE, 0] = 33.0                   # T_diff inside a hole
    b["x"][:, 2, HOLE, 2] = 1.0                    # observed inside a hole
    oa, ob = model(a), model(b)
    assert torch.allclose(oa["trajectory_state"], ob["trajectory_state"])

    c = make_batch()
    c["x"][:, 2, HOLE, 1] = 0.5
    assert torch.allclose(oa["trajectory_state"], model(c)["trajectory_state"])
    c["bin_valid"][:, 2, HOLE] = True
    assert not torch.allclose(oa["trajectory_state"], model(c)["trajectory_state"])


def test_bin_valid_is_a_real_input_channel(model):
    """Unknown T_diff and a real zero differ through the independent
    attention/pooling mask, while x remains exactly three features."""
    a, b = make_batch(), make_batch()
    for t in (a, b):
        t["x"][:, 3, 5] = torch.tensor([0.0, 1.0, 1.0])
    a["bin_valid"][:, 3, 5] = False
    b["bin_valid"][:, 3, 5] = True
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["trajectory_state"][:, 3],
                              ob["trajectory_state"][:, 3])


def test_aggregate_ablation_equals_both_individual_ablations(model):
    a = make_batch()
    on = model(a)
    no_cls = model(a, ablate_cls=True)
    no_group_bins = model(a, ablate_group_bins=True)
    off = model(a, ablate_aggregate=True)
    off_explicit = model(a, ablate_cls=True, ablate_group_bins=True)

    # Each switch changes the prediction, and the legacy aggregate switch is
    # exactly equivalent to turning both independently named channels off.
    assert not torch.allclose(on["prediction"], no_cls["prediction"])
    assert not torch.allclose(on["prediction"], no_group_bins["prediction"])
    assert not torch.allclose(on["prediction"], off["prediction"])
    assert torch.allclose(off["prediction"], off_explicit["prediction"])
    assert torch.isfinite(off["prediction"]).all()


def test_masked_prediction_reads_visible_per_bin_group_state(model):
    """The decoder must receive a spatially indexed group channel. Editing a
    visible bin changes that channel and the masked trajectory reconstruction;
    the masked trajectory itself remains excluded by the leakage test above."""
    a, b = make_batch(), make_batch()
    b["x"][:, 3, 12, 0] += 25.0
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["group_bin_state"], ob["group_bin_state"])
    assert not torch.allclose(oa["prediction"][:, 0], ob["prediction"][:, 0])


def reference_loss(out, batch):
    """The documented rule, spelled out here so the test does not just restate
    the implementation: Huber over the bins of MASKED trajectories that have a
    known T_diff; equal weight per group; a group with nothing to reconstruct is
    dropped, never counted as a zero."""
    mask = reconstruction_mask(batch)
    err = torch.nn.functional.huber_loss(out["prediction"], out["target"],
                                         reduction="none").detach()
    per_group = []
    for b in range(mask.shape[0]):
        rows = []
        for m in range(mask.shape[1]):
            cnt = int(mask[b, m].sum())
            if cnt:
                rows.append(float((err[b, m] * mask[b, m]).sum()) / cnt)
        if rows:
            per_group.append(sum(rows) / len(rows))
    return sum(per_group) / len(per_group) if per_group else 0.0


def test_loss_matches_the_documented_rule(model):
    a = make_batch()
    output = model(a)
    got = float(masked_reconstruction_loss(output, a).detach())
    assert got == pytest.approx(reference_loss(output, a), rel=1e-5)


def test_reconstruction_mask_is_masked_and_valid_bins_only():
    batch = make_batch()
    expected = batch["mae_mask"].unsqueeze(-1) & batch["bin_valid"]
    torch.testing.assert_close(reconstruction_mask(batch), expected)


@pytest.mark.parametrize("transform", ["raw", "log1p"])
def test_reported_errors_are_in_seconds_and_use_reconstruction_mask(transform):
    batch = make_batch()
    prediction_s = torch.full((B, M, N), 4.0)
    target_s = torch.full((B, M, N), 2.0)
    output = {"prediction": prediction_s, "target": target_s}
    if transform == "log1p":
        output = {name: torch.log1p(value) for name, value in output.items()}
    absolute, squared, count = reconstruction_error_sums(output, batch, transform)
    expected = int(reconstruction_mask(batch).sum())
    assert count == expected
    assert float(absolute) == pytest.approx(2.0 * expected)
    assert float(squared) == pytest.approx(4.0 * expected)


def test_seconds_metrics_use_float64_for_large_log_predictions():
    batch = make_batch()
    output = {"prediction": torch.full((B, M, N), 100.0),
              "target": torch.zeros(B, M, N)}
    absolute, squared, count = reconstruction_error_sums(output, batch, "log1p")
    assert count > 0
    assert absolute.dtype == squared.dtype == torch.float64
    assert torch.isfinite(absolute) and torch.isfinite(squared)


def test_nan_outside_reconstruction_mask_does_not_poison_loss(model):
    batch = make_batch()
    output = model(batch)
    expected = masked_reconstruction_loss(output, batch)
    mask = reconstruction_mask(batch)
    with_nan = dict(output)
    with_nan["prediction"] = torch.where(
        mask, output["prediction"], torch.full_like(output["prediction"], float("nan")))
    torch.testing.assert_close(masked_reconstruction_loss(with_nan, batch), expected)


def test_nothing_to_reconstruct_is_dropped_not_zeroed(model):
    a = make_batch()
    a["mae_mask"] = torch.zeros_like(a["mae_mask"])   # nothing masked
    assert masked_reconstruction_loss(model(a), a).item() == pytest.approx(0.0)

    b = make_batch()
    b["bin_valid"][:, 0] = False                      # the only masked traj
    assert masked_reconstruction_loss(model(b), b).item() == pytest.approx(0.0)


def test_per_group_loss_keeps_empty_groups_out(model):
    a = make_batch()
    a["mae_mask"][1] = False
    out = model(a)
    loss, has = reconstruction_loss_by_group(out, a)
    assert loss.shape == (B,)
    assert has.tolist() == [True, False]
    actual = float(masked_reconstruction_loss(out, a).detach())
    assert actual == pytest.approx(float(loss[0].detach()))


def test_backward_is_finite(model):
    m = CellMAE(d_model=32, heads=4, traj_layers=1, level2_layers=1, dropout=0.0)
    m.train()
    a = make_batch()
    masked_reconstruction_loss(m(a), a).backward()
    norms = [p.grad.norm() for p in m.parameters() if p.grad is not None]
    assert norms, "no gradients reached any parameter"
    assert all(torch.isfinite(g) for g in norms)
    assert sum(float(g) ** 2 for g in norms) ** 0.5 > 0.0


def test_feature_axis_matches_the_reader():
    from target_link_v1.data.cell_corpus import FEATURES
    from target_link_v1.models.cell_mae import N_FEATURES
    assert tuple(FEATURES) == ("T_diff", "ratio", "observed")
    assert N_FEATURES == len(FEATURES)
