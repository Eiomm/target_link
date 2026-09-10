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

from target_link_v1.models.cell_mae import CellMAE, masked_reconstruction_loss

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


def test_visible_trajectory_moves_the_aggregate(model):
    """The complement: editing a visible trajectory must move the group state
    and therefore the masked trajectory's reconstruction target."""
    a, b = make_batch(), make_batch()
    b["x"][:, 3] = torch.rand_like(b["x"][:, 3]) * 40.0
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["representation"], ob["representation"])
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


def test_invalid_bin_still_attends(model):
    """presence != validity. A bin whose piece exists but whose T_diff is NaN
    keeps ratio/observed and MUST stay in the attention set: it is real geometry
    with an unknown time. If this fails, someone collapsed the two masks and the
    trajectory silently lost part of its shape."""
    a, b = make_batch(), make_batch()
    b["x"][:, 1, UNKNOWN, 1] = 0.4                 # ratio of the invalid bin
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["trajectory_state"][:, 1],
                              ob["trajectory_state"][:, 1])


def test_hole_is_invisible_but_its_ratio_is_the_switch(model):
    """A hole (ratio == 0) is masked out of attention, so its other channels are
    unreadable; giving it a ratio makes it appear. This is the `present =
    ratio > 0` invariant the reader is expected to maintain."""
    a, b = make_batch(), make_batch()
    b["x"][:, 2, HOLE, 0] = 33.0                   # T_diff inside a hole
    b["x"][:, 2, HOLE, 2] = 1.0                    # observed inside a hole
    oa, ob = model(a), model(b)
    assert torch.allclose(oa["trajectory_state"], ob["trajectory_state"])

    c = make_batch()
    c["x"][:, 2, HOLE, 1] = 0.5                    # now it is present
    assert not torch.allclose(oa["trajectory_state"],
                              model(c)["trajectory_state"])


def test_bin_valid_is_a_real_input_channel(model):
    """§10 regression. "T_diff unknown" (x = [0, 1, 1], bin_valid = 0) and
    "T_diff is genuinely 0 seconds" (same x, bin_valid = 1) are different bins.
    They are only distinguishable because bin_valid rides along as the 4th
    channel -- drop it and this test goes flat."""
    a, b = make_batch(), make_batch()
    for t in (a, b):
        t["x"][:, 3, 5] = torch.tensor([0.0, 1.0, 1.0])
    a["bin_valid"][:, 3, 5] = False
    b["bin_valid"][:, 3, 5] = True
    oa, ob = model(a), model(b)
    assert not torch.allclose(oa["trajectory_state"][:, 3],
                              ob["trajectory_state"][:, 3])


def test_ablation_actually_disables_the_aggregate(model):
    a = make_batch()
    on, off = model(a), model(a, ablate_aggregate=True)
    assert not torch.allclose(on["prediction"], off["prediction"])
    assert torch.isfinite(off["prediction"]).all()


def reference_loss(out, batch):
    """The documented rule, spelled out here so the test does not just restate
    the implementation: Huber over the bins of MASKED trajectories that have a
    known T_diff; equal weight per group; a group with nothing to reconstruct is
    dropped, never counted as a zero."""
    mask = batch["mae_mask"].unsqueeze(-1) & batch["bin_valid"]
    err = torch.nn.functional.huber_loss(out["prediction"], out["target"],
                                         reduction="none")
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
    got = float(masked_reconstruction_loss(model(a), a))
    assert got == pytest.approx(reference_loss(model(a), a), rel=1e-5)


def test_nothing_to_reconstruct_is_dropped_not_zeroed(model):
    a = make_batch()
    a["mae_mask"] = torch.zeros_like(a["mae_mask"])   # nothing masked
    assert masked_reconstruction_loss(model(a), a).item() == pytest.approx(0.0)

    b = make_batch()
    b["bin_valid"][:, 0] = False                      # the only masked traj
    assert masked_reconstruction_loss(model(b), b).item() == pytest.approx(0.0)


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
    from target_link_v1.models.cell_mae import I_RATIO, N_FEATURES
    assert tuple(FEATURES) == ("T_diff", "ratio", "observed")
    assert N_FEATURES == len(FEATURES)
    assert FEATURES[I_RATIO] == "ratio"
