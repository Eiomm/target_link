"""Contract checks for the isolated trajectory MLP experiment."""
import copy
import importlib.util
from pathlib import Path

import torch
import pytest


MODEL_PATH = Path(__file__).parents[1] / "model.py"
spec = importlib.util.spec_from_file_location("trajectory_mlp_model", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
TrajectoryMLPMAE = model_module.TrajectoryMLPMAE


def make_batch(B=2, M=4):
    torch.manual_seed(4)
    x = torch.rand(B, M, 50, 3)
    valid = torch.ones(B, M, 50, dtype=torch.bool)
    valid[:, :, 9] = False
    x[..., 0] *= 5
    mae_mask = torch.zeros(B, M, dtype=torch.bool)
    mae_mask[:, 0] = True
    return dict(x=x, bin_valid=valid, traj_valid=torch.ones(B, M, dtype=torch.bool),
                delta_t=torch.rand(B, M) * 600, mae_mask=mae_mask)


def small_model(dropout=0):
    return TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=dropout).eval()


def test_hidden_travel_times_and_validity_cannot_change_output():
    model, batch = small_model(), make_batch()
    before = model(batch)
    changed = copy.deepcopy(batch)
    hidden = changed["mae_mask"]
    changed["x"][..., 0][hidden] = float("nan")
    changed["bin_valid"][hidden] = ~changed["bin_valid"][hidden]
    after = model(changed)
    torch.testing.assert_close(after["prediction_seconds"], before["prediction_seconds"])
    torch.testing.assert_close(after["representation"], before["representation"])


def test_observed_is_not_an_input_feature():
    model, batch = small_model(), make_batch()
    before = model(batch)
    batch["x"][..., 2].uniform_(-1e6, 1e6)
    after = model(batch)
    torch.testing.assert_close(after["prediction_seconds"], before["prediction_seconds"])
    torch.testing.assert_close(after["representation"], before["representation"])


def test_padded_nan_time_is_safe_and_padding_order_does_not_matter():
    model, batch = small_model(), make_batch(B=1, M=5)
    batch["traj_valid"][0, 3:] = False
    batch["delta_t"][0, 3:] = float("nan")
    batch["x"][0, 3:] = float("nan")
    out = model(batch)
    assert torch.isfinite(out["prediction_seconds"]).all()
    perm = torch.tensor([2, 1, 0, 4, 3])
    permuted = {key: value[:, perm].clone() if value.ndim >= 2 else value
                for key, value in batch.items()}
    got = model(permuted)
    torch.testing.assert_close(got["representation"], out["representation"])


def test_packed_batch_matches_each_group_and_variable_visible_counts():
    model, batch = small_model(), make_batch(B=3, M=5)
    batch["mae_mask"][0, 1:] = True
    batch["mae_mask"][1, [0, 2, 4]] = True
    batch["mae_mask"][2] = True
    packed = model(batch)
    for group in range(3):
        single = {key: value[group:group + 1].clone() for key, value in batch.items()}
        one = model(single)
        torch.testing.assert_close(packed["representation"][group], one["representation"][0])
        torch.testing.assert_close(packed["prediction_seconds"][group], one["prediction_seconds"][0])
    assert torch.isfinite(packed["representation"]).all()


def test_nonfinite_time_on_a_real_trajectory_is_rejected():
    model, batch = small_model(), make_batch(B=1, M=3)
    batch["delta_t"][0, 1] = float("nan")
    try:
        model(batch)
    except ValueError as error:
        assert "delta_t" in str(error)
    else:
        raise AssertionError("real trajectory NaN delta_t was silently accepted")


@pytest.mark.parametrize("encoding", ["seconds", "bucket30"])
def test_shared_cls_receives_mean_of_independent_group_reconstruction_gradients(encoding):
    model = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0,
                             time_encoding=encoding).eval()
    batch = make_batch(B=2, M=4)
    # Different visible counts exercise padding without changing loss weights:
    # each group's reconstruction loss is averaged before the group mean.
    batch["mae_mask"][1, 1] = True

    def loss_for(data):
        prediction = model(data)["prediction_seconds"]
        selected = data["mae_mask"].unsqueeze(-1) & data["bin_valid"]
        error = (prediction - data["x"][..., 0]).abs()
        return ((error * selected).sum((1, 2)) / selected.sum((1, 2))).mean()

    initial_cls = model.cls_token.detach().clone()
    packed_gradient, = torch.autograd.grad(loss_for(batch), model.cls_token)
    individual_gradients = []
    for group in range(2):
        single = {key: value[group:group + 1] for key, value in batch.items()}
        gradient, = torch.autograd.grad(loss_for(single), model.cls_token)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        individual_gradients.append(gradient)
    torch.testing.assert_close(packed_gradient, torch.stack(individual_gradients).mean(0),
                               atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(model.cls_token, initial_cls)


def test_zero_valid_bin_differs_from_invalid_bin_feature():
    model = small_model()
    x = torch.zeros(2, 50, 3)
    valid = torch.ones(2, 50, dtype=torch.bool)
    valid[1, 0] = False
    features = model._visible_features(x, valid)
    assert features[0, 2] == 1
    assert features[1, 2] == 0


def test_variable_groups_gradients_and_state_reload():
    model, batch = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0), make_batch(B=3, M=7)
    batch["traj_valid"][1, 5:] = False
    batch["mae_mask"][1, 5:] = False
    output = model(batch)
    assert output["prediction_seconds"].shape == (3, 7, 50)
    assert not any("group_bin" in key for key in output)
    output["prediction_seconds"].mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    reloaded = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0).eval()
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded(batch)["prediction_seconds"], model.eval()(batch)["prediction_seconds"])


@pytest.mark.parametrize("encoding", ["seconds", "bucket30"])
def test_decoder_uses_visible_states_even_with_fixed_cls(encoding):
    model = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0,
                             time_encoding=encoding).eval()
    batch = make_batch(B=1, M=4)
    cls, states = model._encode_visible(batch)
    states = states.detach().requires_grad_()
    model._encode_visible = lambda _: (cls.detach(), states)
    out = model(batch)
    out['prediction_seconds'][0, 0].sum().backward()
    assert states.grad[0, 1:].abs().sum() > 0
    assert states.grad[0, 0].abs().sum() == 0


def test_bucket_boundaries_and_same_bucket_predictions():
    model = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0,
                             time_encoding="bucket30").eval()
    times = torch.tensor([0., 29.999, 30., 59.999, 570., 599.999, 600.])
    embedding = model.time_embedding(times)
    for left, right in [(0, 1), (2, 3), (4, 5), (5, 6)]:
        torch.testing.assert_close(embedding[left], embedding[right])
    assert not torch.equal(embedding[1], embedding[2])
    batch = make_batch(B=1, M=4)
    batch['mae_mask'][0, :2] = True
    batch['delta_t'][0, :2] = torch.tensor([31., 49.])
    batch['x'][0, 1, :, 1] = batch['x'][0, 0, :, 1]
    before = model(batch)
    torch.testing.assert_close(before['prediction_seconds'][0, 0], before['prediction_seconds'][0, 1])
    batch['x'][..., 0][batch['mae_mask']] = float('nan')
    batch['bin_valid'][batch['mae_mask']] = False
    torch.testing.assert_close(model(batch)['prediction_seconds'], before['prediction_seconds'])


@pytest.mark.parametrize("encoding", ["seconds", "bucket30"])
def test_time_range_and_padded_decoder_invariance(encoding):
    model = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0,
                             time_encoding=encoding).eval()
    batch = make_batch(B=1, M=4)
    for invalid_time in [-.1, 600.1, float('inf')]:
        batch['delta_t'][0, 3] = invalid_time
        with pytest.raises(ValueError, match='delta_t'):
            model(batch)
    batch['traj_valid'][0, 3] = False
    batch['delta_t'][0, 3] = float('nan')
    batch['x'][0, 3] = float('nan')
    output = model(batch)
    trimmed = {k: v[:, :3] for k, v in batch.items()}
    torch.testing.assert_close(output['prediction_seconds'][:, :3], model(trimmed)['prediction_seconds'])


def test_seconds_preserves_sub_bucket_time_and_model_budget():
    model = TrajectoryMLPMAE()
    assert not torch.equal(model.time_embedding(torch.tensor(31.)),
                           model.time_embedding(torch.tensor(49.)))
    assert sum(p.numel() for p in model.parameters()) == 5_003_058
    bucket = TrajectoryMLPMAE(time_encoding='bucket30')
    assert sum(p.numel() for p in bucket.parameters()) == 4_941_874
    with pytest.raises(ValueError, match='10M'):
        TrajectoryMLPMAE(d_model=512)


def test_invalid_time_zeroed_but_ratio_retained_without_reserved_channel():
    model = small_model()
    x = torch.ones(1, 50, 3)
    valid = torch.ones(1, 50, dtype=torch.bool)
    valid[0, -1] = False
    x[0, -1, 0] = float('nan')
    x[0, -1, 1] = .3
    features = model._visible_features(x, valid)
    assert features.shape == (1, 150)
    torch.testing.assert_close(features[0, -3:], torch.tensor([0., .3, 0.]))
    batch = make_batch()
    before = model(batch)
    batch['x'][..., 0][~batch['bin_valid']] = float('nan')
    torch.testing.assert_close(model(batch)['prediction_seconds'], before['prediction_seconds'])


@pytest.mark.parametrize('encoding', ['seconds', 'bucket30'])
def test_hidden_ratio_conditions_decoder_but_not_encoder(encoding):
    model = TrajectoryMLPMAE(d_model=16, heads=2, layers=1, dropout=0,
                             time_encoding=encoding).eval()
    batch = make_batch(B=1, M=4)
    batch['mae_mask'][0, :2] = True
    batch['delta_t'][0, :2] = 75.
    batch['x'][0, 1, :, 1] = batch['x'][0, 0, :, 1]
    before = model(batch)
    torch.testing.assert_close(before['prediction_seconds'][0, 0], before['prediction_seconds'][0, 1])
    batch['x'][0, 1, -1, 1] += .3
    batch['bin_valid'][0, 1, -1] = False
    after = model(batch)
    torch.testing.assert_close(after['representation'], before['representation'])
    torch.testing.assert_close(after['trajectory_state'], before['trajectory_state'])
    assert not torch.allclose(after['prediction_seconds'][0, 1], before['prediction_seconds'][0, 1])
    batch['x'].requires_grad_()
    model(batch)['prediction_seconds'][0, 1].sum().backward()
    assert batch['x'].grad[0, :2, :, 0].abs().sum() == 0
    assert batch['x'].grad[0, 1, :, 1].abs().sum() > 0
    assert model.ratio_embed.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('row', [0, 1])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1])
def test_unknown_ratio_rejected_even_if_time_invalid(row, bad):
    model, batch = small_model(), make_batch()
    batch['bin_valid'][0, row, -1] = False
    batch['x'][0, row, -1, 1] = bad
    with pytest.raises(ValueError, match='ratio'):
        model(batch)
