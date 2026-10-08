"""The archived result remains separate from the V6 model contract."""
import json
from pathlib import Path
import pytest
import torch
from V4.code import run

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/training/best.pt"


def test_reported_result_provenance():
    result = json.loads((ROOT / "results/comparison/comparison.json").read_text())
    metrics = result["epochs"][0]["metrics"]
    assert metrics["bin_mae_seconds"]["model"] == pytest.approx(0.5550923832636057)
    assert metrics["trajectory_mae_seconds"]["model"] == pytest.approx(6.133908441603787)
    assert metrics["trajectory_rmse_seconds"]["model"] == pytest.approx(15.937815852114007)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Archived server checkpoint is not distributed in Git")
def test_original_checkpoint_loads_and_isolates_hidden_labels():
    torch.set_num_threads(1)
    model, checkpoint = run.restore(CHECKPOINT, "cpu")
    assert checkpoint["format"] == "trajectory_mlp_mae_known_ratio_v4"
    assert checkpoint["model_kwargs"]["input_channels"] == 3
    model.eval()
    batch = dict(x=torch.rand(1, 4, 50, 3), bin_valid=torch.ones(1, 4, 50, dtype=torch.bool),
                 traj_valid=torch.ones(1, 4, dtype=torch.bool),
                 mae_mask=torch.tensor([[True, False, False, False]]),
                 delta_t=torch.tensor([[0., 30., 60., 90.]]))
    with torch.no_grad():
        before = model(batch)
        batch["x"][:, 0, :, 0] = float("nan")
        batch["bin_valid"][:, 0] = False
        after = model(batch)
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
