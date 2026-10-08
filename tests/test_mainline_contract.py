"""Mainline isolation and strict checkpoint protocol."""
import ast
from pathlib import Path
import pytest
import torch
from trajectory_mae import run
from trajectory_mae.model import TrajectoryMLPMAE

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_never_imports_archived_models():
    for root in [ROOT / "trajectory_mae", ROOT / "tools"]:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            assert len(path.read_text().splitlines()) <= 600, path
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith(("V4", "target_link_v1", "legacy", "experiments")), path
                if isinstance(node, ast.Import):
                    assert all(not a.name.startswith(("V4", "target_link_v1", "legacy", "experiments"))
                               for a in node.names), path


@pytest.mark.parametrize("format_name", ["trajectory_mlp_mae_known_ratio_v4", "trajectory_mlp_mae_time_ratio_v5"])
def test_mainline_rejects_other_checkpoint_protocols(tmp_path, format_name):
    path = tmp_path / "weights.pt"
    torch.save({"format": format_name}, path)
    with pytest.raises(ValueError, match="format"):
        run.restore(path, "cpu")


def test_checkpoint_roundtrip(tmp_path):
    kwargs = dict(d_model=16, heads=2, layers=1, dropout=0, input_channels=1)
    original = TrajectoryMLPMAE(**kwargs)
    path = tmp_path / "weights.pt"
    torch.save(dict(format=run.FORMAT, model_kwargs=kwargs, model=original.state_dict()), path)
    restored, _ = run.restore(path, "cpu")
    for expected, actual in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
