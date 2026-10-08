"""Strict V6 checkpoint persistence."""
import hashlib
from pathlib import Path
import torch
from .model import TrajectoryMLPMAE

FORMAT = "trajectory_mlp_mae_encoder_time_only_decoder_ratio_v6"

def source_hashes():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py")) if "tests" not in path.parts}


def save_checkpoint(path, model, optimizer, kwargs, a, epoch, manifest, best_score):
    torch.save(dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    model_kwargs=kwargs, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    epoch=epoch, data_manifest=manifest, source_sha256=a.source_sha256,
                    target="raw_seconds", loss="micro_valid_bin_MAE", best_val_bin_mae_seconds=best_score), path)


def restore(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    format_name = checkpoint["format"]
    expected_channels = 1
    if format_name != FORMAT:
        raise ValueError("Checkpoint is not this raw-MAE whole-trajectory MLP format")
    kwargs = checkpoint["model_kwargs"]
    if kwargs["input_channels"] != expected_channels:
        raise ValueError(
            f"Checkpoint format {format_name} requires input_channels={expected_channels}"
        )
    model = TrajectoryMLPMAE(**kwargs).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint
