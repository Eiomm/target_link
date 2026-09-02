"""Debug the overfit stall: A/B encoder out_dim, longer horizon (500 epochs).

Reproduces the exact overfit subset (2048 train samples, seed 0, full batch,
dropout off) twice — once with the config encoder (out_dim=128) and once with
the default identity output (out_dim=None) — to tell "slow escape from the
mean-prediction plateau" apart from "the out_proj layer stalls learning".

Usage: python tools/debug_overfit.py [--epochs 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.data.eta_data import build_eta_data  # noqa: E402
from target_link_v1.models.eta import ETAModel  # noqa: E402
from target_link_v1.utils import load_config, seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--config", default="configs/eta.yaml")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    data = build_eta_data(cfg["data"])
    tr = np.random.default_rng(0).permutation(data.rows_of(0))[: int(cfg["overfit"]["n_samples"])]
    L = torch.from_numpy(data.L_n[tr]).to(device)
    V = torch.from_numpy(data.v_n[tr]).to(device)
    Y = torch.from_numpy(data.log_y[tr]).to(device)
    B = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
         for k, v in data.batch(tr).items()}
    print(f"[debug] {len(tr)} samples | log_y var {Y.var().item():.4f} (mean-pred floor)",
          flush=True)

    for name, ekw in (
        ("out_dim=128 (cfg)", {**cfg["model"]["encoder"], "dropout": 0.0}),
        ("out_dim=None      ", {"dropout": 0.0}),
    ):
        seed_everything(0)
        model = ETAModel("ours", dropout=0.0, encoder_kwargs=ekw).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["overfit"]["lr"]))
        for step in range(1, args.epochs + 1):
            loss = torch.nn.functional.mse_loss(model(L, V, B), Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % 50 == 0:
                print(f"[{name}] step {step:>3} loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
