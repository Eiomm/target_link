"""Train/eval the ETA downstream harness (spec §9) — Stage 1 variants.

Usage:
  python tools/train_eta.py --config configs/eta.yaml --variant ours --seed 0
  python tools/train_eta.py --config configs/eta.yaml --variant ours --overfit   # debug subset

Protocol (spec §17): split by link, shared backbone/optimizer/epochs, metrics
MAE/RMSE/MAPE in seconds on the never-seen test links, stratified by the
link-window trajectory count K (K=1 vs K>=2). Physics baseline y_hat = L/v_bar
is reported for scale. One metrics json per run under output.dir.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.data.eta_data import ETAData, build_eta_data  # noqa: E402
from target_link_v1.models.eta import ETAModel  # noqa: E402
from target_link_v1.utils import dump_json, load_config, mae, mape, rmse, seed_everything  # noqa: E402

ENC_KEYS = ("speeds", "valid", "lengths", "prof_group", "n_groups",
            "edge_group", "edge_sample", "n_samples")


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def to_device(batch: Dict[str, np.ndarray], device: str) -> Dict[str, object]:
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
            for k, v in batch.items()}


def predict(model: ETAModel, data: ETAData, rows: np.ndarray, device: str,
            bs: int) -> np.ndarray:
    """Model log-y predictions for sample rows (eval mode, no grad)."""
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            b = to_device(data.batch(r), device)
            pred = model(
                torch.from_numpy(data.L_n[r]).to(device),
                torch.from_numpy(data.v_n[r]).to(device),
                b if model.variant == "ours" else None,
            )
            out[i:i + bs] = pred.float().cpu().numpy()
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mae_s": mae(y_pred, y_true), "rmse_s": rmse(y_pred, y_true),
            "mape": mape(y_pred, y_true)}


def run(cfg: Dict, variant: str, seed: int, overfit: bool, device: str) -> Dict:
    seed_everything(seed)
    data = build_eta_data(cfg["data"])
    for a, b in ((0, 1), (0, 2), (1, 2)):  # no link may span two splits
        assert not (set(data.link_id[data.split == a]) & set(data.link_id[data.split == b]))
    tr_cfg = cfg["overfit"] if overfit else cfg["train"]

    if overfit:  # wiring sanity: memorise a small train subset, dropout off
        rng = np.random.default_rng(seed)
        tr = rng.permutation(data.rows_of(0))[: int(tr_cfg["n_samples"])]
        va = tr
    else:
        tr, va = data.rows_of(0), data.rows_of(1)

    model = ETAModel(
        variant, hidden=cfg["model"]["hidden"], depth=cfg["model"]["depth"],
        dropout=0.0 if overfit else cfg["model"]["dropout"],
        encoder_kwargs=cfg["model"]["encoder"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]),
                           weight_decay=float(cfg["train"]["weight_decay"]))
    bs = int(tr_cfg.get("batch_size", cfg["train"]["batch_size"]))
    epochs = int(tr_cfg["epochs"])

    best = {"val_mae": float("inf"), "state": None}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.default_rng(seed * 1000 + epoch).permutation(tr)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            b = to_device(data.batch(r), device)
            log_y = torch.from_numpy(data.log_y[r]).to(device)
            pred = model(
                torch.from_numpy(data.L_n[r]).to(device),
                torch.from_numpy(data.v_n[r]).to(device),
                b if model.variant == "ours" else None,
            )
            loss = torch.nn.functional.mse_loss(pred, log_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        if epoch == 1 or epoch % int(cfg["train"]["log_every"]) == 0 or epoch == epochs:
            val_mae = evaluate(data.y[va], np.exp(predict(model, data, va, device, bs)))["mae_s"]
            print(f"[{variant} s{seed}] epoch {epoch:>3} loss {tot / nb:.4f} "
                  f"{'train' if overfit else 'val'}_mae {val_mae:.3f}s")
            if not overfit and val_mae < best["val_mae"]:
                best = {"val_mae": val_mae,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if not overfit and best["state"] is not None:
        model.load_state_dict(best["state"])

    result: Dict = {
        "variant": variant, "seed": seed, "overfit": overfit, "params": n_params(model),
        "train_links": len(np.unique(data.link_id[data.split == 0])),
        "n_train": len(tr), "epochs": epochs, "minutes": round((time.time() - t0) / 60, 2),
    }
    if overfit:
        p = np.exp(predict(model, data, tr, device, bs))
        result["overfit_train"] = evaluate(data.y[tr], p)
        assert result["overfit_train"]["mae_s"] < 1.0, "overfit failed: wiring bug"
        print(f"[{variant} s{seed}] OVERFIT OK: train MAE "
              f"{result['overfit_train']['mae_s']:.3f}s on {len(tr)} samples")
        return result

    te = data.rows_of(2)
    p = np.exp(predict(model, data, te, device, bs))
    result["test"] = evaluate(data.y[te], p)
    k_strat = data.n_trajs_lw[te]
    for name, mask in (("k1", k_strat == 1), ("k_ge2", k_strat >= 2)):
        if mask.any():
            result[f"test_{name}"] = evaluate(data.y[te][mask], p[mask])
    y_phys = data.L_m[te] / np.maximum(data.v_bar[te], 0.3)  # zero-learning ETA
    result["physics_L_over_v"] = evaluate(data.y[te], y_phys)
    result["best_val_mae"] = best["val_mae"]
    print(f"[{variant} s{seed}] test {result['test']} | physics "
          f"{result['physics_L_over_v']['mae_s']:.2f}s | K=1 "
          f"{result.get('test_k1', {}).get('mae_s', float('nan')):.2f}s")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eta.yaml")
    ap.add_argument("--variant", default="ours", choices=["speed", "speed-mlp", "ours"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = run(cfg, args.variant, args.seed, args.overfit, device)
    tag = "overfit_" if args.overfit else ""
    dump_json(out, Path(cfg["output"]["dir"]) / f"{tag}{args.variant}_s{args.seed}" / "metrics.json")
