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
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
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


def _ragged_t(starts: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """torch twin of eta_data._ragged — device-side ragged range expansion."""
    total = int(sizes.sum())
    offs = torch.cumsum(sizes, 0) - sizes
    return (torch.arange(total, device=starts.device)
            - torch.repeat_interleave(offs, sizes)
            + torch.repeat_interleave(starts, sizes))


class GPUData:
    """GPU-resident copy of the hot-loop arrays (profile tables + the group
    CSR + the scalar columns). batch() is pure device index arithmetic — the
    CPU assembler's np.unique sort per batch, fancy-index copies and
    synchronous H2D transfers were what the GPU idled on.

    Numerically identical to ETAData.batch: both produce sorted unique group
    ids, so local group numbering and row order match. run() only enables it
    when the copy fits VRAM (cap-5M corpora are ~2GB); opt out with
    train.gpu_data: false.
    """

    def __init__(self, data: ETAData, device: str) -> None:
        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a).to(device)
        self.speeds, self.valid, self.lengths = t(data.speeds), t(data.valid), t(data.lengths)
        self.group_bounds = t(data.group_bounds)
        self.samp_ptr, self.samp_groups = t(data.samp_ptr), t(data.samp_groups)
        self.L_n, self.v_n, self.z_y = t(data.L_n), t(data.v_n), t(data.z_y)

    @staticmethod
    def nbytes(data: ETAData) -> int:
        return sum(a.nbytes for a in (data.speeds, data.valid, data.lengths,
                                      data.group_bounds, data.samp_ptr, data.samp_groups,
                                      data.L_n, data.v_n, data.z_y))

    def batch(self, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        deg = self.samp_ptr[rows + 1] - self.samp_ptr[rows]
        edge_sample = torch.repeat_interleave(
            torch.arange(len(rows), device=rows.device), deg)
        g = self.samp_groups[_ragged_t(self.samp_ptr[rows], deg)]
        ug, edge_group = torch.unique(g, return_inverse=True)
        gsize = self.group_bounds[ug + 1] - self.group_bounds[ug]
        prof_rows = _ragged_t(self.group_bounds[ug], gsize)
        return {
            "speeds": self.speeds.index_select(0, prof_rows),
            "valid": self.valid.index_select(0, prof_rows),
            "lengths": self.lengths.index_select(0, prof_rows),
            "prof_group": torch.repeat_interleave(
                torch.arange(len(ug), device=g.device), gsize),
            "n_groups": int(len(ug)),
            "edge_group": edge_group,
            "edge_sample": edge_sample,
            "n_samples": int(len(rows)),
        }


def predict(model: ETAModel, data: ETAData, rows: np.ndarray, device: str,
            bs: int, gd: GPUData | None = None) -> np.ndarray:
    """Model predictions in SECONDS for sample rows (eval mode, no grad):
    the model outputs z-scored log-y, so invert z then exp."""
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            if gd is not None:
                rt = torch.from_numpy(r).to(device)
                b, Lb, Vb = gd.batch(rt), gd.L_n[rt], gd.v_n[rt]
            else:
                b = to_device(data.batch(r), device)
                Lb = torch.from_numpy(data.L_n[r]).to(device)
                Vb = torch.from_numpy(data.v_n[r]).to(device)
            pred = model(Lb, Vb, b if model.variant == "ours" else None)
            out[i:i + bs] = pred.float().cpu().numpy()
    return np.exp(out * data.y_sd + data.y_mu)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mae_s": mae(y_pred, y_true), "rmse_s": rmse(y_pred, y_true),
            "mape": mape(y_pred, y_true)}


def run(cfg: Dict, variant: str, seed: int, overfit: bool, device: str) -> Dict:
    seed_everything(seed)
    data = build_eta_data(cfg["data"])
    if data.split_mode == "link":  # link-split: no link may span two splits
        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert not (set(data.link_id[data.split == a]) & set(data.link_id[data.split == b]))
    # time-split: links intentionally repeat across days — the leakage check
    # does not apply; deployment realism is the point
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
        encoder_kwargs={**cfg["model"]["encoder"],
                        **({"dropout": 0.0} if overfit else {})},
        aggregation=cfg["model"].get("aggregation", "cls"),
        level2_kwargs={**cfg["model"].get("level2", {}),
                       **({"dropout": 0.0} if overfit else {})},
    ).to(device)
    base = model  # state_dict IO handle (torch.compile wraps in _orig_mod)
    if bool(tr_cfg.get("compile", False)):
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]),
                           weight_decay=0.0 if overfit else float(cfg["train"]["weight_decay"]),
                           fused=(device == "cuda"))
    # overfit memorisation: constant full-batch lr bounces around the minimum
    # (loss tail oscillated ±10% at 3000 epochs) — cosine decay settles it in
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tr_cfg["epochs"])) \
        if overfit else None
    bs = int(tr_cfg.get("batch_size", cfg["train"]["batch_size"]))
    epochs = int(tr_cfg["epochs"])

    best = {"val_mae": float("inf"), "state": None}
    # overfit = full-batch memorisation on a fixed subset: assemble the batch
    # ONCE — per-step CPU assembly + H2D copies were the wall-clock bottleneck
    # (one core saturated launching tiny kernels; batch order is loss-invariant)
    ob = (to_device(data.batch(tr), device),
          torch.from_numpy(data.L_n[tr]).to(device),
          torch.from_numpy(data.v_n[tr]).to(device),
          torch.from_numpy(data.z_y[tr]).to(device)) if overfit else None
    t0 = time.time()
    # GPU-resident assembler: everything the hot loop touches sits in VRAM;
    # enabled when it fits (0.6x total as headroom for params+activations)
    gd: GPUData | None = None
    if device == "cuda" and cfg["train"].get("gpu_data", True):
        if GPUData.nbytes(data) <= torch.cuda.get_device_properties(0).total_memory * 0.6:
            try:
                gd = GPUData(data, device)
            except RuntimeError:  # OOM mid-copy: fall back to the CPU assembler
                torch.cuda.empty_cache()
                gd = None

    amp = bool(tr_cfg.get("amp", False)) and device == "cuda"
    actx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    # large corpora: cap rows per epoch (aggregation features v̄/r still come
    # from group-complete batches — subsampling rows never corrupts groups)
    max_rows = int(cfg["train"].get("max_rows_per_epoch", 0) or 0)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.default_rng(seed * 1000 + epoch).permutation(tr)
        if max_rows and len(perm) > max_rows:
            perm = perm[:max_rows]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            if overfit:  # cached tensors (bs >= n_samples: single full batch)
                b, Lb, Vb, Yb = ob
            elif gd is not None:
                rt = torch.from_numpy(r).to(device)
                b, Lb, Vb, Yb = gd.batch(rt), gd.L_n[rt], gd.v_n[rt], gd.z_y[rt]
            else:
                b = to_device(data.batch(r), device)
                Lb = torch.from_numpy(data.L_n[r]).to(device)
                Vb = torch.from_numpy(data.v_n[r]).to(device)
                Yb = torch.from_numpy(data.z_y[r]).to(device)
            with actx():
                pred = model(Lb, Vb, b if model.variant == "ours" else None)
                loss = torch.nn.functional.mse_loss(pred, Yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot += loss.detach()  # stays on device — one sync per epoch, not per step
            nb += 1
        tot = float(tot)
        if epoch == 1 or epoch % int(cfg["train"]["log_every"]) == 0 or epoch == epochs:
            val_mae = evaluate(data.y[va], predict(model, data, va, device, bs, gd))["mae_s"]
            print(f"[{variant} s{seed}] epoch {epoch:>3} loss {tot / nb:.4f} "
                  f"{'train' if overfit else 'val'}_mae {val_mae:.3f}s")
            if not overfit and val_mae < best["val_mae"]:
                best = {"val_mae": val_mae,
                        "state": {k: v.detach().clone() for k, v in base.state_dict().items()}}
    if not overfit and best["state"] is not None:
        base.load_state_dict(best["state"])

    result: Dict = {
        "variant": variant, "seed": seed, "overfit": overfit, "params": n_params(model),
        "aggregation": model.aggregation,
        "train_links": len(np.unique(data.link_id[data.split == 0])),
        "n_train": len(tr), "epochs": epochs, "minutes": round((time.time() - t0) / 60, 2),
    }
    if overfit:
        p = predict(model, data, tr, device, bs, gd)
        result["overfit_train"] = evaluate(data.y[tr], p)
        # samples sharing a (link, window) share r_group and cannot be fit
        # individually — the memorisation check lives on the K=1 subset
        solo = tr[data.n_trajs_lw[tr] == 1]
        result["overfit_train_k1"] = evaluate(
            data.y[solo], predict(model, data, solo, device, bs, gd))
        # physics oracle: y≡td (td = Σ10·ratio/v_i) reconstructs travel time
        # from the profile directly. Measured 1.173s MAE on the seed-0 subset
        # (median 0.21s; 17 long-tail samples y>60s carry ~23s each) — the old
        # fixed 1.0s bar sat BELOW this floor and was unpassable without
        # memorising per-sample residuals. Gate = reach physics reconstruction
        # (1.5×oracle); a model relying only on (L, v̄) lands ~1.65× (speed:
        # 1.94s), a mean-prediction plateau sits 3s+ — both fail.
        oracle_k1 = mae(data.td_s[solo], data.y[solo])
        result["oracle_k1"] = oracle_k1
        print(f"[{variant} s{seed}] OVERFIT {'OK' if variant == 'ours' else 'info'}: "
              f"train MAE {result['overfit_train']['mae_s']:.3f}s | K=1 subset "
              f"({len(solo)}) {result['overfit_train_k1']['mae_s']:.3f}s "
              f"(oracle {oracle_k1:.3f}s)")
        if variant == "ours":
            assert result["overfit_train_k1"]["mae_s"] < 1.5 * oracle_k1, \
                "overfit failed: wiring bug"
        return result

    te = data.rows_of(2)
    p = predict(model, data, te, device, bs, gd)
    result["test"] = evaluate(data.y[te], p)
    k_strat = data.n_trajs_lw[te]
    for name, mask in (("k1", k_strat == 1), ("k_ge2", k_strat >= 2)):
        if mask.any():
            result[f"test_{name}"] = evaluate(data.y[te][mask], p[mask])
    y_phys = data.L_m[te] / np.maximum(data.v_bar[te], 0.3)  # zero-learning ETA
    result["physics_L_over_v"] = evaluate(data.y[te], y_phys)
    result["best_val_mae"] = best["val_mae"]

    # per-sample predictions for eyeballing: deterministic subsample of test
    n_dump = min(200_000, len(te))
    rows_dump = np.random.default_rng(seed).choice(te, size=n_dump, replace=False)
    run_name = f"{variant}-{model.aggregation}" if variant == "ours" else variant
    dump_dir = Path(cfg["output"]["dir"]) / f"{run_name}_s{seed}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": data.sample_id[rows_dump], "link_id": data.link_id[rows_dump],
        "window_id": data.window_id[rows_dump], "K": data.n_trajs_lw[rows_dump],
        "L_m": data.L_m[rows_dump], "v_bar": data.v_bar[rows_dump],
        "y_true": data.y[rows_dump], "y_pred": p[np.searchsorted(te, rows_dump)],
    }).assign(err_s=lambda df: (df.y_pred - df.y_true).abs()).to_parquet(
        dump_dir / "test_samples.parquet", index=False)
    print(f"[{variant} s{seed}] test {result['test']} | physics "
          f"{result['physics_L_over_v']['mae_s']:.2f}s | K=1 "
          f"{result.get('test_k1', {}).get('mae_s', float('nan')):.2f}s")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eta.yaml")
    ap.add_argument("--variant", default="ours", choices=["speed", "speed-mlp", "ours"])
    ap.add_argument("--seeds", default="0", help="comma-separated seed list, e.g. 0,1,2")
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["output"]["dir"])
    tag = "overfit_" if args.overfit else ""
    # "ours" exists in two aggregation arms (v2 h_CLS vs scatter_mean baseline):
    # tag the run dir so one arm never overwrites the other's metrics
    name = (f"{args.variant}-{cfg['model'].get('aggregation', 'cls')}"
            if args.variant == "ours" else args.variant)

    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        try:  # one seed hitting the overfit gate (bistability, md/9.4progress
            # §1.4) must not abort the remaining seeds of the queue
            out = run(cfg, args.variant, seed, args.overfit, device)
        except AssertionError as e:
            print(f"[{name} s{seed}] FAILED gate: {e}")
            continue
        dump_json(out, out_dir / f"{tag}{name}_s{seed}" / "metrics.json")
        runs.append(out)

    if len(runs) > 1:  # spec §17: report mean ± std across seeds
        keys = ("test", "test_k1", "test_k_ge2")
        summary: Dict = {"variant": args.variant, "aggregation": runs[0].get("aggregation"),
                         "seeds": [r["seed"] for r in runs]}
        for key in keys:
            if key not in runs[0]:
                continue
            summary[key] = {
                m: {"mean": round(float(np.mean([r[key][m] for r in runs])), 3),
                    "std": round(float(np.std([r[key][m] for r in runs])), 3)}
                for m in runs[0][key]
            }
        dump_json(summary, out_dir / f"{tag}{name}_summary.json")
        print(f"[{name}] summary: {summary.get('test')}")
