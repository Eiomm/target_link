"""Train the stage-1 self-supervised CurveMAE (repV2 §4.1).

Usage:
  python tools/train_pretrain.py --config legacy/configs/pretrain.yaml --overfit  # wiring
  python tools/train_pretrain.py --config legacy/configs/pretrain.yaml --seeds 0  # real run

Monitoring per log interval (repV2 §4.1, all three must move):
  - val L_rec (fresh masks) + train loss;
  - attr head accuracies (random floors: 1/16 buckets, 1/24 hour);
  - CLS liveness: L_rec with use_cls=False (r zeroed) — its RATIO over val L_rec
    is the information share flowing through the CLS token. Overfit gates:
    rec < 0.5*cls0 (the CLS carries real curve information, no mask-query
    shortcut) and y-bucket acc > 0.25 (free labels are learnable at all).

Checkpoint {"model", "epoch", "encoder_kwargs", ...} lands in output.dir:
stage-2 (train_eta --init-from) loads the model.encoder.* and model.cls_token
entries, which map 1:1 onto TrajectoryEncoder submodules + curve_representation
(repV2 §5.4 reuse contract). Overfit masks are REGENERATED every epoch — a
fixed mask would let the decoder memorise positions instead of routes.
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
from target_link_v1.models.pretrain import CurveMAE, span_mask  # noqa: E402
from target_link_v1.utils import dump_json, load_config, seed_everything  # noqa: E402

ATTR_KEYS = ("y", "v", "len", "hour")


def rec_loss(out: Dict, tgt: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(out["rec"].masked_select(out["mask"]), tgt)


def attr_loss(out: Dict, labels: Dict) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy
    return sum(ce(out["attr"][k], labels[k]) for k in ATTR_KEYS) / len(ATTR_KEYS)


def run(cfg: Dict, seed: int, overfit: bool, device: str) -> Dict:
    seed_everything(seed)
    tcfg = cfg["overfit"] if overfit else cfg["train"]
    mcfg = cfg["train"]
    # data.mode=stream: consume build_curves_spark shards lazily (no npz); the
    # overfit wiring check stays on the npz path (needs a fixed subset)
    stream = str(cfg["data"].get("mode", "npz")) == "stream" and not overfit
    tr_idx = va_idx = None
    n = 0
    if stream:
        import glob as globmod
        import pyarrow.parquet as pq
        from target_link_v1.data.pretrain_stream import (CurveShardDataset,
                                                         stream_batch_to_tensors,
                                                         time_cutoff)
        shards_dir = cfg["data"]["shards"]
        files = sorted(globmod.glob(f"{shards_dir}/curves/part-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no curve shards under {shards_dir}/curves")
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        sm = cfg["data"].get("split", {}) or {}
        frac = float(sm.get("val_fraction", mcfg.get("val_fraction", 0.02)))
        w_cut, n_train_rows, _ = ((None, 0, 0) if sm.get("mode", "time") != "time"
                                  else time_cutoff(files, frac))
        if w_cut is not None:
            # past->future: val = the latest windows. A sample's curves share its
            # single window_id, so the cut keeps every sample on one side — no
            # train/val sample leakage (shards are round-robin and cannot).
            print(f"[pretrain s{seed}] time-split: val = windows >= {w_cut} "
                  f"({n - n_train_rows:,}/{n:,} rows)")
            train_files = val_files = files
            train_kw, val_kw = dict(window_cut=w_cut, side="train"), dict(window_cut=w_cut, side="val")
        else:  # smoke corpora whose windows cannot honour the fraction
            print(f"[pretrain s{seed}] shard-split fallback (time split degenerate "
                  f"on this corpus): WARNING — val metrics may be optimistic "
                  f"(round-robin shards can leak samples across the split)")
            n_val_f = max(1, int(round(frac * len(files))))
            val_files, train_files = files[-n_val_f:], files[:-n_val_f]
            n_train_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)
            train_kw = val_kw = {}

        def stream_batches(side_kw, files_, seed_, cap=0):
            """Yield (s, v, l, m, y) on device from a shard list."""
            dl = torch.utils.data.DataLoader(
                CurveShardDataset(shards_dir, int(mcfg["batch_size"]),
                                  seed=seed_, shard_list=files_, **side_kw),
                batch_size=None, num_workers=0,
                collate_fn=lambda x: x)  # keep numpy dicts (default_convert would tensorise)
            seen = 0
            for batch in dl:
                m = span_mask(torch.from_numpy(batch["valid"]),
                              torch.from_numpy(batch["lengths"]),
                              int(mcfg["mask_span"]), float(mcfg["mask_ratio"])).to(device)
                s, v, l, y = stream_batch_to_tensors(batch, device)
                yield s, v, l, m, y
                seen += len(batch["lengths"])
                if cap and seen >= cap:
                    break
    else:
        z = np.load(cfg["data"]["corpus"])
        n = len(z["lengths"])
        speeds = torch.from_numpy(z["speeds"])
        valid = torch.from_numpy(z["valid"])
        lengths = torch.from_numpy(z["lengths"].astype(np.int64))
        labels = {k: torch.from_numpy(z[f"attr_{k}"].astype(np.int64)) for k in ATTR_KEYS}

        # hold out at SAMPLE level: no curve of a held-out sample is ever trained on
        rng = np.random.default_rng(seed)
        sid = z["sample_id"]
        if overfit:
            tr_idx = torch.from_numpy(rng.permutation(n)[: int(tcfg["n_rows"])])
            va_idx = tr_idx
        else:
            uniq = np.unique(sid)
            held = uniq[rng.random(len(uniq)) < float(mcfg["val_fraction"])]
            va_idx = torch.from_numpy(np.flatnonzero(np.isin(sid, held)))
            tr_idx = torch.from_numpy(np.flatnonzero(~np.isin(sid, held)))

    model = CurveMAE(
        encoder_kwargs={**cfg["model"]["encoder"],
                        **({"dropout": 0.0} if overfit else {})},
        n_buckets=int(cfg["model"]["n_buckets"]),
        dec_layers=int(cfg["model"]["dec_layers"]),
    ).to(device)
    base = model  # checkpoint IO handle (torch.compile wraps in _orig_mod)
    if bool(tcfg.get("compile", False)):
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]),
                           weight_decay=0.0 if overfit else float(mcfg["weight_decay"]),
                           fused=(device == "cuda"))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tcfg["epochs"]))
             if overfit else None)
    bs = int(tcfg.get("batch_size", mcfg["batch_size"]))
    epochs = int(tcfg["epochs"])
    lam = float(mcfg["lambda_attr"])

    def batch(rows: torch.Tensor):
        # mask generation on CPU: span_mask's per-row python loop is dominated
        # by kernel-launch overhead on device tensors
        sv, vv, lv = speeds[rows], valid[rows], lengths[rows]
        m = span_mask(vv, lv, int(mcfg["mask_span"]), float(mcfg["mask_ratio"])).to(device)
        s, v, l = sv.to(device), vv.to(device), lv.to(device)
        y = {k: labels[k][rows].to(device) for k in ATTR_KEYS}
        return s, v, l, m, y

    def _metrics(batches) -> Dict[str, float]:
        """Common eval over (s, v, l, m, y) batches — shared by npz and stream."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # return train-phase blocks before eval
        model.eval()
        rec = cls0 = 0.0
        acc = {k: [0, 0] for k in ATTR_KEYS}
        nb = 0
        with torch.no_grad():
            for s, v, l, m, y in batches:
                tgt = model.rec_targets(s, m)
                out = model(s, v, l, m)
                rec += float(rec_loss(out, tgt))
                cls0 += float(rec_loss(model(s, v, l, m, use_cls=False), tgt))
                for k in ATTR_KEYS:
                    p = out["attr"][k].argmax(dim=1)
                    acc[k][0] += int((p == y[k]).sum())
                    acc[k][1] += len(y[k])
                nb += 1
        model.train()
        r, c = rec / max(nb, 1), cls0 / max(nb, 1)
        return {"val_rec": round(r, 5), "val_rec_cls0": round(c, 5),
                "cls_info_ratio": round(c / max(r, 1e-9), 2),
                **{f"acc_{k}": round(a / max(t, 1), 3) for k, (a, t) in acc.items()}}

    def evaluate(rows: torch.Tensor) -> Dict[str, float]:
        return _metrics(batch(rows[i:i + bs]) for i in range(0, len(rows), bs))

    cached = batch(tr_idx) if overfit else None  # full-batch H2D once (train_eta)
    max_rows = int(mcfg.get("max_rows_per_epoch", 0) or 0)
    amp = bool(tcfg.get("amp", False)) and device == "cuda"
    from contextlib import nullcontext
    actx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tot, nb = 0.0, 0

        def step(s, v, l, m, y) -> torch.Tensor:
            with actx():
                out = model(s, v, l, m)
                loss = rec_loss(out, model.rec_targets(s, m)) + lam * attr_loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            return loss.detach()  # stays on device; sync once per epoch below

        if overfit:
            s, v, l, _, y = cached                       # mask regenerated below
            m = span_mask(v.cpu(), l.cpu(), int(mcfg["mask_span"]),
                          float(mcfg["mask_ratio"])).to(device)
            tot, nb = step(s, v, l, m, y), 1
        elif stream:
            for s, v, l, m, y in stream_batches(train_kw, train_files,
                                                seed * 1000 + epoch, max_rows):
                tot += step(s, v, l, m, y)
                nb += 1
        else:
            perm = torch.randperm(len(tr_idx))
            if max_rows and len(perm) > max_rows:
                perm = perm[:max_rows]
            for i in range(0, len(perm), bs):
                tot += step(*batch(tr_idx[perm[i:i + bs]]))
                nb += 1
        tot = float(tot)

        log_every = int(mcfg["log_every"]) if not overfit else max(1, epochs // 20)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            ev = _metrics(stream_batches(val_kw, val_files, seed)) if stream \
                else evaluate(va_idx)
            print(f"[pretrain s{seed}] epoch {epoch:>4} loss {tot / max(nb, 1):.4f} "
                  f"val_rec {ev['val_rec']:.4f} cls0 {ev['val_rec_cls0']:.4f} "
                  f"(x{ev['cls_info_ratio']}) acc y/v/len/hour "
                  f"{ev['acc_y']}/{ev['acc_v']}/{ev['acc_len']}/{ev['acc_hour']}")

    ev = _metrics(stream_batches(val_kw, val_files, seed)) if stream else evaluate(va_idx)
    result = {"seed": seed, "overfit": overfit, "data_mode": "stream" if stream else "npz",
              "n_corpus_rows": int(n),
              "n_train_rows": int(n_train_rows if stream else len(tr_idx)),
              "n_val_rows": int((n - n_train_rows) if stream else len(va_idx)),
              "epochs": epochs, "minutes": round((time.time() - t0) / 60, 2), **ev}
    name = ("overfit_" if overfit else "") + f"pretrain_s{seed}"
    run_dir = Path(cfg["output"]["dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base.state_dict(), "epoch": epochs, "seed": seed,
                "encoder_kwargs": cfg["model"]["encoder"],
                "n_buckets": int(cfg["model"]["n_buckets"])},
               run_dir / "pretrain.pt")
    dump_json(result, run_dir / "metrics.json")
    if overfit:
        assert ev["val_rec"] < 0.5 * ev["val_rec_cls0"], \
            f"cls carries too little (info ratio x{ev['cls_info_ratio']})"
        assert ev["acc_y"] > 0.25, "y-bucket acc near random — attr heads broken"
        print(f"[pretrain s{seed}] OVERFIT OK: rec {ev['val_rec']:.4f} "
              f"cls0 {ev['val_rec_cls0']:.4f} (x{ev['cls_info_ratio']}) "
              f"acc_y {ev['acc_y']}")
    return result


if __name__ == "__main__":
    torch.set_num_threads(8)  # pod default 64 thrashes small CPU ops (span_mask)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="legacy/configs/pretrain.yaml")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--corpus", default=None, help="override cfg data.corpus (job entrypoint)")
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.corpus:
        cfg["data"]["corpus"] = args.corpus
    # MHA fused fastpath OFF: in eval mode torch's _transformer_encoder_layer_fwd
    # materialises a mask expansion that scales with the EVAL SET SIZE (~[N,N]
    # fp32 — a [73k x 73k] = 20GiB request OOM'd the 45G pod GPU after one
    # 400k-row epoch; quadratic, so an A100-80G dies too on a 7-day corpus).
    # Measured: eval forward peak 1.095G -> 0.135G with it off. Train-mode
    # forwards never take this path, so training throughput is unchanged.
    torch.backends.mha.set_fastpath_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in [int(s) for s in args.seeds.split(",")]:
        try:  # one seed failing a gate must not abort the queue (train_eta §)
            run(cfg, seed, args.overfit, device)
        except AssertionError as e:
            print(f"[pretrain s{seed}] FAILED gate: {e}")
