"""Train the v1 window MAE on local/NFS window shards (no HDFS client assumed)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from target_link_v1.data.window_stream import WindowDataset, collate_windows
from target_link_v1.models.window_mae import WindowMAE, reconstruction_mask, reconstruction_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="flat YAML config; explicit CLI flags override it")
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--train-end", type=int, help="exclusive UTC epoch seconds")
    p.add_argument("--val-start", type=int)
    p.add_argument("--val-end", type=int)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="whole road snapshots")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="smoke cap per split; 0=all")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--group-layers", type=int, default=2)
    p.add_argument("--time-features", choices=["none", "curve", "bin"], default="curve")
    p.add_argument("--age-bucket-seconds", type=int, default=60)
    p.add_argument("--max-curves-per-snapshot", type=int, default=512)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--whole-pass-probability", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        import yaml
        config = yaml.safe_load(Path(preliminary.config).read_text())
        if not isinstance(config, dict):
            p.error("config must be a flat mapping")
        known = {action.dest for action in p._actions} - {"help", "config"}
        if set(config) - known:
            p.error("unknown config keys: " + str(sorted(set(config) - known)))
        p.set_defaults(**config)
    a = p.parse_args()
    if any(getattr(a, k) is None for k in ["data", "out", "train_end", "val_start", "val_end"]):
        p.error("data, out, train-end, val-start, val-end are required")
    if a.epochs <= 0 or a.batch_size <= 0 or a.max_batches < 0 or a.workers < 0:
        p.error("invalid training sizes")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    ds_kwargs = dict(age_bucket_seconds=a.age_bucket_seconds,
                     max_curves_per_snapshot=a.max_curves_per_snapshot)
    ds = WindowDataset(a.data, **ds_kwargs)
    w = ds.meta["lookback_seconds"]
    if a.val_start < a.train_end + w or a.val_end <= a.val_start:
        p.error("require val-start >= train-end + lookback, and val-end > val-start")
    if not ds.meta["anchor_start"] < a.train_end <= ds.meta["anchor_end"]:
        p.error("train-end is outside corpus anchor range")
    if a.val_end > ds.meta["anchor_end"] or a.val_start < ds.meta["anchor_start"]:
        p.error("validation interval is outside corpus anchor range")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    model_kwargs = dict(d_model=a.d_model, heads=a.heads, layers=a.layers,
                        group_layers=a.group_layers, time_features=a.time_features)
    model = WindowMAE(**model_kwargs).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for epoch in range(a.epochs):
        metrics = {"epoch": epoch}
        for side in ["train", "val"]:
            train = side == "train"
            model.train(train)
            dataset = WindowDataset(a.data, start_ts=None if train else a.val_start,
                                    end_ts=a.train_end if train else a.val_end,
                                    seed=a.seed + epoch if train else a.seed, **ds_kwargs)
            loader = DataLoader(dataset, batch_size=a.batch_size, collate_fn=collate_windows,
                                num_workers=a.workers)
            total, num, batches, supervised, masked_bins = 0.0, 0, 0, 0, 0
            # Fixed validation masks, without resetting the training RNG stream.
            state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            if not train:
                torch.manual_seed(a.seed)
            with torch.set_grad_enabled(train):
                for b in loader:
                    b = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in b.items()}
                    mask = reconstruction_mask(b, a.mask_ratio, a.whole_pass_probability)
                    pred = model(b, mask)
                    loss = reconstruction_loss(pred, mask, b["curve_group"], b["n_groups"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    eligible = sum(bool(mask[b["curve_group"] == g].any()) for g in range(b["n_groups"]))
                    if train and eligible:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += float(loss.detach()) * eligible
                    supervised += eligible
                    masked_bins += int(mask.sum())
                    num += b["n_groups"]
                    batches += 1
                    if a.max_batches and batches >= a.max_batches:
                        break
            if not train:
                torch.random.set_rng_state(state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
            if not num:
                raise ValueError("No snapshots in " + side + "; inspect explicit split times")
            if not supervised:
                raise ValueError("No reconstructable snapshots in " + side + "; inspect available bin counts")
            metrics[side + "_loss"] = total / supervised
            metrics[side + "_snapshots"] = num
            metrics[side + "_supervised_snapshots"] = supervised
            metrics[side + "_masked_bins"] = masked_bins
        print(json.dumps(metrics), flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": vars(a), "data_meta": ds.meta, "epoch": epoch,
                    "model_kwargs": model_kwargs,
                    "format": "target_link_window_mae_v1"}, out / "last.pt")


if __name__ == "__main__":
    main()
