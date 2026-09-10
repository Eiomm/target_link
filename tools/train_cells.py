"""Train the cell-level whole-trajectory MAE on a LOCAL cell corpus (no HDFS client assumed).

Reads whatever `CellCorpusDataset` can see under --data (one or more
(day, bucket) partitions of observations_v2/ + training_groups_k3/), so the training
split is whatever the caller fetched. --val-data is a second corpus root, which
keeps the split explicit instead of hiding it in a time filter.

    python tools/train_cells.py --data runtime/cell_smoke/train \
        --val-data runtime/cell_smoke/val --out runtime/cell_train_smoke --epochs 1
"""
from __future__ import annotations

import argparse
from collections import deque
import functools
import json
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from target_link_v1.data.cell_corpus import CellCorpusDataset, collate_cells
from target_link_v1.models.cell_mae import (
    CellMAE,
    masked_reconstruction_loss,
    reconstruction_loss_by_group,
)

VAL_EPOCH = 10 ** 6   # validation masks are drawn from a fixed epoch: one eval set


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="local cell corpus root (train)")
    p.add_argument("--val-data", help="local cell corpus root (validation)")
    p.add_argument("--out", required=True)
    p.add_argument("--obs-dir", default="observations_v2",
                   help="observations_v2 is the build that carries bin_pos")
    p.add_argument("--groups-dir", default="training_groups_k3")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4, help="training groups per step")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0,
                   help="per split per epoch; 0 or -1 = full pass")
    p.add_argument("--max-groups", type=int, default=0,
                   help="dataset-level cap; 0 or -1 = all")
    p.add_argument("--m-max", type=int, default=16)
    p.add_argument("--probe-batches", type=int, default=2,
                   help="val batches for the aggregate-ablation probe; 0 = off")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--traj-layers", type=int, default=4)
    p.add_argument("--level2-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target-transform", choices=["raw", "log1p"], default="log1p",
                   help="T_diff is seconds per 10m bin; log1p compresses the curb-side tail")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if a.epochs <= 0 or a.batch_size <= 0:
        p.error("invalid training sizes")
    # -1 is the shell convention for "no cap"; the loop tests truthiness, so
    # normalise it to 0 here rather than rejecting it (0 means full pass).
    a.max_batches = max(a.max_batches, 0)
    a.max_groups = max(a.max_groups, 0)
    if a.workers < 0 or a.probe_batches < 0:
        p.error("workers and probe-batches must be nonnegative")
    return a


def make_loader(root, a, epoch, train):
    ds = CellCorpusDataset(root, obs_dir=a.obs_dir, groups_dir=a.groups_dir,
                           seed=a.seed, epoch=epoch, shuffle_groups=True,
                           max_groups=a.max_groups or None, m_max=a.m_max)
    # the mask seed must be the same epoch the dataset shuffles with, or a
    # re-run of one epoch would not reproduce its own batches
    collate = functools.partial(collate_cells, m_max=a.m_max, epoch=epoch)
    return ds, DataLoader(ds, batch_size=a.batch_size, collate_fn=collate,
                          num_workers=a.workers)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_split(model, optimizer, root, a, epoch, train):
    # validation keeps one fixed mask set so epoch-to-epoch losses are comparable
    ds, loader = make_loader(root, a, epoch if train else VAL_EPOCH, train)
    model.train(train)
    loss_sum, groups, supervised_groups, supervised_batches = 0.0, 0, 0, 0
    masked_bins, seen = 0, 0
    recent_losses = deque(maxlen=100)
    bucket_sums = {"k3": 0.0, "k4_16": 0.0, "k17_plus": 0.0}
    bucket_counts = {name: 0 for name in bucket_sums}
    started = time.monotonic()
    probe, probe_ablated, probe_n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = to_device(batch, a.device)
            out = model(batch)
            per_group_loss, group_has = reconstruction_loss_by_group(out, batch)
            group_weight = group_has.to(per_group_loss.dtype)
            loss = ((per_group_loss * group_weight).sum()
                    / group_weight.sum().clamp_min(1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            reconstruction_mask = batch["mae_mask"].unsqueeze(-1) & batch["bin_valid"]
            n_supervised = int(group_has.sum())
            has = n_supervised > 0
            k = batch["K"]
            bucket_masks = {
                "k3": k == 3,
                "k4_16": (k >= 4) & (k <= 16),
                "k17_plus": k >= 17,
            }
            for name, in_bucket in bucket_masks.items():
                selected = group_has & in_bucket
                count = int(selected.sum())
                if count:
                    bucket_sums[name] += float(per_group_loss[selected].sum().detach())
                    bucket_counts[name] += count
            if not train and probe_n < a.probe_batches and has:
                # Branch-death monitor: if the decoder ignores the level-2 state,
                # zeroing it costs nothing and the gap collapses towards 0.
                ablated = masked_reconstruction_loss(
                    model(batch, ablate_aggregate=True), batch)
                probe += float(loss.detach())
                probe_ablated += float(ablated.detach())
                probe_n += 1
            if train and has:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
            batch_loss = float(loss.detach())
            loss_sum += batch_loss * n_supervised
            supervised_groups += n_supervised
            supervised_batches += has
            if has:
                recent_losses.append(batch_loss)
            masked_bins += int(reconstruction_mask.sum())
            groups += batch["x"].shape[0]
            seen += 1
            if a.log_every and seen % a.log_every == 0:
                print(json.dumps({"split": "train" if train else "val", "epoch": epoch,
                                  "batch": seen, "batch_loss": batch_loss if has else None,
                                  "moving_avg_loss": (sum(recent_losses) / len(recent_losses)
                                                      if recent_losses else None),
                                  "epoch_avg_loss": (loss_sum / max(supervised_groups, 1))}),
                      flush=True)
            if a.max_batches and seen >= a.max_batches:
                break
    if not groups:
        raise ValueError("no groups in " + ("train" if train else "val") + " split")
    if not supervised_groups:
        raise ValueError("no group had a masked bin to reconstruct in "
                         + ("train" if train else "val") + " split")
    elapsed = time.monotonic() - started
    m = {"loss": loss_sum / supervised_groups, "groups": groups,
         "supervised_groups": supervised_groups,
         "supervised_batches": supervised_batches, "masked_bins": masked_bins,
         "batches": seen, "partitions": ds.n_partitions(),
         "seconds": elapsed, "batches_per_second": seen / max(elapsed, 1e-9)}
    m["loss_by_k"] = {
        name: {"loss": (bucket_sums[name] / bucket_counts[name]
                        if bucket_counts[name] else None),
               "groups": bucket_counts[name]}
        for name in bucket_sums
    }
    if probe_n:
        m["aggregate_ablated_loss"] = probe_ablated / probe_n
        m["aggregate_gap"] = (probe_ablated - probe) / probe_n
    return m


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    model_kwargs = dict(d_model=a.d_model, heads=a.heads, traj_layers=a.traj_layers,
                        level2_layers=a.level2_layers, n_bins=50, dropout=a.dropout,
                        target_transform=a.target_transform)
    model = CellMAE(**model_kwargs).to(a.device)
    n_param = sum(p.numel() for p in model.parameters())
    print(json.dumps({"device": a.device, "params": n_param,
                      "train": a.data, "val": a.val_data,
                      "model": model_kwargs}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    for epoch in range(a.epochs):
        metrics = {"epoch": epoch, "params": n_param}
        metrics["train"] = run_split(model, optimizer, a.data, a, epoch, True)
        if a.val_data:
            metrics["val"] = run_split(model, optimizer, a.val_data, a, epoch, False)
        if a.device.startswith("cuda"):
            metrics["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        print(json.dumps(metrics), flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "args": vars(a), "epoch": epoch, "model_kwargs": model_kwargs,
                    "format": "target_link_cell_mae_v1_group_bins"}, out / "last.pt")
    saved = torch.load(out / "last.pt", map_location=a.device, weights_only=False)
    restored = CellMAE(**saved["model_kwargs"]).to(a.device)
    restored.load_state_dict(saved["model"], strict=True)
    print(json.dumps({"checkpoint_reload_ok": True, "epoch": saved["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
