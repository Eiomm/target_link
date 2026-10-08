"""V6 training, model evaluation, baseline evaluation and bounded CPU smoke.
"""
from __future__ import annotations

import csv
import functools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajectory_mae.data import CellDataset, collate_cells
from trajectory_mae.evaluation import Evaluator, reconstruction_loss
from trajectory_mae.model import TrajectoryMLPMAE
from trajectory_mae.tools.plot_loss import plot_loss, plot_steps

VAL_EPOCH = 1_000_000
from trajectory_mae.checkpoints import FORMAT, source_hashes, save_checkpoint, restore
from trajectory_mae.cli import parse_args
from trajectory_mae.manifests import file_manifest, json_write
from trajectory_mae.progress import StageETA, progress


def make_loader(a, train, epoch=0):
    roots = a.data if train else (a.val_data or a.data)
    days = a.train_days if train else a.val_days
    ds = CellDataset(roots=roots, days=days, m_max=a.m_max, seed=a.data_seed,
                     epoch=epoch if train else VAL_EPOCH,
                     max_groups=(a.max_groups or None),
                     groups_per_partition=(a.groups_per_partition or None),
                     freeze_selection=not train)
    return ds, DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                          collate_fn=functools.partial(collate_cells, m_max=a.m_max,
                                                       epoch=epoch if train else VAL_EPOCH))


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def collect_partition_stats(records, batch):
    for stats in batch.get("partition_stats", []):
        if stats is not None:
            records[(stats["day"], stats["bucket"])] = stats


def eval_split(model, a, output_dir=None, bootstrap=0):
    ds, loader = make_loader(a, train=False)
    eta = StageETA(ds.n_partitions(), a.max_groups or None, a.val_group_total)
    evaluator = Evaluator(output_dir=output_dir, include_baseline=model is None)
    # Model validation has no mean baseline or paired comparison to bootstrap.
    bootstrap = bootstrap if model is None else 0
    started = time.monotonic()
    partitions = {}
    stage = "BASELINE" if model is None else "VALIDATION"
    steps, groups = 0, 0
    progress(stage, status="starting; loading first partition")
    if model is not None:
        model.eval()
    with torch.no_grad(), tqdm(total=a.val_group_total,
                              desc='均值基线' if model is None else '验证', unit='组',
                              mininterval=5, dynamic_ncols=True) as bar:
        for batch in loader:
            collect_partition_stats(partitions, batch)
            # Keep the loader's CPU tensors for metrics. Only model inference
            # needs GPU inputs; the baseline never makes a GPU round trip.
            prediction = None if model is None else model(to_device(batch, a.device))["prediction_seconds"]
            evaluator.update(prediction, batch)
            eta.update(batch)
            steps += 1
            groups += batch["x"].shape[0]
            bar.update(batch['x'].shape[0])
            if steps == 1 or (a.log_every and steps % a.log_every == 0):
                progress(stage, step=steps, groups=groups, partitions_seen=len(partitions),
                         **eta.fields(time.monotonic()-started))
    a.val_group_total = groups
    progress(stage, status="aggregating metrics", bootstrap=bootstrap, eta="unknown_for_aggregation")
    metrics = evaluator.finalize(bootstrap=bootstrap, seed=20260921)
    if not metrics["bins"]:
        raise ValueError("No supervised validation bins; cannot select a checkpoint")
    metrics["seconds_including_loading"] = time.monotonic() - started
    metrics["grouping_qc"] = list(partitions.values())
    progress(stage, status="complete", bins=metrics["bins"], elapsed_s=round(time.monotonic()-started, 1))
    return metrics


def run_smoke(a, model, optimizer, kwargs, manifest):
    """Fixed REAL training subset; deliberately has no fake validation split."""
    loading_started = time.monotonic()
    ds = CellDataset(roots=a.data, days=a.train_days, m_max=a.m_max, seed=a.data_seed,
                     epoch=0, max_groups=a.smoke_groups, freeze_selection=True)
    items = list(ds)
    loading_seconds = time.monotonic() - loading_started
    if not items:
        raise ValueError("No usable training groups")
    json_write(a.out / "smoke_group_manifest.json", [dict(group_id=x["group_id"],
                cell_id=int(x["cell_id"]), sample_ids=x["sample_ids"],
                K=x["K"], K_raw=x["K_raw"], group_size=x["group_size"],
                dropped_no_valid=x["dropped_no_valid"], dropped_tail=x["dropped_tail"]) for x in items])
    losses = []
    step, epoch = 0, 0
    started = time.monotonic()
    model.train()
    while step < a.smoke_steps:
        order = np.random.default_rng([a.data_seed, epoch]).permutation(len(items))
        for start in range(0, len(order), a.batch_size):
            batch = collate_cells([items[i] for i in order[start:start + a.batch_size]], m_max=a.m_max, epoch=epoch)
            batch = to_device(batch, a.device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(batch)["prediction_seconds"], batch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if step == 1 or step % 5 == 0:
                print(json.dumps(dict(mode="smoke", step=step, loss_seconds=losses[-1], grad_norm=float(norm))), flush=True)
            if step >= a.smoke_steps:
                break
        epoch += 1
    save_checkpoint(a.out / "last.pt", model, optimizer, kwargs, a, epoch, manifest, None)
    reloaded, _ = restore(a.out / "last.pt", a.device)
    model.eval(); reloaded.eval()
    probe = to_device(collate_cells(items[:a.batch_size], m_max=a.m_max, epoch=VAL_EPOCH), a.device)
    with torch.no_grad():
        expected = model(probe)
        actual = reloaded(probe)
        reload_diff = float((expected["prediction_seconds"] - actual["prediction_seconds"]).abs().max())
        changed = {k: v.clone() if torch.is_tensor(v) else v for k, v in probe.items()}
        hidden = changed["mae_mask"]
        changed["x"][..., 0][hidden] = float("nan")
        changed["bin_valid"][hidden] = ~changed["bin_valid"][hidden]
        counterfactual = model(changed)
        leakage_diff = float((expected["prediction_seconds"] - counterfactual["prediction_seconds"]).abs().max())
        representation_diff = float((expected["representation"] - counterfactual["representation"]).abs().max())
    if reload_diff != 0 or leakage_diff != 0 or representation_diff != 0:
        raise AssertionError("Checkpoint or hidden-content isolation check failed")
    evaluator = Evaluator(output_dir=a.out / "training_subset_diagnostic", include_baseline=False)
    with torch.no_grad():
        for start in range(0, len(items), a.batch_size):
            b = to_device(collate_cells(items[start:start + a.batch_size], m_max=a.m_max, epoch=VAL_EPOCH), a.device)
            evaluator.update(model(b)["prediction_seconds"], b)
    diagnostic = evaluator.finalize(bootstrap=0)
    summary = dict(kind="training-only smoke; NOT validation or evidence of generalization", groups=len(items),
                   steps=step, losses_seconds=losses, seconds=time.monotonic() - started,
                   data_loading_seconds=loading_seconds,
                   grouping_qc=list({(x["day"], x["bucket"]): x["partition_stats"] for x in items if x.get("partition_stats")}.values()),
                   checkpoint_max_abs_diff=reload_diff, hidden_input_prediction_max_abs_diff=leakage_diff,
                   hidden_input_representation_max_abs_diff=representation_diff, training_subset_diagnostic=diagnostic)
    json_write(a.out / "smoke.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["losses_seconds", "training_subset_diagnostic"]}), flush=True)


def main(argv=None):
    a = parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    a.source_sha256 = source_hashes()
    progress("PREFLIGHT", mode=a.mode, device=a.device, batch_groups=a.batch_size)
    # Preflight before creating outputs or starting expensive work.
    manifest = {}
    if a.mode in ("train", "smoke"):
        manifest["train"] = file_manifest(a.data, a.train_days)
    if a.mode in ("train", "baseline", "evaluate"):
        manifest["validation"] = file_manifest(a.val_data or a.data, a.val_days)
    if a.expected_buckets:
        expected = {f"bucket={i}" for i in range(a.expected_buckets)}
        for split in manifest.values():
            for day in split["days"]:
                actual = {f["bucket"] for f in split["files"] if f["day"] == day}
                if actual != expected:
                    raise ValueError(f"Incomplete partition set for {day}: {len(actual)} buckets")
    if a.dry_run:
        print("[DRY RUN] Paths checked; no model or GPU allocation.", flush=True)
        return
    a.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(format=FORMAT, validation_mask_epoch=VAL_EPOCH, label_provenance="existing upstream piece-time labels",
                  source_sha256=a.source_sha256,
                  created_at=datetime.now(timezone.utc).isoformat(), no_sealed_test=True)
    json_write(a.out / "config.json", config)
    json_write(a.out / "data_manifest.json", manifest)
    if a.mode == "baseline":
        metrics = eval_split(None, a, a.out / "predictions", a.bootstrap)
        json_write(a.out / "metrics.json", metrics)
        return
    if a.mode == "evaluate":
        model, saved = restore(a.checkpoint, a.device)
        config.update(format=saved["format"], input_channels=saved["model_kwargs"]["input_channels"])
        json_write(a.out / "config.json", config)
        for key in ("m_max", "data_seed", "val_days"):
            if saved["config"][key] != vars(a)[key]:
                raise ValueError(f"Evaluation {key} differs from checkpoint protocol")
        metrics = eval_split(model, a, a.out / "predictions", a.bootstrap)
        json_write(a.out / "metrics.json", metrics)
        return
    kwargs = dict(d_model=a.d_model, heads=a.heads, layers=a.layers, dropout=a.dropout,
                  n_bins=50, input_channels=a.input_channels,
                  time_encoding=a.time_encoding, decoder_layers=a.decoder_layers)
    model = TrajectoryMLPMAE(**kwargs).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    print(f"模型参数：{sum(p.numel() for p in model.parameters()):,} / 上限10,000,000；"
          f"设备：{a.device}；时间编码：{a.time_encoding}；"
          f"Encoder {a.layers}层 / Decoder {a.decoder_layers}层 / {a.d_model}维。", flush=True)
    if a.mode == "smoke":
        run_smoke(a, model, optimizer, kwargs, manifest)
        return
    # Start training immediately; the first validation pass anchors its identity.
    validation_mask_id = None
    best, best_epoch = float("inf"), None
    previous_train_groups = None
    global_step = 0
    training_started = time.monotonic()
    for epoch in range(a.epochs):
        ds, loader = make_loader(a, train=True, epoch=epoch)
        eta = StageETA(ds.n_partitions(), a.max_groups or None, previous_train_groups)
        model.train()
        progress("TRAIN", epoch=f"{epoch+1}/{a.epochs}", status="loading first partition")
        steps, bins, weighted_loss, groups = 0, 0, 0., 0
        partitions = {}
        started = time.monotonic()
        grad_sum, grad_max = 0., 0.
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(a.device))
        bar = tqdm(total=previous_train_groups, desc=f'训练 {epoch+1}/{a.epochs}',
                   unit='组', mininterval=5, dynamic_ncols=True)
        for batch in loader:
            collect_partition_stats(partitions, batch)
            batch = to_device(batch, a.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)["prediction_seconds"]
            loss = reconstruction_loss(prediction, batch)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            grad_sum += grad_norm
            grad_max = max(grad_max, grad_norm)
            optimizer.step()
            n = int((batch["mae_mask"].unsqueeze(-1) & batch["bin_valid"] & batch["traj_valid"].unsqueeze(-1)).sum())
            bins += n; weighted_loss += float(loss.detach()) * n; steps += 1
            groups += batch["x"].shape[0]
            global_step += 1
            bar.update(batch['x'].shape[0])
            eta.update(batch)
            log_due = steps == 1 or (a.log_every and steps % a.log_every == 0)
            plot_due = a.plot_every and steps % a.plot_every == 0
            if log_due or plot_due:
                elapsed_now = time.monotonic() - started
                batch_loss = float(loss.detach())
                running_loss = weighted_loss / max(bins, 1)
                lr = optimizer.param_groups[0]['lr']
                gpu_peak = (torch.cuda.max_memory_allocated(torch.device(a.device))/2**30
                            if a.device.startswith('cuda') else 0.)
                sample = dict(global_step=global_step, epoch=epoch+1, step=steps,
                              batch_mae_seconds=batch_loss, running_mae_seconds=running_loss,
                              lr=lr, grad_norm=grad_norm, groups_per_second=groups/max(elapsed_now, 1e-9),
                              gpu_peak_GiB=gpu_peak)
                csv_path = a.out / 'step_metrics.csv'
                needs_header = not csv_path.exists()
                with csv_path.open('a', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(sample))
                    if needs_header:
                        writer.writeheader()
                    writer.writerow(sample)
                bar.set_postfix({'批loss秒':f'{batch_loss:.4f}', '累计秒':f'{running_loss:.4f}',
                                 '梯度':f'{grad_norm:.3f}', 'lr':f'{lr:.2g}'}, refresh=False)
                progress("TRAIN", epoch=f"{epoch+1}/{a.epochs}", step=steps, groups=groups,
                         batch_mae_s=round(batch_loss, 6),
                         running_mae_s=round(running_loss, 6), grad_norm=round(grad_norm, 4), lr=lr,
                         groups_per_s=round(groups/max(elapsed_now, 1e-9), 2),
                         gpu_peak_GiB=round(torch.cuda.max_memory_allocated()/2**30, 2)
                         if a.device.startswith("cuda") else 0,
                         **eta.fields(elapsed_now))
                if plot_due or steps == 1:
                    plot_steps(a.out)
        bar.close()
        if not bins:
            raise ValueError("No supervised training bins")
        elapsed = time.monotonic() - started
        parameter_norm = float(torch.sqrt(sum(p.detach().float().square().sum() for p in model.parameters())))
        train_gpu_peak = (torch.cuda.max_memory_allocated(torch.device(a.device))/2**30
                          if a.device.startswith('cuda') else 0.)
        previous_train_groups = groups
        validation = eval_split(model, a)
        if validation_mask_id is None:
            validation_mask_id = validation["eval_mask_id"]
        elif validation["eval_mask_id"] != validation_mask_id:
            raise AssertionError("Validation groups, masks or targets changed across epochs")
        score = validation["ours"]["bin_mae_seconds"]
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        progress("CHECKPOINT", status="保存本轮结果", epoch=epoch+1, best_updated=improved)
        save_checkpoint(a.out / "last.pt", model, optimizer, kwargs, a, epoch, manifest, best)
        if improved:
            save_checkpoint(a.out / "best.pt", model, optimizer, kwargs, a, epoch, manifest, best)
        record = dict(epoch=epoch, train=dict(loss_mae_seconds=weighted_loss/bins, bins=bins, groups=groups,
                      steps=steps, seconds=elapsed, groups_per_second=groups/max(elapsed, 1e-9),
                      lr=optimizer.param_groups[0]['lr'], grad_norm_mean=grad_sum/steps,
                      grad_norm_max=grad_max, parameter_norm=parameter_norm, gpu_peak_GiB=train_gpu_peak,
                      grouping_qc=list(partitions.values())), val=validation,
                      best_epoch=best_epoch)
        with (a.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        curve = plot_loss(a.out)
        plot_steps(a.out)
        progress("LOSS CURVE", path=curve)
        progress("EPOCH COMPLETE", epoch=epoch+1, train_mae_s=weighted_loss/bins, val_mae_s=score, best_epoch=best_epoch+1, best_updated=improved)
    restored, _ = restore(a.out / "best.pt", a.device)
    final = eval_split(restored, a, a.out / "best_val_predictions", a.bootstrap)
    if final["eval_mask_id"] != validation_mask_id:
        raise AssertionError("Final evaluation set changed")
    final.update(best_epoch=best_epoch, elapsed_total_seconds=time.monotonic()-training_started,
                 gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if a.device.startswith("cuda") else None,
                 device_name=torch.cuda.get_device_name() if a.device.startswith("cuda") else "CPU",
                 interpretation="Validation selected checkpoint; not an independent test estimate")
    json_write(a.out / "best_val_metrics.json", final)
    progress("DONE", best_epoch=best_epoch+1, output=a.out)


if __name__ == "__main__":
    main()
