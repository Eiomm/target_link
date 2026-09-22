"""Standalone trajectory-MLP training, paired evaluation, and bounded CPU smoke.

Run as ``python experiments/trajectory_mlp_v1/run.py --help``. Original model,
reader, precomputed group files, and source observations are never modified.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells
from experiments.trajectory_mlp_v1.evaluation import Evaluator, reconstruction_loss
from experiments.trajectory_mlp_v1.model import TrajectoryMLPMAE
from experiments.trajectory_mlp_v1.tools.plot_loss import plot_loss, plot_steps

VAL_EPOCH = 1_000_000
FORMAT = "trajectory_mlp_mae_known_ratio_v4"


def duration(seconds):
    seconds = max(0, round(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


class StageETA:
    def __init__(self, partitions, max_groups=None, known_total=None):
        self.partitions = partitions
        self.cap = max_groups
        self.known_total = known_total
        self.consumed = Counter()
        self.sizes = {}
        self.groups = 0

    def update(self, batch):
        # Counts only groups consumed by the main loop, not prefetched groups.
        for day, bucket in zip(batch['day'], batch['bucket']):
            self.consumed[(day, bucket)] += 1
            self.groups += 1
        for stats in batch.get('partition_stats', []):
            if stats is not None:
                self.sizes[(stats['day'], stats['bucket'])] = stats['selected_groups']

    def fields(self, elapsed):
        done = sum(self.consumed[k] >= n for k,n in self.sizes.items())
        fields = dict(elapsed=duration(elapsed), partitions_completed=f'{done}/{self.partitions}')
        total = self.known_total
        basis = 'previous_full_pass' if total is not None else 'sampled_partition_sizes'
        if total is None:
            if len(self.sizes) < min(4, self.partitions) or not done:
                return dict(fields, eta='estimating', eta_basis='waiting_for_completed_partitions')
            total = sum(self.sizes.values()) / len(self.sizes) * self.partitions
        if self.cap:
            total = min(total, self.cap)
        if self.groups <= 0 or elapsed <= 0:
            return dict(fields, eta='estimating')
        if total < self.groups:
            return dict(fields, eta='re-estimating', eta_basis=basis)
        return dict(fields, eta=duration((total-self.groups)*elapsed/self.groups),
                    estimated_total_groups=round(total), eta_basis=basis)


def progress(stage, **fields):
    names = {"PREFLIGHT": "数据与配置检查", "BASELINE": "计算均值基线", "TRAIN": "模型训练",
             "VALIDATION": "验证模型", "EPOCH COMPLETE": "本轮训练完成", "LOSS CURVE": "保存收敛曲线",
             "DONE": "全部完成", "CHECKPOINT": "保存模型与指标"}
    statuses = {"starting; loading first partition": "开始，正在加载首个分桶",
                "loading first partition": "正在加载首个分桶",
                "aggregating metrics": "遍历结束，正在汇总指标", "complete": "阶段完成"}
    if "status" in fields:
        fields["status"] = statuses.get(fields["status"], fields["status"])
    if "status" in fields or stage in ("PREFLIGHT", "EPOCH COMPLETE", "DONE"):
        print("\n" + "=" * 20 + " 【" + names.get(stage, stage) + "】 " + "=" * 20, flush=True)
    stage = names.get(stage, stage)
    labels = dict(epoch="轮次", step="批次", groups="已处理group", bins="监督bin数",
                  batch_mae_s="本批MAE(秒)", running_mae_s="累计MAE(秒)",
                  groups_per_s="group/秒", gpu_peak_GiB="显存峰值GiB", elapsed="已用时间",
                  partitions_completed="完成分区", partitions_seen="已读取分区", eta="预计剩余",
                  eta_basis="估计依据", estimated_total_groups="预计group总数", status="状态",
                  train_mae_s="训练MAE(秒)", val_mae_s="验证MAE(秒)", best_epoch="最优轮次",
                  best_updated="更新最优模型", path="文件", output="输出目录", elapsed_s="耗时秒",
                  grad_norm="梯度范数(裁剪前)", lr="学习率")
    translations = dict(estimating="估计中", **{'re-estimating':"重新估计中"},
                        waiting_for_completed_partitions="等待完整分区", sampled_partition_sizes="按已读分区估算",
                        previous_full_pass="上一轮总量", unknown_for_aggregation="指标汇总阶段无法估计")
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    tqdm.write(f"[{stamp}] [{stage}] " + " | ".join(
        f"{labels.get(k,k)}={translations.get(v,v) if isinstance(v,str) else v}" for k, v in fields.items()), file=sys.stdout)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "baseline", "evaluate", "smoke"])
    p.add_argument("--data", nargs="+", required=True, help="corpus roots containing observations_v2; may combine old train and val roots")
    p.add_argument("--val-data", nargs="+", help="defaults to --data roots; validation is selected by --val-days")
    p.add_argument("--train-days", nargs="+", default=[f"202608{d}" for d in range(17, 23)])
    p.add_argument("--val-days", nargs="+", default=["20260823"])
    p.add_argument("--out", type=Path, required=True, help="new output directory; refuses to overwrite")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--m-max", type=int, default=64, help="cell chunk size; ablate e.g. 16,32,64; discard tails <3")
    p.add_argument("--input-channels", type=int, choices=[3], default=3, help="T_clean,ratio,valid; known ratio is retained when time is invalid")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--time-encoding", choices=["seconds", "bucket30"], default="seconds")
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42, help="model seed; does not change validation membership or masks")
    p.add_argument("--data-seed", type=int, default=20260921)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-groups", type=int, default=0, help="explicit diagnostic cap; default is full pass")
    p.add_argument("--groups-per-partition", type=int, default=0, help="explicit sampling cap; default is all groups")
    p.add_argument("--bootstrap", type=int, default=2000, help="paired cell bootstrap replicates on final evaluation")
    p.add_argument("--smoke-groups", type=int, default=128)
    p.add_argument("--smoke-steps", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--plot-every", type=int, default=1000, help="refresh sampled batch curves every N steps; 0 means epoch end only")
    a = p.parse_args(argv)
    if min(a.batch_size, a.epochs, a.threads, a.smoke_groups, a.smoke_steps) < 1 or a.m_max < 3:
        p.error("sizes must be positive and m-max >=3")
    if min(a.workers, a.max_groups, a.groups_per_partition, a.bootstrap) < 0:
        p.error("counts cannot be negative")
    if min(a.log_every, a.plot_every) < 0:
        p.error("log-every and plot-every must be nonnegative")
    if a.max_groups and a.workers:
        p.error("max-groups requires workers=0")
    if a.mode == "smoke" and a.workers:
        p.error("smoke requires workers=0 to fix the small dataset")
    if a.mode == "evaluate" and not a.checkpoint:
        p.error("evaluate requires --checkpoint")
    if set(a.train_days) & set(a.val_days):
        p.error("training and validation days must be disjoint")
    return a


def file_manifest(roots, days):
    """Metadata fingerprint, not a claim to have hashed all corpus contents."""
    entries, seen, available = [], set(), set()
    prepared_artifacts = []
    for root in roots:
        tensor_marker = Path(root) / '_TENSORS_SUCCESS.json'
        if (Path(root) / '_TENSORS_BUILDING').exists():
            raise ValueError(f'Tensor data not published: {root}')
        if tensor_marker.exists():
            ready = json.loads(tensor_marker.read_text())
            from experiments.trajectory_mlp_v1.tensor_corpus import manifest as tensor_manifest
            tensor_manifest(root, ready['m_max'], ready['data_seed'])
            prepared_artifacts.append(dict(path=str(tensor_marker.resolve()),
                sha256=hashlib.sha256(tensor_marker.read_bytes()).hexdigest(),
                format=ready['format'], m_max=ready['m_max'], data_seed=ready['data_seed']))
            for key, receipt in ready['partitions'].items():
                day, bucket = key.split('/')
                if day not in days:
                    continue
                key = (day, 'bucket=' + bucket)
                if key in seen:
                    raise ValueError(f'Duplicate observation partition {key}')
                seen.add(key)
                available.add(day)
                for rec in [*receipt['arrays'].values(), receipt['payload']]:
                    f = Path(root) / rec['path']
                    st = f.stat()
                    if st.st_size != rec['bytes']:
                        raise ValueError(f'Tensor artifact changed: {f}')
                    entries.append(dict(path=str(f.resolve()), day=day, bucket=key[1],
                                        bytes=st.st_size, mtime_ns=st.st_mtime_ns))
            continue
        marker = Path(root) / "_FINAL_SUCCESS.json"
        if (Path(root) / "_BUILDING").exists():
            raise ValueError(f"Prepared data not published: {root}")
        if marker.exists():
            ready = json.loads(marker.read_text())
            prepared_artifacts.append(dict(path=str(marker.resolve()), sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                                           format=ready["format"], m_max=ready["m_max"], data_seed=ready["data_seed"]))
        for day in days:
            for bucket in sorted((Path(root) / "observations_v2" / f"day={day}").glob("bucket=*")):
                key = (day, bucket.name)
                if key in seen:
                    raise ValueError(f"Duplicate observation partition {key}; do not pass overlapping corpus roots")
                files = sorted(bucket.glob("*.parquet"))
                if not files:
                    continue
                seen.add(key)
                available.add(day)
                for f in files:
                    st = f.stat()
                    entries.append(dict(path=str(f.resolve()), day=day, bucket=bucket.name,
                                        bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    missing = sorted(set(days) - available)
    if missing:
        raise ValueError(f"Requested days are missing: {missing}. Supply the actual corpus; no substitution is performed.")
    return dict(days=days, files=entries, metadata_sha256=digest(dict(files=entries, prepared_artifacts=prepared_artifacts)),
                prepared_artifacts=prepared_artifacts, partitions=len(seen), fingerprint_kind="path/size/mtime metadata, not parquet content hash")


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
    eta = StageETA(ds.n_partitions(), a.max_groups or None, getattr(a, "_val_group_total", None))
    evaluator = Evaluator(output_dir=output_dir)
    started = time.monotonic()
    partitions = {}
    stage = "BASELINE" if model is None else "VALIDATION"
    steps, groups = 0, 0
    progress(stage, status="starting; loading first partition")
    if model is not None:
        model.eval()
    with torch.no_grad(), tqdm(total=getattr(a, '_val_group_total', None),
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
    a._val_group_total = groups
    progress(stage, status="aggregating metrics", bootstrap=bootstrap, eta="unknown_for_aggregation")
    metrics = evaluator.finalize(bootstrap=bootstrap, seed=20260921)
    if not metrics["bins"]:
        raise ValueError("No supervised validation bins; cannot select a checkpoint")
    metrics["seconds_including_loading"] = time.monotonic() - started
    metrics["grouping_qc"] = list(partitions.values())
    progress(stage, status="complete", bins=metrics["bins"], elapsed_s=round(time.monotonic()-started, 1))
    return metrics


def source_hashes():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ["run.py", "data.py", "model.py", "evaluation.py", "prepared.py", "tensor_corpus.py", "tools/plot_loss.py"]}


def save_checkpoint(path, model, optimizer, kwargs, a, epoch, manifest, best_score):
    torch.save(dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    model_kwargs=kwargs, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    epoch=epoch, data_manifest=manifest, source_sha256=a.source_sha256,
                    target="raw_seconds", loss="micro_valid_bin_MAE", best_val_bin_mae_seconds=best_score), path)


def restore(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Checkpoint is not this raw-MAE whole-trajectory MLP format")
    model = TrajectoryMLPMAE(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


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
    evaluator = Evaluator(output_dir=a.out / "training_subset_diagnostic")
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
    a.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(format=FORMAT, validation_mask_epoch=VAL_EPOCH, label_provenance="existing valid upstream-interpolated times accepted by user",
                  upstream_interpolation_audit="out of scope by user instruction", source_sha256=a.source_sha256,
                  created_at=datetime.now(timezone.utc).isoformat(), no_sealed_test=True)
    json_write(a.out / "config.json", config)
    json_write(a.out / "data_manifest.json", manifest)
    if a.mode == "baseline":
        metrics = eval_split(None, a, a.out / "predictions", a.bootstrap)
        json_write(a.out / "metrics.json", metrics)
        return
    if a.mode == "evaluate":
        model, saved = restore(a.checkpoint, a.device)
        for key in ("m_max", "data_seed", "val_days"):
            if saved["config"][key] != getattr(a, key):
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
    # Baseline first on exactly the same deterministic validation groups/masks.
    baseline = eval_split(None, a)
    json_write(a.out / "baseline.json", baseline)
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
        if validation["eval_mask_id"] != baseline["eval_mask_id"]:
            raise AssertionError("Validation groups, masks or targets changed across methods/epochs")
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
    if final["eval_mask_id"] != baseline["eval_mask_id"]:
        raise AssertionError("Final evaluation set changed")
    final.update(best_epoch=best_epoch, elapsed_total_seconds=time.monotonic()-training_started,
                 gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if a.device.startswith("cuda") else None,
                 device_name=torch.cuda.get_device_name() if a.device.startswith("cuda") else "CPU",
                 interpretation="Validation selected checkpoint; not an independent test estimate")
    json_write(a.out / "best_val_metrics.json", final)
    progress("DONE", best_epoch=best_epoch+1, output=a.out)


if __name__ == "__main__":
    main()
