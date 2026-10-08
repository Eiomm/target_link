"""Compare CPU/GPU mean calculation on identical complete validation partitions.

All existing metrics, strata, identity checks and fallback rules are retained.
Only visible_mean is moved to CUDA in the GPU candidate. Transfers are timed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from trajectory_mae import evaluation, run


@contextmanager
def timed_mean(device):
    original = evaluation.visible_mean
    timing = {"seconds": 0.0}

    def calculate(batch):
        start = time.perf_counter()
        if device == "cpu":
            result = original(batch)
        else:
            # The baseline only reads the time channel, never ratio/metadata.
            inputs = {"x": batch["x"][..., :1].contiguous().to(device)}
            inputs.update({k: batch[k].to(device)
                           for k in ("bin_valid", "traj_valid", "mae_mask")})
            gpu = original(inputs)
            # Return one mean per group/bin, not M identical rows over PCIe.
            means = gpu["prediction_seconds"][:, 0].contiguous().cpu()
            result = {
                "prediction_seconds": means[:, None].expand_as(batch["x"][..., 0]),
                "same_bin_support": gpu["same_bin_support"].cpu(),
                "group_has_visible": gpu["group_has_visible"].cpu(),
            }
            torch.cuda.synchronize(device)
        timing["seconds"] += time.perf_counter() - start
        return result

    evaluation.visible_mean = calculate
    try:
        yield timing
    finally:
        evaluation.visible_mean = original


def one_pass(args, partition_count, device):
    started = time.perf_counter()
    ds, loader = run.make_loader(args, train=False)
    if partition_count > len(ds.partitions):
        raise ValueError("Requested more partitions than are available")
    ds.partitions = sorted(ds.partitions, key=lambda p: (p[2], int(p[3])))[:partition_count]
    selected = [{"root": p[1], "day": p[2], "bucket": p[3]} for p in ds.partitions]
    scorer = evaluation.Evaluator(include_baseline=True)
    seen = {}
    groups = steps = 0
    load_seconds = update_seconds = 0.0
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), timed_mean(device) as mean_timing:
        iterator = iter(loader)
        while True:
            tick = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                load_seconds += time.perf_counter() - tick
                break
            load_seconds += time.perf_counter() - tick
            run.collect_partition_stats(seen, batch)
            tick = time.perf_counter()
            scorer.update(None, batch)
            update_seconds += time.perf_counter() - tick
            groups += batch["x"].shape[0]
            steps += 1
            if steps == 1 or steps % 20 == 0:
                print(f"[{device}] batch={steps}, groups={groups}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        tick = time.perf_counter()
        metrics = scorer.finalize(bootstrap=0)
        finalize_seconds = time.perf_counter() - tick
    if device != "cpu":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started
    expected_groups = sum(p["selected_groups"] for p in seen.values())
    if len(seen) != partition_count or groups != expected_groups or not metrics["bins"]:
        raise ValueError("Incomplete or empty partition pass; benchmark is invalid")
    return dict(device=device, partitions=selected, complete_partitions=len(seen),
                groups=groups, batches=steps, total_seconds=total_seconds,
                groups_per_second=groups / total_seconds,
                loader_wait_seconds=load_seconds,
                mean_and_transfer_seconds=mean_timing["seconds"],
                other_metrics_seconds=update_seconds - mean_timing["seconds"],
                finalize_seconds=finalize_seconds,
                other_seconds=total_seconds-load_seconds-update_seconds-finalize_seconds,
                gpu_peak_GiB=(torch.cuda.max_memory_allocated(device)/2**30 if device != "cpu" else None),
                metrics=metrics)


def check_equal(reference, candidate):
    if reference["partitions"] != candidate["partitions"]:
        raise ValueError("Selected partitions changed")

    def compare(a, b, path):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f"Metric keys changed: {path}")
            for key in a:
                compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, float):
            if not np.isclose(a, b, rtol=1e-5, atol=1e-6):
                raise ValueError(f"CPU/GPU metrics differ: {path}: {a} != {b}")
        elif a != b:
            raise ValueError(f"Protocol/count mismatch: {path}: {a} != {b}")

    compare(reference["metrics"], candidate["metrics"], "metrics")


def render(result):
    lines = ["# 可见均值 CPU/GPU 小分区对比", "", result["scope"], "",
             f"状态：{result['status']}。GPU：{result['gpu']}", "",
             "| 执行次序 | 设备 | 完整分区 | group 数 | 总秒数 | 读取等待秒 | 均值及传输秒 | 其他指标秒 | group/秒 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, p in enumerate(result["passes"], 1):
        lines.append(f"| {i} | {p['device']} | {p['complete_partitions']} | {p['groups']} | {p['total_seconds']:.2f} | {p['loader_wait_seconds']:.2f} | {p['mean_and_transfer_seconds']:.2f} | {p['other_metrics_seconds']:.2f} | {p['groups_per_second']:.1f} |")
    if result.get("speedup") is not None:
        lines += ["", f"CPU/GPU 总耗时中位数之比：{result['speedup']:.3f}；大于 1 表示 GPU 候选更快。"]
    else:
        lines += ["", "没有 GPU 实测结果，不能据此判断 GPU 是否提速。"]
    lines += ["", "总耗时包括分区读取、组装、均值计算、CPU/GPU 传输、完整评估统计和汇总；不含进程导入及一次性 CUDA 初始化。",
              "读取等待与后台 worker 工作重叠，不能直接视为全部 CPU 预处理时间。未清除文件缓存；有 GPU 时交替 CPU→GPU、GPU→CPU，分别报告各次耗时。",
              "保留原基线的分层统计与遮挡身份检查；只把均值计算迁到 GPU，不代表整套评估迁到 GPU 的潜力。", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partitions", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--m-max", type=int, default=64)
    p.add_argument("--data-seed", type=int, default=20260921)
    p.add_argument("--val-days", nargs="+", default=["20260823"])
    c = p.parse_args(argv)
    if min(c.partitions, c.repeats, c.threads, c.batch_size) < 1:
        p.error("partition/repeat/thread/batch counts must be positive")
    a = run.parse_args(["baseline", "--data", *c.data, "--out", str(c.out),
                       "--val-days", *c.val_days, "--workers", str(c.workers),
                       "--threads", str(c.threads), "--batch-size", str(c.batch_size),
                       "--m-max", str(c.m_max), "--data-seed", str(c.data_seed), "--device", "cpu"])
    torch.set_num_threads(c.threads)
    available = torch.cuda.is_available()
    gpu, setup_seconds = None, None
    if available:
        tick = time.perf_counter()
        torch.cuda.init()
        torch.zeros(1, device="cuda").sum().item()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter()-tick
        gpu = torch.cuda.get_device_name()
    else:
        print("CUDA unavailable: running CPU only; GPU comparison will remain incomplete.", flush=True)
    c.out.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", gpu=gpu, cuda_available=available,
                  gpu_setup_seconds=setup_seconds,
                  scope="固定少量完整验证分区、相同 batch/workers 和冻结遮挡；对比原 CPU 基线与仅均值计算上 GPU 的候选。",
                  config={k: str(v) if isinstance(v, Path) else v for k,v in vars(c).items()},
                  source_sha256=run.source_hashes(), passes=[], speedup=None)
    for repeat in range(c.repeats):
        devices = (["cpu", "cuda"] if repeat % 2 == 0 else ["cuda", "cpu"]) if available else ["cpu"]
        for device in devices:
            current = one_pass(a, c.partitions, device)
            if result["passes"]:
                check_equal(result["passes"][0], current)
            result["passes"].append(current)
            run.json_write(c.out / "benchmark.json", result)
    result["status"] = "paired_complete" if available else "cpu_only_gpu_unavailable"
    if available:
        medians = {d: statistics.median(x["total_seconds"] for x in result["passes"] if x["device"] == d)
                   for d in ("cpu", "cuda")}
        result["speedup"] = medians["cpu"] / medians["cuda"]
    run.json_write(c.out / "benchmark.json", result)
    report = render(result)
    (c.out / "benchmark.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    return result


if __name__ == "__main__":
    main()
