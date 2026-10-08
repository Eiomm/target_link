"""Evaluate the visible same-bin mean against completed model epochs on CPU.

Reuse the training validation reader, frozen masks and evaluator. Do not export
per-bin CSVs: a full validation pass can contain hundreds of millions of bins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trajectory_mae import run

METRICS = {
    "bin_mae_seconds": "单 bin MAE（秒）",
    "bin_rmse_seconds": "单 bin RMSE（秒）",
    "trajectory_mae_seconds": "轨迹有效 bin 总耗时 MAE（秒）",
    "trajectory_rmse_seconds": "轨迹有效 bin 总耗时 RMSE（秒）",
    "group_balanced_bin_mae_seconds": "组等权、组内轨迹等权 bin MAE（秒）",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_epochs(root):
    """Snapshot only fully written epoch records, including an ongoing run."""
    history = root / "metrics.jsonl"
    if history.exists():
        records = []
        for line in history.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):
                continue  # The training process may currently be appending it.
            record = json.loads(line)
            records.append((record["epoch"] + 1, record["val"]))
        if records:
            return records
    final = root / "best_val_metrics.json"
    if final.exists():
        metrics = read_json(final)
        return [(metrics["best_epoch"] + 1, metrics)]
    raise ValueError("对比实验还没有完整保存的验证指标，请在一轮验证结束后运行。")


def compare(baseline, records):
    comparisons = []
    for epoch, model in records:
        # Never compare metrics from different groups/masks/denominators.
        for key in ("eval_mask_id", "bins", "trajectories", "groups", "groups_total"):
            if baseline[key] != model[key]:
                raise ValueError(f"第 {epoch} 轮验证集不匹配：{key}；拒绝计算改善率。")
        if model.get("ours") is None:
            raise ValueError(f"第 {epoch} 轮缺少模型验证指标。")
        values = {}
        for key in METRICS:
            base, ours = baseline["baseline"][key], model["ours"][key]
            values[key] = dict(baseline=base, model=ours,
                               model_minus_baseline=ours - base,
                               improvement_percent=(100 * (base - ours) / base if base else None))
        comparisons.append(dict(epoch=epoch, metrics=values))
    return dict(eval_mask_id=baseline["eval_mask_id"], epochs=comparisons,
                interpretation="正改善率表示模型误差更低；这是同一验证集上的点估计，不是独立测试或显著性检验。")


def report(result):
    lines = ["# 可见均值与模型验证对比", "",
             "对每个隐藏 bin，取同组可见轨迹在同一 bin 上的有效原始耗时算术均值。",
             "同一 bin 无可见有效值时，回退到组内所有可见有效 bin 的均值；不使用隐藏标签填补。",
             "不按 ratio 归一化；未改变原模型的监督位置。", "",
             "轨迹 MAE = 各轨迹 abs(sum(有效监督 bin 的预测值 − 真实值)) 的平均。", "",
             "| 轮次 | 指标 | 均值基线 | 模型 | 模型误差降低 |",
             "|---|---|---:|---:|---:|"]
    for entry in result["epochs"]:
        for key, label in METRICS.items():
            v = entry["metrics"][key]
            percent = v["improvement_percent"]
            gain = "不适用（基线为零）" if percent is None else f"{percent:.2f}%"
            lines.append(f"| {entry['epoch']} | {label} | {v['baseline']:.6f} | {v['model']:.6f} | {gain} |")
    lines += ["", result["interpretation"], "",
              "仅包含本脚本启动时已完整保存的验证轮次；后续轮次未纳入本次报告。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    cli = parser.parse_args(argv)
    root = cli.model_run.resolve()
    if not (root / "config.json").exists() and (root / "artifacts/config.json").exists():
        root = root / "artifacts"
    config = read_json(root / "config.json")
    if config.get("validation_mask_epoch") != run.VAL_EPOCH:
        raise ValueError("验证遮挡版本与当前代码不一致，不能直接比较。")
    records = reference_epochs(root)
    data = config.get("val_data") or config["data"]
    max_groups = config.get("max_groups", 0)
    args = run.parse_args([
        "baseline", "--data", *data, "--val-days", *config["val_days"],
        "--train-days", *config["train_days"], "--out", str(cli.out),
        "--device", "cpu", "--batch-size", str(cli.batch_size),
        "--workers", str(0 if max_groups else cli.workers), "--threads", str(cli.threads),
        "--m-max", str(config["m_max"]), "--data-seed", str(config["data_seed"]),
        "--seed", str(config["seed"]), "--bootstrap", "0",
        "--max-groups", str(max_groups),
        "--groups-per-partition", str(config.get("groups_per_partition", 0)),
    ])
    manifest = run.file_manifest(data, args.val_days)
    expected = read_json(root / "data_manifest.json")["validation"]
    if manifest != expected:
        raise ValueError("验证数据文件或元数据与模型运行时不同；拒绝进行不配对比较。")
    cli.out.mkdir(parents=True, exist_ok=False)
    run.json_write(cli.out / "config.json", dict(
        model_run=str(root), validation_data=data, val_days=args.val_days,
        m_max=args.m_max, data_seed=args.data_seed, validation_mask_epoch=run.VAL_EPOCH,
        batch_size=args.batch_size, workers=args.workers, threads=args.threads,
        device="cpu", compared_epochs=[epoch for epoch, _ in records],
        source_sha256=run.source_hashes(), predictions_exported=False,
        baseline="visible_same_bin_raw_seconds_arithmetic_mean_with_group_fallback"))
    run.json_write(cli.out / "data_manifest.json", {"validation": manifest})
    run.torch.set_num_threads(args.threads)
    print(f"验证配置已与模型对齐：M={args.m_max}，data_seed={args.data_seed}，workers={args.workers}", flush=True)
    print(f"对比已保存轮次：{[e for e, _ in records]}；仅使用 CPU，不加载模型权重。", flush=True)
    baseline = run.eval_split(None, args, output_dir=None, bootstrap=0)
    run.json_write(cli.out / "metrics.json", baseline)
    result = compare(baseline, records)
    run.json_write(cli.out / "comparison.json", result)
    rendered = report(result)
    (cli.out / "comparison.md").write_text(rendered, encoding="utf-8")
    print(rendered, flush=True)
    print(f"结果已保存：{cli.out / 'comparison.md'}", flush=True)
    return result


if __name__ == "__main__":
    main()
