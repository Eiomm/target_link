"""Summarize three model seeds evaluated on exactly the same validation set."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(paths):
    runs = []
    for path in paths:
        root = Path(path)
        runs.append((root, json.loads((root / "config.json").read_text()),
                     json.loads((root / "best_val_metrics.json").read_text())))
    seeds = [c["seed"] for _, c, _ in runs]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Provide at least three distinct completed model seeds")
    if len({m["eval_mask_id"] for _, _, m in runs}) != 1:
        raise ValueError("Validation groups or masks differ: cannot aggregate these runs as seeds")
    protocol_keys = ["m_max", "input_channels", "d_model", "heads", "layers", "dropout", "epochs",
                     "lr", "weight_decay", "batch_size", "data_seed", "max_groups", "groups_per_partition",
                     "train_days", "val_days", "source_sha256"]
    for key in protocol_keys:
        if any(c[key] != runs[0][1][key] for _, c, _ in runs[1:]):
            raise ValueError(f"Seed runs have different protocol: {key}")
    for key, default in (("time_encoding", "seconds"), ("decoder_layers", 2)):
        if any(c.get(key, default) != runs[0][1].get(key, default) for _, c, _ in runs[1:]):
            raise ValueError(f"Seed runs have different protocol: {key}")
    for _, _, m in runs[1:]:
        if m["baseline"] != runs[0][2]["baseline"]:
            raise ValueError("Baseline/targets changed between runs")
    keys = ["bin_mae_seconds", "bin_rmse_seconds", "trajectory_mae_seconds",
            "trajectory_rmse_seconds", "group_balanced_bin_mae_seconds"]
    return dict(seeds=seeds, eval_mask_id=runs[0][2]["eval_mask_id"],
                baseline=runs[0][2]["baseline"],
                ours={k: dict(mean=statistics.mean(m["ours"][k] for _, _, m in runs),
                              std=statistics.stdev(m["ours"][k] for _, _, m in runs)) for k in keys},
                runs=[dict(path=str(p), seed=c["seed"], best_epoch=m["best_epoch"],
                           ours=m["ours"], paired_ci=m["paired_ci"]) for p, c, m in runs],
                interpretation="Validation-selected checkpoints; seed SD is not an independent-test CI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
