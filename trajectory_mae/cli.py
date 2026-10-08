"""CLI options; an explicit TOML preset can override defaults."""
import argparse
from pathlib import Path
import tomllib

def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preset, _ = config_parser.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    p.add_argument("mode", choices=["train", "baseline", "evaluate", "smoke"])
    p.add_argument("--data", nargs="+", required=True, help="observation-v2, published observation-v3, or tensor corpus roots")
    p.add_argument("--val-data", nargs="+", help="defaults to --data roots; validation is selected by --val-days")
    p.add_argument("--train-days", nargs="+", default=[f"202608{d}" for d in range(17, 23)])
    p.add_argument("--val-days", nargs="+", default=["20260823"])
    p.add_argument("--out", type=Path, required=True, help="new output directory; refuses to overwrite")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--m-max", type=int, default=64, help="cell chunk size; ablate e.g. 16,32,64; discard tails <3")
    p.add_argument("--input-channels", type=int, choices=[1], default=1, help="encoder T_clean only; hidden ratio conditions decoder; bin_valid is a loss mask")
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-groups", type=int, default=0, help="explicit diagnostic cap; default is full pass")
    p.add_argument("--groups-per-partition", type=int, default=0, help="explicit sampling cap; default is all groups")
    p.add_argument("--bootstrap", type=int, default=2000, help="legacy paired-bootstrap setting; model-only training/evaluation does not use it")
    p.add_argument("--smoke-groups", type=int, default=128)
    p.add_argument("--smoke-steps", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--plot-every", type=int, default=1000, help="refresh sampled batch curves every N steps; 0 means epoch end only")
    p.add_argument("--dry-run", action="store_true", help="validate data without training")
    p.add_argument("--expected-buckets", type=int, default=0)
    if preset.config is not None:
        with preset.config.open("rb") as stream:
            defaults = tomllib.load(stream)
        supported = {action.dest for action in p._actions}
        unknown = set(defaults) - supported
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**defaults)
    a = p.parse_args(argv)
    a.val_group_total = None
    if min(a.batch_size, a.epochs, a.threads, a.smoke_groups, a.smoke_steps) < 1 or a.m_max < 3:
        p.error("sizes must be positive and m-max >=3")
    if min(a.workers, a.max_groups, a.groups_per_partition, a.bootstrap) < 0:
        p.error("counts cannot be negative")
    if min(a.log_every, a.plot_every, a.expected_buckets) < 0:
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
