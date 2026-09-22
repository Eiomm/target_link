"""Generate a tiny, deterministic observations_v2 corpus without private data.

Run from the repository root with ``python -m
experiments.trajectory_mlp_v1.tools.make_synthetic_corpus --out PATH``.
The output directory must not already exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def make_corpus(out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for day, cells in (("20260817", (10, 11)), ("20260823", (20, 21))):
        rows = []
        for cell in cells:
            for member in range(5):
                rows.append({
                    "cell_id": cell,
                    "sample_id": f"SYNTHETIC-{day}-{cell}-{member}",
                    "dt": float(member * 30),
                    "T_diff": [1.0 + member / 10, 2.0 + member / 10],
                    # Historical name: 10 represents full coverage, not 10%.
                    "ratio_pct": [10, 7],
                    "valid": [True, True],
                    "bin_pos": [0, 12],
                })
        partition = out / "observations_v2" / f"day={day}" / "bucket=0"
        partition.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), partition / "part-0.parquet")
    (out / "synthetic_manifest.json").write_text(
        json.dumps({
            "synthetic": True,
            "purpose": "CLI smoke testing only; not a scientific benchmark",
            "train_days": ["20260817"],
            "val_days": ["20260823"],
            "cells_per_day": 2,
            "trajectories_per_cell": 5,
        }, indent=2) + "\n", encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    make_corpus(args.out)
    print(f"Synthetic corpus written to {args.out}")


if __name__ == "__main__":
    main()
