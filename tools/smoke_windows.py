"""Server-side synthetic smoke: SQL build -> grouped training -> representation export.

Uses the current Python/Java environment; installs nothing and submits no YARN
jobs. The synthetic available_ts and geometry are known by construction; this
does not validate the production producer's timestamp contract.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="new local directory")
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    repo = Path(__file__).resolve().parents[1]
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for period, origin in [("train", 540.), ("val", 1260.)]:
        for road in ["A", "B"]:
            for sample in range(3):
                for idx, pos in [(0, 0.), (1, 30.), (2, 210.)]:
                    start = origin + idx * 10 + sample
                    duration = 5. + sample
                    rows.append(dict(map_version="synthetic", target_link_id=road,
                        sample_id="%s-%s-%s" % (period, road, sample), bin_idx=idx,
                        bin_size_m=10., ratio=0.5 if idx == 1 else 1., T_diff=duration,
                        L_link_m=300., observed=1, seg_mark=1, spatial_start_m=pos,
                        bin_start_ts=start, bin_end_ts=start+duration, available_ts=start+duration+1))
                # A still-in-progress next bin: it must NOT enter the first anchor.
                rows.append(dict(rows[-1], bin_idx=3, spatial_start_m=220.,
                                  bin_start_ts=origin+50, bin_end_ts=origin+80,
                                  T_diff=30., available_ts=origin+81))
    pq.write_table(pa.Table.from_pylist(rows), root / "events.parquet")
    env = os.environ.copy()
    env["PYSPARK_PYTHON"] = sys.executable
    env.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    def run(script, *args):
        subprocess.run([sys.executable, str(repo / "tools" / script), *map(str, args)],
                       cwd=repo, env=env, check=True)

    run("build_windows_spark.py", "--inputs", root / "events.parquet", "--out", root / "corpus",
        "--anchor-start", 600, "--anchor-end", 1440, "--lookback-seconds", 600,
        "--stride-seconds", 60, "--max-passes", 2, "--partitions", 2, "--shuffle-partitions", 2)
    curves = pq.read_table(root / "corpus/window_curves").to_pylist()
    for row in curves:
        if row["anchor_ts"] in [600, 1320]:
            assert all(b["bin_idx"] != 3 for b in row["bins"]), "in-progress bin leaked"
    run("train_windows.py", "--data", root / "corpus", "--out", root / "training",
        "--train-end", 720, "--val-start", 1320, "--val-end", 1440,
        "--epochs", 1, "--batch-size", 2, "--d-model", 16, "--heads", 2,
        "--layers", 1, "--group-layers", 1, "--device", "cpu")
    metrics = json.loads((root / "training/metrics.jsonl").read_text().strip())
    assert metrics["train_supervised_snapshots"] > 0 and metrics["val_supervised_snapshots"] > 0
    run("encode_windows.py", "--data", root / "corpus", "--checkpoint", root / "training/last.pt",
        "--out", root / "representations.parquet", "--start", 1320, "--end", 1440)
    exported = pq.read_table(root / "representations.parquet").to_pylist()
    # 2 roads x 2 validation anchors x 2 modeling units (positions 0/30 and 210 of a
    # 300m link), one representation per (modeling unit, anchor).
    assert len(exported) == 8 and all(len(r["representation"]) == 16 for r in exported)
    assert {r["sub_id"] for r in exported} == {0, 1}
    print("PASS: local SQL build, partial-bin boundaries, train/val, and representation export")


if __name__ == "__main__":
    main()
