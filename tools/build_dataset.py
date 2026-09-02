"""Build the raw V1 dataset from the synthetic city.

Output layout (under --out-dir):
    raw/
        links.parquet       link geometry + metadata (the "road map")
        cells_train.parquet raw per-vehicle, per-(link, window) traversals
        cells_val.parquet
        cells_test.parquet
        stats.json          basic statistics (spec section 11)

Each cell row is one vehicle traversing one link during one 5-minute window:
    link_id, window_id, order_id,
    obs_t    [P] float32   seconds since window start
    obs_s    [P] float32   projected arc length on the link (map-matched)
    mean_speed float32     production traffic feature of that (link, window)
    eta_label  float32     next-window mean speed of the link (regression)
    rp_label   int32       next-window congestion class of the link (0..3)

Labels come strictly from the FUTURE window; features from the current one,
so there is no future-information leakage.

Splits are by window id (disjoint time): train / val / test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from target_link_v1.data.synthetic_city import build_default_city  # noqa: E402

RP_THRESHOLDS = (0.45, 0.60, 0.75)  # ratios of mean_speed / free-flow speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V1 synthetic dataset")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-links", type=int, default=400)
    parser.add_argument("--n-windows", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--val-windows", type=int, default=12)
    parser.add_argument("--test-windows", type=int, default=12)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-coverage", type=float, default=0.60,
                        help="keep traversal only if it covers >= this fraction of link length")
    parser.add_argument("--max-gap-sec", type=float, default=30.0)
    parser.add_argument("--noise-gps-m", type=float, default=6.0)
    parser.add_argument("--mean-vehicles", type=float, default=4.0,
                        help="poisson mean vehicles per (link, window)")
    parser.add_argument("--max-vehicles", type=int, default=12)
    return parser.parse_args()


def project_onto_link(points: np.ndarray, cum: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Project positions onto the polyline -> arc length s for each point."""
    a = points[:-1]  # [M-1, 2]
    b = points[1:]
    ab = b - a
    seg_len = np.maximum(np.linalg.norm(ab, axis=1), 1e-6)
    d = xy[:, None, :] - a[None, :, :]  # [P, M-1, 2]
    t = np.clip(np.sum(d * ab[None], axis=2) / (seg_len[None] ** 2), 0.0, 1.0)
    proj = a[None] + t[:, :, None] * ab[None]
    dist = np.linalg.norm(xy[:, None, :] - proj, axis=2)
    j = np.argmin(dist, axis=1)
    s = cum[:-1][j] + t[np.arange(len(xy)), j] * seg_len[j]
    return s.astype(np.float32)


def main() -> None:
    args = parse_args()
    out = args.out_dir / "raw"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    city = build_default_city(n_links=args.n_links, n_windows=args.n_windows, seed=args.seed)

    # ---- split by window id (disjoint time) --------------------------------
    n_test = args.test_windows
    n_val = args.val_windows
    n_train = city.n_windows - n_test - n_val
    assert n_train > 0, "not enough windows for the requested splits"
    split_of_window = (
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )

    # per-(link, window) vehicle generation + mean speed ----------------------
    # lambda_k varies with link "popularity" to create realistic sparsity
    popularity = rng.uniform(0.6, 4.0, size=args.n_links)

    rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    # first pass: generate all vehicles, keep window mean speeds for labels
    window_mean_speed: Dict[int, np.ndarray] = {}  # window -> [n_links]
    cells: Dict[int, List[Dict[str, Any]]] = {}    # window -> cell dicts

    total_trajs = 0
    dropped_short = 0
    empty_cells = 0
    total_cells = 0

    for w in range(city.n_windows):
        cells[w] = []
        speeds = np.full(args.n_links, np.nan, dtype=np.float32)
        for link in city.links:
            total_cells += 1
            lam = args.mean_vehicles * popularity[link.link_id] / popularity.mean()
            k = int(min(np.clip(rng.poisson(lam), 0, args.max_vehicles), args.max_vehicles))
            trip_speeds: List[float] = []
            for _ in range(k):
                traj = city.generate_trajectory(link, w, rng, noise_gps_m=args.noise_gps_m)
                if not traj:
                    dropped_short += 1
                    continue
                obs_t = traj["obs_t"]
                # project noisy GPS onto link -> arc length; enforce monotonicity
                obs_s = project_onto_link(link.points, link.cum, traj["obs_xy"])
                obs_s = np.maximum.accumulate(obs_s)
                obs_s = np.clip(obs_s, 0.0, link.length)
                coverage = (obs_s[-1] - obs_s[0]) / max(link.length, 1e-3)
                gaps = np.diff(obs_t)
                ok = (
                    len(obs_t) >= args.min_points
                    and coverage >= args.min_coverage
                    and (gaps.max() if len(gaps) else 0.0) <= args.max_gap_sec
                    and obs_t[-1] - obs_t[0] >= 3.0
                )
                if not ok:
                    dropped_short += 1
                    continue
                trip_speed = float((obs_s[-1] - obs_s[0]) / max(obs_t[-1] - obs_t[0], 1e-3))
                trip_speeds.append(trip_speed)
                total_trajs += 1
                cells[w].append(
                    {
                        "link_id": link.link_id,
                        "window_id": w,
                        "obs_t": obs_t.astype(np.float32),
                        "obs_s": obs_s.astype(np.float32),
                    }
                )
            if trip_speeds:
                # production-style mean speed: average vehicle trip speed + noise
                speeds[link.link_id] = float(np.mean(trip_speeds)) * (
                    1.0 + float(rng.normal(0.0, 0.03))
                )
            else:
                empty_cells += 1
        window_mean_speed[w] = speeds

    # ---- labels from the NEXT window; write rows ---------------------------
    for w in range(city.n_windows - 1):  # last window has no future label
        split = split_of_window[w]
        speeds_now = window_mean_speed[w]
        speeds_next = window_mean_speed[w + 1]
        for cell in cells[w]:
            lid = cell["link_id"]
            if np.isnan(speeds_now[lid]) or np.isnan(speeds_next[lid]):
                continue  # the cell itself (or its label window) has no traffic
            link = city.links[lid]
            ratio = speeds_next[lid] / link.base_speed
            rp_label = int(np.searchsorted(RP_THRESHOLDS, ratio))
            rows[split].append(
                {
                    **cell,
                    "order_id": f"{w}-{lid}-{len(rows[split])}",
                    "mean_speed": np.float32(speeds_now[lid]),
                    "eta_label": np.float32(speeds_next[lid]),
                    "rp_label": np.int32(rp_label),
                }
            )

    # ---- persist -------------------------------------------------------------
    def to_table(split_rows: List[Dict[str, Any]]) -> pa.Table:
        return pa.table(
            {
                "order_id": pa.array([r["order_id"] for r in split_rows], type=pa.string()),
                "link_id": pa.array([r["link_id"] for r in split_rows], type=pa.int32()),
                "window_id": pa.array([r["window_id"] for r in split_rows], type=pa.int32()),
                "obs_t": pa.array([r["obs_t"].tolist() for r in split_rows], type=pa.list_(pa.float32())),
                "obs_s": pa.array([r["obs_s"].tolist() for r in split_rows], type=pa.list_(pa.float32())),
                "mean_speed": pa.array([float(r["mean_speed"]) for r in split_rows], type=pa.float32()),
                "eta_label": pa.array([float(r["eta_label"]) for r in split_rows], type=pa.float32()),
                "rp_label": pa.array([int(r["rp_label"]) for r in split_rows], type=pa.int32()),
            }
        )

    for split in ("train", "val", "test"):
        pq.write_table(to_table(rows[split]), out / f"cells_{split}.parquet", compression="snappy")

    link_lengths = np.array([lk.length for lk in city.links], dtype=np.float32)
    links_table = pa.table(
        {
            "link_id": pa.array([lk.link_id for lk in city.links], type=pa.int32()),
            "points": pa.array([lk.points.flatten().tolist() for lk in city.links], type=pa.list_(pa.float32())),
            "base_speed": pa.array([lk.base_speed for lk in city.links], type=pa.float32()),
            "length": pa.array([lk.length for lk in city.links], type=pa.float32()),
            "bottleneck_center": pa.array(
                [lk.bottleneck_center if not np.isnan(lk.bottleneck_center) else -1.0 for lk in city.links],
                type=pa.float32(),
            ),
        }
    )
    pq.write_table(links_table, out / "links.parquet", compression="snappy")

    stats = {
        "seed": args.seed,
        "n_links": args.n_links,
        "n_windows": args.n_windows,
        "window_seconds": city.window_seconds,
        "split_windows": {"train": n_train, "val": n_val, "test": n_test},
        "n_rows": {split: len(rows[split]) for split in rows},
        "n_vehicles_generated": total_trajs,
        "n_dropped_by_filter": dropped_short,
        "n_empty_cells": empty_cells,
        "n_total_cells": total_cells,
        "link_length_mean_m": float(link_lengths.mean()),
        "link_length_p50_m": float(np.percentile(link_lengths, 50)),
        "link_length_p95_m": float(np.percentile(link_lengths, 95)),
        "link_length_max_m": float(link_lengths.max()),
        "rp_thresholds": list(RP_THRESHOLDS),
        "note": "labels are derived from window+1 only; features from window w",
    }
    with open(out / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
