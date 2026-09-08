"""Build the self-supervised pretraining corpus from profiles (repV2 §2.2).

Pretraining unit = one profile row: a single sample's 10m-bin speed curve on
one 200m sub-link — the EXACT input unit TrajectoryEncoder eats downstream,
so the encoder weights transfer verbatim (no second encoder, repV2 §5.4).

The profiles npz already carries per-row free labels (y_travel_s, v_sample,
n_bins_target), so no join is needed. This tool:
  1. loads each input npz (one per day, sequential — peak memory = one day),
  2. optionally subsamples WHOLE samples per day (--per-day-cap, uniform,
     seeded — the 7-day corpus at ~1.7e9 rows does not fit one GPU epoch),
  3. derives quantile-bucket labels (y / v / len, n buckets each) and
     Beijing hour-of-day from window_id (epoch hours: hour = (w+8) % 24),
  4. writes one compact npz: curve arrays + int8 label columns + bucket edges.

Masking itself is done at train time (encoder-native "invalid bin" semantics:
masked positions attend with v=0,m=0 — see models/encoder.py docstring); the
corpus only carries valid/observed so the trainer masks valid positions only.

Run:
  python tools/build_pretrain_corpus.py \
      --inputs data/processed_day20260820/profiles_l200.npz ... \
      --out data/pretrain_corpus/corpus.npz --per-day-cap 5000000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CURVE_KEYS = ["speeds", "valid", "observed", "lengths"]      # per-bin arrays
META_KEYS = ["sample_id", "link_id", "window_id", "sub_id"]  # provenance
LABEL_KEYS = ["y_travel_s", "v_sample", "n_bins_target"]     # free-label sources


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", nargs="+", required=True,
                   help="profiles npz paths, one per day (e.g. data/processed_dayX/profiles_l200.npz)")
    p.add_argument("--out", required=True, help="output corpus npz")
    p.add_argument("--per-day-cap", type=int, default=0,
                   help="max SAMPLES kept per input (0 = keep all); whole samples are kept")
    p.add_argument("--n-buckets", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bucketize(values: np.ndarray, n: int) -> tuple:
    """Balanced quantile buckets + the edges used (stored for eval reuse)."""
    edges = np.quantile(values.astype(np.float64), np.linspace(0.0, 1.0, n + 1))
    labels = np.clip(np.digitize(values, edges[1:-1]), 0, n - 1).astype(np.int8)
    return labels, edges.astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    keys = CURVE_KEYS + META_KEYS + LABEL_KEYS
    parts: dict[str, list] = {k: [] for k in keys}
    kept_rows, kept_samples = [], []

    for path in args.inputs:
        z = np.load(path)
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise SystemExit(f"{path}: missing keys {missing}")
        keep = np.ones(len(z["lengths"]), dtype=bool)
        if args.per_day_cap and args.per_day_cap > 0:
            # subsample at SAMPLE level: unique ids -> pick cap -> row mask
            uniq, inverse = np.unique(z["sample_id"], return_inverse=True)
            if len(uniq) > args.per_day_cap:
                pick = np.zeros(len(uniq), dtype=bool)
                pick[rng.choice(len(uniq), args.per_day_cap, replace=False)] = True
                keep = pick[inverse]
        for k in keys:
            parts[k].append(z[k][keep])
        kept_rows.append(int(keep.sum()))
        kept_samples.append(int(np.unique(z["sample_id"][keep]).size))
        print(f"[corpus] {path}: kept {keep.sum():,}/{len(keep):,} rows "
              f"({kept_samples[-1]:,} samples)")
        del z, keep

    out = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(out["lengths"])
    if n == 0:
        raise SystemExit("empty corpus — nothing kept")

    out["attr_y"], out["y_edges"] = bucketize(out.pop("y_travel_s"), args.n_buckets)
    out["attr_v"], out["v_edges"] = bucketize(out.pop("v_sample"), args.n_buckets)
    out["attr_len"], out["len_edges"] = bucketize(
        out.pop("n_bins_target").astype(np.float64), args.n_buckets)
    # window_id = floor(t_enter/3600) is UTC epoch hours -> Beijing hour
    out["attr_hour"] = ((out["window_id"] % 24 + 8) % 24).astype(np.int8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out,
                        src_paths=np.array(args.inputs),
                        per_day_kept_rows=np.array(kept_rows),
                        per_day_kept_samples=np.array(kept_samples),
                        seed=np.int64(args.seed), n_buckets=np.int64(args.n_buckets))
    print(f"[corpus] wrote {args.out}: {n:,} rows; labels attr_y/attr_v/attr_len "
          f"({args.n_buckets} quantile buckets) + attr_hour(24, Beijing)")


if __name__ == "__main__":
    main()
