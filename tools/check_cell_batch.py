"""Field-by-field acceptance of the training-side cell reader.

Pulls real groups out of the corpus, runs them through `CellCorpusDataset` and
`collate_cells`, and checks every field of the resulting batch: shapes, dtypes,
the padding contract, the masks, and -- independent of the reader -- the
piece-to-bin fold recomputed with plain Python dicts.

On this pod pyarrow needs the Hadoop JVM classpath to reach HDFS:

    export CLASSPATH=$(hadoop classpath --glob)
    export ARROW_LIBHDFS_DIR=/usr/local/hadoop-current/lib/native

Run:
    python tools/check_cell_batch.py --root hdfs://DClusterNmg3/.../corpus_v1 \
        --obs-dir observations_v2 --batch 4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

sys.path.insert(0, ".")
from target_link_v1.data.cell_corpus import (  # noqa: E402
    CellCorpusDataset, collate_cells)

_OBS = ["cell_id", "sample_id", "dt", "T_diff", "ratio_pct", "observed",
        "valid", "bin_pos"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--days", default=None, help="comma-separated, default all")
    p.add_argument("--batch", type=int, default=4, help="groups in the batch")
    p.add_argument("--epoch", type=int, default=0)
    return p.parse_args()


def check(cond, label, detail=""):
    print("   [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                            ("  " + detail) if detail else ""))
    if not cond:
        raise AssertionError(label)


def refold(ds, item):
    """Independent piece->bin fold for one group, straight from Parquet."""
    day, bucket = item["day"], item["bucket"]
    tab = []
    for f in ds.obs.files("%s/%s" % ("day=" + day, "bucket=" + bucket)):
        tab.append(ds.obs.read(f, _OBS))
    t = pc.concat_tables(tab) if len(tab) > 1 else tab[0]
    cid = t["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    lo = int(np.searchsorted(cid, item["cell_id"], "left"))
    hi = int(np.searchsorted(cid, item["cell_id"], "right"))
    cell = t.slice(lo, hi - lo)
    rows = {r["sample_id"]: r for r in cell.to_pylist()}
    x = np.zeros((len(item["sample_ids"]), ds.n_bins, 3), dtype=np.float64)
    bv = np.zeros((len(item["sample_ids"]), ds.n_bins), dtype=bool)
    for j, sid in enumerate(item["sample_ids"]):
        r = rows[sid]
        if abs(r["dt"] - float(item["delta_t"][j])) > 1e-6:
            raise AssertionError("delta_t mismatch")
        bins = {}
        for b, T, R, O, V in zip(r["bin_pos"], r["T_diff"], r["ratio_pct"],
                                 r["observed"], r["valid"]):
            d = bins.setdefault(int(b), {"T": 0.0, "R": 0.0, "O": False,
                                         "n": 0, "nv": 0})
            d["n"] += 1
            # only T_diff is gated on `valid`; ratio/observed are geometry and
            # a GPS fix, so they stay whatever the pieces say
            d["R"] += float(R) / 10.0
            d["O"] = d["O"] or bool(O)
            if V:
                d["nv"] += 1
                d["T"] += float(T)
        for b, d in bins.items():
            ok = d["nv"] == d["n"]        # every piece of the bin has a time
            bv[j, b] = ok
            x[j, b] = (d["T"] if ok else 0.0, d["R"], float(d["O"]))
    return x, bv


def main():
    a = parse_args()
    days = None if not a.days else [d.strip() for d in a.days.split(",")]
    ds = CellCorpusDataset(a.root, obs_dir=a.obs_dir, days=days, seed=0,
                           epoch=a.epoch)
    print("corpus: %d (day,bucket) partitions, %d bins, m_max=%d"
          % (ds.n_partitions(), ds.n_bins, ds.m_max))
    it = iter(ds)
    items = [next(it) for _ in range(a.batch)]
    b = collate_cells(items, epoch=a.epoch)
    B, M, NB, F = b["x"].shape

    print("\n[1] shapes / dtypes")
    print("   x%s %s  bin_valid%s %s  traj_valid%s %s  delta_t%s %s  mae_mask%s %s"
          % (tuple(b["x"].shape), b["x"].dtype, tuple(b["bin_valid"].shape),
             b["bin_valid"].dtype, tuple(b["traj_valid"].shape), b["traj_valid"].dtype,
             tuple(b["delta_t"].shape), b["delta_t"].dtype,
             tuple(b["mae_mask"].shape), b["mae_mask"].dtype))
    check((B, M, NB, F) == (a.batch, ds.m_max, ds.n_bins, 3), "batch is [B,16,50,3]")
    check(b["x"].dtype == b["delta_t"].dtype == torch.float32 and
          b["bin_valid"].dtype == b["traj_valid"].dtype ==
          b["mae_mask"].dtype == torch.bool,
          "x/delta_t float32; masks bool")

    print("\n[2] group sizes and padding contract")
    sizes = np.array([len(it["sample_ids"]) for it in items])
    Ks = np.array([it["K"] for it in items])
    print("   group sizes %s   K %s" % (sizes.tolist(), Ks.tolist()))
    check(bool((sizes >= 4).all() and (sizes <= M).all()), "4 <= size <= 16")
    check(bool((sizes <= Ks).all()), "size <= K (K>16 must be split)")
    check(bool((Ks[sizes < M] >= sizes[sizes < M]).all()), "no group over K")
    pad = torch.ones((B, M), dtype=torch.bool)
    for i, size in enumerate(sizes):
        pad[i, :size] = False
    check(bool((b["x"][pad] == 0).all()), "pad slots have x == 0")
    check(bool((~b["mae_mask"][pad]).all()), "pad slots never masked")
    n_pad = int(pad.sum())
    print("   padded slots %d / %d" % (n_pad, B * M))

    print("\n[3] bin / trajectory masks")
    nb = b["bin_valid"].sum(-1)
    tv = int(b["traj_valid"].sum())
    print("   valid bins per trajectory: min %d max %d mean %.2f"
          % (int(nb[b["traj_valid"]].min()), int(nb.max()),
             float(nb[b["traj_valid"]].float().mean())))
    check(bool((nb[b["traj_valid"]] >= 1).all()), "every valid trajectory has >=1 bin")
    check(bool(b["traj_valid"][~pad].all()), "every real trajectory has >=1 valid bin")
    check(bool((nb[~b["traj_valid"]] == 0).all()), "invalid trajectory has no bins")
    # only T_diff is gated on the mask: an invalid bin may still carry the
    # geometry (ratio) and the GPS flag (observed)
    check(bool((b["x"][..., 0][~b["bin_valid"]] == 0).all()),
          "invalid bins have T_diff == 0")
    gi = ~b["bin_valid"]
    print("   invalid bins carrying ratio>0: %d, observed>0: %d"
          % (int((b["x"][..., 1][gi] > 0).sum()), int((b["x"][..., 2][gi] > 0).sum())))
    check(bool((nb <= ds.n_bins).all()), "bins <= 50")
    print("   trajectories: valid %d, invalid %d" % (tv, int((~b["traj_valid"]).sum())))

    print("\n[4] delta_t")
    d = b["delta_t"][b["traj_valid"]]
    print("   range [%.3f, %.3f)  mean %.2f" % (float(d.min()), float(d.max()),
                                               float(d.mean())))
    check(bool((d >= 0).all() and (d < 600).all()), "delta_t in [0, 600)")

    print("\n[5] 50%% whole-trajectory MAE mask")
    nv = b["traj_valid"].sum(-1).numpy()
    nm = b["mae_mask"].sum(-1).numpy()
    print("   per group: valid %s -> masked %s" % (nv.tolist(), nm.tolist()))
    check(bool((b["mae_mask"] & ~b["traj_valid"]).sum() == 0), "mask only on valid")
    thin = nv < 3
    check(bool((nm[thin] == 0).all()), "groups with <3 valid left unmasked")
    exp = np.minimum(np.maximum(1, np.round(0.5 * nv).astype(int)), nv - 2)
    check(bool((nm[~thin] == exp[~thin]).all()), "masked ~= 50%, with >=2 visible")
    check(bool(((nv - nm)[~thin] >= 2).all()), "at least two trajectories visible")

    print("\n[6] mask reproducibility (same epoch) and epoch sensitivity")
    again = collate_cells(items, epoch=a.epoch)
    check(bool((again["mae_mask"] == b["mae_mask"]).all()), "same epoch -> same mask")
    changed = []
    for e in range(a.epoch + 1, a.epoch + 9):
        other = collate_cells(items, epoch=e)
        changed.append((other["mae_mask"] != b["mae_mask"]).any(-1))
    n_changed = int(torch.stack(changed).any(0).sum())
    print("   mask positions change within next 8 epochs for %d/%d groups" % (n_changed, B))
    check(n_changed > 0, "different epoch can change mask positions")

    print("\n[7] independent piece->bin refold (plain dicts vs bincount)")
    for i, item in enumerate(items):
        rx, rb = refold(ds, item)
        m = len(item["sample_ids"])
        dx = np.abs(b["x"][i, :m].numpy().astype(np.float64) - rx).max()
        check(bool(dx <= 5e-7) and bool((b["bin_valid"][i, :m].numpy() == rb).all()),
              "group %d (%d traj, K=%d) folds piece-for-piece"
              % (i, m, item["K"]), "max|x diff| = %.1e" % dx)

    print("\n[8] K semantics")
    for i, item in enumerate(items):
        check(item["K"] >= len(item["sample_ids"]),
              "group %d K=%d >= size=%d" % (i, item["K"], len(item["sample_ids"])))
    print("\nALL CHECKS PASS  (%d groups, %d trajectories, %d valid bins)"
          % (B, int(b["traj_valid"].sum()), int(b["bin_valid"].sum())))


if __name__ == "__main__":
    main()
