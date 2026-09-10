"""Trace one real cell training group back to raw HDFS piece by piece."""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.fs as pfs

sys.path.insert(0, ".")
from target_link_v1.data.cell_corpus import CellCorpusDataset  # noqa: E402


def _numeric(value):
    return value is not None and not math.isnan(float(value))


def _same_float(a, b, atol=5e-7):
    if not _numeric(a) and not _numeric(b):
        return True
    return _numeric(a) and _numeric(b) and abs(float(a) - float(b)) <= atol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="local corpus root")
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--raw-base", required=True)
    p.add_argument("--seed", type=int, default=20260910)
    a = p.parse_args()

    ds = CellCorpusDataset(a.corpus, obs_dir=a.obs_dir, seed=a.seed, epoch=0)
    item = next(iter(ds))
    day, bucket = "day=" + item["day"], "bucket=" + item["bucket"]
    tables = [ds.obs.read(f, None) for f in ds.obs.files(day + "/" + bucket)]
    obs = pc.concat_tables(tables) if len(tables) > 1 else tables[0]
    rows = {r["sample_id"]: r for r in obs.to_pylist()
            if r["cell_id"] == item["cell_id"] and r["sample_id"] in item["sample_ids"]}
    if set(rows) != set(item["sample_ids"]):
        raise AssertionError("group sample_ids do not resolve to observation rows")

    hours = sorted({sid.rsplit("#", 1)[-1] for sid in item["sample_ids"]})
    if any(len(hour) != 10 or not hour.isdigit() for hour in hours):
        raise ValueError("sample_id does not end in YYYYMMDDHH")
    fs, raw_base = pfs.FileSystem.from_uri(a.raw_base)
    files = []
    for hour in hours:
        root = raw_base.rstrip("/") + "/event_hour=" + hour
        files.extend(i.path for i in fs.get_file_info(
            pfs.FileSelector(root, recursive=False))
                     if i.type == pfs.FileType.File and i.path.endswith(".parquet"))
    if not files:
        raise ValueError("no raw parquet for " + ",".join(hours))
    raw_ds = pds.dataset(files, filesystem=fs, format="parquet")
    filt = pds.field("sample_id").isin(item["sample_ids"]) & (pds.field("seg_mark") == 1)
    cols = ["map_version", "target_link_id", "sample_id", "seg_idx", "bin_idx",
            "sub_idx", "t_ref", "T_cum", "T_diff", "ratio", "observed"]
    raw = raw_ds.to_table(columns=cols, filter=filt).to_pylist()
    grouped = {}
    for r in raw:
        key = (r["map_version"], r["target_link_id"], int(r["seg_idx"]), r["sample_id"])
        grouped.setdefault(key, []).append(r)

    multi_bins = partial_bins = invalid_pieces = long_seg = 0
    for j, sid in enumerate(item["sample_ids"]):
        got = rows[sid]
        key = (got["map_version"], got["target_link_id"], int(got["seg_idx"]), sid)
        pieces = grouped.get(key)
        if not pieces:
            raise AssertionError("raw pieces missing for " + sid)
        pieces.sort(key=lambda r: (
            int(r["bin_idx"]) - 50 * int(r["seg_idx"]) - 10, int(r["sub_idx"])))
        expected = []
        starts = []
        t_refs = []
        by_bin = {}
        for r in pieces:
            bp = int(r["bin_idx"]) - 50 * int(r["seg_idx"]) - 10
            if not 0 <= bp < 50:
                raise AssertionError("raw bin_pos outside [0,49]")
            valid = _numeric(r["T_diff"])
            t = np.float32(r["T_diff"]) if valid else np.float32(np.nan)
            ratio_pct = int(math.floor(float(r["ratio"]) * 10.0 + 0.5))
            expected.append((bp, t, ratio_pct, bool(r["observed"]), valid))
            d = by_bin.setdefault(bp, {"n": 0, "R": 0, "valid": True})
            d["n"] += 1
            d["R"] += ratio_pct
            d["valid"] = d["valid"] and valid
            invalid_pieces += int(not valid)
            if _numeric(r["T_cum"]) and valid:
                starts.append(float(r["T_cum"]) - float(r["T_diff"]))
            if r["t_ref"] is not None:
                t_refs.append(float(r["t_ref"]))
        actual = list(zip(got["bin_pos"], got["T_diff"], got["ratio_pct"],
                          got["observed"], got["valid"]))
        if len(actual) != got["n_pieces"] or len(actual) != len(expected):
            raise AssertionError("ragged length mismatch for " + sid)
        for i, (want, have) in enumerate(zip(expected, actual)):
            if (int(have[0]) != want[0] or not _same_float(have[1], want[1]) or
                    int(have[2]) != want[2] or bool(have[3]) != want[3] or
                    bool(have[4]) != want[4]):
                raise AssertionError("piece mismatch %s index %d: %r != %r"
                                     % (sid, i, have, want))
        dt = np.float32(max(t_refs) + min(starts) - float(item["window"]))
        if not _same_float(got["dt"], dt, atol=1e-4):
            raise AssertionError("delta_t mismatch for %s: %r != %r" % (sid, got["dt"], dt))
        if not 0 <= float(got["dt"]) < 600:
            raise AssertionError("delta_t outside [0,600)")
        multi_bins += sum(d["n"] > 1 for d in by_bin.values())
        partial_bins += sum(d["R"] < 10 for d in by_bin.values())
        long_seg += int(got["seg_idx"] > 0)

        # Compare the raw-derived bin fold with the reader tensor as well.
        for bp, d in by_bin.items():
            indices = [i for i, e in enumerate(expected) if e[0] == bp]
            valid = all(expected[i][4] for i in indices)
            td = sum(float(expected[i][1]) for i in indices) if valid else 0.0
            ratio = sum(expected[i][2] for i in indices) / 10.0
            observed = any(expected[i][3] for i in indices)
            np.testing.assert_allclose(item["x"][j, bp], [td, ratio, observed], atol=5e-7)
            assert bool(item["bin_valid"][j, bp]) == valid

    print("RAW GROUP CHECK PASS")
    print("group_id=%s cell_id=%s K=%d size=%d day=%s bucket=%s hours=%s"
          % (item["group_id"], item["cell_id"], item["K"], len(item["sample_ids"]),
             item["day"], item["bucket"], ",".join(hours)))
    print("raw_pieces=%d multi_piece_bins=%d partial_ratio_bins=%d invalid_pieces=%d seg_idx_gt0_trajectories=%d"
          % (len(raw), multi_bins, partial_bins, invalid_pieces, long_seg))


if __name__ == "__main__":
    main()
