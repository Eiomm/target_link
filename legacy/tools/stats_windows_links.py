"""Per-link trajectory (passage) count distribution, read straight from HDFS.

A "trajectory on a link" is one passage = one distinct sample_id. The same
passage shows up in several rows of window_curves (one per sub_id it covers,
and one per anchor window its bins straddle), so the count must be a DISTINCT
count over the whole corpus, not a row count and not a sum of per-window
counts (the latter inflates by every passage that crosses a window boundary).

Exactness: keys are (link id, sample_id) compared as fixed-width bytes, so no
hashing and no collisions. Read in parallel, deduped per file, then merged with
one in-place sort of a preallocated structured array.

  link id  = (map_version, target_link_id) -- build_windows_spark's LINK_GROUP
             identity; a link id may exist under two map versions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.fs as fs
import pyarrow.parquet as pq

COLS = ["map_version", "target_link_id", "sample_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="corpus root, local path or hdfs:// URI")
    p.add_argument("--out", required=True, help="output directory for results")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-files", type=int, default=0, help="debug: cap file count")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--sid-width", type=int, default=0,
                   help="fixed key width; 0 = measure it with a sample_id-only pass")
    return p.parse_args()


def filesystem(data):
    """Return (fs, curves_dir, label). Local paths keep the plain glob behaviour."""
    if data.startswith("hdfs://"):
        f, base = fs.FileSystem.from_uri(data)
        curves = base.rstrip("/") + "/window_curves"
        infos = f.get_file_info(fs.FileSelector(curves, recursive=False))
        return f, curves, [s.path for s in infos
                           if s.is_file and os.path.basename(s.path).startswith("part-")
                           and s.path.endswith(".parquet")]
    import glob
    curves = os.path.join(data, "window_curves")
    return None, curves, sorted(glob.glob(os.path.join(curves, "part-*.parquet")))


def sid_bytes(col, width):
    """(n,) fixed-width byte key: every sample_id NUL-padded to `width`.

    sample_ids are 38..50 ASCII chars with no NUL, so padding is unambiguous and
    bytewise comparison is exact -- no hash, no collisions.
    """
    import pyarrow as pa
    a = col.combine_chunks()
    n = len(a)
    offs = np.frombuffer(a.buffers()[1],
                         dtype=np.int64 if pa.types.is_large_string(a.type) else np.int32
                         )[a.offset:a.offset + n + 1]
    raw = np.frombuffer(a.buffers()[2], dtype=np.uint8)
    lens = offs[1:] - offs[:-1]
    if len(lens) and lens.max() > width:
        raise ValueError("sample_id longer than the fixed key width %d" % width)
    mat = np.zeros((n, width), dtype=np.uint8)
    row = np.repeat(np.arange(n), lens)
    col_idx = np.arange(int(lens.sum())) - np.repeat(offs[:-1], lens)
    mat[row, col_idx] = raw[offs[0]:offs[-1]]
    return mat.view("V%d" % width).ravel()


def read_file(args):
    """Read one file; return (dict values, per-row code, sid keys, per-file distinct)."""
    path, f, width = args
    tbl = pq.read_table(path, filesystem=f, columns=COLS)
    keys = pc.binary_join_element_wise(
        tbl["map_version"].combine_chunks(),
        tbl["target_link_id"].combine_chunks(), "|")
    enc = pc.dictionary_encode(keys)
    # Per-file distinct (link, sample_id) pairs: count_distinct on the joined
    # pair, not on `keys` alone -- the latter counts distinct LINKS per file,
    # which says how spread links are across files, not how much a pair repeats.
    pair_key = pc.binary_join_element_wise(keys, tbl["sample_id"].combine_chunks(), "#")
    return (path, enc.dictionary.to_pylist(),
            enc.indices.to_numpy(zero_copy_only=False),
            sid_bytes(tbl["sample_id"], width),
            int(pc.count_distinct(pair_key).as_py()))


def main():
    a = parse_args()
    t0 = time.time()
    f, curves, files = filesystem(a.data)
    if a.max_files:
        files = files[:a.max_files]
    if not files:
        raise SystemExit("no window_curves part files under %s" % curves)
    print("files=%d  list %.1fs" % (len(files), time.time() - t0), flush=True)

    import concurrent.futures as cf

    def footer(path):
        return pq.ParquetFile(path, filesystem=f).metadata.num_rows

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = sum(ex.map(footer, files))
    print("rows=%d  footers %.1fs" % (rows, time.time() - t0), flush=True)

    # Record width = longest sample_id in the corpus, measured on a sample_id-only
    # pass. The key is compared as fixed-width bytes, so an under-sized width is a
    # hard error rather than a silent collision; measuring beats guessing, and this
    # pass only touches one column.
    width = a.sid_width
    if not width:
        t0 = time.time()

        def longest(path):
            col = pq.read_table(path, filesystem=f, columns=["sample_id"])
            return int(pc.max(pc.binary_length(col["sample_id"])).as_py())

        with cf.ThreadPoolExecutor(a.workers) as ex:
            width = max(ex.map(longest, files))
        print("max sample_id width=%d  %.1fs" % (width, time.time() - t0), flush=True)

    rec = np.zeros(rows, dtype=np.dtype([("link", np.int32), ("sid", "V%d" % width)]))
    link_ids = {}          # "map_version|target_link_id" -> int32
    link_names = []        # int32 -> the same key string
    pos = 0
    sum_file_distinct = 0

    # Every row is appended (no per-file dedupe: intra-file duplicates are ~0.03%
    # and the numpy sort it would cost dominates the run); the single global sort
    # below does the exact distinct. Files are processed in bounded batches because
    # a completed Future keeps its result alive -- submitting all 800 at once held
    # ~13 GB of dead per-file buffers and got the job OOM-killed at 24.7 GB RSS.
    chunk = max(2 * a.workers, 32)
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for start in range(0, len(files), chunk):
            batch = [(p, f, width) for p in files[start:start + chunk]]
            for path, values, codes, sids, n_file_distinct in ex.map(read_file, batch):
                n = len(codes)
                sum_file_distinct += n_file_distinct
                # Map this file's small dictionary into global link ids (main
                # thread only: the dict must not be touched concurrently).
                remap = np.empty(len(values), dtype=np.int32)
                for i, key in enumerate(values):
                    gid = link_ids.get(key)
                    if gid is None:
                        gid = link_ids[key] = len(link_names)
                        link_names.append(key)
                    remap[i] = gid
                rec["link"][pos:pos + n] = remap[codes]
                rec["sid"][pos:pos + n] = sids
                pos += n
                done += 1
            del batch
            print("  read %d/%d  rows=%d  links=%d  %.1fs"
                  % (done, len(files), pos, len(link_names), time.time() - t0), flush=True)
    rec = rec[:pos]
    print("read %.1fs  rows=%d  links=%d  per-file distinct sum=%d"
          % (time.time() - t0, pos, len(link_names), sum_file_distinct), flush=True)

    t0 = time.time()
    rec.sort()
    print("global sort %.1fs" % (time.time() - t0), flush=True)
    keep = np.ones(pos, dtype=bool)
    keep[1:] = rec[1:] != rec[:-1]
    n_distinct = int(keep.sum())
    counts = np.bincount(rec["link"][keep].astype(np.int64), minlength=len(link_names))
    print("distinct (link, sample_id) pairs = %d  (per-file sum %d, duplicate rate %.3f%%)"
          % (n_distinct, sum_file_distinct,
             100.0 * (sum_file_distinct - n_distinct) / max(sum_file_distinct, 1)), flush=True)

    # ---- report -----------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    order = np.argsort(-counts, kind="stable")
    with open(os.path.join(a.out, "link_trajectory_counts.csv"), "w") as fh:
        fh.write("map_version,target_link_id,n_trajectories\n")
        for i in order:
            mv, link = link_names[i].split("|", 1)
            fh.write("%s,%s,%d\n" % (mv, link, counts[i]))

    nz = counts[counts > 0]
    edges = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 10 ** 12]
    hist = [(edges[i], edges[i + 1], int(((nz >= edges[i]) & (nz < edges[i + 1])).sum()))
            for i in range(len(edges) - 1)]
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    quant = {("p%d" % q): float(np.percentile(nz, q)) for q in qs}
    summary = dict(
        corpus=a.data, files=len(files), rows=int(rows),
        distinct_pairs=int(n_distinct), per_file_distinct_sum=int(sum_file_distinct),
        links_total=len(link_names), links_with_trajectories=int(len(nz)),
        links_zero=int((counts == 0).sum()),
        trajectories_total=int(nz.sum()), mean_per_link=float(nz.mean()),
        median_per_link=float(np.median(nz)), max_per_link=int(nz.max()),
        quantiles=quant,
        histogram=[dict(lo=lo, hi=(None if hi >= 10 ** 12 else hi), links=n)
                   for lo, hi, n in hist],
        top=[dict(key=link_names[i], n=int(counts[i])) for i in order[:a.top]],
    )
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
