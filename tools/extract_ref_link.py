"""Extract one target_link's raw rows into a self-contained reference package.

Use when an agent (or a reviewer) needs to see the real data shape but cannot
read HDFS. Reads the flat samples table, keeps every row of one
target_link_id, and writes:

  <out>/samples_flat.parquet   all columns, all rows of that link
  <out>/samples_nested.jsonl   one line per sample, list fields nested as in
                               the producer's nested export (bin_link_ids /
                               bin_link_code / ratio / T_cum / T_diff /
                               bin_pt_ts / seg_mark / seg_idx / bin_link_obs)
  <out>/README.md              link stats + the contract checks that passed

Only the standard library + pyarrow are needed (no Spark, no pandas required
for the write path).
"""
from __future__ import annotations

import argparse
import json
import math
import os


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, help="comma-separated Parquet files/globs")
    p.add_argument("--link", required=True, help="target_link_id to extract")
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--nested-limit", type=int, default=0,
                   help="max samples in samples_nested.jsonl; 0 = all")
    p.add_argument("--batch-rows", type=int, default=1_000_000)
    return p


def expand(paths):
    """Expand local globs; pass hdfs:// paths through untouched."""
    import glob as _glob
    out = []
    for p in paths:
        if p.startswith("hdfs://") or not any(ch in p for ch in "*?["):
            out.append(p)
        else:
            out.extend(sorted(_glob.glob(p)))
    if not out:
        raise SystemExit("no input files matched: %s" % paths)
    return out


def rows_for_link(paths, link, batch_rows):
    """Stream batches, keep rows whose target_link_id matches. Memory-bounded."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    kept = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows):
            hit = batch.filter(pc.equal(batch.column("target_link_id"), link))
            if hit.num_rows:
                kept.append(hit)
    if not kept:
        raise SystemExit("no rows for target_link_id=%s in the given inputs" % link)
    return pa.Table.from_batches(kept)


def nest_sample(rows):
    """One sample's rows -> the producer's nested (one row per sample) shape."""
    rows = sorted(rows, key=lambda r: (r["bin_idx"], r["sub_idx"]))
    n_bins = int(rows[0]["n_bins"])
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        bins[int(r["bin_idx"])].append(r)
    out = {
        "sample_id": rows[0]["sample_id"],
        "traj_id": rows[0]["traj_id"],
        "pass_idx": rows[0]["pass_idx"],
        "target_link_id": rows[0]["target_link_id"],
        "map_version": rows[0]["map_version"],
        "bin_size_m": rows[0]["bin_size_m"],
        "L_link_m": rows[0]["L_link_m"],
        "n_bins": n_bins,
        "n_segs": rows[0]["n_segs"],
        "t_ref": rows[0]["t_ref"],
        "t_enter": rows[0]["t_enter"],
        "y_travel_s": rows[0]["y_travel_s"],
        "corridor_time_s": rows[0]["corridor_time_s"],
        "X_in_m": rows[0]["X_in_m"],
        "X_out_m": rows[0]["X_out_m"],
        "truncated_in": rows[0]["truncated_in"],
        "truncated_out": rows[0]["truncated_out"],
        "n_pts_corr": rows[0]["n_pts_corr"],
        "n_pts_raw": rows[0]["n_pts_raw"],
        "n_pts_kept": rows[0]["n_pts_kept"],
        "n_gap_bins": rows[0]["n_gap_bins"],
        "bin_link_ids": [], "bin_link_code": [], "ratio": [], "T_cum": [], "T_diff": [],
        "bin_pt_ts": [], "seg_mark": [], "seg_idx": [], "bin_link_obs": [],
    }
    for cell in bins:
        out["bin_link_ids"].append([c["link_id"] for c in cell])
        out["bin_link_code"].append([c["link_code"] for c in cell])
        out["ratio"].append([c["ratio"] for c in cell])
        out["T_cum"].append([c["T_cum"] for c in cell])
        out["T_diff"].append([c["T_diff"] for c in cell])
        # per-bin fields (not expanded to sub-segments)
        out["bin_pt_ts"].append(cell[0]["bin_pt_ts"])
        out["seg_mark"].append(cell[0]["seg_mark"])
        out["seg_idx"].append(cell[0]["seg_idx"])
        out["bin_link_obs"].append(cell[0]["observed"])
    return out


def checks(table, link):
    """Contract checks a reference package must satisfy to be trustworthy."""
    rows = table.to_pylist()
    by_sample = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], []).append(r)
    res = {"n_rows": len(rows), "n_samples": len(by_sample), "link": link}
    bad_closure, bad_obs, bad_cont, bad_seg = [], [], [], []
    for sid, rs in by_sample.items():
        seg1 = [r for r in rs if r["seg_mark"] == 1]
        if not seg1:
            continue
        # 1. geometry closes: sum(bin_size*ratio) over target bins == L_link_m
        dist = sum(r["bin_size_m"] * r["ratio"] for r in seg1)
        if abs(dist - seg1[0]["L_link_m"]) > 0.01:
            bad_closure.append((sid, dist, seg1[0]["L_link_m"]))
        # 2. observed == 1 <=> bin_pt_ts != '-1'
        for r in rs:
            if (r["observed"] == 1) != (r["bin_pt_ts"] not in (None, "", "-1")):
                bad_obs.append((sid, r["bin_idx"]))
                break
        # 3. bin_idx strictly contiguous inside seg_mark==1
        idx = sorted({r["bin_idx"] for r in seg1})
        if idx != list(range(idx[0], idx[0] + len(idx))):
            bad_cont.append(sid)
        # 4. n_segs == ceil(L/500); seg_idx in [-1, n_segs)
        n_segs = seg1[0]["n_segs"]
        if n_segs != math.ceil(seg1[0]["L_link_m"] / 500.0):
            bad_seg.append((sid, n_segs))
        elif any(not (-1 <= r["seg_idx"] < n_segs) for r in rs):
            bad_seg.append((sid, "seg_idx out of range"))
    res.update(
        closure_ok=len(bad_closure) == 0, closure_bad=bad_closure[:3],
        observed_ok=len(bad_obs) == 0, observed_bad=bad_obs[:3],
        contiguous_ok=len(bad_cont) == 0, contiguous_bad=bad_cont[:3],
        n_segs_ok=len(bad_seg) == 0, n_segs_bad=bad_seg[:3])
    return res


def main():
    a = parser().parse_args()
    if os.path.exists(a.out):
        raise SystemExit("output exists; choose a new directory: " + a.out)
    paths = expand([p.strip() for p in a.inputs.split(",") if p.strip()])
    table = rows_for_link(paths, a.link, a.batch_rows)
    os.makedirs(a.out)
    import pyarrow.parquet as pq
    pq.write_table(table, os.path.join(a.out, "samples_flat.parquet"), compression="snappy")

    rows = table.to_pylist()
    by_sample = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], []).append(r)
    sids = sorted(by_sample)
    limit = a.nested_limit if a.nested_limit > 0 else len(sids)
    with open(os.path.join(a.out, "samples_nested.jsonl"), "w") as fh:
        for sid in sids[:limit]:
            fh.write(json.dumps(nest_sample(by_sample[sid]), ensure_ascii=False) + "\n")

    res = checks(table, a.link)
    lines = [
        "# 单 link 参考数据包 — target_link_id = %s" % a.link,
        "",
        "由 `tools/extract_ref_link.py` 从扁平 samples 表抽取，自包含（无需 HDFS/Spark）。",
        "",
        "## 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `samples_flat.parquet` | 该 link 的全部原始行（所有列，一行 = 一个 bin 子片段） |",
        "| `samples_nested.jsonl` | 每个 sample 一行，序列字段按 bin 嵌套（producer 嵌套格式；%d 行） |" % limit,
        "",
        "## 规模",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 行数 | %d |" % res["n_rows"],
        "| 样本数（经过次数） | %d |" % res["n_samples"],
        "| 输入 | %s |" % a.inputs,
        "",
        "## 契约校验（全部应为 true）",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        "| 几何闭合 Σ(bin_size×ratio) == L_link_m | %s |" % res["closure_ok"],
        "| observed==1 ⟺ bin_pt_ts != '-1' | %s |" % res["observed_ok"],
        "| seg_mark==1 内 bin_idx 严格连续 | %s |" % res["contiguous_ok"],
        "| n_segs == ceil(L/500) 且 seg_idx 合法 | %s |" % res["n_segs_ok"],
        "",
        "失败样例（最多 3 条）：closure=%s observed=%s contiguous=%s n_segs=%s"
        % (res["closure_bad"], res["observed_bad"], res["contiguous_bad"], res["n_segs_bad"]),
        "",
        "## 怎么读",
        "",
        "```python",
        "import pyarrow.parquet as pq",
        "t = pq.read_table('%s/samples_flat.parquet')" % a.out,
        "print(t.schema)",
        "df = t.to_pandas()",
        "one = df[df.sample_id == df.sample_id.iloc[0]].sort_values(['bin_idx', 'sub_idx'])",
        "```",
        "",
        "字段语义见 `md/data.md`（生产端权威定义）与 `数据说明.md`（本项目实测补充）。",
        "",
    ]
    with open(os.path.join(a.out, "README.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("[ref] wrote %s: %d rows / %d samples" % (a.out, res["n_rows"], res["n_samples"]))
    print("[ref] checks: closure=%s observed=%s contiguous=%s n_segs=%s"
          % (res["closure_ok"], res["observed_ok"], res["contiguous_ok"], res["n_segs_ok"]))


if __name__ == "__main__":
    main()
