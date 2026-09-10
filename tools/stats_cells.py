"""Per (link, seg, 10-min window) trajectory-count census over the raw corridor table.

The new spec (md/最新讨论想法.md) makes the training unit a CELL:

    cell = (map_version, target_link_id, seg_idx, window)
    window = floor(t_seg_enter / 600) * 600,  t_seg_enter = t_ref + (T_cum - T_diff)

at the segment's FIRST bin -- `T_cum` is the END of a piece, so `T_cum - T_diff`
is its start (verified on raw: within one bin the two piece rows satisfy
T_cum_a - T_diff_a == T_cum_b - T_diff_b - ... i.e. T_cum is cumulative to the
piece end). T_diff is already ratio-scaled per piece, so no extra weighting.

What is counted: K = number of trajectory OBSERVATIONS in the cell. One
observation = one `sample_id`. Upstream `sample_id` is
`traj_id#target_link_id#pass_idx#event_hour`, so a re-entry is already a
distinct sample_id -- the spec's "same traj_id twice = two observations" needs
no extra dedup. A (link, sample_id, seg_idx) triple yields exactly one entry
time (min over the segment's bins), hence exactly one cell, so K is a plain
row count after collapsing bins to one row per observation -- no countDistinct
over strings anywhere.

Rows are taken from the marked span (`seg_mark = 1`, identical to `seg_idx >= 0`).
NOTE: `link_id == target_link_id` must NOT be used as the span selector -- on
raw, 21.9% of pairs then pick up the target id 40-100 m past the span (measured,
see md/9.9progress.md §20).

T_diff/T_cum NaN (0.60% / 0.47% of raw rows) are treated as "no timing": they
are dropped from the min, so the entry time is the first bin that has one. They
are NOT imputed (decision 2026-09-09); downstream they become valid=0.

Outputs to <out>/:
  cells.parquet   (map_version, target_link_id, seg_idx, window, K) -- the
                  sampling frame for the build step; one row per non-empty cell
  k_hist.parquet  (K, n_cells) -- exact K histogram, so any percentile/coverage
                  can be recomputed without re-reading the corpus
  summary.json.d  meta + K percentiles + coverage table for the requested M list

Run yarn: MODE=yarn (see scripts/submit_cells_stats_yarn.sh). Pure Spark SQL,
no executor python, so minipy3 is enough.
"""
from __future__ import annotations

import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", required=True, help="raw parquet glob (hdfs:// or local)")
    p.add_argument("--out", required=True)
    p.add_argument("--master", default="local[8]")
    p.add_argument("--driver-memory", default="6g")
    p.add_argument("--shuffle-partitions", type=int, default=2000)
    p.add_argument("--window-seconds", type=int, default=600)
    p.add_argument("--m-list", default="1,2,4,8,16,32,64,128,256,512",
                   help="sample sizes M for the coverage table")
    p.add_argument("--no-cells", action="store_true",
                   help="skip writing cells.parquet (histogram/coverage only)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ms = [int(x) for x in a.m_list.split(",") if x.strip()]

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (SparkSession.builder.appName("tl_cells_census")
             .master(a.master)
             .config("spark.driver.memory", a.driver_memory)
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    def clean(c):
        return F.col(c).isNotNull() & ~F.isnan(F.col(c))

    raw = (spark.read.parquet(a.inputs)
           .selectExpr("map_version", "target_link_id", "sample_id", "seg_idx",
                       "seg_mark", "t_ref", "T_cum", "T_diff")
           .where("seg_mark = 1"))

    # one row per (link, observation, segment); entry = first bin with timing
    seg = (raw
           .withColumn("t_start", F.when(clean("T_cum") & clean("T_diff"),
                                         F.col("T_cum") - F.col("T_diff")))
           .groupBy("map_version", "target_link_id", "sample_id", "seg_idx")
           .agg(F.min("t_start").alias("t_start_min"),
                F.max("t_ref").alias("t_ref")))

    ws = float(a.window_seconds)
    obs = (seg.where(F.col("t_start_min").isNotNull())
           .withColumn("t_enter", F.col("t_ref") + F.col("t_start_min"))
           .withColumn("window", (F.floor(F.col("t_enter") / ws) * ws).cast("long")))

    cells = (obs.groupBy("map_version", "target_link_id", "seg_idx", "window")
             .agg(F.count(F.lit(1)).alias("K")))
    # the cell table is reused by four aggregations below; persisting it to
    # parquet (rather than caching 1e8 rows) keeps the driver/executor heap flat
    if a.no_cells:
        cells = cells.cache()
    else:
        cells.write.mode("overwrite").parquet(f"{a.out}/cells.parquet")
        cells = spark.read.parquet(f"{a.out}/cells.parquet").cache()

    n_cells = cells.count()
    n_obs = cells.agg(F.sum("K").alias("s")).head()["s"]
    n_windows = cells.select("window").distinct().count()
    n_segs = cells.select("map_version", "target_link_id", "seg_idx").distinct().count()
    n_links = cells.select("map_version", "target_link_id").distinct().count()

    # exact K histogram -> percentiles / coverage in the driver
    hist = (cells.groupBy("K").count()
            .withColumnRenamed("count", "n_cells").orderBy("K").collect())
    ks = [int(r["K"]) for r in hist]
    ns = [int(r["n_cells"]) for r in hist]
    (spark.createDataFrame([(int(k), int(n)) for k, n in zip(ks, ns)], "K LONG, n_cells LONG")
     .write.mode("overwrite").parquet(f"{a.out}/k_hist.parquet"))

    def pct(q):
        """K percentile weighted by n_cells (q in [0,100])."""
        want = q / 100.0 * n_cells
        acc = 0
        for k, n in zip(ks, ns):
            acc += n
            if acc >= want:
                return k
        return ks[-1]

    cov = []
    for m in ms:
        cells_ge = sum(n for k, n in zip(ks, ns) if k >= m)
        obs_ge = sum(k * n for k, n in zip(ks, ns) if k >= m)
        obs_sampled = sum(min(k, m) * n for k, n in zip(ks, ns) if k >= m)
        cov.append({
            "M": m,
            "cells_ge_M": cells_ge,
            "cells_cov": round(cells_ge / max(n_cells, 1), 6),
            "obs_ge_M": obs_ge,
            "obs_cov": round(obs_ge / max(n_obs, 1), 6),
            # one sample per cell draws min(K, M) observations
            "obs_used_1sample_per_cell": obs_sampled,
            "obs_used_cov": round(obs_sampled / max(n_obs, 1), 6),
            # floor(K/M) samples per cell uses (almost) every observation
            "samples_floor_K_over_M": sum((k // m) * n for k, n in zip(ks, ns) if k >= m),
        })

    # per-window shape (diurnal effect on how often a cell can be filled)
    agg = [F.count(F.lit(1)).alias("n_cells"), F.sum("K").alias("n_obs"),
           F.max("K").alias("k_max")]
    for m in ms:
        agg.append(F.sum(F.when(F.col("K") >= m, 1).otherwise(0)).alias("cells_ge_%d" % m))
    per_window = (cells.groupBy("window").agg(*agg).orderBy("window").collect())

    edges = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 10 ** 12]
    bucketed = [{"lo": edges[i], "hi": (None if edges[i + 1] >= 10 ** 12 else edges[i + 1]),
                 "cells": sum(n for k, n in zip(ks, ns) if edges[i] <= k < edges[i + 1])}
                for i in range(len(edges) - 1)]

    meta = {
        "inputs": a.inputs,
        "window_seconds": a.window_seconds,
        "n_cells": n_cells, "n_obs": n_obs,
        "n_links": n_links, "n_segments": n_segs, "n_windows": n_windows,
        "mean_K": round(n_obs / max(n_cells, 1), 3),
        "k_max": ks[-1] if ks else 0,
        "k_quantiles": {"p%d" % q: pct(q) for q in
                        (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)},
        "k_histogram": bucketed,
        "coverage": cov,
        "per_window": [{"window": int(r["window"]),
                        "n_cells": int(r["n_cells"]),
                        "n_obs": int(r["n_obs"]),
                        "k_max": int(r["k_max"]),
                        "cells_ge": {str(m): int(r["cells_ge_%d" % m]) for m in ms}}
                       for r in per_window],
    }
    spark.createDataFrame([(json.dumps(meta, ensure_ascii=False),)], "s STRING") \
        .coalesce(1).write.mode("overwrite").text(f"{a.out}/summary.json.d")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_window"},
                     ensure_ascii=False, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
