"""Build the V1 canonical corpus for the trajectory-MAE pipeline.

Spec: md/最新讨论想法.md (unit = cell), updated 2026-09-10:
  M_max=16, K_min=3, deterministic shuffle + ceil(K/16) even split for K>16,
  every trajectory used at most once, 50% whole-trajectory mask (training side).

Three stages, each reading the previous stage's HDFS output (so the big pass
over raw runs exactly once):

  obs    raw -> observations/   ONE ROW PER TRAJECTORY OBSERVATION (= one
         (cell, sample_id)). Ragged, no 16-slot padding materialised:
             cell_id   int64   xxhash64("map_version|target_link_id|seg|window")
             sample_id string  upstream id = traj_id#link#pass#event_hour
             dt        float   t_seg_enter - window_start, in [0, 600)
             n_pieces  int16   ragged length (= #pieces of this observation
                               inside the 500m segment; <=50 bins, a bin split
                               across links adds a piece, measured max 51)
             T_diff[]  float32 per-piece passage time, NaN kept as NaN
             ratio_pct[] uint8 per-piece share of its 10m bin, = ratio*10 in 1..10
                               (raw ratio takes exactly 10 values, verified)
             observed[] bool   raw `observed` flag (a GPS fix exists)
             valid[]   bool    T_diff is not NaN (decision: no imputation)
             bin_pos[] uint8   THE spatial token position, 0..49. A trajectory
                               traverses a 500m segment = 50 x 10m bins, and
                               raw `bin_idx` is the corridor-global bin number
                               while `seg_idx` is the 500m segment index. Both
                               grid and offset are hard constants in the data:
                                 (bin_idx - 10) // 50 == seg_idx   (100% of rows,
                                 6 partitions over 4 days, 57M rows checked)
                               so bin_pos = bin_idx - 50*seg_idx - 10 lands in
                               [0,49] exactly, and pieces of one bin that were
                               split across links share a bin_pos. The grid is
                               ABSOLUTE (shared by every trajectory of a cell):
                               at seg_idx=3, 1959/1960 samples start at
                               bin_idx=160, one joins mid-segment at 163 -- so
                               never re-normalise per trajectory, that would
                               shift that one by 3 bins. Collate asserts 0..49.
         plus map_version/target_link_id/seg_idx/window so the later stages
         never re-read raw; partitioned by (day, bucket=xxhash64(cell_id)%NB),
         sorted within partition by cell_id.
  cells  observations/ -> cells/ (cell_id, keys, window, K) + the acceptance
         baseline (exact K histogram, per-window shape).
  groups observations/ -> training_groups_k3/ (group_id, cell_id, group_idx,
         group_size, K, sample_ids[]) under K_min / M_max. Rebuild this alone
         when the sampling policy changes -- the corpus is policy-free.

Run: MODE=yarn (scripts/submit_build_corpus_yarn.sh). Only Spark SQL + a few
`transform` lambdas, so minipy3 is enough.
"""
from __future__ import annotations

import argparse
import json
import traceback

KEYS = ["map_version", "target_link_id", "seg_idx", "sample_id"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", required=True,
                   help="comma-separated raw parquet globs (hdfs:// or local)")
    p.add_argument("--out", required=True)
    p.add_argument("--obs-dir", default="observations_v2",
                   help="sub-dir of --out holding the ragged corpus; use a "
                        "different one to rebuild obs next to the live copy "
                        "(the row set does not change, so cells/ and "
                        "training_groups/ stay valid)")
    p.add_argument("--groups-dir", default="training_groups_k3",
                   help="sub-dir of --out holding policy-dependent groups")
    p.add_argument("--stages", default="obs,cells,groups")
    p.add_argument("--master", default="local[8]")
    p.add_argument("--driver-memory", default="6g")
    p.add_argument("--shuffle-partitions", type=int, default=8000)
    p.add_argument("--obs-partitions", type=int, default=1024,
                   help="output partitions of observations/, = #(day,bucket) dirs")
    p.add_argument("--buckets", type=int, default=128,
                   help="hash buckets on cell_id; groups/ uses the same so a "
                        "group's observations are co-located")
    p.add_argument("--window-seconds", type=int, default=600)
    p.add_argument("--m-max", type=int, default=16)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--max-records-per-file", type=int, default=4_000_000)
    return p.parse_args()


def _run(a):
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    ws = float(a.window_seconds)

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (SparkSession.builder.appName("tl_build_corpus")
             .master(a.master)
             .config("spark.driver.memory", a.driver_memory)
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             # `day` must be the BEIJING calendar day of the window epoch
             .config("spark.sql.session.timeZone", "Asia/Shanghai")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    paths = [x for x in a.inputs.split(",") if x.strip()]

    # xxhash64 is signed and this pyspark build has no F.pmod, so take the
    # non-negative remainder by hand; buckets=128 is a power of two anyway
    def bucket_of(col: str):
        nb = F.lit(a.buckets)
        return (((F.col(col) % nb) + nb) % nb).cast("int")

    def dump(name: str, meta: dict) -> None:
        (spark.createDataFrame([(json.dumps(meta, ensure_ascii=False),)], "s STRING")
         .coalesce(1).write.mode("overwrite").text(f"{a.out}/_qc/{name}.json.d"))
        print("[%s] %s" % (name, json.dumps(meta, ensure_ascii=False)), flush=True)

    def check(df, name: str) -> None:
        # touches .schema to force analysis: a bad expression fails here in
        # seconds instead of after a 10-minute scan, and log aggregation is
        # off on this cluster so the only cheap feedback loop is the driver
        df.schema
        print("[%s] plan analysed" % name, flush=True)

    if "obs" in stages:
        raw = (spark.read.parquet(*paths)
               .selectExpr("map_version", "target_link_id", "sample_id", "seg_idx",
                           "seg_mark", "bin_idx", "sub_idx", "t_ref", "T_cum",
                           "T_diff", "ratio", "observed")
               .where("seg_mark = 1"))
        num = lambda c: F.col(c).isNotNull() & ~F.isnan(F.col(c))
        piece = (raw
                 .withColumn("has_time", num("T_cum") & num("T_diff"))
                 .withColumn("t_start", F.when(F.col("has_time"),
                                               F.col("T_cum") - F.col("T_diff")))
                 # decision: T_diff=NaN is NOT imputed, it becomes valid=0
                 .withColumn("valid", num("T_diff"))
                 .withColumn("ratio_pct", F.round(F.col("ratio") * 10).cast("byte"))
                 .withColumn("obs_flag", F.col("observed").cast("boolean"))
                 # spatial token position inside the 500m segment: see the
                 # module docstring, (bin_idx-10)//50 == seg_idx is a data
                 # invariant, so this is in [0,49] or the data broke
                 .withColumn("bin_pos",
                             (F.col("bin_idx") - 50 * F.col("seg_idx") - 10).cast("byte")))

        # one row per (link, segment, observation); pieces sorted by (bin, sub)
        # T_diff is cast to float32 here: it is ~8 bytes of the ~11 per piece
        # and the model consumes float32 anyway, so the cast pays for bin_pos
        g = (piece.groupBy(*KEYS).agg(
            F.min("t_start").alias("t_start_min"),
            F.max("t_ref").alias("t_ref"),
            F.array_sort(F.collect_list(F.struct(
                F.col("bin_pos").alias("b"), F.col("sub_idx").alias("s"),
                F.col("T_diff").cast("float").alias("T"),
                F.col("ratio_pct").alias("R"),
                F.col("obs_flag").alias("O"), F.col("valid").alias("V")))).alias("pieces")))

        obs = (g.where(F.col("t_start_min").isNotNull())
               .withColumn("t_seg_enter", F.col("t_ref") + F.col("t_start_min"))
               .withColumn("window", (F.floor(F.col("t_seg_enter") / ws) * ws).cast("long"))
               .withColumn("dt", (F.col("t_seg_enter") - F.col("window")).cast("float"))
               .withColumn("cell_id", F.xxhash64(F.concat_ws(
                   "|", F.col("map_version"), F.col("target_link_id"),
                   F.col("seg_idx").cast("string"), F.col("window").cast("string"))))
               .withColumn("day", F.from_unixtime("window", "yyyyMMdd"))
               .withColumn("bucket", bucket_of("cell_id")))

        out = obs.select(
            "cell_id", "sample_id", "dt",
            F.size("pieces").cast("short").alias("n_pieces"),
            F.transform("pieces", lambda x: x["T"]).alias("T_diff"),
            F.transform("pieces", lambda x: x["R"]).alias("ratio_pct"),
            F.transform("pieces", lambda x: x["O"]).alias("observed"),
            F.transform("pieces", lambda x: x["V"]).alias("valid"),
            F.transform("pieces", lambda x: x["b"]).alias("bin_pos"),
            "map_version", "target_link_id", "seg_idx", "window", "day", "bucket")

        check(out, "obs")
        (out.repartition(a.obs_partitions, "day", "bucket")
            .sortWithinPartitions("cell_id", "sample_id")
            .write.mode("overwrite").partitionBy("day", "bucket")
            .option("maxRecordsPerFile", a.max_records_per_file)
            .parquet(f"{a.out}/{a.obs_dir}"))
        dump("obs", {"inputs": paths, "out": f"{a.out}/{a.obs_dir}",
                     "buckets": a.buckets, "obs_partitions": a.obs_partitions,
                     "window_seconds": a.window_seconds,
                     "cell_id": "xxhash64(map_version|target_link_id|seg_idx|window)",
                     "ratio_pct": "round(ratio*10), uint8 in 1..10",
                     "bin_pos": "bin_idx - 50*seg_idx - 10, uint8 in 0..49",
                     "T_diff": "float32, NaN kept"})

    if "cells" in stages:
        corpus = spark.read.parquet(f"{a.out}/{a.obs_dir}")
        cells = (corpus.select("cell_id", "map_version", "target_link_id", "seg_idx", "window")
                 .groupBy("cell_id", "map_version", "target_link_id", "seg_idx", "window")
                 .agg(F.count(F.lit(1)).alias("K"))
                 .withColumn("day", F.from_unixtime("window", "yyyyMMdd")))
        check(cells, "cells")
        (cells.write.mode("overwrite").partitionBy("day")
            .option("maxRecordsPerFile", a.max_records_per_file)
            .parquet(f"{a.out}/cells"))

        hist = (cells.groupBy("K").count().withColumnRenamed("count", "n_cells")
                .orderBy("K").collect())
        ks = [int(r["K"]) for r in hist]
        ns = [int(r["n_cells"]) for r in hist]
        n_cells = sum(ns)
        n_obs = sum(k * n for k, n in zip(ks, ns))
        (spark.createDataFrame([(int(k), int(n)) for k, n in zip(ks, ns)], "K LONG, n_cells LONG")
         .write.mode("overwrite").parquet(f"{a.out}/k_hist.parquet"))

        def pct(q):
            want = q / 100.0 * n_cells
            acc = 0
            for k, n in zip(ks, ns):
                acc += n
                if acc >= want:
                    return k
            return ks[-1]

        # pieces-per-observation shape (ragged length, needed by collate)
        pcs = corpus.select("n_pieces").agg(F.min("n_pieces").alias("mn"),
                                            F.max("n_pieces").alias("mx"),
                                            F.avg("n_pieces").alias("av"),
                                            F.sum(F.col("n_pieces").cast("long")).alias("tot")).head()
        per_day = (cells.groupBy("day").agg(F.count(F.lit(1)).alias("n_cells"),
                                            F.sum("K").alias("n_obs"),
                                            F.max("K").alias("k_max"))
                   .orderBy("day").collect())
        edges = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 10 ** 12]
        dump("cells", {
            "n_cells": n_cells, "n_obs": n_obs,
            "n_links": cells.select("map_version", "target_link_id").distinct().count(),
            "n_segments": cells.select("map_version", "target_link_id", "seg_idx").distinct().count(),
            "n_windows": cells.select("window").distinct().count(),
            "mean_K": round(n_obs / max(n_cells, 1), 3), "k_max": ks[-1] if ks else 0,
            "k_quantiles": {"p%d" % q: pct(q) for q in (50, 75, 90, 95, 99)},
            "k_histogram": [{"lo": edges[i], "hi": (None if edges[i + 1] >= 10 ** 12 else edges[i + 1]),
                             "cells": sum(n for k, n in zip(ks, ns) if edges[i] <= k < edges[i + 1])}
                            for i in range(len(edges) - 1)],
            "pieces_per_obs": {"min": int(pcs["mn"]), "max": int(pcs["mx"]),
                               "mean": round(float(pcs["av"]), 3), "total": int(pcs["tot"])},
            "per_day": [{"day": r["day"], "n_cells": int(r["n_cells"]),
                         "n_obs": int(r["n_obs"]), "k_max": int(r["k_max"])} for r in per_day],
        })

    if "groups" in stages:
        M, KMIN = a.m_max, a.k_min
        c = (spark.read.parquet(f"{a.out}/{a.obs_dir}")
             .select("cell_id", "sample_id", "window")
             # deterministic shuffle: hash of (cell_id, sample_id), no RNG
             .withColumn("h", F.xxhash64(F.concat_ws(
                 "|", F.col("cell_id").cast("string"), F.col("sample_id")))))
        w = Window.partitionBy("cell_id").orderBy("h")
        o = (c.withColumn("rn", (F.row_number().over(w) - 1).cast("int"))
             .withColumn("K", F.count(F.lit(1)).over(Window.partitionBy("cell_id")).cast("int"))
             .where(F.col("K") >= KMIN))
        ng = F.ceil(F.col("K") / M).cast("int")              # ceil(K/16) groups
        base = F.floor(F.col("K") / ng).cast("int")
        rem = (F.col("K") % ng).cast("int")
        flen = base + 1                                       # first `rem` groups are 1 longer
        o = o.withColumn("group_idx", F.when(
            F.col("rn") < rem * flen, F.floor(F.col("rn") / flen))
            .otherwise(rem + F.floor((F.col("rn") - rem * flen) / base)).cast("int"))

        groups = (o.groupBy("cell_id", "group_idx")
                  .agg(F.first("K").alias("K"), F.first("window").alias("window"),
                       F.count(F.lit(1)).alias("group_size"),
                       F.transform(F.array_sort(F.collect_list(F.struct("h", "sample_id"))),
                                   lambda x: x["sample_id"]).alias("sample_ids"))
                  .withColumn("group_id", F.concat_ws("_", F.col("cell_id").cast("string"),
                                                      F.col("group_idx").cast("string")))
                  .withColumn("day", F.from_unixtime("window", "yyyyMMdd"))
                  .withColumn("bucket", bucket_of("cell_id")))
        check(groups, "groups")
        (groups.repartition(a.obs_partitions, "day", "bucket")
            .sortWithinPartitions("cell_id", "group_idx")
            .write.mode("overwrite").partitionBy("day", "bucket")
            .option("maxRecordsPerFile", a.max_records_per_file)
            .parquet(f"{a.out}/{a.groups_dir}"))

        st = (groups.agg(F.count(F.lit(1)).alias("n_groups"),
                         F.sum("group_size").alias("n_used"),
                         F.max("group_size").alias("max_group_size"),
                         F.avg("group_size").alias("avg_group_size")).head())
        gs = (groups.groupBy("group_size").count().orderBy("group_size").collect())
        qc_name = "groups_" + a.groups_dir.replace("/", "_")
        dump(qc_name, {"groups_dir": a.groups_dir, "m_max": M, "k_min": KMIN,
                        "n_groups": int(st["n_groups"]),
                        "n_used_obs": int(st["n_used"]),
                        "max_group_size": int(st["max_group_size"]),
                        "avg_group_size": round(float(st["avg_group_size"]), 3),
                        "group_size_hist": {str(int(r["group_size"])): int(r["count"]) for r in gs}})

    return spark


def main() -> None:
    a = parse_args()
    spark = None
    try:
        spark = _run(a)
    except BaseException:
        tb = traceback.format_exc()
        print(tb, flush=True)
        # log aggregation is off on this cluster, so persist the traceback
        # where it can be read back from the laptop
        try:
            if spark is not None:
                (spark.createDataFrame([(tb,)], "s STRING").coalesce(1)
                 .write.mode("overwrite").text(f"{a.out}/_qc/failure.txt.d"))
        except Exception as e:  # noqa: BLE001
            print("could not persist traceback: %s" % e, flush=True)
        raise
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
