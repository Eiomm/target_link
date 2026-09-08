"""ONE Spark job: raw corridor parquet -> per-(sample, sub) curve shards (v2.1).

Merges the whole pretraining data path into a single cluster pass (repV2 §2.1
streaming variant) — nothing materialises locally, no intermediate npz:

  raw@hdfs
    ├─ dedup window + per-sample window agg -> samples.parquet / link_window
    │   (ingest contract; the supervised line keeps consuming these)
    └─ pure-SQL curve generation (NO executor python — minipy3 has no pandas):
        pre-cum ratio window -> sub assignment (spec §3) -> collect_list of
        sorted per-(sample, sub) structs -> transform() speed/valid arrays,
        aggregate() eff_len — same math as build_profiles.process_shard.

Training streams the shards (target_link_v1/data/pretrain_stream.py); the
streaming trainer derives y/v/len buckets from curves_meta.json.d
(percentile_approx in-cluster) and hour from window_id.

Precision note: speeds keep DOUBLE inputs end-to-end (raw T_diff is double);
the old bins_shards chain cast T_diff to float32 at ingest, so its speeds
carry f32 rounding (measured ≤3.8e-6 m/s, rel 2.5e-7 — this path is the more
accurate one; keys/labels match bit-for-bit).

Run local:
  env -u SPARK_HOME ... python tools/build_curves_spark.py \
      --inputs 'data/raw_hdfs/event_hour=2026082007/part-*.parquet' \
      --out data/curves_spark/smoke3 --master 'local[8]'
Run yarn: three-tier wrapper as scripts/submit_curves_yarn.sh.
"""
from __future__ import annotations

import argparse
import json

RAW_COLS = [
    "sample_id", "target_link_id", "t_enter", "bin_idx",
    "ratio", "T_diff", "observed",
    "y_travel_s", "L_link_m", "seg_mark",
]


def value_at_min_bin(col: str):
    """value at the smallest bin_idx of a sample (ingest 'first' semantics)."""
    from pyspark.sql import functions as F
    s = F.min(F.struct(F.col("bin_idx").alias("_k"), F.col(col).alias("_v")))
    return s.getField("_v").alias(col)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--master", default="local[8]")
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--bin-size-m", type=float, default=10.0)
    ap.add_argument("--l-sub-m", type=float, default=200.0)
    ap.add_argument("--max-bins", type=int, default=40)
    ap.add_argument("--v-invalid-above", type=float, default=33.3)
    ap.add_argument("--min-td", type=float, default=0.1)
    ap.add_argument("--curves-partitions", type=int, default=200)
    ap.add_argument("--shuffle-partitions", type=int, default=800)
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark import StorageLevel

    spark = (SparkSession.builder
             .appName("target_link_curves").master(args.master)
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    L = float(args.l_sub_m)
    bs, cap = float(args.bin_size_m), float(args.v_invalid_above)

    raw = spark.read.parquet(args.inputs).select(*RAW_COLS) \
        .where(F.col("seg_mark") == 1).drop("seg_mark")
    if args.sample_fraction < 1.0:
        raw = raw.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)
    n_rows_raw = raw.count()

    # dedup (sample_id, bin_idx): prefer valid T_diff, then max ratio (ingest)
    td = F.col("T_diff")
    w_dedup = Window.partitionBy("sample_id", "bin_idx") \
        .orderBy((td.isNotNull() & ~F.isnan(td) & (td > 0)).desc(), F.col("ratio").desc())
    dedup = (raw.withColumn("_rn", F.row_number().over(w_dedup))
             .where(F.col("_rn") == 1).drop("_rn")
             .persist(StorageLevel.DISK_ONLY))

    # [a] samples / link_window — native agg, ingest semantics (nanvl sum)
    td_null = F.nanvl(F.col("T_diff"), F.lit(None).cast("double"))
    agg = dedup.groupBy("sample_id").agg(
        value_at_min_bin("target_link_id"), F.sum(td_null).alias("td_target"),
        F.min("bin_idx").alias("bin_idx_min"), F.count(F.lit(1)).alias("n_bins_target"),
        F.sum("observed").alias("n_observed"), F.sum("ratio").alias("eff_len_ratio"),
        value_at_min_bin("t_enter"), value_at_min_bin("y_travel_s"),
        value_at_min_bin("L_link_m"))
    samples = (agg.where(F.col("td_target") > args.min_td)
               .withColumn("v_sample", F.col("L_link_m") / F.col("td_target"))
               .withColumn("observed_ratio", F.col("n_observed")
                           / F.greatest(F.col("n_bins_target"), F.lit(1)))
               .withColumn("window_id",
                           (F.floor(F.col("t_enter") / 3600)).cast("long")))
    samples.write.mode("overwrite").parquet(f"{args.out}/samples.parquet")
    samples.groupBy("target_link_id", "window_id").agg(
        F.mean("v_sample").alias("mean_speed"), F.stddev_samp("v_sample").alias("speed_std"),
        F.count(F.lit(1)).alias("n_trajs"), F.mean("y_travel_s").alias("y_travel_mean")
    ).write.mode("overwrite").parquet(f"{args.out}/link_window.parquet")

    # [b] curves — pure Spark SQL, zero executor python
    w_pre = Window.partitionBy("sample_id").orderBy("bin_idx") \
        .rowsBetween(Window.unboundedPreceding, -1)               # cum(ratio) BEFORE row
    w_all = Window.partitionBy("sample_id").orderBy("bin_idx") \
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    d2 = (dedup
          .withColumn("pre_cum", F.coalesce(F.sum("ratio").over(w_pre), F.lit(0.0)))
          # sample-level values at the min-bin row (static cols: any-row == max)
          .withColumn("link_id", F.max("target_link_id").over(w_all))
          .withColumn("y", F.max("y_travel_s").over(w_all))
          .withColumn("L_link", F.max("L_link_m").over(w_all))
          .withColumn("td", F.sum(td_null).over(w_all))
          .withColumn("n_bins", F.count(F.lit(1)).over(w_all))
          # t_enter/y at the FIRST (min bin_idx) row = ingest value_at_min_bin
          .withColumn("t0", F.first("t_enter").over(w_all))
          .withColumn("n_subs", F.greatest(F.ceil(F.col("L_link") / L), F.lit(1.0)))
          .withColumn("sub", F.least(
              F.greatest(F.floor(F.col("pre_cum") * bs / L), F.lit(0.0)),
              F.col("n_subs") - F.lit(1.0)).cast("int")))

    grouped = d2.groupBy("sample_id", "sub").agg(
        F.sort_array(F.collect_list(F.struct(
            F.col("bin_idx").alias("i"), F.col("ratio").alias("r"),
            F.col("T_diff").alias("t"), F.col("observed").alias("o")))).alias("a"),
        F.max("link_id").alias("link_id"), F.max("y").alias("y"),
        F.max("L_link").alias("L_link"), F.max("td").alias("td_target"),
        F.max("n_bins").alias("n_bins"), F.max("t0").alias("t0"))

    v = f"({bs} * x.r / x.t)"
    bad_td = "x.t IS NULL OR isnan(x.t) OR x.t <= 0"
    v_bad = f"{v} IS NULL OR {v} <= 0 OR {v} > {cap}"
    curves = (grouped
              .where((F.col("td_target") > args.min_td) & (F.size("a") <= args.max_bins))
              .select(
                  "sample_id", "sub", "link_id", "y", "td_target", "n_bins",
                  (F.col("L_link") / F.col("td_target")).alias("v_sample"),
                  (F.floor(F.col("t0") / 3600)).cast("long").alias("window_id"),
                  F.expr(f"transform(a, x -> CASE WHEN {bad_td} OR {v_bad} "
                         f"THEN 0.0 ELSE {v} END)").alias("speeds"),
                  F.expr(f"transform(a, x -> NOT ({bad_td} OR {v_bad}))").alias("valid"),
                  F.expr("transform(a, x -> x.o = 1)").alias("observed"),
                  F.size("a").alias("length"),
                  F.expr("aggregate(a, 0.0D, (acc, x) -> acc + x.r)")
                  .cast("float").alias("eff_len_ratio")))

    # hash(sample_id) partitioning: every curve of a sample lands in ONE shard,
    # so the streaming trainer's shard-level val split stays sample-level
    # (round-robin repartition would leak train samples' curves into val)
    (curves.repartition(args.curves_partitions, F.col("sample_id"))
     .write.mode("overwrite").option("compression", "snappy")
     .parquet(f"{args.out}/curves"))
    n_curves = spark.read.parquet(f"{args.out}/curves").count()

    # bucket edges for the streaming trainer's free labels (percentile_approx)
    def edges(col):
        row = samples.agg(F.percentile_approx(col, [i / 16 for i in range(17)],
                                              10000).alias("p")).head()
        return [round(float(x), 4) for x in row["p"]]

    meta = {"n_rows_raw": int(n_rows_raw), "n_curves": int(n_curves),
            "bin_size_m": bs, "l_sub_m": L, "max_bins": args.max_bins,
            "v_invalid_above": cap,
            "y_edges": edges("y_travel_s"), "v_edges": edges("v_sample"),
            "len_edges": edges("n_bins_target")}
    spark.createDataFrame([(json.dumps(meta, ensure_ascii=False),)], "s STRING") \
        .coalesce(1).write.mode("overwrite").text(f"{args.out}/curves_meta.json.d")
    spark.stop()
    print(f"[curves] wrote {args.out}: samples/link_window (ingest contract) + "
          f"{n_curves:,} curve rows in {args.curves_partitions} shards")


if __name__ == "__main__":
    main()
