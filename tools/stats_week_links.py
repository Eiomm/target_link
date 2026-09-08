"""One-off 7-day link/trajectory census over the raw beijing_week_biz table.

Counts at PASS level (a "pass" = one (traj_id, target_link_id), i.e. one vehicle
crossing one target link once) using seg_mark==1 rows only — the same row
semantics tools/build_curves_spark.py filters on. Pure Spark SQL, no executor
python (minipy3 is enough).

Outputs to <out>/:
  link_stats.parquet   per target_link_id: n_passes (distinct sample_id),
                       n_trajs (distinct traj_id)
  day_counts.parquet   per event_day: distinct links / distinct trajs / passes
  summary.json.d       overall distinct links / trajs / passes + day table

Run local (one hour):
  env -u SPARK_HOME ... python tools/stats_week_links.py \
    --inputs 'data/raw_hdfs/event_hour=2026082007/part-*.parquet' \
    --out data/_stats/week_local --master 'local[8]'
Run yarn (full 7 days): MODE=yarn INPUT_GLOB=... (see submit_week_stats_yarn.sh)
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="glob of raw parquet (hdfs:// or local)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--master", default="local[8]")
    ap.add_argument("--driver-memory", default="6g")
    ap.add_argument("--shuffle-partitions", type=int, default=1000)
    args = ap.parse_args()

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (SparkSession.builder.appName("tl_week_links_census")
             .master(args.master)
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = (spark.read.parquet(args.inputs)
          # projection only; day = Beijing-local calendar date from t_enter
          # (works for HDFS partitioned & local copies alike)
          .selectExpr("sample_id", "traj_id", "target_link_id", "seg_mark",
                      "t_enter")
          .where("seg_mark = 1")                      # target-link pass rows
          .drop("seg_mark")
          .withColumn("day", F.date_format(
              F.from_unixtime(F.col("t_enter") + 8 * 3600), "yyyyMMdd")
              .cast("int")))
    # one row per 10 m bin -> collapse to one row per pass
    ps = df.select("sample_id", "traj_id", "target_link_id", "day").distinct()

    # per-link census
    per_link = (ps.groupBy("target_link_id").agg(
        F.count("sample_id").alias("n_passes"),
        F.countDistinct("traj_id").alias("n_trajs")))
    per_link.write.mode("overwrite").parquet(f"{args.out}/link_stats.parquet")

    # per-day + overall census
    per_day = (ps.groupBy("day").agg(
        F.countDistinct("target_link_id").alias("n_links"),
        F.countDistinct("traj_id").alias("n_trajs"),
        F.count("sample_id").alias("n_passes"))
        .orderBy("day"))
    per_day.write.mode("overwrite").parquet(f"{args.out}/day_counts.parquet")
    day_rows = per_day.collect()
    overall = ps.agg(
        F.countDistinct("target_link_id").alias("distinct_links"),
        F.countDistinct("traj_id").alias("distinct_trajs"),
        F.count("sample_id").alias("passes"))
    o = overall.head()

    meta = {
        "distinct_target_links_7d": int(o["distinct_links"]),
        "distinct_trajs_7d": int(o["distinct_trajs"]),
        "passes_7d": int(o["passes"]),
        "per_day": [{"day": int(r["day"]), "n_links": int(r["n_links"]),
                     "n_trajs": int(r["n_trajs"]), "n_passes": int(r["n_passes"])}
                    for r in day_rows],
    }
    spark.createDataFrame([(json.dumps(meta, ensure_ascii=False),)], "s STRING") \
        .coalesce(1).write.mode("overwrite").text(f"{args.out}/summary.json.d")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
