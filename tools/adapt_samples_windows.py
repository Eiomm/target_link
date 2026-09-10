"""Adapt beijing_week_biz/samples into the causal window contract.

Derives the four missing columns from fields already present in the raw
corridor table; nothing is invented:

  bin_end_ts  = t_ref + T_cum          (audited 9.8 §19: fix time, ~0.1s)
  bin_start_ts = bin_end_ts - T_diff
  components   = physical link_id pieces can SPLIT one bin_idx (6m + 4m in one
                 10m bin); rows sharing a bin_idx are merged (distance summed,
                 end = last component's t_ref+T_cum, observed = max) so the
                 event key (link, sample, bin_idx) is unique
  spatial_start_m = prefix sum of merged bin distances over the pass's
                    contiguous seg_mark==1 bins (geometry closes: sum -
                    L_link_m has median -1mm, p1..p99 within +-5mm)
  available_ts = bin_end of the nearest downstream observed==1 bin; a bin's
                    duration is only final once the closing GPS fix exists
                    (9.8 §19). This is a causal LOWER bound: it ignores
                    vehicle->server pipeline delay. Bins with no downstream
                    fix (tail extrapolation, ~7%) are causally unavailable
                    online and are dropped here, not silently kept.

Rows with nonfinite T_diff/t_ref (~0.4%) are dropped. All counts land in
adapt_meta.json.d; build_windows_spark still enforces its own contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json


def positive(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, help="comma-separated Parquet paths/globs")
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--master", default="local[2]")
    p.add_argument("--partitions", type=positive, default=200)
    p.add_argument("--shuffle-partitions", type=positive, default=800)
    return p


def adapt(raw, a):
    """Return (events_df, audit dict). Events carry the explicit 4 columns."""
    from pyspark.sql import functions as F
    from pyspark.sql import Window
    needed = ["map_version", "target_link_id", "sample_id", "bin_idx", "ratio",
              "bin_size_m", "T_diff", "L_link_m", "observed", "seg_mark", "t_ref", "T_cum",
              "link_id"]
    missing = sorted(set(needed) - set(raw.columns))
    if missing:
        raise ValueError("Missing raw columns: %s" % missing)
    tgt = raw.where(F.col("seg_mark") == 1)
    part = ["map_version", "target_link_id", "sample_id"]
    # A target_link is produced as several physical link components that can
    # SPLIT one bin (e.g. bin 18 = 6m of comp A + 4m of comp B). First work at
    # component-row level, then merge rows that share a bin_idx, so the event
    # key (link, sample, bin_idx) is unique, as the window contract requires.
    w_row = Window.partitionBy(*part).orderBy("bin_idx", F.col("t_ref") + F.col("T_cum"))
    rows = (tgt.withColumn("row_end", (F.col("t_ref") + F.col("T_cum")).cast("double"))
               .withColumn("distance_m", F.col("bin_size_m") * F.col("ratio"))
               # nearest downstream (or own) observed fix closes this component row
               .withColumn("avail_row", F.min(F.when(F.col("observed") == 1, F.col("row_end")))
                           .over(w_row.rowsBetween(0, Window.unboundedFollowing))))
    merged = (rows.groupBy(*part, "bin_idx").agg(
        F.sum("distance_m").alias("distance_m"),
        F.max("row_end").alias("bin_end_ts"),
        F.min(F.col("row_end") - F.col("T_diff")).alias("bin_start_ts"),
        # merged duration is fixed once the LAST component row's end is fixed;
        # max() is conservative when a fix lands mid-bin (row avail < bin end)
        F.max("avail_row").alias("available_ts"),
        F.max(F.col("observed")).cast("int").alias("observed"),
        F.max("bin_size_m").alias("bin_size_m"),
        F.max("L_link_m").alias("L_link_m")))
    w = Window.partitionBy(*part).orderBy("bin_idx")
    df = merged.withColumn("spatial_start_m",
                           F.coalesce(F.sum("distance_m").over(
                               w.rowsBetween(Window.unboundedPreceding, -1)), F.lit(0.0)))
    # df carries both window functions plus the component merge -- by far the
    # expensive part. Every later action (stats / closure / write) used to
    # recompute it from the raw parquet; cache it once instead.
    from pyspark import StorageLevel
    df = df.persist(StorageLevel.DISK_ONLY)
    quality = (F.col("distance_m").isNotNull() & (F.col("distance_m") > 0)
               & ~F.isnan("distance_m") & ~F.isnan("bin_end_ts") & ~F.isnan("bin_start_ts")
               & ~F.isnan("spatial_start_m") & ~F.isnan("bin_size_m")
               & (F.col("bin_size_m") > 0) & (F.col("bin_end_ts") > F.col("bin_start_ts"))
               & (F.abs("bin_end_ts") < float("inf")))
    kept = quality & F.col("available_ts").isNotNull() & ~F.isnan("available_ts") \
        & (F.abs("available_ts") < float("inf"))
    stats = df.agg(F.count("*").alias("rows"), F.sum(quality.cast("long")).alias("q"),
                   F.sum(kept.cast("long")).alias("k")).first()
    good = df.where(kept)
    # availability must not precede the interval it certifies (float safety)
    good = good.withColumn("available_ts", F.greatest("available_ts", "bin_end_ts"))
    closure = good.groupBy("map_version", "target_link_id", "sample_id").agg(
        F.sum("distance_m").alias("s"), F.max("L_link_m").alias("L"))
    closure = closure.select(
        F.percentile_approx((F.col("s") - F.col("L")) * 1000.0, [0.001, 0.5, 0.999], 1000)
        .alias("closure_mm"))
    events = good.select(
        *[F.col(k).cast("string").alias(k) for k in ["map_version", "target_link_id", "sample_id"]],
        F.col("bin_idx").cast("long").alias("bin_idx"),
        (F.col("distance_m") / F.col("bin_size_m")).cast("double").alias("ratio"),
        F.col("bin_size_m").cast("double").alias("bin_size_m"),
        (F.col("bin_end_ts") - F.col("bin_start_ts")).cast("double").alias("T_diff"),
        F.col("L_link_m").cast("double").alias("L_link_m"),
        F.col("spatial_start_m").cast("double").alias("spatial_start_m"),
        F.col("bin_start_ts").cast("double").alias("bin_start_ts"),
        F.col("bin_end_ts").cast("double").alias("bin_end_ts"),
        F.col("available_ts").cast("double").alias("available_ts"),
        F.col("observed").cast("int").alias("observed"),
        F.lit(1).cast("int").alias("seg_mark"))
    audit = dict(seg1_rows=int(stats["rows"]),
                 dropped_quality=int(stats["rows"] - stats["q"]),
                 dropped_tail_no_fix=int(stats["q"] - stats["k"]),
                 closure_mm=closure.first()[0])
    return events, audit


def main():
    a = parser().parse_args()
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master(a.master).appName("adapt-samples-windows")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        path = spark._jvm.org.apache.hadoop.fs.Path(a.out)
        if path.getFileSystem(spark._jsc.hadoopConfiguration()).exists(path):
            raise ValueError("Output already exists; choose a new version directory: " + a.out)
        raw = spark.read.parquet(*[p.strip() for p in a.inputs.split(",") if p.strip()])
        events, audit = adapt(raw, a)
        # events is a projection of df.where(kept) with no further filtering, so
        # the written row count is exactly the audit's kept count -- a separate
        # count() would rescan the whole cached frame for a number already known
        # (identity holds on the 25h run: 2296301711-9018711-133095514=rows).
        n = audit["seg1_rows"] - audit["dropped_quality"] - audit["dropped_tail_no_fix"]
        (events.repartition(a.partitions, "sample_id").sortWithinPartitions(
            "map_version", "target_link_id", "sample_id", "bin_idx")
         .write.mode("errorifexists").parquet(a.out + "/events"))
        with open(__file__, "rb") as source:
            sha = hashlib.sha256(source.read()).hexdigest()
        meta = dict(vars(a), format="target_link_adapted_events_v1", audit=audit,
                    rows=int(n), builder_sha256=sha,
                    availability="derived: nearest downstream observed fix bin_end "
                                 "(causal lower bound, pipeline delay excluded)",
                    tail_bins_without_fix="dropped (causally unavailable online)")
        spark.createDataFrame([(json.dumps(meta),)], "value string").coalesce(1).write.text(
            a.out + "/adapt_meta.json.d")
        print("[adapt] wrote %d events to %s; audit=%s" % (n, a.out, audit))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
