"""Build causal, partial-passage snapshots using native Spark SQL only.

The strict input contract is documented in md/窗口管线使用.md. No whole-pass
label, duration, or speed participates in filtering. Epoch timestamps are UTC.
An observation must be immutable; conflicting revisions fail instead of letting
future corrections silently replace the values used at an earlier anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

KEY = ["map_version", "target_link_id", "sample_id", "bin_idx"]
GROUP = ["map_version", "target_link_id", "anchor_ts"]
FORMAT = "target_link_windows_v1"


def positive(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, help="comma-separated Parquet paths/globs")
    p.add_argument("--out", required=True)
    p.add_argument("--master", default="local[2]")
    p.add_argument("--anchor-start", type=int, required=True, help="inclusive UTC epoch seconds")
    p.add_argument("--anchor-end", type=int, required=True, help="exclusive UTC epoch seconds")
    p.add_argument("--lookback-seconds", type=positive, default=600)
    p.add_argument("--stride-seconds", type=positive, default=60)
    p.add_argument("--max-passes", type=int, default=0, help="per full link per anchor; 0=all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sub-length-m", type=float, default=200.0)
    p.add_argument("--max-bins", type=positive, default=40)
    p.add_argument("--max-speed", type=float, default=33.3)
    p.add_argument("--partitions", type=positive, default=200)
    p.add_argument("--shuffle-partitions", type=positive, default=800)
    p.add_argument("--availability-column", default="available_ts")
    p.add_argument("--position-column", default="spatial_start_m")
    p.add_argument("--time-source", choices=["explicit", "cumulative"], default="explicit",
                   help="cumulative asserts producer contract: end=t_ref+T_cum")
    p.add_argument("--links", help="optional map_version,target_link_id table for empty snapshots")
    return p


def validate_args(a):
    if a.lookback_seconds <= 0 or a.stride_seconds <= 0 or a.max_bins <= 0:
        raise ValueError("lookback, stride and max-bins must be positive")
    if a.anchor_start >= a.anchor_end:
        raise ValueError("anchor-start must be before anchor-end")
    if a.anchor_start % a.stride_seconds or a.anchor_end % a.stride_seconds:
        raise ValueError("anchor bounds must align with stride-seconds on the epoch grid")
    if a.max_passes < 0:
        raise ValueError("max-passes must be >=0")
    if not math.isfinite(a.sub_length_m) or a.sub_length_m <= 0:
        raise ValueError("sub-length-m must be finite and positive")
    if not math.isfinite(a.max_speed) or a.max_speed <= 0:
        raise ValueError("max-speed must be finite and positive")


def finite(c):
    from pyspark.sql import functions as F
    return c.isNotNull() & ~F.isnan(c) & (F.abs(c) < float("inf"))


def prepare_events(raw, a):
    """Strict, immutable target-link component contract. Return events + audit."""
    from pyspark.sql import functions as F
    needed = KEY + ["ratio", "bin_size_m", "T_diff", "L_link_m", "observed", "seg_mark",
                    a.availability_column, a.position_column]
    needed += ["t_ref", "T_cum"] if a.time_source == "cumulative" else ["bin_start_ts", "bin_end_ts"]
    missing = sorted(set(needed) - set(raw.columns))
    if missing:
        raise ValueError("Missing causal input columns: %s. Do not substitute t_enter or "
                         "bin_end for availability; see md/窗口管线使用.md" % missing)
    target = raw.where(F.col("seg_mark") == 1)
    end = (F.col("t_ref").cast("double") + F.col("T_cum").cast("double")
           if a.time_source == "cumulative" else F.col("bin_end_ts").cast("double"))
    start = (end - F.col("T_diff").cast("double") if a.time_source == "cumulative"
             else F.col("bin_start_ts").cast("double"))
    df = target.select(
        *[F.col(k).cast("string").alias(k) for k in KEY[:-1]],
        F.col("bin_idx").cast("long").alias("bin_idx"),
        F.col("ratio").cast("double").alias("ratio_space"),
        F.col("bin_size_m").cast("double").alias("bin_size_m"),
        F.col("T_diff").cast("double").alias("duration"),
        F.col("L_link_m").cast("double").alias("link_length_m"),
        F.col(a.position_column).cast("double").alias("position_m"),
        F.col(a.availability_column).cast("double").alias("available_ts"),
        F.col("observed").cast("int").alias("observed"),
        start.alias("bin_start_ts"), end.alias("bin_end_ts"))
    # No quality decision is based on a full passage or its future suffix.
    df = df.withColumn("distance_m", F.col("bin_size_m") * F.col("ratio_space"))
    ok = F.lit(True)
    for k in KEY:
        ok = ok & F.col(k).isNotNull()
    for k in KEY[:-1]:
        ok = ok & (F.length(F.col(k)) > 0)
    ok = ok & (F.col("bin_idx") >= 0)
    for k in ["ratio_space", "bin_size_m", "duration", "link_length_m", "position_m",
              "available_ts", "bin_start_ts", "bin_end_ts", "distance_m"]:
        ok = ok & finite(F.col(k))
    ok = (ok & (F.col("ratio_space") > 0) & (F.col("ratio_space") <= 1.000001)
          & (F.col("bin_size_m") > 0) & (F.col("duration") > 0)
          & (F.col("link_length_m") > 0) & (F.col("position_m") >= 0)
          & (F.col("position_m") + F.col("distance_m") <= F.col("link_length_m") + 1e-3)
          & (F.col("available_ts") >= F.col("bin_end_ts"))
          & (F.col("bin_end_ts") > F.col("bin_start_ts"))
          & (F.abs(F.col("bin_end_ts") - F.col("bin_start_ts") - F.col("duration")) <= 1e-3)
          & F.col("observed").isin(0, 1)
          & (F.col("distance_m") / F.col("duration") <= a.max_speed))
    df = df.withColumn("_ok", F.coalesce(ok, F.lit(False)))
    audit = df.agg(F.count("*").alias("target_rows"),
                   F.sum(F.when(~F.col("_ok"), 1).otherwise(0)).alias("rejected_rows")).first().asDict()
    # Missing availability is not a speed outlier: fail rather than silently
    # turning a source with no trustworthy timestamps into an empty corpus.
    if df.where(~finite(F.col("available_ts")) | (F.col("available_ts") < 0)).limit(1).count():
        raise ValueError("Missing/nonfinite available_ts on target rows; upstream audit required")
    good = df.where("_ok").drop("_ok")
    values = [k for k in good.columns if k not in KEY + ["available_ts"]]
    variants = good.groupBy(*KEY).agg(F.countDistinct(F.struct(*values)).alias("n"))
    if variants.where("n > 1").limit(1).count():
        raise ValueError("Conflicting bin versions/components: normalize immutable events upstream")
    # Duplicate delivery of identical observations: first actual availability.
    events = good.groupBy(*(KEY + values)).agg(F.min("available_ts").alias("available_ts"))
    events = events.withColumn("event_id", F.sha2(F.to_json(F.struct(*KEY)), 256))
    return events, {k: int(v or 0) for k, v in audit.items()}


def window_members(events, a):
    """Expand only to eligible anchors, never to a full link x time cross join."""
    from pyspark.sql import functions as F
    from pyspark.sql import Window
    stride = a.stride_seconds
    first = F.greatest(
        (F.floor(F.col("bin_end_ts") / stride) + 1) * stride,
        F.ceil(F.col("available_ts") / stride) * stride, F.lit(a.anchor_start)).cast("long")
    last = F.least(F.floor((F.col("bin_start_ts") + a.lookback_seconds) / stride) * stride,
                   F.lit(a.anchor_end - stride)).cast("long")
    eligible = events.withColumn("_first", first).withColumn("_last", last).where("_first <= _last")
    members = (eligible.withColumn("anchor_ts", F.explode(F.sequence("_first", "_last", F.lit(stride))))
               .drop("_first", "_last"))
    passes = members.select(*(GROUP + ["sample_id"])).distinct()
    counts = passes.groupBy(*GROUP).agg(F.count("*").alias("n_passes_before_cap"))
    if a.max_passes:
        rank = Window.partitionBy(*GROUP).orderBy(
            F.hash(F.lit(a.seed), F.col("sample_id")), F.col("sample_id"))
        passes = passes.withColumn("_rank", F.row_number().over(rank)).where(
            F.col("_rank") <= a.max_passes).drop("_rank")
    counts = counts.join(passes.groupBy(*GROUP).agg(F.count("*").alias("n_passes_kept")), GROUP)
    members = members.join(passes, GROUP + ["sample_id"], "inner")
    members = members.withColumn("snapshot_id", F.sha2(F.to_json(F.struct(*GROUP)), 256))
    # Absolute geometry, never cumsum over surviving bins (which closes holes).
    members = members.withColumn("sub_id", F.floor(F.col("position_m") / a.sub_length_m).cast("int"))
    return members, counts


def build_curves(members, a):
    from pyspark.sql import functions as F
    curve_keys = ["snapshot_id"] + GROUP + ["sample_id", "sub_id"]
    fields = ["position_m", "bin_idx", "duration", "distance_m", "observed", "event_id",
              "bin_start_ts", "bin_end_ts", "available_ts"]
    curves = members.groupBy(*curve_keys).agg(
        F.sort_array(F.collect_list(F.struct(*fields))).alias("bins"),
        F.max("link_length_m").alias("link_length_m"))
    if curves.where(F.size("bins") > a.max_bins).limit(1).count():
        raise ValueError("A curve exceeds max-bins; adjust geometry/config, never silently truncate")
    # This is coverage of the selected observations, not future pass completion.
    return (curves.withColumn("length", F.size("bins"))
            .withColumn("available_distance_m", F.expr("aggregate(bins, 0.0D, (s,x) -> s+x.distance_m)")))


def snapshots(members, counts, links, spark, a):
    from pyspark.sql import functions as F
    stats = members.groupBy(*GROUP).agg(
        F.count("*").alias("n_bin_observations"),
        F.sum("distance_m").alias("distance_observations_m"),
        (F.sum("distance_m") / F.sum("duration")).alias("distance_over_time_speed"))
    table = counts.join(stats, GROUP)
    if links is not None:
        # Explicit static road universe is required for exhaustive empty windows.
        roads = links.select("map_version", "target_link_id").distinct()
        roads = roads.select(*[F.col(k).cast("string").alias(k) for k in roads.columns])
        if members.select("map_version", "target_link_id").distinct().join(
                roads, ["map_version", "target_link_id"], "left_anti").limit(1).count():
            raise ValueError("Input events include roads outside --links")
        anchors = spark.range(a.anchor_start, a.anchor_end, a.stride_seconds).select(
            F.col("id").alias("anchor_ts"))
        table = roads.crossJoin(anchors).join(table, GROUP, "left")
    return (table.fillna(0, subset=["n_passes_before_cap", "n_passes_kept", "n_bin_observations",
                                   "distance_observations_m"])
            .withColumn("empty_flag", F.col("n_passes_kept") == 0)
            .withColumn("snapshot_id", F.sha2(F.to_json(F.struct(*GROUP)), 256))
            .withColumn("lookback_seconds", F.lit(a.lookback_seconds)))


def main():
    a = parser().parse_args()
    validate_args(a)
    from pyspark.sql import SparkSession
    from pyspark import StorageLevel
    spark = (SparkSession.builder.master(a.master).appName("target_link_windows")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        # Fail before any write. Each build gets a new directory, including local.
        path = spark._jvm.org.apache.hadoop.fs.Path(a.out)
        if path.getFileSystem(spark._jsc.hadoopConfiguration()).exists(path):
            raise ValueError("Output already exists; choose a new version directory: " + a.out)
        raw = spark.read.parquet(*[p.strip() for p in a.inputs.split(",") if p.strip()])
        events, audit = prepare_events(raw, a)
        events = events.persist(StorageLevel.DISK_ONLY)
        members, counts = window_members(events, a)
        members = members.persist(StorageLevel.DISK_ONLY)
        if not a.links and not members.limit(1).count():
            raise ValueError("No eligible observations in anchor range; inspect audit, input history, and times")
        curves = build_curves(members, a)
        links = spark.read.parquet(a.links) if a.links else None
        table = snapshots(members, counts, links, spark, a)
        for name, df, order in [("window_curves", curves, ["snapshot_id", "sub_id", "sample_id"]),
                                ("snapshots", table, ["snapshot_id"]),
                                ("window_members", members, ["snapshot_id", "sample_id", "bin_idx"])]:
            (df.repartition(a.partitions, "snapshot_id").sortWithinPartitions(*order)
             .write.mode("errorifexists").option("maxRecordsPerFile", 0)
             .parquet(a.out + "/" + name))
        with open(__file__, "rb") as source:
            builder_sha256 = hashlib.sha256(source.read()).hexdigest()
        meta = dict(vars(a), format=FORMAT, audit=audit, builder_sha256=builder_sha256,
                    availability="producer_supplied; causal contract requires upstream audit",
                    boundary="start>=anchor-W,end<anchor,available<=anchor",
                    snapshot_scope="road_universe" if a.links else "nonempty_only")
        spark.createDataFrame([(json.dumps(meta),)], "value string").coalesce(1).write.text(
            a.out + "/window_meta.json.d")
        # Readers require metadata: a failed intermediate write isn't a valid corpus.
        print("[windows] wrote " + a.out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
