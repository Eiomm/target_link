"""Build a small, representative Cell-MAE dataset for local visual analysis.

The output has two layers:
  link_bin_window_stats/  all selected links, every observed 10-minute window, bin stats
  sampled_observations/   two windows/hour, <=N deterministic trajectories/cell
plus selected_links/, selected_link_days/, and manifest.json.d/.  A physical link
is selected by target_link_id alone: map_version rotates during the week and is
retained as a per-day observation attribute.  Row order is deliberately irrelevant.
"""
from __future__ import annotations

import argparse
import json


def parse_days(value):
    days = [x.strip() for x in value.split(",") if x.strip()]
    if not days or any(len(x) != 8 or not x.isdigit() for x in days):
        raise ValueError("--days must be comma-separated YYYYMMDD values")
    return days


def parse_slots(value):
    slots = sorted(set(int(x) for x in value.split(",") if x.strip()))
    if not slots or any(x < 0 or x > 5 for x in slots):
        raise ValueError("--detail-slots must contain 10-minute slots in 0..5")
    return slots


def tier_quotas(total, tiers=4):
    return [total // tiers + int(i < total % tiers) for i in range(tiers)]


def selection_keys():
    """Stable physical-link key used for cross-day eligibility and filtering."""
    return ["target_link_id"]


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, help="corpus_v1 HDFS root")
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--out", required=True, help="new HDFS output directory")
    p.add_argument("--days", default="20260817,20260818,20260819,20260820,20260821,20260822,20260823")
    p.add_argument("--n-links", type=int, default=48)
    p.add_argument("--min-observations", type=int, default=100)
    p.add_argument("--min-active-days", type=int, default=7)
    p.add_argument("--min-hour-slots", type=int, default=24,
                   help="distinct Beijing hour-of-day values required (1..24)")
    p.add_argument("--min-active-hours", type=int, default=84,
                   help="distinct dated hours required across the requested days")
    p.add_argument("--detail-slots", default="1,4", help="within-hour slots; 1,4 = :10,:40")
    p.add_argument("--max-trajectories-per-cell", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--shuffle-partitions", type=int, default=2000)
    p.add_argument("--master", default="yarn")
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    a = parser().parse_args()
    try:
        days, slots = parse_days(a.days), parse_slots(a.detail_slots)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if a.n_links < 4 or a.max_trajectories_per_cell <= 0:
        raise SystemExit("--n-links must be >=4 and trajectory cap must be positive")
    if not 1 <= a.min_hour_slots <= 24:
        raise SystemExit("--min-hour-slots must be in 1..24")
    if not 1 <= a.min_active_days <= len(days):
        raise SystemExit("--min-active-days must be in 1..number of requested days")

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.storagelevel import StorageLevel

    spark = (SparkSession.builder.appName("tl_extract_representative_cells")
             .master(a.master)
             .config("spark.sql.session.timeZone", "Asia/Shanghai")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    mode = "overwrite" if a.overwrite else "errorifexists"
    root, out = a.corpus.rstrip("/"), a.out.rstrip("/")
    keys = selection_keys()

    # Select a fixed link panel for the whole week. Quartiles are computed only
    # among links with enough temporal support to be useful for visualisation.
    cells = (spark.read.parquet(root + "/cells")
             .where(F.col("day").cast("string").isin(days))
             .select("cell_id", "map_version", "target_link_id", "seg_idx",
                     "window", "K", F.col("day").cast("string").alias("day")))
    link_stats = (cells.groupBy(*keys).agg(
        F.sum("K").alias("n_observations"),
        F.count(F.lit(1)).alias("n_cells"),
        F.countDistinct(F.floor(F.col("window") / 3600)).alias("n_active_hours"),
        F.countDistinct("day").alias("n_active_days"),
        F.countDistinct(F.pmod(F.floor(F.col("window") / 3600) + 8, 24))
        .alias("n_hour_slots"),
        F.countDistinct("seg_idx").alias("n_segments"),
        F.countDistinct("map_version").alias("n_map_versions"),
        F.sort_array(F.collect_set(F.col("map_version").cast("string")))
        .alias("map_versions")))
    candidates = link_stats.where(
        (F.col("n_observations") >= a.min_observations) &
        (F.col("n_active_days") >= a.min_active_days) &
        (F.col("n_hour_slots") >= a.min_hour_slots) &
        (F.col("n_active_hours") >= a.min_active_hours))
    if candidates.limit(a.n_links).count() < a.n_links:
        raise SystemExit("fewer than %d links satisfy the coverage thresholds" % a.n_links)
    q = candidates.approxQuantile("n_observations", [0.25, 0.5, 0.75], 0.01)
    if len(q) != 3:
        raise SystemExit("could not compute activity quartiles")
    tier = (F.when(F.col("n_observations") <= q[0], 0)
            .when(F.col("n_observations") <= q[1], 1)
            .when(F.col("n_observations") <= q[2], 2).otherwise(3))
    ranked = (candidates.withColumn("volume_tier", tier)
              .withColumn("pick_hash", F.xxhash64(F.concat_ws(
                  "|", F.lit(str(a.seed)), F.col("target_link_id")))))
    selected_parts = []
    for i, quota in enumerate(tier_quotas(a.n_links)):
        selected_parts.append(ranked.where(F.col("volume_tier") == i)
                              .orderBy("pick_hash", "target_link_id")
                              .limit(quota))
    selected_links = selected_parts[0]
    for part in selected_parts[1:]:
        selected_links = selected_links.unionByName(part)
    selected_links = selected_links.drop("pick_hash").persist(StorageLevel.MEMORY_AND_DISK)
    actual_links = selected_links.count()
    if actual_links != a.n_links:
        raise SystemExit("activity tiers produced %d/%d links; relax thresholds" %
                         (actual_links, a.n_links))
    (selected_links.withColumn("map_versions", F.to_json("map_versions"))
     .coalesce(1).write.mode(mode).option("header", True).csv(
         out + "/selected_links"))

    selected_cells = (cells.join(F.broadcast(selected_links.select(*keys)), keys, "inner")
                      .persist(StorageLevel.MEMORY_AND_DISK))

    # A link can have multiple map versions even within one day.  Keep the
    # exact day/version membership as rows instead of pretending that the
    # selected physical link has one fixed map_version for the whole week.
    selected_link_days = (selected_cells.groupBy("target_link_id", "day").agg(
        F.sort_array(F.collect_set(F.col("map_version").cast("string")))
        .alias("map_versions"),
        F.sum("K").alias("n_observations"),
        F.count(F.lit(1)).alias("n_cells"),
        F.sum(F.when(F.col("K") >= 3, 1).otherwise(0)).alias("n_cells_k_ge_3"))
        .withColumn("cell_ratio_k_ge_3",
                    F.col("n_cells_k_ge_3") / F.col("n_cells"))
        .withColumn("map_versions", F.to_json("map_versions")))
    selected_link_days.coalesce(1).write.mode(mode).option("header", True).csv(
        out + "/selected_link_days")

    # HDFS v2's row order is untrusted, but its row set is intact. This job uses
    # relational joins/groupBy only and never relies on physical order.
    obs = (spark.read.parquet(root + "/" + a.obs_dir)
           .where(F.col("day").cast("string").isin(days))
           .join(F.broadcast(selected_links.select(*keys)), keys, "inner")
           .select("cell_id", "map_version", "target_link_id", "seg_idx", "window",
                   "sample_id", "dt", "n_pieces", "T_diff", "ratio_pct",
                   "observed", "valid", "bin_pos",
                   F.col("day").cast("string").alias("day"))
           .persist(StorageLevel.MEMORY_AND_DISK))

    # Fold pieces into absolute 10m bins using the same all-valid rule as the reader.
    piece = (obs.select("cell_id", "map_version", "target_link_id", "seg_idx",
                        "window", "sample_id", "day",
                        F.explode(F.arrays_zip("T_diff", "ratio_pct", "valid", "bin_pos")).alias("p"))
             .select("cell_id", "map_version", "target_link_id", "seg_idx", "window",
                     "sample_id", "day", F.col("p.bin_pos").cast("int").alias("bin_pos"),
                     F.col("p.T_diff").cast("double").alias("T"),
                     F.col("p.ratio_pct").cast("double").alias("R"),
                     F.col("p.valid").cast("boolean").alias("V")))
    valid_piece = F.col("V") & F.col("T").isNotNull() & ~F.isnan(F.col("T"))
    obs_bin = (piece.groupBy("cell_id", "map_version", "target_link_id", "seg_idx",
                             "window", "sample_id", "day", "bin_pos").agg(
        F.count(F.lit(1)).alias("piece_count"),
        F.sum(F.when(valid_piece, 1).otherwise(0)).alias("valid_piece_count"),
        F.sum(F.when(valid_piece, F.col("T")).otherwise(0.0)).alias("td_sum"),
        F.sum("R").alias("ratio_pct_sum"))
        .withColumn("bin_valid", F.col("valid_piece_count") == F.col("piece_count"))
        .withColumn("t10", F.when(
            F.col("bin_valid") & (F.col("ratio_pct_sum") > 0),
            F.col("td_sum") / (F.col("ratio_pct_sum") / 10.0))))

    quantiles = F.percentile_approx("t10", [0.25, 0.5, 0.75, 0.90], 10000)
    stats = (obs_bin.groupBy("cell_id", "map_version", "target_link_id", "seg_idx",
                             "window", "day", "bin_pos").agg(
        F.count("t10").alias("n_valid"), quantiles.alias("q"))
        .join(selected_cells.select("cell_id", F.col("K").alias("K_cell")), "cell_id", "left")
        .select("cell_id", "map_version", "target_link_id", "seg_idx", "window", "day",
                "bin_pos", "K_cell", "n_valid", F.col("q")[0].alias("p25_t10"),
                F.col("q")[1].alias("p50_t10"), F.col("q")[2].alias("p75_t10"),
                F.col("q")[3].alias("p90_t10")))
    stats.repartition(len(days), "day").write.mode(mode).partitionBy("day").parquet(
        out + "/link_bin_window_stats")

    # Keep detailed rows for two representative windows per hour, capped per cell.
    second_in_hour = ((F.col("window") % 3600) + 3600) % 3600
    detail = obs.withColumn("slot", F.floor(second_in_hour / 600).cast("int")).where(
        F.col("slot").isin(slots))
    order = Window.partitionBy("cell_id").orderBy(F.xxhash64(F.concat_ws(
        "|", F.lit(str(a.seed)), F.col("cell_id").cast("string"), F.col("sample_id"))))
    detail = (detail.withColumn("sample_rank", F.row_number().over(order))
              .where(F.col("sample_rank") <= a.max_trajectories_per_cell)
              .join(selected_cells.select("cell_id", F.col("K").alias("K_original")),
                    "cell_id", "left").drop("slot", "sample_rank"))
    detail.repartition(len(days), "day").write.mode(mode).partitionBy("day").parquet(
        out + "/sampled_observations")

    stats_rows = spark.read.parquet(out + "/link_bin_window_stats").count()
    detail_rows = spark.read.parquet(out + "/sampled_observations").count()
    per_day = (spark.read.parquet(out + "/link_bin_window_stats").groupBy("day").agg(
        F.countDistinct("window").alias("windows"),
        F.countDistinct("target_link_id").alias("links"),
        F.count(F.lit(1)).alias("rows")).orderBy("day").collect())
    manifest = {
        "format": "target_link_representative_extract_v1",
        "corpus": root, "observations": a.obs_dir, "days": days,
        "n_links": actual_links, "seed": a.seed,
        "selection": {"key": "target_link_id",
                      "map_version_semantics": "per-day attribute, not a cross-day key",
                      "min_observations": a.min_observations,
                      "min_active_days": a.min_active_days,
                      "min_hour_slots": a.min_hour_slots,
                      "min_active_hours": a.min_active_hours,
                      "volume_quartiles": q, "tier_quotas": tier_quotas(a.n_links)},
        "detail": {"slots": slots, "max_trajectories_per_cell": a.max_trajectories_per_cell},
        "rows": {"link_bin_window_stats": stats_rows,
                 "sampled_observations": detail_rows},
        "per_day": [{"day": str(r["day"]), "windows": int(r["windows"]),
                     "links": int(r["links"]), "rows": int(r["rows"])} for r in per_day],
        "metric": "t10=sum(valid piece T_diff)/sum(piece ratio); bin requires all pieces valid",
        "order_contract": "row order is not significant",
    }
    spark.createDataFrame([(json.dumps(manifest, ensure_ascii=False),)], "value string") \
        .coalesce(1).write.mode(mode).text(out + "/manifest.json.d")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    selected_links.unpersist(); selected_cells.unpersist(); obs.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
