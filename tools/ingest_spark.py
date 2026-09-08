"""Spark ingest: raw corridor parquet -> target-link tables (ingest_streaming contract).

Distributed rewrite of tools/ingest_streaming.py — same three outputs + stats,
so build_profiles / split_random / build_eta_data consume them unchanged.
Spark reads the corridor files directly, and the input paths can be hdfs://
URIs when submitted to the company cluster, which removes the 8.9GB/hour
local pull entirely.

Equivalence notes vs ingest_streaming.py:
  - (sample_id, bin_idx) dedup keeps max(_td_ok, ratio): pandas does a stable
    sort on [_td_ok, ratio] ascending and keeps the LAST row per key — same
    winner; exact (td_ok, ratio) ties are decided by file order in pandas and
    arbitrarily in Spark (verify with the smoke equivalence check).
  - per-sample "first" aggregations = value at the smallest bin_idx
    (min over struct(bin_idx, value); bin_idx is unique per sample after
    dedup). Uses the struct trick instead of min_by() so it also runs on the
    system Spark 3.2.0 (min_by needs 3.3+).
  - NaN AND null T_diff are both skipped in td_target (pandas skipna):
    nanvl -> null, and Spark sum ignores nulls.
  - bins are sharded by hash(sample_id): a sample's bins never span shards —
    the same trajectory-sharded guarantee build_profiles' streaming relies on.
  - stats quantiles are computed numpy-side on the written (small) tables, so
    they match pandas .quantile rather than Spark percentile_approx.

Run local (this pod, qwen12 + pip pyspark 3.5.9 + Java 8; see scripts/submit_ingest_yarn.sh):
  env -u SPARK_HOME SPARK_LOCAL_DIRS=~/sparktmp PYSPARK_PYTHON=<qwen12>/bin/python \
      <qwen12>/bin/python tools/ingest_spark.py \
      --inputs 'data/raw_hdfs/event_hour=2026082007/part-*.parquet' \
      --out data/processed_spark/day0820_07 --master 'local[8]'

Run on the company cluster (yarn, direct HDFS read/write, no local data pull):
  MODE=yarn DAY=20260820 OUT_DIR=hdfs://.../processed_spark/day20260820 \
      bash scripts/submit_ingest_yarn.sh
  (submit script handles minipy3 env distribution; this file keeps its
   cluster-mode imports to stdlib-only until Spark starts, so minipy3
   needs nothing beyond pyspark itself)
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

RAW_COLS = [
    "sample_id", "target_link_id", "t_enter", "seg_mark", "bin_idx",
    "ratio", "T_diff", "observed",
    "y_travel_s", "L_link_m",
    "link_fc", "link_speed_class", "link_kind", "link_lane", "link_direction", "link_urban",
]
STATIC_COLS = [
    "y_travel_s", "L_link_m",
    "link_fc", "link_speed_class", "link_kind", "link_lane", "link_direction", "link_urban",
]
SAMPLES_OUT_COLS = [
    "sample_id", "target_link_id", "window_id", "t_enter", "td_target", "v_sample",
    "bin_idx_min", "n_bins_target", "observed_ratio", "eff_len_ratio",
] + STATIC_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", required=True,
                   help="glob or single path of raw corridor parquet (local or hdfs://)")
    p.add_argument("--out", required=True, help="output dir (samples/link_window/bins_shards)")
    p.add_argument("--master", default="local[8]", help="local[N] now, yarn via spark-submit later")
    p.add_argument("--sample-fraction", type=float, default=1.0,
                   help="debug: sample raw rows (seed=42) before processing")
    p.add_argument("--stats", choices=["auto", "local", "spark", "off"], default="auto",
                   help="ingest_stats computation: auto=local for local paths, spark for "
                        "remote (hdfs://) outputs; spark mode runs percentile_approx "
                        "in-cluster and writes ingest_stats.json.d/ (a text dir)")
    p.add_argument("--window-s", type=int, default=3600)
    p.add_argument("--min-td", type=float, default=0.1)
    p.add_argument("--bins-shards", type=int, default=64, help="hash(sample_id) shard count")
    p.add_argument("--shuffle-partitions", type=int, default=None,
                   help="spark.sql.shuffle.partitions (default: 4x bins-shards)")
    p.add_argument("--driver-memory", default="4g", help="local mode runs the executor in-driver")
    p.add_argument("--app-name", default="target_link_ingest")
    return p.parse_args()


def value_at_min_bin(col: str):
    """pandas groupby 'first' after [sample_id, bin_idx] sort == value at min bin_idx.

    struct(bin_idx, value) min compares bin_idx first; bin_idx is unique per
    sample after the dedup, so ties never reach the value field.
    """
    from pyspark.sql import functions as F
    s = F.min(F.struct(F.col("bin_idx").alias("_k"), F.col(col).alias("_v")))
    return s.getField("_v").alias(col)


def _stats_via_spark(spark, samples, lw, args, out, n_rows_raw, n_bins,
                     n_bins_nan_td, n_samples_with_nan):
    """Diagnostics computed in-cluster, for remote (hdfs://) outputs the driver
    cannot pyarrow-read. Quantiles use percentile_approx(accuracy=10000) —
    close to, but not bit-identical with, local runs' exact numpy semantics."""
    from pyspark.sql import functions as F

    def pct(df, col):
        row = df.agg(F.percentile_approx(col, [0.1, 0.5, 0.9], 10000).alias("p")).head()
        return [round(float(x), 3) for x in row["p"]]

    return {
        "engine": "spark",
        "master": args.master,
        "n_input_files": None,
        "sample_fraction": args.sample_fraction,
        "n_rows_raw": int(n_rows_raw),
        "n_rows_target_bins": int(n_bins),
        "n_bins_nan_td": int(n_bins_nan_td),
        "n_samples_with_nan_td_bins": int(n_samples_with_nan),
        "n_samples": samples.count(),
        "n_links": samples.agg(F.countDistinct("target_link_id")).head()[0],
        "n_windows": samples.agg(F.countDistinct("window_id")).head()[0],
        "n_link_windows": lw.count(),
        "bins_per_sample_p10_p50_p90_mean":
            pct(samples, "n_bins_target")
            + [round(float(samples.agg(F.mean("n_bins_target")).head()[0]), 2)],
        "observed_ratio_p10_p50_p90": pct(samples, "observed_ratio"),
        "v_sample_mps_p10_p50_p90": pct(samples, "v_sample"),
        "L_link_m_p10_p50_p90": pct(samples, "L_link_m"),
        "y_travel_s_p10_p50_p90": pct(samples, "y_travel_s"),
        "trajs_per_link_window_p10_p50_p90": pct(lw, "n_trajs"),
        "bins_shard_dir": f"{out}/bins_shards",
        "n_bins_shards": args.bins_shards,
        "note": "spark ingest (remote output): quantiles via percentile_approx(acc=10000); "
                "bins sharded by hash(sample_id)",
    }


def main() -> None:
    args = parse_args()
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (SparkSession.builder
             .appName(args.app_name)
             .master(args.master)
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions",
                     args.shuffle_partitions or 4 * args.bins_shards)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    from pyspark import StorageLevel

    raw_all = spark.read.parquet(args.inputs)
    if args.sample_fraction < 1.0:
        raw_all = raw_all.sample(withReplacement=False,
                                 fraction=args.sample_fraction, seed=42)
    n_rows_raw = raw_all.count()
    tgt = raw_all.select(*RAW_COLS).where(F.col("seg_mark") == 1)

    # --- (sample_id, bin_idx) dedup: prefer valid T_diff, then max ratio ------
    # (matches pandas stable-sort on [_td_ok, ratio] asc + keep='last'; isNotNull
    # first makes the flag null-free, null-safe on T_diff)
    td = F.col("T_diff")
    tgt = tgt.withColumn("_td_ok", td.isNotNull() & ~F.isnan(td) & (td > 0))
    w = Window.partitionBy("sample_id", "bin_idx") \
        .orderBy(F.col("_td_ok").desc(), F.col("ratio").desc())
    dedup = (tgt.withColumn("_rn", F.row_number().over(w))
             .where(F.col("_rn") == 1).drop("_rn", "_td_ok"))
    # single materialisation on the CLUSTER: bins/samples/lw all branch off
    # dedup+agg, and every action (diagnostics, writes, stats) would otherwise
    # re-run the raw->filter->dedup-shuffle DAG from scratch (~18x on yarn with
    # stats=spark). Skipped on local masters: this pod's root disk is 20G-quota'd
    # and DISK_ONLY blocks fight the shuffle spill for the same space (blows the
    # quota on an hour of corridor data). The small agg table persists anywhere.
    if not args.master.startswith("local"):
        dedup = dedup.persist(StorageLevel.DISK_ONLY)

    # --- per-sample aggregation (one row per trajectory-pass) -----------------
    td_skipna = F.nanvl(td, F.lit(None).cast("double"))  # NaN -> null; sum skips nulls
    agg = dedup.groupBy("sample_id").agg(
        value_at_min_bin("target_link_id"),
        F.sum(td_skipna).alias("td_target"),
        F.min("bin_idx").alias("bin_idx_min"),
        F.count(F.lit(1)).alias("n_bins_target"),
        F.sum("observed").alias("n_observed"),
        F.sum("ratio").alias("eff_len_ratio"),
        value_at_min_bin("t_enter"),
        *[value_at_min_bin(c) for c in STATIC_COLS],
    ).persist(StorageLevel.DISK_ONLY)

    # bins: NOT filtered by min_td (same as ingest_streaming; build_profiles
    # drops orphans against the samples table)
    bins = (dedup.join(agg.select("sample_id", "bin_idx_min"), "sample_id")
            .select("sample_id",
                    (F.col("bin_idx") - F.col("bin_idx_min")).cast("int").alias("rel_bin_idx"),
                    F.col("ratio").cast("float"),
                    F.col("T_diff").cast("float"),
                    F.col("observed").cast("byte")))
    # all three bin diagnostics in ONE action — each .count() used to be a
    # separate full pass over the bins DAG
    bad_td = td.isNull() | F.isnan(td)
    diag = bins.agg(
        F.count(F.lit(1)).alias("n_bins"),
        F.count(F.when(bad_td, F.lit(1))).alias("n_bins_nan_td"),
        F.countDistinct(F.when(bad_td, F.col("sample_id"))).alias("n_samples_with_nan"),
    ).head()
    n_bins, n_bins_nan_td, n_samples_with_nan = (
        int(diag[k]) for k in ("n_bins", "n_bins_nan_td", "n_samples_with_nan"))

    samples = (agg.where(F.col("td_target") > F.lit(args.min_td))
               .withColumn("v_sample", F.col("L_link_m") / F.col("td_target"))
               .withColumn("observed_ratio",
                           F.col("n_observed") / F.greatest(F.col("n_bins_target"), F.lit(1)))
               .withColumn("window_id",
                           F.floor(F.col("t_enter") / F.lit(args.window_s)).cast("long")))
    lw = samples.groupBy("target_link_id", "window_id").agg(
        F.mean("v_sample").alias("mean_speed"),          # pandas mean
        F.stddev_samp("v_sample").alias("speed_std"),    # pandas std (ddof=1)
        F.count(F.lit(1)).alias("n_trajs"),
        F.mean("y_travel_s").alias("y_travel_mean"),
    )

    # --- write ------------------------------------------------------------------
    out = str(args.out)
    stats_mode = args.stats
    if stats_mode == "auto":
        stats_mode = "spark" if "://" in out else "local"
    (bins.repartition(args.bins_shards, "sample_id").write.mode("overwrite")
     .option("compression", "snappy").parquet(f"{out}/bins_shards"))
    (samples.select(*SAMPLES_OUT_COLS).write.mode("overwrite")
     .option("compression", "snappy").parquet(f"{out}/samples.parquet"))
    lw.write.mode("overwrite").option("compression", "snappy") \
        .parquet(f"{out}/link_window.parquet")

    if stats_mode == "spark":
        # stats re-read the WRITTEN parquet (column-pruned scans of the small
        # output tables) instead of running ~10 more actions on the pipeline
        stats = _stats_via_spark(spark, spark.read.parquet(f"{out}/samples.parquet"),
                                 spark.read.parquet(f"{out}/link_window.parquet"),
                                 args, out, n_rows_raw, n_bins,
                                 n_bins_nan_td, n_samples_with_nan)
        spark.createDataFrame(
            [(json.dumps(stats, indent=2, ensure_ascii=False, default=str),)], "stats STRING"
        ).coalesce(1).write.mode("overwrite").text(f"{out}/ingest_stats.json.d")
        spark.stop()
        print(f"[ingest_spark] wrote {out}/ : {stats['n_samples']} samples, {n_bins} bin rows "
              f"({args.bins_shards} shards), {stats['n_link_windows']} link-windows")
        return

    spark.stop()
    if stats_mode == "off":
        print(f"[ingest_spark] wrote {out}/ : {n_bins} bin rows ({args.bins_shards} shards)")
        return

    # --- stats: numpy on the written small tables (pandas .quantile semantics) --
    # (local outputs only; heavy imports stay lazy so the yarn driver — whose
    # minipy3 env only guarantees pyspark — never needs numpy/pandas/pyarrow)
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    def pct(col: str, ps=(10, 50, 90)):
        a = pq.read_table(f"{out}/samples.parquet", columns=[col]).column(col).to_numpy()
        return [round(float(np.percentile(a, x)), 3) for x in ps]

    lw_pd = pd.read_parquet(f"{out}/link_window.parquet", columns=["n_trajs"])
    samples_cols = pq.read_table(f"{out}/samples.parquet",
                                 columns=["n_bins_target", "observed_ratio", "v_sample",
                                          "L_link_m", "y_travel_s", "target_link_id",
                                          "window_id"])
    stats = {
        "engine": "spark",
        "master": args.master,
        "n_input_files": len(globmod.glob(args.inputs)) if "://" not in args.inputs else None,
        "sample_fraction": args.sample_fraction,
        "n_rows_raw": int(n_rows_raw),
        "n_rows_target_bins": int(n_bins),
        "n_bins_nan_td": int(n_bins_nan_td),
        "n_samples_with_nan_td_bins": int(n_samples_with_nan),
        "n_samples": samples_cols.num_rows,
        "n_links": int(samples_cols.column("target_link_id").to_pandas().nunique()),
        "n_windows": int(samples_cols.column("window_id").to_pandas().nunique()),
        "bins_per_sample_p10_p50_p90_mean":
            pct("n_bins_target") + [round(float(np.asarray(
                samples_cols.column("n_bins_target").to_numpy()).mean()), 2)],
        "observed_ratio_p10_p50_p90": pct("observed_ratio"),
        "v_sample_mps_p10_p50_p90": pct("v_sample"),
        "L_link_m_p10_p50_p90": pct("L_link_m"),
        "y_travel_s_p10_p50_p90": pct("y_travel_s"),
        "trajs_per_link_window_p10_p50_p90":
            [round(float(np.percentile(lw_pd.n_trajs.to_numpy(), x)), 3) for x in (10, 50, 90)],
        "bins_shard_dir": f"{out}/bins_shards",
        "n_bins_shards": args.bins_shards,
        "note": "spark ingest: bins sharded by hash(sample_id)",
    }
    stats_path = Path(f"{out}/ingest_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
    print(f"[ingest_spark] wrote {out}/ : {stats['n_samples']} samples, {n_bins} bin rows "
          f"({args.bins_shards} shards), {len(lw_pd)} link-windows")


if __name__ == "__main__":
    main()
