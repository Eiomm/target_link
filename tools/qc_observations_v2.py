"""Full Spark QC for the side-built observations_v2 corpus."""
from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--old-dir", default="observations")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, ByteType, FloatType

    spark = SparkSession.builder.appName("tl_qc_observations_v2").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    new = spark.read.parquet(a.root.rstrip("/") + "/" + a.obs_dir)
    old = spark.read.parquet(a.root.rstrip("/") + "/" + a.old_dir)
    schema_ok = (
        "bin_pos" in new.columns
        and isinstance(new.schema["bin_pos"].dataType, ArrayType)
        and isinstance(new.schema["bin_pos"].dataType.elementType, ByteType)
        and isinstance(new.schema["T_diff"].dataType, ArrayType)
        and isinstance(new.schema["T_diff"].dataType.elementType, FloatType)
    )
    arrays = ["T_diff", "ratio_pct", "observed", "valid", "bin_pos"]
    aligned = F.lit(True)
    for col in arrays:
        aligned = aligned & (F.size(col) == F.col("n_pieces"))
    row = new.agg(
        F.count("*").alias("new_rows"),
        F.min("dt").alias("dt_min"), F.max("dt").alias("dt_max"),
        F.sum(F.when((F.col("dt") < 0) | (F.col("dt") >= 600), 1).otherwise(0)).alias("dt_bad_strict"),
        F.sum(F.when(F.col("dt") == 600, 1).otherwise(0)).alias("dt_float32_edge"),
        F.sum(F.when(~aligned, 1).otherwise(0)).alias("ragged_length_bad"),
        F.min("n_pieces").alias("n_pieces_min"), F.max("n_pieces").alias("n_pieces_max"),
    ).first().asDict()
    row["old_rows"] = old.count()
    row["row_count_equal"] = row["new_rows"] == row["old_rows"]

    piece = new.select(F.explode(F.arrays_zip(*arrays)).alias("p"))
    expected_valid = F.col("p.T_diff").isNotNull() & ~F.isnan(F.col("p.T_diff"))
    ps = piece.agg(
        F.count("*").alias("pieces"),
        F.min("p.bin_pos").alias("bin_pos_min"), F.max("p.bin_pos").alias("bin_pos_max"),
        F.sum(F.when((F.col("p.bin_pos") < 0) | (F.col("p.bin_pos") > 49), 1).otherwise(0)).alias("bin_pos_bad"),
        F.min("p.ratio_pct").alias("ratio_pct_min"), F.max("p.ratio_pct").alias("ratio_pct_max"),
        F.sum(F.when((F.col("p.ratio_pct") < 1) | (F.col("p.ratio_pct") > 10), 1).otherwise(0)).alias("ratio_pct_bad"),
        F.sum(F.when(F.col("p.valid") != expected_valid, 1).otherwise(0)).alias("valid_tdiff_bad"),
        F.sum(F.when(~F.col("p.valid"), 1).otherwise(0)).alias("invalid_pieces"),
    ).first().asDict()
    result = {"schema": new.schema.simpleString(), "schema_ok": schema_ok,
              "rows": row, "pieces": ps}
    hard_ok = (schema_ok and row["row_count_equal"] and row["ragged_length_bad"] == 0
               and ps["bin_pos_bad"] == 0 and ps["ratio_pct_bad"] == 0
               and ps["valid_tdiff_bad"] == 0 and row["dt_bad_strict"] == 0)
    result["pass"] = hard_ok
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(payload, flush=True)
    (spark.createDataFrame([(payload,)], "value string").coalesce(1)
     .write.mode("overwrite").text(a.out))
    spark.stop()
    if not hard_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
