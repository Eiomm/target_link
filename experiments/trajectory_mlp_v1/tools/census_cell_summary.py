"""Build restartable, exact per-partition cell summaries from compact census rows.

Each source compact file is physically co-located by ``pmod(cell_id, 128)``.
The tool verifies that invariant before grouping, so its outputs can later be
combined without a global 1.19B-row cell aggregation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPO / "experiments/trajectory_mlp_v1/reports/seven_day_p0_20260817_23"
PART_NAME = re.compile(r"^(\d{8})_(\d{3})\.parquet$")
SQL = """SELECT cell_id,
    min("window") AS window_start,
    max("window") AS window_end,
    count(*) AS k_raw,
    cast(sum(usable::BIGINT) AS BIGINT) AS k_usable,
    min(n_present) AS min_present,
    max(n_present) AS max_present,
    min(covered_m) AS min_covered_m,
    max(covered_m) AS max_covered_m
  FROM {source}
  GROUP BY cell_id"""


def _quoted(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _fingerprint(path: Path) -> dict:
    info = path.stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _part_identity(path: Path) -> tuple[str, int]:
    match = PART_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid compact part name: {path.name}")
    return match.group(1), int(match.group(2))


def _connect(spill: Path):
    import duckdb

    spill.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='512MB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET temp_directory=?", [str(spill)])
    return connection


def _sql_hash() -> str:
    return hashlib.sha256((SQL + "|contract-v1").encode()).hexdigest()


def _receipt_matches(receipt: Path, output: Path, source: Path, duckdb_version: str) -> bool:
    if not (receipt.exists() and output.exists()):
        return False
    try:
        old = json.loads(receipt.read_text())
    except json.JSONDecodeError:
        return False
    return (old.get("input") == _fingerprint(source) and old.get("sql_hash") == _sql_hash() and
            old.get("duckdb_version") == duckdb_version and old.get("output_bytes") == output.stat().st_size and
            old.get("rows", -1) >= 0 and old.get("observations", -1) >= 0 and
            old.get("sum_k_usable", -1) >= 0)


def _smoke_compare(connection, source_sql: str, output: Path, day: str) -> None:
    """Compare exactly with the existing aggregate() cells SQL for one part."""
    original = f"""SELECT '{day}' AS day,cell_id,any_value("window") AS window_start,
        count(*) AS k_raw,sum(usable::BIGINT) AS k_usable,
        min(n_present) AS min_present,max(n_present) AS max_present,
        min(covered_m) AS min_covered_m,max(covered_m) AS max_covered_m
        FROM {source_sql} GROUP BY cell_id"""
    output_sql = f"SELECT * FROM read_parquet({_quoted(output)})"
    mismatch = connection.execute(
        f"SELECT count(*) FROM (({original}) EXCEPT ({output_sql}) UNION ALL "
        f"({output_sql}) EXCEPT ({original}))"
    ).fetchone()[0]
    if mismatch:
        raise AssertionError(f"smoke comparison differs in {mismatch} cell rows")


def build_one(task: tuple[str, str, bool]) -> dict:
    """Build or verify one native ``(day,bucket)`` compact parquet summary."""
    source_name, output_dir_name, smoke = task
    source = Path(source_name)
    output_dir = Path(output_dir_name)
    day, bucket = _part_identity(source)
    stem = source.stem
    output = output_dir / f"{stem}.parquet"
    receipt = output_dir / f"{stem}.json"
    spill = output_dir / "spill" / stem

    import duckdb

    version = duckdb.__version__
    if _receipt_matches(receipt, output, source, version):
        return {"part": stem, "cached": True, "rows": json.loads(receipt.read_text())["rows"]}

    start = time.time()
    connection = _connect(spill)
    source_sql = f"read_parquet({_quoted(source)})"
    try:
        raw = connection.execute(f"""SELECT count(*) AS observations,
            coalesce(sum(usable::BIGINT), 0) AS sum_k_usable,
            count(*) FILTER (WHERE (((cell_id % 128) + 128) % 128) <> {bucket}) AS wrong_bucket,
            count(*) FILTER (WHERE strftime(to_timestamp("window") AT TIME ZONE 'Asia/Shanghai','%Y%m%d') <> '{day}') AS wrong_day
            FROM {source_sql}""").fetchone()
        observations, input_usable, wrong_bucket, wrong_day = map(int, raw)
        if wrong_bucket or wrong_day:
            raise ValueError(f"{stem}: wrong_bucket={wrong_bucket}, wrong_day={wrong_day}")

        connection.execute("CREATE TEMP TABLE summary AS " + SQL.format(source=source_sql))
        multi_window = connection.execute(
            "SELECT count(*) FROM summary WHERE window_start <> window_end"
        ).fetchone()[0]
        if multi_window:
            raise ValueError(f"{stem}: {multi_window} cells contain multiple windows")
        summary = connection.execute(
            "SELECT count(*) AS rows,coalesce(sum(k_raw),0) AS observations,"
            "coalesce(sum(k_usable),0) AS sum_k_usable FROM summary"
        ).fetchone()
        rows, output_observations, output_usable = map(int, summary)
        if output_observations != observations or output_usable != input_usable:
            raise AssertionError(f"{stem}: summary reconciliation failed")

        temporary = output.with_suffix(".parquet.tmp")
        connection.execute(
            f"COPY (SELECT '{day}' AS day,cell_id,window_start,k_raw,k_usable,min_present,max_present,"
            f"min_covered_m,max_covered_m FROM summary) TO {_quoted(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary.replace(output)
        if smoke:
            _smoke_compare(connection, source_sql, output, day)
        receipt_data = {
            "part": stem, "day": day, "bucket": bucket, "input": _fingerprint(source),
            "sql_hash": _sql_hash(), "duckdb_version": version, "rows": rows,
            "observations": observations, "sum_k_usable": input_usable,
            "output_bytes": output.stat().st_size, "seconds": round(time.time() - start, 3),
        }
        _dump(receipt, receipt_data)
        return {"part": stem, "cached": False, "rows": rows, "seconds": receipt_data["seconds"]}
    finally:
        connection.close()


def discover(parts: Path) -> list[Path]:
    files = sorted(parts.glob("????????_???.parquet"))
    keys = [_part_identity(path) for path in files]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate compact day/bucket part")
    return files


def complete_marker(output_dir: Path, files: list[Path]) -> None:
    expected = {_part_identity(path) for path in files}
    found = {_part_identity(path) for path in output_dir.glob("????????_???.parquet")}
    if found != expected:
        raise ValueError("cell summary outputs do not cover every input part")
    receipts = [output_dir / f"{path.stem}.json" for path in files]
    if not all(path.exists() for path in receipts):
        raise ValueError("cell summary receipt missing")
    summaries = [json.loads(path.read_text()) for path in receipts]
    _dump(output_dir.parent / "cell_summary_complete.json", {
        "status": "complete", "partitions": len(files), "sql_hash": _sql_hash(),
        "cells": sum(item["rows"] for item in summaries),
        "observations": sum(item["observations"] for item in summaries),
        "usable_observations": sum(item["sum_k_usable"] for item in summaries),
        "source_fingerprints": {
            path.stem: [_fingerprint(path)["bytes"], _fingerprint(path)["mtime_ns"]]
            for path in files
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, default=DEFAULT_REPORT / "parts")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT / "cell_parts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="compare the first selected part to aggregate() SQL")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    files = discover(args.parts)
    if not files:
        parser.error("no compact parquet parts found")
    selected = files[:args.limit] if args.limit else files
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), str(args.out), args.smoke and index == 0)
             for index, path in enumerate(selected)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, future in enumerate(concurrent.futures.as_completed(
                [pool.submit(build_one, task) for task in tasks]), 1):
            result = future.result()
            print(f"{index}/{len(tasks)} {result['part']} rows={result['rows']} cached={result['cached']} "
                  f"seconds={result.get('seconds', 0):.3f}", flush=True)
    if len(selected) == len(files):
        complete_marker(args.out, files)
        print("CELL SUMMARY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
