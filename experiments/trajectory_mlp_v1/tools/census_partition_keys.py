"""Create exact, restartable key reductions for compact census partitions."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
REPORT = REPO / "experiments/trajectory_mlp_v1/reports/seven_day_p0_20260817_23"
NAME = re.compile(r"^(\d{8})_(\d{3})\.parquet$")
SQL_CONTRACT = "keys-v2/base-all-scalars/traj-window/link-window/geometry-weighted/coverage"


def quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def identity(path: Path) -> tuple[str, int]:
    match = NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid compact part name: {path.name}")
    return match.group(1), int(match.group(2))


def sql_hash() -> str:
    return hashlib.sha256(SQL_CONTRACT.encode()).hexdigest()


def outputs(root: Path, stem: str) -> dict[str, Path]:
    return {"traj": root / "traj" / f"{stem}.parquet",
            "link": root / "link" / f"{stem}.parquet",
            "geometry": root / "geometry" / f"{stem}.parquet",
            "coverage": root / "coverage" / f"{stem}.parquet",
            "stats": root / "stats" / f"{stem}.json"}


def connect(spill: Path):
    import duckdb
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET memory_limit='512MB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory=?", [str(spill)])
    return con


def cached(receipt: Path, paths: dict[str, Path], source: Path, version: str) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        value = json.loads(receipt.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (value.get("input") == fingerprint(source) and value.get("sql_hash") == sql_hash() and
            value.get("duckdb_version") == version and
            all(value.get("output_bytes", {}).get(key) == path.stat().st_size
                for key, path in paths.items()))


def copy(con, query: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".parquet.tmp")
    con.execute(f"COPY ({query}) TO {quote(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    temp.replace(output)


def smoke(con, paths: dict[str, Path], day: str) -> None:
    """Exact one-part comparison to aggregate() daily/weekly/window ingredients."""
    checks = con.execute(f"""SELECT
      (SELECT count(DISTINCT traj_id) FROM base) =
        (SELECT count(DISTINCT traj_id) FROM read_parquet({quote(paths['traj'])})) AS traj_daily_weekly,
      (SELECT count(DISTINCT target_link_id) FROM base) =
        (SELECT count(DISTINCT target_link_id) FROM read_parquet({quote(paths['link'])})) AS link_daily_weekly,
      (SELECT count(*) FROM (SELECT window_start,traj_id FROM base GROUP BY 1,2)) =
        (SELECT count(*) FROM read_parquet({quote(paths['traj'])})) AS traj_window,
      (SELECT count(*) FROM (SELECT window_start,target_link_id FROM base GROUP BY 1,2)) =
        (SELECT count(*) FROM read_parquet({quote(paths['link'])})) AS link_window,
      (SELECT count(*) FROM base) =
        (SELECT sum(observations) FROM read_parquet({quote(paths['geometry'])})) AS geometry_observations,
      (SELECT count(*) FROM base) =
        (SELECT sum(observations) FROM read_parquet({quote(paths['coverage'])})) AS coverage_observations""").fetchone()
    if not all(checks):
        raise AssertionError(f"{day}: smoke mismatch {checks}")


def build_one(task: tuple[str, str, bool]) -> dict:
    source = Path(task[0]); root = Path(task[1]); do_smoke = task[2]
    day, bucket = identity(source); stem = source.stem
    paths = outputs(root, stem); receipt = root / "receipts" / f"{stem}.json"
    import duckdb
    version = duckdb.__version__
    if cached(receipt, paths, source, version) and not do_smoke:
        return {"part": stem, "cached": True, "observations": json.loads(receipt.read_text())["observations"]}
    started = time.time(); con = connect(root / "spill" / stem)
    src = f"read_parquet({quote(source)})"
    try:
        # Keep only columns used by the reductions.  Retaining sample_id and
        # unrelated compact metrics here needlessly enlarges DuckDB's temporary
        # table and can force spills under the 512MB per-worker budget.
        con.execute(f"""CREATE TEMP TABLE base AS SELECT "window" AS window_start,
            map_version,target_link_id,seg_idx,covered_m,n_pieces,n_present,n_valid,
            has_internal_gap,usable,split_part(sample_id,'#',1) AS traj_id FROM {src}""")
        raw = con.execute("""SELECT count(*) observations,coalesce(sum(n_pieces),0) pieces,
            coalesce(sum(n_present),0) recorded_bins,coalesce(sum(n_valid),0) valid_bins,
            coalesce(sum(has_internal_gap::BIGINT),0) internal_gap_observations,
            coalesce(sum(usable::BIGINT),0) usable_observations,coalesce(sum(covered_m),0) covered_m
            FROM base""").fetchone()
        fields = ("observations", "pieces", "recorded_bins", "valid_bins", "internal_gap_observations",
                  "usable_observations", "covered_m")
        stats = dict(zip(fields, raw))
        copy(con, f"SELECT '{day}' AS day,window_start,traj_id FROM base GROUP BY window_start,traj_id", paths["traj"])
        copy(con, f"SELECT '{day}' AS day,window_start,target_link_id FROM base GROUP BY window_start,target_link_id", paths["link"])
        copy(con, f"""SELECT '{day}' AS day,map_version,target_link_id,seg_idx,covered_m,count(*) observations
            FROM base GROUP BY map_version,target_link_id,seg_idx,covered_m""", paths["geometry"])
        copy(con, f"SELECT '{day}' AS day,n_present,n_valid,count(*) observations FROM base GROUP BY n_present,n_valid", paths["coverage"])
        geometry_observations = con.execute(
            f"SELECT coalesce(sum(observations),0) FROM read_parquet({quote(paths['geometry'])})"
        ).fetchone()[0]
        if int(geometry_observations) != int(stats["observations"]):
            raise AssertionError(f"{stem}: geometry observation reconciliation failed")
        coverage_observations = con.execute(
            f"SELECT coalesce(sum(observations),0) FROM read_parquet({quote(paths['coverage'])})"
        ).fetchone()[0]
        if int(coverage_observations) != int(stats["observations"]):
            raise AssertionError(f"{stem}: coverage observation reconciliation failed")
        if do_smoke:
            smoke(con, paths, day)
        paths["stats"].parent.mkdir(parents=True, exist_ok=True)
        dump(paths["stats"], {"day": day, **stats})
        receipt_data = {"part": stem, "day": day, "bucket": bucket, "input": fingerprint(source),
                        "sql_hash": sql_hash(), "duckdb_version": version, **stats,
                        "output_bytes": {key: path.stat().st_size for key, path in paths.items()},
                        "seconds": round(time.time() - started, 3)}
        receipt.parent.mkdir(parents=True, exist_ok=True); dump(receipt, receipt_data)
        return {"part": stem, "cached": False, "observations": int(stats["observations"]),
                "seconds": receipt_data["seconds"]}
    finally:
        con.close()


def discover(parts: Path) -> list[Path]:
    files = sorted(parts.glob("????????_???.parquet"))
    if len({identity(path) for path in files}) != len(files):
        raise ValueError("duplicate part key")
    return files


def marker(root: Path, files: list[Path]) -> None:
    receipts = [root / "receipts" / f"{path.stem}.json" for path in files]
    if not all(path.exists() for path in receipts):
        raise ValueError("missing key receipt")
    for path in files:
        if not all(candidate.exists() for candidate in outputs(root, path.stem).values()):
            raise ValueError(f"missing key output: {path.stem}")
    values = [json.loads(path.read_text()) for path in receipts]
    dump(root.parent / "key_summary_complete.json", {
        "status": "complete", "partitions": len(files), "observations": sum(int(x["observations"]) for x in values),
        "sql_hash": sql_hash(), "source_fingerprints": {path.stem: [fingerprint(path)["bytes"], fingerprint(path)["mtime_ns"]] for path in files},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, default=REPORT / "parts")
    parser.add_argument("--out", type=Path, default=REPORT / "key_parts")
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    files = discover(args.parts); selected = files[:args.limit] if args.limit else files
    if not selected: parser.error("no compact parts")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), str(args.out), args.smoke and index == 0) for index, path in enumerate(selected)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            print(f"{index}/{len(tasks)} {result['part']} obs={result['observations']} cached={result['cached']} seconds={result.get('seconds',0):.3f}", flush=True)
    if len(selected) == len(files):
        marker(args.out, files); print("KEY SUMMARY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
