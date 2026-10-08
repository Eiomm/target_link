"""Verify that a YARN observation-v3 copy exactly matches a local reference.

The check deliberately reads every persisted partition payload and index, but
only materializes the first three groups through ``CellDataset``.  It is an
acceptance check for a completed copy, not a training-data scan.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FORMAT = "trajectory_mlp_observation_v3"
MARKER = "_OBSERVATION_V3_SUCCESS.json"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root: Path, record: dict) -> Path:
    path = root / record["path"]
    resolved_root, resolved = root.resolve(), path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("artifact path escapes corpus root: " + record["path"])
    return path


def _load_manifest(root: Path, side: str) -> dict:
    path = root / side / MARKER
    try:
        marker = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError("missing manifest: " + str(path)) from exc
    if marker.get("format") != FORMAT or not isinstance(marker.get("partitions"), dict):
        raise ValueError("invalid observation-v3 manifest: " + str(path))
    return marker


def _array_equal(left, right) -> bool:
    if left.type != right.type or len(left) != len(right):
        return False
    left, right = left.combine_chunks(), right.combine_chunks()
    if pa.types.is_list(left.type) or pa.types.is_large_list(left.type):
        if not left.offsets.equals(right.offsets):
            return False
        left, right = left.values, right.values
    if not left.is_null().equals(right.is_null()):
        return False
    a = left.to_numpy(zero_copy_only=False)
    b = right.to_numpy(zero_copy_only=False)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def parquet_equal(reference: Path, candidate: Path) -> tuple[bool, str | None]:
    left, right = pq.ParquetFile(reference).read(), pq.ParquetFile(candidate).read()
    if left.schema != right.schema:
        return False, "schema"
    if len(left) != len(right):
        return False, "row_count"
    for name in left.column_names:
        if not _array_equal(left[name], right[name]):
            return False, "column:" + name
    return True, None


def npz_equal(reference: Path, candidate: Path) -> tuple[bool, str | None]:
    with np.load(reference, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            return False, "array_names"
        for name in left.files:
            if not np.array_equal(left[name], right[name], equal_nan=True):
                return False, "array:" + name
    return True, None


def _candidate_record_valid(root: Path, record: dict) -> tuple[bool, str | None]:
    try:
        path = _artifact(root, record)
        if path.stat().st_size != record["bytes"]:
            return False, "bytes"
        if sha256(path) != record["sha256"]:
            return False, "sha256"
    except (KeyError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, None


def compare_partition(reference_root: Path, candidate_root: Path, key: str,
                      reference: dict, candidate: dict) -> dict:
    result = {"partition": key, "ok": True, "checks": {}}
    for field in ("stored_rows", "stats"):
        equal = reference.get(field) == candidate.get(field)
        result["checks"][field] = equal
        result["ok"] &= equal
    for name in ("observations", "index", "cells"):
        valid, detail = _candidate_record_valid(candidate_root, candidate.get(name, {}))
        result["checks"]["candidate_" + name + "_sha256"] = valid
        result["ok"] &= valid
        if not valid:
            result.setdefault("details", {})["candidate_" + name] = detail
            continue
        try:
            left = _artifact(reference_root, reference[name])
            right = _artifact(candidate_root, candidate[name])
            equal, detail = (parquet_equal(left, right) if name == "observations"
                             else npz_equal(left, right))
        except (KeyError, OSError, ValueError) as exc:
            equal, detail = False, str(exc)
        result["checks"][name] = equal
        result["ok"] &= equal
        if not equal:
            result.setdefault("details", {})[name] = detail
    return result


def reader_samples(reference_root: Path, candidate_root: Path, side: str, marker: dict) -> dict:
    """Compare a bounded reader sample; full payload equivalence is checked above."""
    from trajectory_mae.data import CellDataset

    days = sorted({key.split("/", 1)[0] for key in marker["partitions"]})
    options = dict(m_max=int(marker["m_max"]), seed=int(marker["data_seed"]), epoch=0)
    before = list(itertools.islice(CellDataset([str(reference_root / side)], days, **options), 3))
    after = list(itertools.islice(CellDataset([str(candidate_root / side)], days, **options), 3))
    checks = {"count": len(before) == len(after)}
    for index, (left, right) in enumerate(zip(before, after)):
        checks["group_%d_metadata" % index] = all(left[name] == right[name]
                                                    for name in ("group_id", "sample_ids"))
        checks["group_%d_x" % index] = np.array_equal(left["x"], right["x"], equal_nan=True)
        checks["group_%d_bin_valid" % index] = np.array_equal(left["bin_valid"], right["bin_valid"])
        checks["group_%d_delta_t" % index] = np.array_equal(left["delta_t"], right["delta_t"], equal_nan=True)
    return {"groups_compared": len(before), "ok": all(checks.values()), "checks": checks}


def compare(reference_root: Path, candidate_root: Path) -> dict:
    report = {"format": FORMAT, "reference": str(reference_root), "candidate": str(candidate_root),
              "ok": True, "sides": {}}
    for side in ("train", "val"):
        reference = _load_manifest(reference_root, side)
        candidate = _load_manifest(candidate_root, side)
        side_report = {"ok": True, "manifest": {}, "partitions": []}
        for field in ("format", "m_max", "data_seed"):
            equal = reference.get(field) == candidate.get(field)
            side_report["manifest"][field] = equal
            side_report["ok"] &= equal
        reference_keys, candidate_keys = set(reference["partitions"]), set(candidate["partitions"])
        side_report["manifest"]["partition_keys"] = reference_keys == candidate_keys
        side_report["ok"] &= reference_keys == candidate_keys
        for key in sorted(reference_keys | candidate_keys):
            if key not in reference["partitions"] or key not in candidate["partitions"]:
                entry = {"partition": key, "ok": False, "details": {"missing": "reference" if key not in reference["partitions"] else "candidate"}}
            else:
                entry = compare_partition(reference_root / side, candidate_root / side, key,
                                          reference["partitions"][key], candidate["partitions"][key])
            side_report["partitions"].append(entry)
            side_report["ok"] &= entry["ok"]
        if side_report["manifest"].get("m_max") and side_report["manifest"].get("data_seed"):
            sample = reader_samples(reference_root, candidate_root, side, reference)
        else:
            sample = {"groups_compared": 0, "ok": False, "checks": {"protocol": False}}
        side_report["reader_samples"] = sample
        side_report["ok"] &= sample["ok"]
        report["sides"][side] = side_report
        report["ok"] &= side_report["ok"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="runtime/observationv3_pilot")
    parser.add_argument("--candidate", default="runtime/observationv3_yarn_smoke")
    parser.add_argument("--report", help="optional path to write the JSON report")
    args = parser.parse_args(argv)
    try:
        report = compare(Path(args.reference), Path(args.candidate))
    except Exception as exc:
        report = {"format": FORMAT, "reference": args.reference, "candidate": args.candidate,
                  "ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
