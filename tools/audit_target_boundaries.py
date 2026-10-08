"""Read-only audit of local raw reference packages; no production selection rule.

Only audits data/ref_links_biz_strata/link_*/samples_flat.parquet. These are
purposively selected links, so the rates are NOT population estimates.
Requires pyarrow; writes a new JSON report and refuses to overwrite it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


COLS = ["map_version", "target_link_id", "sample_id", "bin_idx", "sub_idx",
        "seg_mark", "seg_idx", "link_id", "ratio", "bin_size_m", "L_link_m",
        "X_in_m", "T_diff", "T_cum"]


def json_safe(value):
    """Keep missing/nonfinite timings visible without nonstandard JSON NaNs."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def audit_file(path):
    import pyarrow.parquet as pq

    samples = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(columns=COLS, batch_size=65536):
        for row in batch.to_pylist():
            key = (row["map_version"], row["target_link_id"], row["sample_id"])
            samples[key].append(row)
    counts = Counter()
    examples = {}
    length_errors = []
    for key, rows in samples.items():
        rows.sort(key=lambda r: (r["bin_idx"], r["sub_idx"]))
        marked = [r for r in rows if r["seg_mark"] == 1]
        own = [r for r in rows if r["link_id"] == key[1]]
        own_marked = [r for r in marked if r["link_id"] == key[1]]
        foreign = [r for r in marked if r["link_id"] != key[1]]
        outside = [r for r in own if r["seg_mark"] != 1]
        dist = lambda rs: math.fsum(r["ratio"] * r["bin_size_m"] for r in rs)
        length = rows[0]["L_link_m"]
        error = dist(marked) - length
        length_errors.append(error)
        counts["samples"] += 1
        counts["rows"] += len(rows)
        counts["marked_distance_closes_1cm"] += abs(error) <= 0.01
        counts["marked_own_distance_closes_1cm"] += abs(dist(own_marked)-length) <= 0.01
        counts["all_own_distance_closes_1cm"] += abs(dist(own)-length) <= 0.01
        counts["all_own_exceeds_length_gt_10m"] += dist(own)-length > 10.01
        counts["samples_with_foreign_marked_piece"] += bool(foreign)
        counts["samples_with_own_outside_marked"] += bool(outside)
        counts["samples_with_partial_own_outside_marked"] += any(
            0 < r["ratio"] < 0.999999 for r in outside)
        counts["samples_with_invalid_ratio"] += any(
            r["ratio"] is None or not math.isfinite(r["ratio"])
            or not 0 < r["ratio"] <= 1.000001 for r in rows)
        counts["samples_with_duplicate_piece_key"] += len({
            (r["bin_idx"], r["sub_idx"]) for r in rows}) != len(rows)
        counts["samples_with_non_multiple_10_X_in"] += abs(
            rows[0]["X_in_m"] / 10 - round(rows[0]["X_in_m"] / 10)) > 1e-6
        counts["samples_violating_old_marked_grid"] += any(
            (r["bin_idx"] - 10) // 50 != r["seg_idx"] for r in marked)
        first = min((r["bin_idx"] for r in marked), default=None)
        first_own = [r for r in own_marked if r["bin_idx"] == first]
        first_partial = bool(first_own) and dist(first_own) < 9.99999
        counts["samples_with_partial_own_in_first_marked_bin"] += first_partial
        kinds = []
        if foreign:
            kinds.append("foreign_marked_piece")
        if any(0 < r["ratio"] < 0.999999 for r in outside):
            kinds.append("partial_own_outside_marked")
        if dist(own) - length > 10.01:
            kinds.append("all_own_overcounts_gt_10m")
        if first_partial:
            kinds.append("partial_first_marked_bin")
        for kind in kinds:
            if kind in examples:
                continue
            edge_bins = {first, max((r["bin_idx"] for r in marked), default=-1)}
            selected = [r for r in rows if r["bin_idx"] in edge_bins or r in outside]
            examples[kind] = dict(
                key=key, length_m=length, marked_m=dist(marked),
                marked_own_m=dist(own_marked), all_own_m=dist(own),
                rows=selected[:35], rows_truncated=len(selected) > 35)
    return dict(path=str(path), bytes=path.stat().st_size, counts=dict(counts),
                marked_length_error_m=dict(min=min(length_errors), max=max(length_errors)),
                examples=examples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/ref_links_biz_strata"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists; choose a new path")
    files = sorted(a.root.glob("link_*/samples_flat.parquet"))
    if not files:
        p.error("No reference package Parquet files found")
    results = []
    total = Counter()
    for path in files:
        result = audit_file(path)
        results.append(result)
        total.update(result["counts"])
        print(path.parent.name, json.dumps(result["counts"], sort_keys=True), flush=True)
    report = dict(
        scope="Local 13-link purposive reference packages; not population rates; "
              "no fresh HDFS verification or upstream coordinate assumption.",
        nonfinite_encoding="Nonfinite timing examples are strings: nan, inf, -inf.",
        total=dict(total), files=results)
    encoded = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        f.write(encoded + "\n")
    print("TOTAL", json.dumps(dict(total), sort_keys=True))


if __name__ == "__main__":
    main()
