"""Validate local train/val Cell-MAE corpus partitions before training.

This is the reusable form of the server preflight. It deliberately reads local
Parquet only: deployment scripts fetch with the Hadoop CLI first, avoiding the
PyArrow/libhdfs environment mismatch documented for the training server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.normalize_cell_partition import (  # noqa: E402
    REQUIRED_GROUPS,
    REQUIRED_OBS,
    _require,
    _single,
    _validate_groups,
    _validate_ragged,
)


OBS_TYPES = {
    "cell_id": "int64", "sample_id": "string", "dt": "float",
    "n_pieces": "int16", "T_diff": "float", "ratio_pct": "int8",
    "observed": "bool", "valid": "bool", "bin_pos": "int8",
}
GROUP_TYPES = {
    "group_id": "string", "cell_id": "int64", "K": "int32",
    "group_size": "int64", "sample_ids": "string", "window": "int64",
}
LIST_COLUMNS = {"T_diff", "ratio_pct", "observed", "valid", "bin_pos",
                "sample_ids"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True,
                   help="root containing train/ and val/ corpus directories")
    p.add_argument("--obs-dir", default="observations_v2")
    p.add_argument("--groups-dir", default="training_groups_k3")
    p.add_argument("--m-max", type=int, default=16)
    return p.parse_args(argv)


def _directories(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"missing directory: {path}")
    return sorted((entry for entry in path.iterdir() if entry.is_dir()),
                  key=lambda entry: entry.name)


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files under {path}")
    return files


def _read_tables(files: list[Path]):
    tables = [pq.ParquetFile(path).read() for path in files]
    schema = tables[0].schema
    for path, table in zip(files[1:], tables[1:]):
        if not table.schema.equals(schema, check_metadata=False):
            raise ValueError(f"schema differs across input files: {path}")
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _check_reader_keys(observations, label):
    if observations["cell_id"].null_count or observations["sample_id"].null_count:
        raise ValueError(f"cell_id/sample_id contains null: {label}")
    cell_id = observations["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    if cell_id.size > 1 and np.any(cell_id[1:] < cell_id[:-1]):
        raise ValueError(f"cell_id is not sorted: {label}")
    keys = observations.select(["cell_id", "sample_id"])
    order = pc.sort_indices(
        keys, sort_keys=[("cell_id", "ascending"), ("sample_id", "ascending")])
    keys = keys.take(order)
    same_cell = pc.equal(_single(keys["cell_id"]).slice(1),
                         _single(keys["cell_id"]).slice(0, len(keys) - 1))
    same_sample = pc.equal(_single(keys["sample_id"]).slice(1),
                           _single(keys["sample_id"]).slice(0, len(keys) - 1))
    if len(keys) > 1 and bool(pc.any(pc.and_(same_cell, same_sample)).as_py()):
        raise ValueError(f"duplicate (cell_id, sample_id): {label}")


def _check_types(table, expected, label):
    for name, want in expected.items():
        if name not in table.column_names:
            raise ValueError(f"{label} lacks column {name}")
        field_type = table.schema.field(name).type
        if name in LIST_COLUMNS:
            if not pa.types.is_list(field_type) or str(field_type.value_type) != want:
                raise ValueError(f"{label}.{name} is {field_type}, want list<{want}>")
        elif str(field_type) != want:
            raise ValueError(f"{label}.{name} is {field_type}, want {want}")


def check_partition(obs_path: Path, groups_path: Path, m_max: int) -> dict:
    obs_files, group_files = _parquet_files(obs_path), _parquet_files(groups_path)
    observations = _read_tables(obs_files)
    groups = _read_tables(group_files)
    _require(observations, REQUIRED_OBS, "observations")
    _require(groups, REQUIRED_GROUPS + ("window",), "groups")
    _check_types(observations, OBS_TYPES, str(obs_path))
    _check_types(groups, GROUP_TYPES, str(groups_path))
    if not observations.num_rows or not groups.num_rows:
        raise ValueError(f"empty observations/groups partition: {obs_path}")

    _validate_ragged(observations)
    _check_reader_keys(observations, str(obs_path))
    group_stats = _validate_groups(observations, groups)

    group_size = groups["group_size"].to_numpy(zero_copy_only=False)
    if group_size.min() < 1 or group_size.max() > m_max:
        raise ValueError(f"group_size outside 1..{m_max}: {groups_path}")

    bin_pos = pc.list_flatten(_single(observations["bin_pos"])).to_numpy(
        zero_copy_only=False)
    if not bin_pos.size or bin_pos.min() < 0 or bin_pos.max() > 49:
        raise ValueError(f"bin_pos empty or outside [0,49]: {obs_path}")
    valid = pc.list_flatten(_single(observations["valid"])).to_numpy(
        zero_copy_only=False)
    duration = pc.list_flatten(_single(observations["T_diff"])).to_numpy(
        zero_copy_only=False)
    bad_duration = int((valid & ~np.isfinite(duration)).sum())
    if bad_duration:
        raise ValueError(
            f"{bad_duration} valid pieces have nonfinite T_diff: {obs_path}")
    delta_t = observations["dt"].to_numpy(zero_copy_only=False)
    if not np.isfinite(delta_t).all() or delta_t.min() < 0 or delta_t.max() > 600:
        raise ValueError(f"dt is nonfinite or outside [0,600]: {obs_path}")

    return {
        "observation_files": len(obs_files),
        "group_files": len(group_files),
        "observations": observations.num_rows,
        "groups": groups.num_rows,
        "bin_pos_min": int(bin_pos.min()),
        "bin_pos_max": int(bin_pos.max()),
        "group_size_min": int(group_size.min()),
        "group_size_max": int(group_size.max()),
        "dt_at_upper_edge": int((delta_t >= 600).sum()),
        **group_stats,
    }


def check_corpus(data, obs_dir="observations_v2",
                 groups_dir="training_groups_k3", m_max=16) -> dict:
    if m_max <= 0:
        raise ValueError("m_max must be positive")
    data = Path(data)
    result = {"data": str(data), "obs_dir": obs_dir,
              "groups_dir": groups_dir, "splits": {}}
    for split in ("train", "val"):
        obs_root, group_root = data / split / obs_dir, data / split / groups_dir
        obs_days, group_days = _directories(obs_root), _directories(group_root)
        if [p.name for p in obs_days] != [p.name for p in group_days]:
            raise ValueError(f"{split}: observation/group days differ")
        split_stats = []
        for obs_day, group_day in zip(obs_days, group_days):
            obs_buckets, group_buckets = _directories(obs_day), _directories(group_day)
            if [p.name for p in obs_buckets] != [p.name for p in group_buckets]:
                raise ValueError(f"{split}/{obs_day.name}: buckets differ")
            for obs_bucket, group_bucket in zip(obs_buckets, group_buckets):
                stats = check_partition(obs_bucket, group_bucket, m_max)
                stats["partition"] = f"{obs_day.name}/{obs_bucket.name}"
                split_stats.append(stats)
                print(json.dumps({"split": split, **stats}, sort_keys=True), flush=True)
        if not split_stats:
            raise ValueError(f"{split}: no aligned partitions")
        result["splits"][split] = {
            "partitions": len(split_stats),
            "observations": sum(s["observations"] for s in split_stats),
            "groups": sum(s["groups"] for s in split_stats),
        }
    return result


def main():
    a = parse_args()
    result = check_corpus(a.data, a.obs_dir, a.groups_dir, a.m_max)
    print(json.dumps({"corpus_ok": True, **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
