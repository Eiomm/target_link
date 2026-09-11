"""Create one locally sorted, training-ready Cell-MAE observation partition.

The HDFS observations_v2 row set is accepted, but its physical parquet order is
not a safe contract.  This tool operates on ONE downloaded ``day/bucket`` at a
time, sorts complete rows by ``(cell_id, sample_id)``, validates the matching
training_groups_k3 partition exactly, and atomically publishes one parquet file.

It intentionally accepts local paths only.  Use
``scripts/prepare_full_cell_corpus_local.sh`` to download all HDFS partitions
with the Hadoop CLI; this avoids PyArrow/libhdfs JNI compatibility problems.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


REQUIRED_OBS = (
    "cell_id", "sample_id", "dt", "n_pieces", "T_diff", "ratio_pct",
    "observed", "valid", "bin_pos",
)
REQUIRED_GROUPS = ("group_id", "cell_id", "K", "group_size", "sample_ids")
RAGGED_COLUMNS = ("T_diff", "ratio_pct", "observed", "valid", "bin_pos")
MARKER = "_TRAINREADY.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--observations", required=True,
                   help="downloaded local observations day/bucket directory")
    p.add_argument("--groups", required=True,
                   help="downloaded local training_groups_k3 day/bucket directory")
    p.add_argument("--out", required=True,
                   help="new local output partition; must not already exist")
    p.add_argument("--row-group-size", type=int, default=131072)
    p.add_argument("--compression", default="snappy")
    return p


def _files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files under {directory}")
    return files


def _read_exact(files):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.ParquetFile(path).read() for path in files]
    schema = tables[0].schema
    for path, table in zip(files[1:], tables[1:]):
        if not table.schema.equals(schema, check_metadata=True):
            raise ValueError(f"schema differs across input files: {path}")
    return (pa.concat_tables(tables) if len(tables) > 1 else tables[0]), schema


def _single(column):
    return column.combine_chunks() if hasattr(column, "combine_chunks") else column


def _require(table, columns, label):
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise ValueError(f"{label} lacks columns: {','.join(missing)}")


def _validate_ragged(table) -> None:
    import numpy as np

    expected = table["n_pieces"].to_numpy(zero_copy_only=False).astype(np.int64)
    if np.any(expected < 0):
        raise ValueError("n_pieces contains a negative value")
    for name in RAGGED_COLUMNS:
        values = _single(table[name])
        if values.null_count:
            raise ValueError(f"{name} contains null lists")
        lengths = values.value_lengths().to_numpy(zero_copy_only=False)
        if not np.array_equal(lengths, expected):
            raise ValueError(f"{name} list lengths differ from n_pieces")


def _assert_sorted_unique(table) -> None:
    import numpy as np
    import pyarrow.compute as pc

    if table["cell_id"].null_count or table["sample_id"].null_count:
        raise ValueError("cell_id/sample_id contains null")
    cid = table["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    if cid.size > 1 and np.any(cid[1:] < cid[:-1]):
        raise ValueError("cell_id is not monotonic after persistence")
    if table.num_rows > 1:
        c = _single(table["cell_id"])
        s = _single(table["sample_id"])
        adjacent_cell = pc.equal(c.slice(1), c.slice(0, len(c) - 1))
        adjacent_sample = pc.equal(s.slice(1), s.slice(0, len(s) - 1))
        if bool(pc.any(pc.and_(adjacent_cell, adjacent_sample)).as_py()):
            raise ValueError("duplicate (cell_id, sample_id) observation rows")


def _sort_observations(table):
    import pyarrow.compute as pc

    order = pc.sort_indices(
        table, sort_keys=[("cell_id", "ascending"), ("sample_id", "ascending")])
    out = table.take(order)
    _assert_sorted_unique(out)
    return out


def _validate_groups(observations, groups) -> dict:
    """Exact K and member-pair equality for every grouped (K>=3) cell."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    _require(groups, REQUIRED_GROUPS, "groups")
    if groups["cell_id"].null_count or groups["sample_ids"].null_count:
        raise ValueError("groups cell_id/sample_ids contains null")

    sid_lists = _single(groups["sample_ids"])
    group_sizes = groups["group_size"].to_numpy(zero_copy_only=False).astype(np.int64)
    actual_sizes = sid_lists.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(group_sizes, actual_sizes):
        raise ValueError("group_size differs from len(sample_ids)")
    if group_sizes.size and (group_sizes.min() < 1 or group_sizes.max() > 16):
        raise ValueError("group_size is outside 1..16")

    cid = observations["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    obs_cells, obs_counts = np.unique(cid, return_counts=True)
    count_of = dict(zip(obs_cells.tolist(), obs_counts.tolist()))
    group_cells = groups["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    group_k = groups["K"].to_numpy(zero_copy_only=False).astype(np.int64)
    if group_k.size and group_k.min() < 3:
        raise ValueError("training_groups_k3 contains K < 3")
    for cell, k in zip(group_cells, group_k):
        if count_of.get(int(cell), 0) != int(k):
            raise ValueError(
                f"group cell {cell} has K={k}, observations hold {count_of.get(int(cell), 0)}")

    # Explode the group membership into (cell_id, sample_id), sort it, and
    # compare with observations restricted to cells represented by groups.
    # This proves exact membership without building a billion-row Python set.
    parent = pc.list_parent_indices(sid_lists)
    member_cells = pc.take(_single(groups["cell_id"]), parent)
    member_samples = pc.list_flatten(sid_lists)
    members = pa.table({"cell_id": member_cells, "sample_id": member_samples})
    member_order = pc.sort_indices(
        members, sort_keys=[("cell_id", "ascending"), ("sample_id", "ascending")])
    members = members.take(member_order)

    unique_group_cells = pa.array(np.unique(group_cells), type=_single(observations["cell_id"]).type)
    in_group = pc.is_in(_single(observations["cell_id"]), value_set=unique_group_cells)
    obs_members = observations.select(["cell_id", "sample_id"]).filter(in_group)
    if obs_members.num_rows != members.num_rows:
        raise ValueError(
            f"group members cover {members.num_rows} rows, expected {obs_members.num_rows}")
    for name in ("cell_id", "sample_id"):
        equal = pc.equal(_single(obs_members[name]), _single(members[name]))
        if len(equal) and not bool(pc.all(equal).as_py()):
            raise ValueError(f"groups have different observation membership ({name})")

    return {
        "group_rows": groups.num_rows,
        "group_cells": int(len(np.unique(group_cells))),
        "group_members": members.num_rows,
    }


def normalize_partition(observations, groups, out, row_group_size=131072,
                        compression="snappy") -> dict:
    import pyarrow.parquet as pq

    observations = Path(observations)
    groups = Path(groups)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    obs_files, group_files = _files(observations), _files(groups)
    obs, schema = _read_exact(obs_files)
    group_table, _ = _read_exact(group_files)
    _require(obs, REQUIRED_OBS, "observations")
    _validate_ragged(obs)
    sorted_obs = _sort_observations(obs)
    del obs
    group_stats = _validate_groups(sorted_obs, group_table)
    del group_table

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        part = temp / "part-00000.parquet"
        pq.write_table(sorted_obs, part, compression=compression,
                       row_group_size=row_group_size)
        persisted = pq.ParquetFile(part).read(columns=["cell_id", "sample_id"])
        _assert_sorted_unique(persisted)
        if persisted.num_rows != sorted_obs.num_rows:
            raise AssertionError("persisted row count changed")
        if not pq.ParquetFile(part).schema_arrow.equals(schema, check_metadata=True):
            raise AssertionError("persisted schema changed")

        import numpy as np
        cid = sorted_obs["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        manifest = {
            "format": "target_link_observations_trainready_v1",
            "sort_keys": ["cell_id", "sample_id"],
            "source_files": len(obs_files),
            "source_group_files": len(group_files),
            "rows": sorted_obs.num_rows,
            "cells": int(len(np.unique(cid))),
            "cell_id_min": int(cid[0]) if cid.size else None,
            "cell_id_max": int(cid[-1]) if cid.size else None,
            "compression": compression,
            "row_group_size": row_group_size,
            **group_stats,
        }
        (temp / MARKER).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
        os.replace(temp, out)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    a = parser().parse_args()
    result = normalize_partition(a.observations, a.groups, a.out,
                                 a.row_group_size, a.compression)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
