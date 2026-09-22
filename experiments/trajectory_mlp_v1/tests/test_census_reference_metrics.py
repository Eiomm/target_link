from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells
from experiments.trajectory_mlp_v1.tools.census_reference_metrics import reference_metrics


DAY = "20260817"
BUCKET = "0"
SEED = 99


def _write_source(root, rows):
    partition = root / "observations_v2" / f"day={DAY}" / f"bucket={BUCKET}"
    partition.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), partition / "part.parquet")


def _reduced(rows):
    reduced = []
    for row in rows:
        bitset = 0
        for bin_pos in set(row["bin_pos"]):
            at = [i for i, value in enumerate(row["bin_pos"]) if value == bin_pos]
            if all(row["valid"][i] for i in at):
                bitset |= 1 << bin_pos
        reduced.append(dict(cell_id=row["cell_id"], sample_id=row["sample_id"],
                            valid_bits=np.uint64(bitset),
                            n_valid=np.int16(sum(row["valid"])), usable=bool(bitset)))
    return pa.Table.from_pylist(reduced)


def _expected_from_dataset(root, *, m_max, epoch):
    items = list(CellDataset([str(root)], [DAY], m_max=m_max, seed=SEED, epoch=epoch))
    loaded = CellDataset([str(root)], [DAY], m_max=m_max, seed=SEED)._load_partition(
        CellDataset([str(root)], [DAY], m_max=m_max, seed=SEED).partitions[0][0], DAY, BUCKET
    )
    # _load_partition's stats are completed only when _group_specs is consumed.
    list(CellDataset([str(root)], [DAY], m_max=m_max, seed=SEED)._group_specs(loaded, DAY, BUCKET))
    batch = collate_cells(items, m_max=m_max, epoch=epoch)
    hidden = batch["mae_mask"].numpy()
    valid = batch["bin_valid"].numpy()
    visible = batch["traj_valid"].numpy() & ~hidden
    visible_by_bin = (visible[:, :, None] & valid).any(axis=1)
    supported = hidden[:, :, None] & valid & visible_by_bin[:, None, :]
    return {
        "retained_observations": sum(item["group_size"] for item in items),
        "dropped_no_valid": loaded["partition_stats"]["dropped_no_valid"],
        "dropped_tail": loaded["partition_stats"]["dropped_tail"],
        "groups": len(items),
        "hidden_trajectories": int(hidden.sum()),
        "supervised_bins": int((hidden[:, :, None] & valid).sum()),
        "supported_supervised_bins": int(supported.sum()),
    }


def _row(cell, sample, bins, valid):
    return dict(cell_id=cell, sample_id=sample, dt=np.float32(1),
                T_diff=[np.float32(i + 1) if ok else np.float32(np.nan)
                        for i, ok in enumerate(valid)],
                ratio_pct=[np.float32(10)] * len(bins), valid=valid, bin_pos=bins)


def test_matches_dataset_for_64_member_groups_and_all_tail_rules(tmp_path):
    rows = []
    # 65 and 66 discard tails of one and two; 67 keeps a three-member tail.
    for cell, count in ((10, 65), (20, 66), (30, 67)):
        rows.extend(_row(cell, f"{cell}-{i:03d}", [cell % 50], [True]) for i in range(count))
    # A no-valid member and a candidate with no fully-valid bin exercise both
    # branches counted by production's dropped_no_valid statistic.
    rows.extend([_row(40, "40-invalid", [1], [False]),
                 _row(40, "40-partial", [2, 2], [True, False])])
    rows.extend(_row(40, f"40-good-{i}", [3], [True]) for i in range(3))
    np.random.default_rng(8).shuffle(rows)
    _write_source(tmp_path, rows)

    actual = reference_metrics(_reduced(rows), DAY, BUCKET, seed=SEED, epoch=4)
    assert actual == _expected_from_dataset(tmp_path, m_max=64, epoch=4)
    assert actual["dropped_tail"] == 3
    assert actual["groups"] == 5


def test_matches_dataset_for_heterogeneous_members_and_epoch_mask(tmp_path):
    rows = []
    patterns = [([0, 1], [True, True]), ([1, 2], [True, True]),
                ([2, 3], [True, False]), ([3], [True]), ([4, 5], [True, True])]
    for i in range(68):
        bins, valid = patterns[i % len(patterns)]
        rows.append(_row(99, f"member-{67 - i:03d}", bins, valid))
    np.random.default_rng(19).shuffle(rows)
    _write_source(tmp_path, rows)
    table = _reduced(rows)

    for epoch in (0, 7):
        assert reference_metrics(table, DAY, BUCKET, seed=SEED, epoch=epoch) == _expected_from_dataset(
            tmp_path, m_max=64, epoch=epoch
        )
