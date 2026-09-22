"""Dataset for the standalone trajectory-MLP experiment.

The input corpus is deliberately only ``observations_v2``.  Old
``training_groups_k3`` files encode the former 16-member, balanced-tail policy
and are therefore not a valid index for this experiment.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


N_BINS = 50
_COLUMNS = ("cell_id", "sample_id", "dt", "T_diff", "ratio_pct", "valid", "bin_pos")


def _seed(*parts: object) -> int:
    """Stable seed; unlike ``hash()``, this is unchanged between processes."""
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")


class _Store:
    def __init__(self, root: str):
        uri = str(root)
        if "://" in uri:
            self.fs, base = pfs.FileSystem.from_uri(uri)
        else:
            self.fs, base = pfs.LocalFileSystem(), os.path.abspath(uri)
        self.base = (base or "").rstrip("/")

    def _infos(self, rel=""):
        path = (self.base + "/" + rel.strip("/")).rstrip("/")
        return self.fs.get_file_info(pfs.FileSelector(path, recursive=False,
                                                       allow_not_found=True))

    def dirs(self, rel=""):
        return sorted(x.base_name for x in self._infos(rel)
                      if x.type == pfs.FileType.Directory)

    def files(self, rel):
        return sorted(x.path for x in self._infos(rel)
                      if x.type == pfs.FileType.File and x.path.endswith(".parquet"))

    def read(self, path):
        # ParquetFile avoids Hive partition columns and reads one physical file.
        with self.fs.open_input_file(path) as source:
            return pq.ParquetFile(source).read(columns=_COLUMNS)


def _array(column):
    return column.combine_chunks() if hasattr(column, "combine_chunks") else column


def _flat(column):
    return pc.list_flatten(_array(column))


def _offsets(column):
    values = _array(column).value_lengths().to_numpy(zero_copy_only=False)
    result = np.empty(len(values) + 1, dtype=np.int64)
    result[0] = 0
    np.cumsum(values, out=result[1:])
    return result


def _row_any(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Vectorized ``any`` for a ragged boolean array."""
    out = np.zeros(len(offsets) - 1, dtype=bool)
    if values.size:
        # Prefix sums also handles empty lists without an invalid reduceat.
        prefix = np.empty(values.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(values.astype(np.int64, copy=False), out=prefix[1:])
        out[:] = prefix[offsets[1:]] > prefix[offsets[:-1]]
    return out


class CellDataset(IterableDataset):
    """Yield freshly-built, fixed-member groups from ``observations_v2``.

    ``partitions`` is a public iterable of ``(root, day, bucket)`` tuples.  A
    day/bucket must occur in at most one supplied corpus root: silent merging
    would duplicate cells and make the split ambiguous.
    """

    def __init__(self, roots: list[str], days: list[str], m_max: int = 64,
                 seed: int = 20260921, epoch: int = 0, max_groups=None,
                 groups_per_partition=None, freeze_selection: bool = False):
        super().__init__()
        if not roots:
            raise ValueError("roots must not be empty")
        if not days:
            raise ValueError("days must not be empty")
        if m_max < 3:
            raise ValueError("m_max must be at least 3")
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be positive or None")
        if groups_per_partition is not None and groups_per_partition <= 0:
            raise ValueError("groups_per_partition must be positive or None")
        self.m_max, self.seed, self.epoch = int(m_max), int(seed), int(epoch)
        self.max_groups = max_groups
        self.groups_per_partition = groups_per_partition
        self.freeze_selection = bool(freeze_selection)

        wanted = {str(day).removeprefix("day=") for day in days}
        found: dict[tuple[str, str], tuple[_Store, str]] = {}
        self._prepared = {}
        self._tensors = {}
        for root in roots:
            root = str(root).rstrip("/")
            if "://" not in root:
                from .tensor_corpus import manifest as tensor_manifest
                tensor_ready = tensor_manifest(root, self.m_max, self.seed)
                if tensor_ready:
                    tensor_root, info = tensor_ready
                    store = _Store(str(tensor_root))
                    for key, receipt in info['partitions'].items():
                        day, bucket = key.split('/')
                        if day not in wanted:
                            continue
                        if (day, bucket) in found:
                            raise ValueError(f'duplicate observations partition day={day}/bucket={bucket}')
                        found[(day, bucket)] = (store, root)
                        self._tensors[(store.base, day, bucket)] = (tensor_root, receipt)
                    continue
            # Accept either a corpus root or observations_v2 itself.
            obs_root = root if root.endswith("/observations_v2") else root + "/observations_v2"
            store = _Store(obs_root)
            if "://" not in root:
                from .prepared import manifest
                ready = manifest(root, self.m_max, self.seed)
                if ready:
                    prepared_root, info = ready
                    for key, receipt in info['partitions'].items():
                        self._prepared[(store.base, *key.split('/'))] = (prepared_root, receipt)
            for day_dir in store.dirs():
                if not day_dir.startswith("day=") or day_dir[4:] not in wanted:
                    continue
                for bucket_dir in store.dirs(day_dir):
                    if not bucket_dir.startswith("bucket="):
                        continue
                    key = (day_dir[4:], bucket_dir[7:])
                    if key in found:
                        raise ValueError("duplicate observations partition day=%s/bucket=%s "
                                         "in corpus roots %s and %s" %
                                         (key[0], key[1], found[key][1], root))
                    if "://" not in root and ready and (store.base, *key) not in self._prepared:
                        raise ValueError(f'Partition not registered in prepared manifest: {key}')
                    found[key] = (store, root)
        self.partitions = [(store, root, day, bucket)
                           for (day, bucket), (store, root) in sorted(found.items())]
        if not self.partitions:
            raise ValueError("no requested observations_v2 day/bucket partitions found")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        return self

    def n_partitions(self):
        return len(self.partitions)

    def _load_partition(self, store, day, bucket):
        tensor = getattr(self, '_tensors', {}).get((store.base, day, bucket))
        if tensor:
            from .tensor_corpus import load
            return load(*tensor, day, bucket, self.m_max)
        prepared = self._prepared.get((store.base, day, bucket))
        if prepared:
            from .prepared import load
            return load(store, day, bucket, *prepared, self.m_max)
        tables = [store.read(path) for path in store.files("day=%s/bucket=%s" % (day, bucket))]
        if not tables:
            return None
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        original_cid = table["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        original_sid = np.asarray(table["sample_id"].to_pylist(), dtype=object)
        identity_order = np.lexsort((original_sid, original_cid))
        identity_cid, identity_sid = original_cid[identity_order], original_sid[identity_order]
        if (len(identity_cid) > 1 and
                np.any((identity_cid[1:] == identity_cid[:-1]) &
                       (identity_sid[1:] == identity_sid[:-1]))):
            raise ValueError("duplicate sample_id within a cell")
        raw_ids, raw_counts = np.unique(original_cid, return_counts=True)
        raw_count = dict(zip(raw_ids.tolist(), raw_counts.tolist()))
        lengths = {name: _array(table[name]).value_lengths().to_numpy(zero_copy_only=False)
                   for name in ("T_diff", "ratio_pct", "valid", "bin_pos")}
        if any(not np.array_equal(lengths["T_diff"], lengths[name])
               for name in ("ratio_pct", "valid", "bin_pos")):
            raise ValueError("ragged observation arrays have different per-row lengths")
        off = _offsets(table["T_diff"])
        valid_flat = _flat(table["valid"]).to_numpy(zero_copy_only=False).astype(bool)
        # Filter before Python-level cell/group construction.  This is important
        # for the 1M-row partitions: trajectories with no possible label cannot
        # participate in a group and need not allocate a 50-bin tensor.
        candidate = _row_any(valid_flat, off)
        if not candidate.any():
            return None
        if not candidate.all():
            table = table.filter(pa.array(candidate))
            off = _offsets(table["T_diff"])
            valid_flat = _flat(table["valid"]).to_numpy(zero_copy_only=False).astype(bool)

        cid = table["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        sid = original_sid[candidate]
        dt = table["dt"].to_numpy(zero_copy_only=False).astype(np.float32)
        if not np.isfinite(dt).all() or (dt < 0).any() or (dt > 600).any():
            raise ValueError("delta_t outside finite [0, 600]")
        flat = {
            "T": _flat(table["T_diff"]).to_numpy(zero_copy_only=False).astype(np.float64),
            "R": _flat(table["ratio_pct"]).to_numpy(zero_copy_only=False).astype(np.float64),
            "V": valid_flat,
            "B": _flat(table["bin_pos"]).to_numpy(zero_copy_only=False).astype(np.int64),
            "off": off,
            # A prefix sum is much cheaper than 50-wide bincounts for the
            # overwhelmingly common all-valid trajectory.
            "row_all_valid": None,
        }
        if flat["B"].size and ((flat["B"] < 0).any() or (flat["B"] >= N_BINS).any()):
            raise ValueError("bin_pos outside 0..49")
        if flat["R"].size and (not np.isfinite(flat["R"]).all() or (flat["R"] < 0).any()):
            raise ValueError("ratio_pct must be finite and nonnegative")
        times = flat["T"][flat["V"]]
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("valid T_diff must be finite and nonnegative")
        valid_prefix = np.empty(len(valid_flat) + 1, dtype=np.int64)
        valid_prefix[0] = 0
        np.cumsum(valid_flat.astype(np.int64), out=valid_prefix[1:])
        flat["row_all_valid"] = ((valid_prefix[off[1:]] - valid_prefix[off[:-1]]) ==
                                 np.diff(off))
        return dict(cid=cid, sid=sid, dt=dt, flat=flat, raw_count=raw_count,
                    partition_stats={"day": day, "bucket": bucket,
                                     "raw_rows": int(len(original_cid)),
                                     "raw_cells": int(len(raw_count)),
                                     "candidate_rows": int(candidate.sum()),
                                     "usable_rows": 0,
                                     "dropped_no_valid": int((~candidate).sum()),
                                     "dropped_tail": 0, "groups": 0,
                                     "full_groups": 0, "tail_groups": 0,
                                     "selected_groups": 0})

    @staticmethod
    def _cells(cid):
        # Stable sorting makes corpus physical ordering irrelevant.
        order = np.argsort(cid, kind="stable")
        sorted_cid = cid[order]
        starts = np.r_[0, np.flatnonzero(sorted_cid[1:] != sorted_cid[:-1]) + 1]
        return order, starts, np.r_[starts[1:], len(order)]

    def _scatter(self, row, flat):
        a, b = int(flat["off"][row]), int(flat["off"][row + 1])
        bp, t, r, v = flat["B"][a:b], flat["T"][a:b], flat["R"][a:b] / 10.0, flat["V"][a:b]
        count = np.bincount(bp, minlength=N_BINS)
        vcount = np.bincount(bp[v], minlength=N_BINS)
        present = count > 0
        bin_valid = present & (count == vcount)
        x = np.zeros((N_BINS, 3), dtype=np.float32)
        x[:, 0] = np.where(bin_valid, np.bincount(bp, weights=np.where(v, t, 0.0), minlength=N_BINS), 0.0)
        x[:, 1] = np.where(present, np.bincount(bp, weights=r, minlength=N_BINS), 0.0)
        # Channel 2 is intentionally zero.  ``observed`` is not even read from
        # parquet, so it cannot leak into this experiment by accident.
        return x, bin_valid

    @staticmethod
    def _has_valid_bin(row, flat):
        a, b = int(flat["off"][row]), int(flat["off"][row + 1])
        bp, v = flat["B"][a:b], flat["V"][a:b]
        count = np.bincount(bp, minlength=N_BINS)
        return bool(np.any((count > 0) & (np.bincount(bp[v], minlength=N_BINS) == count)))

    def _group_specs(self, loaded, day, bucket):
        stats = loaded["partition_stats"]
        order, starts, ends = self._cells(loaded["cid"])
        for lo, hi in zip(starts, ends):
            rows = order[lo:hi]
            cell_id = int(loaded["cid"][rows[0]])
            raw_k = int(loaded["raw_count"][cell_id])
            candidate_k = len(rows)
            packed = []
            for row in rows:
                if (loaded["flat"]["row_all_valid"][row] or
                        self._has_valid_bin(int(row), loaded["flat"])):
                    packed.append((str(loaded["sid"][row]), int(row)))
            usable_k = len(packed)
            stats["usable_rows"] += usable_k
            stats["dropped_no_valid"] += candidate_k - usable_k
            if usable_k < 3:
                # This includes a cell whose raw K was >=3 but became too small
                # after invalid labels were removed, as well as raw K<3 cells.
                stats["dropped_tail"] += usable_k
                continue
            # sample_id order is part of the construction contract; the separate
            # stable RNG is then what defines persistent group membership.
            packed.sort(key=lambda p: p[0])
            if any(a[0] == b[0] for a, b in zip(packed, packed[1:])):
                raise ValueError("duplicate sample_id within cell_id=%d" % cell_id)
            perm = np.random.default_rng(_seed(self.seed, day, bucket, cell_id, "members")).permutation(usable_k)
            packed = [packed[int(i)] for i in perm]
            if usable_k % self.m_max < 3:
                stats["dropped_tail"] += usable_k % self.m_max
            for group_index, begin in enumerate(range(0, usable_k, self.m_max)):
                members = packed[begin:begin + self.m_max]
                if len(members) < 3:  # discard only the undersized tail; never rebalance
                    continue
                stats["groups"] += 1
                if len(members) == self.m_max:
                    stats["full_groups"] += 1
                else:
                    stats["tail_groups"] += 1
                group_id = "groups-v1-m%d/%s/%s/%s/%d" % (self.m_max, day, bucket,
                                                            cell_id, group_index)
                yield dict(
                    cell_id=cell_id, K=usable_k, K_raw=raw_k, group_size=len(members),
                    group_id=group_id, rows=np.asarray([row for _, row in members], dtype=np.int64),
                    day=day, bucket=bucket,
                    dropped_trajectories=raw_k - usable_k,
                    dropped_no_valid=raw_k - usable_k,
                    dropped_tail=(usable_k % self.m_max if usable_k % self.m_max < 3 else 0),
                )

    def _pack(self, loaded, spec):
        if spec.get('tensor_ready'):
            return dict(spec, epoch=self.epoch)
        rows = spec["rows"]
        x, bv = zip(*(self._scatter(int(row), loaded["flat"]) for row in rows))
        result = {key: value for key, value in spec.items() if key != "rows"}
        return dict(result, x=np.stack(x), bin_valid=np.stack(bv),
                    traj_valid=np.ones(len(rows), dtype=bool),
                    delta_t=loaded["dt"][rows].astype(np.float32, copy=True),
                    sample_ids=[str(loaded["sid"][row]) for row in rows], m_max=self.m_max,
                    n_bins=N_BINS, epoch=self.epoch)

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        if worker is not None and self.max_groups is not None:
            raise ValueError("max_groups is only supported with num_workers=0")
        # Partition ownership is disjoint, while the per-epoch order is shared.
        ordering_epoch = 0 if self.freeze_selection else self.epoch
        part_order = np.random.default_rng(_seed(self.seed, ordering_epoch, "partitions")).permutation(len(self.partitions))
        seen = 0
        for pi in part_order[wid::workers]:
            store, root, day, bucket = self.partitions[int(pi)]
            loaded = self._load_partition(store, day, bucket)
            if loaded is None:
                continue
            # Only compact row-index specs are retained for the epoch shuffle.
            # The [member,50,3] tensors are made only for yielded groups.
            groups = loaded.get("cached_groups")
            if groups is None:
                groups = list(self._group_specs(loaded, day, bucket))
            if not groups:
                continue
            group_order = np.random.default_rng(_seed(self.seed, ordering_epoch, day, bucket, "groups")).permutation(len(groups))
            if self.groups_per_partition is not None:
                select_epoch = ordering_epoch
                select = np.random.default_rng(_seed(self.seed, select_epoch, day, bucket, "select")).permutation(len(groups))
                selected = set(int(i) for i in select[:self.groups_per_partition])
                group_order = np.asarray([i for i in group_order if int(i) in selected], dtype=np.int64)
            loaded["partition_stats"]["selected_groups"] = int(len(group_order))
            # A shared finalized dict lets a multi-worker caller de-duplicate
            # stats by (day,bucket) after collate, while every yielded group
            # carries the complete accounting for its scanned partition.
            for gi in group_order:
                spec = groups[int(gi)]
                spec["partition_stats"] = loaded["partition_stats"]
                yield self._pack(loaded, spec)
                seen += 1
                if self.max_groups is not None and seen >= self.max_groups:
                    return


def collate_cells(items, m_max: int = 64, epoch: int = 0):
    """Pad variable groups and mask exactly ``floor(n_valid_traj / 2)`` rows."""
    if not items:
        raise ValueError("empty batch")
    B, M = len(items), int(m_max)
    tensor_ready = all(item.get('tensor_ready', False) for item in items)
    if tensor_ready and any(item['m_max'] != M for item in items):
        raise ValueError('Tensor corpus m_max does not match batch')
    x = np.stack([item['x'] for item in items]) if tensor_ready else np.zeros((B, M, N_BINS, 3), dtype=np.float32)
    bin_valid = np.stack([item['bin_valid'] for item in items]) if tensor_ready else np.zeros((B, M, N_BINS), dtype=bool)
    traj_valid = np.stack([item['traj_valid'] for item in items]) if tensor_ready else np.zeros((B, M), dtype=bool)
    delta_t = np.stack([item['delta_t'] for item in items]) if tensor_ready else np.zeros((B, M), dtype=np.float32)
    mae_mask = np.zeros((B, M), dtype=bool)
    for i, item in enumerate(items):
        n = len(item["sample_ids"])
        if n > M:
            raise ValueError("group size %d exceeds m_max=%d" % (n, M))
        if not tensor_ready:
            x[i, :n], bin_valid[i, :n] = item["x"][:n], item["bin_valid"][:n]
            traj_valid[i, :n], delta_t[i, :n] = item["traj_valid"][:n], item["delta_t"][:n]
        choices = np.flatnonzero(item["traj_valid"])
        hidden = len(choices) // 2
        if hidden:
            rng = np.random.default_rng(_seed(item["group_id"], int(epoch)))
            mae_mask[i, rng.choice(choices, size=hidden, replace=False)] = True
    return dict(
        x=torch.from_numpy(x), bin_valid=torch.from_numpy(bin_valid),
        traj_valid=torch.from_numpy(traj_valid), delta_t=torch.from_numpy(delta_t),
        mae_mask=torch.from_numpy(mae_mask),
        cell_id=torch.tensor([item["cell_id"] for item in items], dtype=torch.int64),
        K=torch.tensor([item["K"] for item in items], dtype=torch.int64),
        K_raw=torch.tensor([item.get("K_raw", item["K"]) for item in items], dtype=torch.int64),
        group_size=torch.tensor([item.get("group_size", len(item["sample_ids"])) for item in items], dtype=torch.int64),
        group_id=[item["group_id"] for item in items], sample_ids=[item["sample_ids"] for item in items],
        day=[item["day"] for item in items], bucket=[item["bucket"] for item in items],
        partition_stats=[item.get("partition_stats") for item in items],
        m_max=M, n_bins=N_BINS,
    )
