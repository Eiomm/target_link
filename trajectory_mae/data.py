"""Partition discovery and iteration for the canonical trajectory groups."""
from __future__ import annotations

import os

import numpy as np
from .columns import _array, _flat, _offsets, _row_any
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


from .grouping import ObservationGroups, N_BINS, _seed
_COLUMNS = ("cell_id", "sample_id", "dt", "T_diff", "ratio_pct", "valid", "bin_pos")


class _Store:
    def __init__(self, root: str):
        uri = str(root)
        if "://" in uri:
            self.fs, base = pfs.FileSystem.from_uri(uri)
        else:
            self.fs, base = pfs.LocalFileSystem(), os.path.abspath(uri)
        self.base = base.rstrip("/")

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


class CellDataset(ObservationGroups, IterableDataset):
    """Yield freshly-built, fixed-member groups from ``observations_v2``.

    ``partitions`` is a public iterable of ``(root, day, bucket)`` tuples.  A
    day/bucket must occur in at most one supplied corpus root: silent merging
    would duplicate cells and make the split ambiguous.
    """

    def __init__(self, roots: list[str], days: list[str], m_max: int = 64,
                 seed: int = 20260921, epoch: int = 0, max_groups=None,
                 groups_per_partition=None, freeze_selection: bool = False):
        super().__init__(m_max, seed, epoch)
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
        self.max_groups = max_groups
        self.groups_per_partition = groups_per_partition
        self.freeze_selection = bool(freeze_selection)

        wanted = {str(day).removeprefix("day=") for day in days}
        found: dict[tuple[str, str], tuple[_Store, str]] = {}
        self._observation_v3 = {}
        self._prepared = {}
        self._tensors = {}
        for root in roots:
            root = str(root).rstrip("/")
            if "://" not in root:
                # A published v3 corpus owns its filtered rows and precomputed
                # group membership.  Detect it before all older protocols.
                from .observation_v3 import manifest as observation_v3_manifest
                observation_v3_ready = observation_v3_manifest(root, self.m_max, self.seed)
                if observation_v3_ready:
                    observation_v3_root, info = observation_v3_ready
                    store = _Store(str(observation_v3_root / 'observations_v3'))
                    for key, receipt in info['partitions'].items():
                        day, bucket = key.split('/')
                        if day not in wanted:
                            continue
                        if (day, bucket) in found:
                            raise ValueError(f'duplicate observations partition day={day}/bucket={bucket}')
                        found[(day, bucket)] = (store, root)
                        self._observation_v3[(store.base, day, bucket)] = (observation_v3_root, receipt)
                    continue
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
        observation_v3 = self._observation_v3.get((store.base, day, bucket))
        if observation_v3:
            from .observation_v3 import load
            return load(store, day, bucket, *observation_v3, self.m_max)
        tensor = self._tensors.get((store.base, day, bucket))
        if tensor:
            from .tensor_corpus import load
            return load(*tensor, day, bucket, self.m_max)
        prepared = self._prepared.get((store.base, day, bucket))
        if prepared:
            from .prepared import load
            return load(store, day, bucket, *prepared, self.m_max)
        return super()._load_partition(store, day, bucket)


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
