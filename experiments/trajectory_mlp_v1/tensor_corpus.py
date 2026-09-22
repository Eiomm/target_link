"""Versioned, padded group tensors. No pickle or observation folding at training time."""
from __future__ import annotations

import json
import zlib
from pathlib import Path
import numpy as np
from .prepared import sha256

FORMAT = 'trajectory_mlp_tensors_v1'
CHANNELS = ['piece_time_sum_seconds', 'piece_ratio_sum', 'zero']
STORAGE = 'zlib-group-blocks-v1'


def manifest(root, m_max, seed):
    root = Path(root)
    marker = root / '_TENSORS_SUCCESS.json'
    if (root / '_TENSORS_BUILDING').exists():
        raise ValueError(f'Tensor corpus is not published: {root}')
    if not marker.exists():
        return None
    info = json.loads(marker.read_text())
    if (info['format'], info['m_max'], info['data_seed'], info['channels']) != (FORMAT, m_max, seed, CHANNELS):
        raise ValueError('Tensor corpus protocol mismatch; use matching m_max/data_seed')
    if info.get('storage') != STORAGE:
        raise ValueError('Tensor storage protocol mismatch')
    return root, info


class TensorGroups:
    def __init__(self, root, receipt, day, bucket, m_max):
        self.day, self.bucket, self.m = day, bucket, m_max
        self.arrays = {}
        for name, rec in receipt['arrays'].items():
            path = root / rec['path']
            if path.stat().st_size != rec['bytes']:
                raise ValueError(f'Tensor artifact changed: {path}')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(a.shape) != rec['shape'] or a.dtype.str != rec['dtype']:
                raise ValueError(f'Tensor schema changed: {path}')
            self.arrays[name] = a
        rec = receipt['payload']
        self.payload_path = root / rec['path']
        if self.payload_path.stat().st_size != rec['bytes']:
            raise ValueError(f'Tensor artifact changed: {self.payload_path}')
        offsets, meta = self.arrays['offsets'], self.arrays['meta']
        if (meta.shape != (len(meta), 5) or offsets.shape != (len(meta)+1,)
                or offsets[0] != 0 or offsets[-1] != rec['bytes']
                or np.any(np.diff(offsets) <= 0)
                or np.any((meta[:,4] < 3) | (meta[:,4] > m_max))):
            raise ValueError('Invalid tensor group index')
        self._payload = self.payload_path.open('rb')
        self.layout = receipt['layout']
        self.stats = dict(receipt['stats'])

    def __del__(self):
        payload = getattr(self, '_payload', None)
        if payload is not None:
            payload.close()

    def __len__(self):
        return len(self.arrays['meta'])

    def __getitem__(self, i):
        a = self.arrays
        cell, k, raw, group_index, n = map(int, a['meta'][i])
        start, stop = map(int, a['offsets'][i:i+2])
        self._payload.seek(start)
        data = zlib.decompress(self._payload.read(stop-start))
        offset, tensors = 0, {}
        for name, spec in self.layout.items():
            dtype = np.dtype(spec['dtype'])
            size = int(np.prod(spec['shape'])) * dtype.itemsize
            tensors[name] = np.frombuffer(data, dtype=dtype, count=size//dtype.itemsize,
                                         offset=offset).reshape(spec['shape'])
            offset += size
        if offset != len(data):
            raise ValueError('Tensor payload size mismatch')
        return dict(cell_id=cell, K=k, K_raw=raw, group_size=n,
                    group_id=f'groups-v1-m{self.m}/{self.day}/{self.bucket}/{cell}/{group_index}',
                    day=self.day, bucket=self.bucket,
                    dropped_trajectories=raw-k, dropped_no_valid=raw-k,
                    dropped_tail=k % self.m if k % self.m < 3 else 0,
                    x=tensors['x'], bin_valid=tensors['bin_valid'],
                    traj_valid=tensors['traj_valid'], delta_t=tensors['delta_t'],
                    sample_ids=[s.decode('utf-8') for s in tensors['sample_ids'][:n]],
                    m_max=self.m, n_bins=50, tensor_ready=True)


def load(root, receipt, day, bucket, m_max):
    groups = TensorGroups(root, receipt, day, bucket, m_max)
    return dict(cached_groups=groups, partition_stats=groups.stats)


def verify(root):
    """Explicit full content verification; training only checks sizes and schemas."""
    root = Path(root)
    info = json.loads((root / '_TENSORS_SUCCESS.json').read_text())
    manifest(root, info['m_max'], info['data_seed'])
    for receipt in info['partitions'].values():
        for rec in [*receipt['arrays'].values(), receipt['payload']]:
            if sha256(root / rec['path']) != rec['sha256']:
                raise ValueError(f'Tensor checksum mismatch: {rec["path"]}')
    return info
