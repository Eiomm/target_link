"""Reader for the published, filtered ``observations_v3`` corpus.

The builder owns the expensive validation, sorting, filtering, and group
construction.  Training verifies publication metadata and the small index on
every load, while treating the immutable parquet payload like the tensor
corpus: its recorded size is checked without re-hashing it each epoch.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .prepared import Groups


FORMAT = 'trajectory_mlp_observation_v3'
_MARKER = '_OBSERVATION_V3_SUCCESS.json'
_BUILDING = '_BUILDING'
_STATS = frozenset((
    'raw_rows', 'raw_cells', 'candidate_rows', 'usable_rows',
    'dropped_no_valid', 'dropped_tail', 'groups', 'full_groups',
    'tail_groups', 'selected_groups', 'day', 'bucket',
))


def _corpus_root(root):
    root = Path(root)
    return root.parent if root.name == 'observations_v3' else root


def _validate_record(receipt, key):
    rec = receipt.get(key)
    if not isinstance(rec, dict) or not {'path', 'bytes', 'sha256'}.issubset(rec):
        raise ValueError(f'Invalid observation-v3 {key} receipt')
    if (not isinstance(rec['path'], str) or not isinstance(rec['bytes'], int)
            or rec['bytes'] < 0 or not isinstance(rec['sha256'], str)):
        raise ValueError(f'Invalid observation-v3 {key} receipt')


def _validate_manifest(info, m_max, seed):
    if not isinstance(info, dict) or (info.get('format'), info.get('m_max'),
                                      info.get('data_seed')) != (FORMAT, m_max, seed):
        raise ValueError('Observation-v3 protocol mismatch; use matching m_max/data_seed')
    partitions = info.get('partitions')
    if not isinstance(partitions, dict):
        raise ValueError('Invalid observation-v3 partition manifest')
    for partition, receipt in partitions.items():
        if not isinstance(partition, str) or len(partition.split('/')) != 2:
            raise ValueError('Invalid observation-v3 partition key')
        day, bucket = partition.split('/')
        if not isinstance(receipt, dict) or not isinstance(receipt.get('stored_rows'), int):
            raise ValueError('Invalid observation-v3 partition receipt')
        _validate_record(receipt, 'observations')
        _validate_record(receipt, 'index')
        stats = receipt.get('stats')
        if (not isinstance(stats, dict) or not _STATS.issubset(stats)
                or stats['day'] != day or str(stats['bucket']) != bucket):
            raise ValueError('Invalid observation-v3 partition statistics')


def manifest(root, m_max, seed):
    """Return ``(corpus_root, marker)`` for a published v3 corpus, if any.

    An existing v3 directory is deliberately not treated as a raw-v2 corpus:
    without its success marker it is an unpublished build.
    """
    root = _corpus_root(root)
    if (root / _BUILDING).exists():
        raise ValueError(f'Observation-v3 corpus is not published: {root}')
    marker = root / _MARKER
    if not marker.exists():
        if (root / 'observations_v3').exists():
            raise ValueError(f'Observation-v3 corpus is not published: {root}')
        return None
    try:
        info = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid observation-v3 marker: {marker}') from exc
    _validate_manifest(info, int(m_max), int(seed))
    return root, info


def _artifact(root, rec, name):
    path = root / rec['path']
    # Receipt paths are publication-relative; do not let a corrupt marker
    # point a training process outside of the corpus root.
    resolved_root, resolved_path = root.resolve(), path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f'Invalid observation-v3 {name} path')
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f'Observation-v3 artifact missing: {path}') from exc
    if size != rec['bytes']:
        raise ValueError(f'Observation-v3 artifact changed: {path}')
    return path


def load(store, day, bucket, root, receipt, m_max):
    """Load the published rows and the immutable compact group index."""
    from .columns import _flat, _offsets

    observation_path = _artifact(root, receipt['observations'], 'observations')
    index_path = _artifact(root, receipt['index'], 'index')
    index_data = index_path.read_bytes()
    if hashlib.sha256(index_data).hexdigest() != receipt['index']['sha256']:
        raise ValueError('Observation-v3 group index checksum mismatch')
    try:
        with np.load(io.BytesIO(index_data), allow_pickle=False) as archive:
            index = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError('Invalid observation-v3 group index') from exc
    required = {'ptr', 'rows', 'cell_id', 'K', 'K_raw', 'group_index'}
    if set(index) != required:
        raise ValueError('Invalid observation-v3 group index schema')
    groups = len(index['cell_id'])
    ptr, rows = index['ptr'], index['rows']
    index_names = ('ptr', 'rows', 'cell_id', 'K', 'K_raw', 'group_index')
    if (any(not np.issubdtype(index[name].dtype, np.integer) for name in index_names)
            or ptr.ndim != 1 or len(ptr) != groups + 1 or rows.ndim != 1
            or any(index[name].ndim != 1 or len(index[name]) != groups
                   for name in ('cell_id', 'K', 'K_raw', 'group_index'))
            or len(ptr) == 0 or ptr[0] != 0 or ptr[-1] != len(rows)
            or np.any(np.diff(ptr) < 3) or np.any(np.diff(ptr) > m_max) or np.any(rows < 0)
            or np.any((index['K'] < 3) | (index['K_raw'] < index['K']))):
        raise ValueError('Invalid observation-v3 group index')

    table = store.read(str(observation_path))
    if len(table) != receipt['stored_rows']:
        raise ValueError('Observation-v3 stored row count mismatch')
    if len(rows) and np.max(rows) >= len(table):
        raise ValueError('Observation-v3 group row index out of range')
    flat = {
        'T': _flat(table['T_diff']).to_numpy(zero_copy_only=False).astype(np.float64),
        'R': _flat(table['ratio_pct']).to_numpy(zero_copy_only=False).astype(np.float64),
        'V': _flat(table['valid']).to_numpy(zero_copy_only=False).astype(bool),
        'B': _flat(table['bin_pos']).to_numpy(zero_copy_only=False).astype(np.int64),
        'off': _offsets(table['T_diff']),
    }
    return dict(
        cid=table['cell_id'].to_numpy(zero_copy_only=False).astype(np.int64),
        sid=np.asarray(table['sample_id'].to_pylist(), dtype=object),
        dt=table['dt'].to_numpy(zero_copy_only=False).astype(np.float32),
        flat=flat,
        partition_stats=dict(receipt['stats']),
        cached_groups=Groups(index, day, bucket, m_max),
    )
