"""Publish filtered ragged observations with immutable, equivalent group indices.

No padding, piece folding, or epoch mask is persisted. Original columns are
retained for traceability; the training reader projects its seven columns.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FORMAT = 'trajectory_mlp_observation_v3'


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def build_partition(job):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch
    from trajectory_mae.grouping import ObservationGroups
    from trajectory_mae.data import CellDataset, _COLUMNS, _flat, _offsets, _row_any
    from trajectory_mae.prepared import file_record
    from trajectory_mae.observation_v3 import load

    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, destination, day, bucket, m_max, seed = job
    started = time.monotonic()
    source, root = Path(source), Path(destination)
    src = source / 'observations_v2' / f'day={day}' / f'bucket={bucket}'
    files = sorted(src.glob('*.parquet'))
    if not files:
        raise ValueError(f'Missing input: {src}')
    sources = [file_record(p, source) for p in files]
    receipt_path = root / 'receipts' / f'{day}_{bucket}.json'
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        if rec['source'] != sources or rec['m_max'] != m_max or rec['data_seed'] != seed:
            raise ValueError(f'Cached partition protocol/source changed: {src}')
        for key in ('observations', 'index', 'cells'):
            actual = file_record(root / rec[key]['path'], root)
            if any(actual[k] != rec[key][k] for k in ('path', 'bytes', 'sha256')):
                raise ValueError(f'Cached artifact changed: {key}, {receipt_path}')
        return day, bucket, rec

    table = pa.concat_tables([pq.ParquetFile(p).read() for p in files]).combine_chunks()
    for name in _COLUMNS:
        if table[name].null_count:
            raise ValueError(f'Null observation field: {name}')
    for name in ('T_diff', 'ratio_pct', 'valid', 'bin_pos'):
        if pc.list_flatten(table[name]).null_count:
            raise ValueError(f'Null array element: {name}')
    cid = table['cell_id'].to_numpy()
    if not np.all(cid % 128 == int(bucket)):
        raise ValueError(f'Wrong cell bucket: {src}')
    # Sorting is done once, offline. Preserve row fields and ragged arrays.
    table = table.take(pc.sort_indices(table, sort_keys=[('cell_id', 'ascending'), ('sample_id', 'ascending')]))
    raw_cells = len(np.unique(cid))
    raw_rows = len(table)
    candidates = np.flatnonzero(_row_any(
        _flat(table['valid']).to_numpy(zero_copy_only=False).astype(bool), _offsets(table['valid'])))

    class MemoryStore:
        base = 'memory'
        def files(self, rel):
            return ['memory']
        def read(self, path):
            return table.select(_COLUMNS)

    ds = ObservationGroups(m_max, seed)
    loaded = ds._load_partition(MemoryStore(), day, bucket)
    groups = list(ds._group_specs(loaded, day, bucket)) if loaded is not None else []
    if loaded is None:
        usable = np.empty(0, dtype=np.int64)
        stats = dict(day=day, bucket=bucket, raw_rows=raw_rows, raw_cells=raw_cells,
                     candidate_rows=0, usable_rows=0, dropped_no_valid=raw_rows,
                     dropped_tail=0, groups=0, full_groups=0, tail_groups=0, selected_groups=0)
    else:
        flat = loaded['flat']
        good = flat['row_all_valid'].copy()
        for i in np.flatnonzero(~good):
            good[i] = ds._has_valid_bin(int(i), flat)
        usable = np.flatnonzero(good)
        stats = dict(loaded['partition_stats'])
    if len(usable) != stats['usable_rows']:
        raise ValueError('Filtering differs from training reader')
    filtered = table.take(pa.array(candidates[usable]))
    remap = np.full(len(candidates), -1, dtype=np.int64)
    remap[usable] = np.arange(len(usable))
    ptr = np.r_[0, np.cumsum([g['group_size'] for g in groups])].astype(np.int64)
    old_rows = np.concatenate([g['rows'] for g in groups]) if groups else np.empty(0, dtype=np.int64)
    rows = remap[old_rows]
    if (rows < 0).any() or len(np.unique(rows)) != len(rows):
        raise ValueError('Group index contains invalid or repeated members')
    if len(rows) + stats['dropped_tail'] != len(filtered):
        raise ValueError('Filtered/group/tail accounting mismatch')
    for g in groups:
        if not np.all(loaded['cid'][g['rows']] == g['cell_id']):
            raise ValueError('Cross-cell group')

    dst = root / 'observations_v3' / f'day={day}' / f'bucket={bucket}'
    idx = root / 'group_indices' / f'day={day}' / f'bucket={bucket}'
    dst.mkdir(parents=True, exist_ok=True)
    idx.mkdir(parents=True, exist_ok=True)
    output = dst / 'part-00000.parquet'
    pq.write_table(filtered, output, compression='snappy', row_group_size=131072)
    # Compare every persisted field including NaNs and list boundaries.
    saved = pq.ParquetFile(output).read().combine_chunks()
    if saved.schema != filtered.schema or len(saved) != len(filtered):
        raise ValueError('Observation schema/row count changed')
    for name in filtered.column_names:
        a, b = filtered[name].combine_chunks(), saved[name].combine_chunks()
        if a.equals(b):
            continue
        if pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
            if not a.offsets.equals(b.offsets):
                raise ValueError(f'List boundaries changed: {name}')
            a, b = a.values, b.values
        if not a.is_null().equals(b.is_null()):
            raise ValueError(f'Null positions changed: {name}')
        av, bv = a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False)
        if av.dtype.kind not in 'fc' or not np.array_equal(av, bv, equal_nan=True):
            raise ValueError(f'Observation values changed: {name}')
    del saved, table
    index_path = idx / 'groups.npz'
    index = dict(ptr=ptr, rows=rows,
                 cell_id=np.array([g['cell_id'] for g in groups], dtype=np.int64),
                 K=np.array([g['K'] for g in groups], dtype=np.int64),
                 K_raw=np.array([g['K_raw'] for g in groups], dtype=np.int64),
                 group_index=np.array([int(g['group_id'].rsplit('/', 1)[1]) for g in groups], dtype=np.int64))
    np.savez(index_path, **index)
    with np.load(index_path, allow_pickle=False) as reread:
        if any(not np.array_equal(value, reread[key]) for key, value in index.items()):
            raise ValueError('Group index round trip changed')
    filtered_cid = filtered['cell_id'].to_numpy()
    starts = (np.r_[0, np.flatnonzero(filtered_cid[1:] != filtered_cid[:-1]) + 1]
              if len(filtered_cid) else np.empty(0, dtype=np.int64))
    cells_path = idx / 'cells.npz'
    np.savez(cells_path, cell_id=filtered_cid[starts], starts=starts,
             ends=np.r_[starts[1:], len(filtered_cid)].astype(np.int64) if len(starts) else starts)
    rec = dict(format=FORMAT, m_max=m_max, data_seed=seed, source=sources,
               observations=file_record(output, root), index=file_record(index_path, root),
               cells=file_record(cells_path, root), stats=stats, stored_rows=len(filtered))

    class DiskStore:
        def read(self, path):
            return pq.ParquetFile(path).read(columns=_COLUMNS)

    cached = load(DiskStore(), day, bucket, root, rec, m_max)
    if len(cached['cached_groups']) != len(groups):
        raise ValueError('Saved group count changed')
    for i in sorted({0, len(groups)//2, len(groups)-1}) if groups else []:
        before = ds._pack(loaded, groups[i])
        after = ds._pack(cached, cached['cached_groups'][i])
        for name in ('x', 'bin_valid', 'delta_t'):
            if not np.array_equal(before[name], after[name], equal_nan=True):
                raise ValueError(f'V3 model input changed: {name}')
        if before['sample_ids'] != after['sample_ids'] or before['group_id'] != after['group_id']:
            raise ValueError('Group identity changed')
    # Reject mutations while building without hashing all inputs a second time.
    for path, recorded in zip(files, sources):
        st = path.stat()
        if st.st_size != recorded['bytes'] or st.st_mtime_ns != recorded['mtime_ns']:
            raise ValueError(f'Source changed during build: {path}')
    rec['elapsed_seconds'] = time.monotonic() - started
    import resource
    rec['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    rec['verification'] = dict(all_columns_roundtrip=True, all_member_indices_roundtrip=True,
                               packed_groups_checked=min(3, len(groups)))
    receipt_path.parent.mkdir(exist_ok=True)
    write_json(receipt_path, rec)
    return day, bucket, rec


def main():
    from trajectory_mae.prepared import memory_status
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='data/cell_mlp_train')
    p.add_argument('--val', default='data/cell_mlp_validation_20260823')
    p.add_argument('--out', default='runtime/observationv3')
    p.add_argument('--train-days', nargs='+', default=[f'202608{d}' for d in range(17, 23)])
    p.add_argument('--val-days', nargs='+', default=['20260823'])
    p.add_argument('--buckets', nargs='+', type=int, default=list(range(128)))
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--m-max', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260921)
    a = p.parse_args()
    if a.workers < 1 or a.m_max < 3 or not a.buckets or any(b < 0 or b >= 128 for b in a.buckets):
        p.error('Invalid worker/group/bucket count')
    if set(a.train_days) & set(a.val_days):
        p.error('Training and validation days overlap')
    out = Path(a.out).resolve()
    sources = [Path(a.train).resolve(), Path(a.val).resolve()]
    if any(out == s or out.is_relative_to(s) or s.is_relative_to(out) for s in sources):
        p.error('Output and input trees must be separate')
    protocol = dict(format=FORMAT, m_max=a.m_max, data_seed=a.seed,
                    train=str(sources[0]), val=str(sources[1]),
                    train_days=sorted(set(a.train_days)), val_days=sorted(set(a.val_days)),
                    buckets=sorted(set(a.buckets)))
    jobs = []
    for side, source, days in [('train', sources[0], protocol['train_days']), ('val', sources[1], protocol['val_days'])]:
        for day in days:
            for b in protocol['buckets']:
                if not list((source / 'observations_v2' / f'day={day}' / f'bucket={b}').glob('*.parquet')):
                    raise ValueError(f'Missing source {side}/{day}/{b}')
                jobs.append((str(source), str(out / side), day, str(b), a.m_max, a.seed))
    state = memory_status()
    if state['used'] + a.workers * 4 * 2**30 > state['limit'] * .85:
        raise RuntimeError(f'Insufficient memory for {a.workers} workers: {state}')
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.build.lock'
    # flock releases automatically after crashes; receipts allow safe resume.
    import fcntl
    with lock.open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = out / 'build_config.json'
        if config.exists() and json.loads(config.read_text()) != protocol:
            raise ValueError('Existing build configuration differs')
        if (out / '_SUCCESS.json').exists():
            raise ValueError('Already published; use a new output directory')
        write_json(config, protocol)
        for side in ('train', 'val'):
            (out / side).mkdir(exist_ok=True)
            (out / side / '_BUILDING').touch()
        started = time.monotonic()
        parts = {'train': {}, 'val': {}}
        print(json.dumps(dict(event='start', partitions=len(jobs), workers=a.workers, protocol=protocol)), flush=True)
        # multiprocessing.Pool replaces workers reliably on Python 3.12;
        # ProcessPoolExecutor(max_tasks_per_child=...) can stall after its
        # initial workers retire on that interpreter. Keep bounded lifetimes.
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(
                processes=a.workers, maxtasksperchild=4) as pool:
            for i, (day, bucket, rec) in enumerate(pool.imap_unordered(build_partition, jobs), 1):
                side = 'train' if day in protocol['train_days'] else 'val'
                parts[side][f'{day}/{bucket}'] = rec
                elapsed = time.monotonic() - started
                progress = dict(completed=i, total=len(jobs), elapsed_seconds=elapsed,
                                estimated_remaining_seconds=elapsed/i*(len(jobs)-i),
                                last_partition=f'{side}/{day}/{bucket}', last_stats=rec['stats'])
                write_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
                state = memory_status()
                if state['used'] > state['limit'] * .90:
                    raise RuntimeError(f'Memory guard: {state}')
        for side, receipts in parts.items():
            result = dict(format=FORMAT, m_max=a.m_max, data_seed=a.seed, partitions=receipts,
                          created_at=datetime.now(timezone.utc).isoformat(),
                          total_source_rows=sum(r['stats']['raw_rows'] for r in receipts.values()),
                          total_stored_rows=sum(r['stored_rows'] for r in receipts.values()),
                          total_groups=sum(r['stats']['groups'] for r in receipts.values()),
                          dropped_no_valid=sum(r['stats']['dropped_no_valid'] for r in receipts.values()))
            write_json(out / side / '_OBSERVATION_V3_SUCCESS.json', result)
            (out / side / '_BUILDING').unlink()
        write_json(out / '_SUCCESS.json', dict(protocol, status='passed', partitions=len(jobs),
                                               elapsed_seconds=time.monotonic()-started))
        print('observationv3 complete', flush=True)


if __name__ == '__main__':
    main()
