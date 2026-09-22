"""Prepare versioned model-ready group tensors from observations_v2, resumably."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from contextlib import contextmanager
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zlib

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from experiments.trajectory_mlp_v1.data import CellDataset, _Store
from experiments.trajectory_mlp_v1.prepared import file_record, sha256
from experiments.trajectory_mlp_v1.tensor_corpus import FORMAT, CHANNELS, STORAGE, TensorGroups, verify


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


class CheckedStore(_Store):
    def read(self, path):
        table = super().read(path)
        for name in table.column_names:
            if table[name].null_count:
                raise ValueError(f'Null observation field: {name}')
        for name in ('T_diff', 'ratio_pct', 'valid', 'bin_pos'):
            if pc.list_flatten(table[name]).null_count:
                raise ValueError(f'Null array element: {name}')
        return table


@contextmanager
def build_lock(out):
    with (out / '.build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another tensor builder owns {out}') from exc
        yield


def partition(ds, source, out, day, bucket):
    store = CheckedStore(str(source / 'observations_v2'))
    paths = [Path(p) for p in store.files(f'day={day}/bucket={bucket}')]
    sources = [file_record(p, source) for p in paths]
    receipt_path = out / 'receipts' / f'{day}_{bucket}.json'
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        if rec['source'] != sources:
            raise ValueError(f'Source changed: {day}/{bucket}')
        for a in [*rec['arrays'].values(), rec['payload']]:
            if sha256(out / a['path']) != a['sha256']:
                raise ValueError(f'Cached tensor changed: {a["path"]}')
        return rec
    # Force canonical observation processing even when the source has old indices.
    loaded = ds._load_partition(store, day, bucket)
    specs = list(ds._group_specs(loaded, day, bucket)) if loaded is not None else []
    if loaded is None:
        ids = [store.read(str(p))['cell_id'].to_numpy() for p in paths]
        cid = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
        stats = dict(day=day, bucket=bucket, raw_rows=len(cid), raw_cells=len(np.unique(cid)),
                     candidate_rows=0, usable_rows=0, dropped_no_valid=len(cid), dropped_tail=0,
                     groups=0, full_groups=0, tail_groups=0, selected_groups=0)
    else:
        stats = dict(loaded['partition_stats'])
    count, m = len(specs), ds.m_max
    width = max((len(str(s).encode('utf-8')) for s in loaded['sid']), default=1) if loaded else 1
    shapes = dict(x=((m,50,3),np.float32), bin_valid=((m,50),np.bool_),
                  traj_valid=((m,),np.bool_), delta_t=((m,),np.float32),
                  sample_ids=((m,),np.dtype(f'S{width}')))
    dest = out / 'tensors' / f'day={day}' / f'bucket={bucket}'
    dest.mkdir(parents=True, exist_ok=True)
    # Compress each group independently: random training order never decompresses
    # unrelated groups, while padding consumes almost no disk space.
    meta = np.empty((count,5), dtype=np.int64)
    offsets = np.zeros(count+1, dtype=np.int64)
    tensors = {name: np.zeros(shape, dtype=dtype) for name,(shape,dtype) in shapes.items()}
    used = 0
    payload_path = dest / 'groups.bin'
    with payload_path.open('wb') as payload:
        for i, spec in enumerate(specs):
            item = ds._pack(loaded, spec)
            n = item['group_size']
            if not np.all(loaded['cid'][spec['rows']] == spec['cell_id']):
                raise ValueError('Cross-cell group')
            for name in ('x','bin_valid','traj_valid','delta_t'):
                tensors[name].fill(0)
                tensors[name][:n] = item[name]
            tensors['sample_ids'].fill(b'')
            tensors['sample_ids'][:n] = [s.encode('utf-8') for s in item['sample_ids']]
            meta[i] = [item['cell_id'], item['K'], item['K_raw'], int(item['group_id'].rsplit('/',1)[1]), n]
            payload.write(zlib.compress(b''.join(a.tobytes() for a in tensors.values()), level=1))
            offsets[i+1] = payload.tell()
            used += n
    if stats['usable_rows'] != used + stats['dropped_tail']:
        raise ValueError('Trajectory accounting mismatch')
    arrays = dict(meta=meta, offsets=offsets)
    for name,a in arrays.items():
        np.save(dest / f'{name}.npy', a, allow_pickle=False)
    records = {name: dict(file_record(dest / f'{name}.npy', out), shape=list(a.shape), dtype=a.dtype.str)
               for name,a in arrays.items()}
    receipt = dict(source=sources, arrays=records, stats=stats,
                   payload=file_record(payload_path,out),
                   layout={name:dict(shape=list(a.shape),dtype=a.dtype.str) for name,a in tensors.items()})
    saved = TensorGroups(out, receipt, day, bucket, m)
    for i in sorted({0,count//2,count-1}) if count else []:
        before, after = ds._pack(loaded, specs[i]), saved[i]
        n = before['group_size']
        for name in ('x','bin_valid','traj_valid','delta_t'):
            np.testing.assert_array_equal(before[name], after[name][:n])
            if after[name][n:].any():
                raise ValueError('Nonzero padding')
        if before['sample_ids'] != after['sample_ids'] or before['group_id'] != after['group_id']:
            raise ValueError('Saved group membership changed')
    if [file_record(p, source) for p in paths] != sources:
        raise ValueError('Source changed during build')
    write_json(receipt_path, receipt)
    return receipt


def build_job(job):
    source, out, day, bucket, m_max, seed = job
    import torch
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    ds = CellDataset.__new__(CellDataset)
    ds.m_max, ds.seed, ds.epoch = m_max, seed, 0
    ds._prepared, ds._tensors = {}, {}
    return day, bucket, partition(ds, Path(source), Path(out), day, bucket)


def build(source, out, days, m_max=64, seed=20260921, workers=1):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Source and output must be separate directory trees')
    if workers < 1 or workers > 2:
        raise ValueError('workers must be 1 or 2')
    days = sorted(set(map(str, days)))
    ds = CellDataset([str(source)], days, m_max=m_max, seed=seed)
    if ds._tensors:
        raise ValueError('Expected observations_v2 source, not a tensor corpus')
    ds._prepared = {}  # independently validate and fold the original observations
    available = {p[2] for p in ds.partitions}
    if set(days) != available:
        raise ValueError(f'Missing requested source days: {set(days)-available}')
    out.mkdir(parents=True, exist_ok=True)
    with build_lock(out):
        config = dict(format=FORMAT, m_max=m_max, data_seed=seed, channels=CHANNELS, storage=STORAGE,
                      source=str(source), days=days, partitions=[f'{d}/{b}' for _,_,d,b in ds.partitions],
                      builder_sha256=sha256(Path(__file__)),
                      reader_sha256=sha256(Path(__file__).resolve().parents[1]/'data.py'))
        path = out / 'build_config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Build configuration changed; choose a new output directory')
        if (out / '_TENSORS_SUCCESS.json').exists():
            result = verify(out)
            for rec in result['partitions'].values():
                for record in rec['source']:
                    if file_record(source / record['path'], source) != record:
                        raise ValueError('Source changed since tensor publication')
            return result
        write_json(path, config)
        (out / '_TENSORS_BUILDING').touch()
        (out / 'receipts').mkdir(exist_ok=True)
        receipts = {}
        started = time.monotonic()
        jobs = [(str(source),str(out),d,b,m_max,seed) for _,_,d,b in ds.partitions]
        def record(i, day, bucket, rec):
            receipts[f'{day}/{bucket}'] = rec
            elapsed = time.monotonic()-started
            print(f'{i}/{len(jobs)} {day}/{bucket} groups={rec["stats"]["groups"]} '
                  f'elapsed={elapsed:.1f}s eta={elapsed/i*(len(jobs)-i):.1f}s', flush=True)
        if workers == 1:
            for i, (_,_,day,bucket) in enumerate(ds.partitions,1):
                record(i,day,bucket,partition(ds,source,out,day,bucket))
        else:
            from experiments.trajectory_mlp_v1.prepared import memory_status
            state = memory_status()
            if state['limit']*.8-state['used'] < workers*4*2**30:
                raise RuntimeError(f'Insufficient memory headroom for {workers} workers: {state}')
            stop = threading.Event()
            # Independent processes bound the number of expanded source partitions.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(build_job,job) for job in jobs]
                def guard():
                    while not stop.wait(1):
                        state = memory_status()
                        if state['used'] > state['limit']*.85:
                            print('Memory guard stopped build; completed receipts are resumable',flush=True)
                            for proc in list((pool._processes or {}).values()):
                                if proc.is_alive():
                                    proc.terminate()
                            return
                monitor = threading.Thread(target=guard,daemon=True)
                monitor.start()
                try:
                    for i,f in enumerate(as_completed(futures),1):
                        day,bucket,rec = f.result()
                        record(i,day,bucket,rec)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    for proc in list((pool._processes or {}).values()):
                        if proc.is_alive():
                            proc.terminate()
                    raise
                finally:
                    stop.set()
                    monitor.join()
        result = dict(config, partitions=receipts, total_groups=sum(r['stats']['groups'] for r in receipts.values()),
                      total_rows=sum(r['stats']['raw_rows'] for r in receipts.values()),
                      output_bytes=sum(a['bytes'] for r in receipts.values() for a in [*r['arrays'].values(), r['payload']]),
                      elapsed_seconds=time.monotonic()-started)
        write_json(out / '_TENSORS_SUCCESS.json', result)
        (out / '_TENSORS_BUILDING').unlink()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source')
    p.add_argument('--out', required=True)
    p.add_argument('--days', nargs='+')
    p.add_argument('--m-max', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.verify_only:
        result = verify(a.out)
    else:
        if not a.source or not a.days:
            p.error('--source and --days are required for building')
        import torch
        torch.set_num_threads(1)
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        result = build(a.source, a.out, a.days, a.m_max, a.seed, a.workers)
    print(json.dumps({k:result[k] for k in ('format','total_groups','total_rows','output_bytes')}, indent=2))


if __name__ == '__main__':
    main()
