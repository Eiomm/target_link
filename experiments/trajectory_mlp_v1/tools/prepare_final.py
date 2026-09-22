"""Build sorted observations and fixed M64 group indices, then atomically publish manifests."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))


def build_partition(job):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from experiments.trajectory_mlp_v1.data import CellDataset, _Store, _flat, _offsets, _row_any
    from experiments.trajectory_mlp_v1.prepared import file_record, load
    import torch
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source, destination, day, bucket, m_max, seed = job
    root=Path(destination); src=Path(source)/'observations_v2'/f'day={day}'/f'bucket={bucket}'
    dst=root/'observations_v2'/f'day={day}'/f'bucket={bucket}'
    receipt_path=root/'receipts'/f'{day}_{bucket}.json'
    source_paths=sorted(src.glob('*.parquet'))
    source_records=[file_record(p,Path(source)) for p in source_paths]
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['source'] != source_records:
            raise ValueError(f'Source changed: {src}')
        for key in ('observations','index'):
            if file_record(root/receipt[key]['path'],root) != receipt[key]:
                raise ValueError(f'Cached artifact changed: {dst}')
        return day,bucket,receipt
    dst.mkdir(parents=True,exist_ok=True)
    table=pa.concat_tables([pq.ParquetFile(p).read() for p in source_paths]).combine_chunks()
    for name in ('cell_id','sample_id','dt','T_diff','ratio_pct','valid','bin_pos'):
        if table[name].null_count:
            raise ValueError(f'Null observation field: {name} in {src}')
    for name in ('T_diff','ratio_pct','valid','bin_pos'):
        if pc.list_flatten(table[name]).null_count:
            raise ValueError(f'Null array element: {name} in {src}')
    if not np.all(table['cell_id'].to_numpy()%128==int(bucket)):
        raise ValueError(f'Wrong bucket: {src}')
    table=table.take(pc.sort_indices(table,sort_keys=[('cell_id','ascending'),('sample_id','ascending')]))
    output=dst/'part-00000.parquet'
    pq.write_table(table,output,compression='snappy')
    reread=pq.ParquetFile(output).read().combine_chunks()
    if not table.equals(reread):
        # Arrow equality treats some NaNs as unequal: compare IEEE values with
        # equal_nan while preserving list boundaries and null locations.
        if table.schema != reread.schema or len(table)!=len(reread):
            raise ValueError('Sorted write/read schema or row count mismatch')
        for name in table.column_names:
            a,b=table[name].combine_chunks(),reread[name].combine_chunks()
            if a.equals(b):continue
            if pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
                if not a.offsets.equals(b.offsets):raise ValueError('List offsets changed')
                a,b=a.values,b.values
            if not a.is_null().equals(b.is_null()):raise ValueError('Null locations changed')
            av,bv=a.to_numpy(zero_copy_only=False),b.to_numpy(zero_copy_only=False)
            if av.dtype.kind not in 'fc' or not np.array_equal(av,bv,equal_nan=True):
                raise ValueError(f'Written values changed: {name}')
    candidates=np.flatnonzero(_row_any(_flat(reread['valid']).to_numpy(zero_copy_only=False).astype(bool),_offsets(reread['T_diff'])))
    raw_rows=len(reread);raw_cells=len(np.unique(reread['cell_id'].to_numpy()))
    del table,reread
    # The legacy reader performs all numerical and duplicate-ID checks once.
    ds=CellDataset.__new__(CellDataset);ds.m_max=m_max;ds.seed=seed;ds.epoch=0;ds._prepared={}
    store=_Store(str(root/'observations_v2'))
    loaded=ds._load_partition(store,day,bucket)
    groups=list(ds._group_specs(loaded,day,bucket)) if loaded is not None else []
    ptr=np.r_[0,np.cumsum([g['group_size'] for g in groups])].astype(np.int64)
    rows=np.concatenate([g['rows'] for g in groups]) if groups else np.empty(0,dtype=np.int64)
    for g in groups:
        if not np.all(loaded['cid'][g['rows']]==g['cell_id']):raise ValueError('Cross-cell group')
    if len(np.unique(rows))!=len(rows):raise ValueError('Repeated trajectory in groups')
    index_dir=root/'group_indices'/f'day={day}'/f'bucket={bucket}';index_dir.mkdir(parents=True,exist_ok=True)
    index_path=index_dir/'groups.npz'
    np.savez(index_path,candidate_rows=candidates,ptr=ptr,rows=rows,
             cell_id=np.array([g['cell_id'] for g in groups],dtype=np.int64),
             K=np.array([g['K'] for g in groups],dtype=np.int64),
             K_raw=np.array([g['K_raw'] for g in groups],dtype=np.int64),
             group_index=np.array([int(g['group_id'].rsplit('/',1)[1]) for g in groups],dtype=np.int64))
    stats=dict(loaded['partition_stats']) if loaded is not None else dict(day=day,bucket=bucket,raw_rows=raw_rows,raw_cells=raw_cells,candidate_rows=0,usable_rows=0,dropped_no_valid=raw_rows,dropped_tail=0,groups=0,full_groups=0,tail_groups=0,selected_groups=0)
    if stats['usable_rows'] != len(rows)+stats['dropped_tail']:raise ValueError('Trajectory accounting mismatch')
    receipt=dict(source=source_records,observations=file_record(output,root),index=file_record(index_path,root),stats=stats)
    cached=load(store,day,bucket,root,receipt,m_max)
    if len(cached['cached_groups'])!=len(groups):raise ValueError('Group count mismatch')
    # Verify the entire saved member index and metadata, not only sample groups.
    with np.load(index_path,allow_pickle=False) as saved:
        if not np.array_equal(saved['rows'],rows) or not np.array_equal(saved['ptr'],ptr):raise ValueError('Index round trip failed')
    for i in sorted(set([0,len(groups)//2,len(groups)-1])) if groups else []:
        before=ds._pack(loaded,groups[i]);after=ds._pack(cached,cached['cached_groups'][i])
        for name in ('x','bin_valid','delta_t'):
            if not np.array_equal(before[name],after[name],equal_nan=True):raise ValueError(f'Packed features changed: {name}')
        if before['sample_ids']!=after['sample_ids'] or before['group_id']!=after['group_id']:raise ValueError('Membership changed')
    if [file_record(p,Path(source)) for p in source_paths] != source_records:
        raise ValueError('Source modified during build')
    receipt_path.parent.mkdir(exist_ok=True)
    temp=receipt_path.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(receipt_path)
    return day,bucket,receipt


def main():
    from experiments.trajectory_mlp_v1.prepared import FORMAT,sha256
    from experiments.trajectory_mlp_v1.prepared import memory_status
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='runtime/cell_mlp_train')
    p.add_argument('--val',default='runtime/cell_mlp_validation_20260823')
    p.add_argument('--out',default='runtime/final')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--m-max',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--train-days',nargs='+',default=[f'202608{d}' for d in range(17,23)])
    p.add_argument('--val-days',nargs='+',default=['20260823'])
    p.add_argument('--buckets',type=int,default=128)
    a=p.parse_args()
    if set(a.train_days)&set(a.val_days):p.error('Overlapping train/validation dates')
    if a.workers<1 or a.m_max<3 or not 1<=a.buckets<=128:p.error('Invalid sizes')
    memory=memory_status()
    # Each process may expand a compressed partition into several GB of Arrow,
    # NumPy and Python objects. Keep room for the parent and other services.
    safe_workers=max(0,min(2,int((memory['limit']*.70-memory['used'])//(8*2**30))))
    if a.workers>safe_workers:
        raise RuntimeError(f'Requested {a.workers} workers exceeds memory-safe limit {safe_workers}; cgroup={memory}')
    print(f"【内存保护】limit_GiB={memory['limit']/2**30:.1f} used_GiB={memory['used']/2**30:.1f} workers={a.workers}",flush=True)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=dict(format=FORMAT,m_max=a.m_max,data_seed=a.seed,train=str(Path(a.train).resolve()),val=str(Path(a.val).resolve()),train_days=a.train_days,val_days=a.val_days,buckets=a.buckets)
    config=out/'build_config.json'
    if config.exists() and json.loads(config.read_text())!=protocol:raise ValueError('Build configuration differs; choose a new output directory')
    config.write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[]
    for side,source,days in [('train',a.train,a.train_days),('val',a.val,a.val_days)]:
        dest=out/side;dest.mkdir(exist_ok=True)
        if (dest/'_FINAL_SUCCESS.json').exists():raise ValueError(f'Already published: {dest}')
        (dest/'_BUILDING').touch()
        for day in days:
            for b in range(a.buckets):
                if not list((Path(source)/'observations_v2'/f'day={day}'/f'bucket={b}').glob('*.parquet')):raise ValueError(f'Missing input {side}/{day}/{b}')
                jobs.append((str(Path(source).resolve()),str(dest),day,str(b),a.m_max,a.seed))
    started=time.monotonic();receipts={'train':{},'val':{}}
    import threading
    stop_monitor=threading.Event()
    def watch_memory(pool):
        while not stop_monitor.wait(.5):
            state=memory_status()
            if state['used']>state['limit']*.80:
                print(f"【内存保护触发】usage_GiB={state['used']/2**30:.2f}; 停止构建，保留已验收分区",flush=True)
                for proc in list((pool._processes or {}).values()):
                    if proc.is_alive():proc.terminate()
                return
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        monitor=threading.Thread(target=watch_memory,args=(pool,),daemon=True);monitor.start()
        tasks={pool.submit(build_partition,j):Path(j[1]).name for j in jobs}
        for i,f in enumerate(as_completed(tasks),1):
            try:
                day,b,rec=f.result()
            except BaseException:
                stop_monitor.set()
                for pending in tasks:pending.cancel()
                for proc in list((pool._processes or {}).values()):
                    if proc.is_alive():proc.terminate()
                raise
            side=tasks[f];receipts[side][f'{day}/{b}']=rec
            elapsed=time.monotonic()-started
            print(f'【最终数据构建】{i}/{len(jobs)} {side}/{day}/bucket={b} rows={rec["stats"]["raw_rows"]} groups={rec["stats"]["groups"]} elapsed_min={elapsed/60:.1f} eta_min={elapsed/i*(len(jobs)-i)/60:.1f}',flush=True)
    stop_monitor.set()
    for side,parts in receipts.items():
        dest=out/side
        result=dict(format=FORMAT,m_max=a.m_max,data_seed=a.seed,partitions=parts,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reader_sha256=sha256(Path(__file__).resolve().parents[1]/'data.py'),
                    total_rows=sum(r['stats']['raw_rows'] for r in parts.values()),total_groups=sum(r['stats']['groups'] for r in parts.values()))
        temp=dest/'_FINAL_SUCCESS.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest/'_FINAL_SUCCESS.json')
        (dest/'_BUILDING').unlink()
    (out/'_SUCCESS.json').write_text(json.dumps(dict(protocol,status='passed',partitions=len(jobs),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    print('【完成】全部分区已排序、生成索引并验收发布',flush=True)


if __name__=='__main__':main()
