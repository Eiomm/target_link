"""Raw Spark rows -> final group tensors in one job; no observation files."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import posixpath
from urllib.parse import urlsplit

# Direct local execution; YARN provides the same modules through --py-files.
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_worker(_):
    import sys
    if sys.version_info < (3, 9):
        raise RuntimeError('Raw tensor builder requires Python >=3.9')
    import numpy
    import pyarrow
    import torch
    return dict(python=sys.version.split()[0],numpy=numpy.__version__,
                pyarrow=pyarrow.__version__,torch=torch.__version__)


def source_inventory(spark, raw):
    result = []
    for uri in sorted(raw.inputFiles()):
        path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(uri)
        fs = path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
        status = fs.getFileStatus(path)
        result.append(dict(uri=uri,bytes=status.getLen(),mtime_ms=status.getModificationTime()))
    return result


def output_path(output, inputs):
    """Normalize destinations before deriving lock/staging paths."""
    def normalized(value, destination=False):
        if value.startswith('hdfs://'):
            uri = urlsplit(value)
            if not uri.netloc or uri.query or uri.fragment:
                raise ValueError('Invalid HDFS URI: '+value)
            path = posixpath.normpath('/'+uri.path.lstrip('/'))
            if destination and path == '/':
                raise ValueError('Output must be a new child directory')
            return 'hdfs://'+uri.netloc, path
        if '://' in value:
            raise ValueError('Only HDFS or local paths supported')
        path = str(Path(value).resolve())
        if destination and path == '/':
            raise ValueError('Output must be a new child directory')
        return '',path
    if not output.strip():
        raise ValueError('Output path is empty')
    authority,path = normalized(output,True)
    for raw in inputs:
        # A wildcard can match future output files: reject overlapping trees.
        magic = [raw.index(c) for c in '*?[' if c in raw]
        static = raw[:min(magic)].rsplit('/',1)[0] if magic else raw
        src_authority,src = normalized(static)
        if authority == src_authority and (path == src or path.startswith(src.rstrip('/')+'/')
                                          or src.startswith(path.rstrip('/')+'/')):
            raise ValueError('Input and output must be separate trees')
    return authority+path


def observations(raw, buckets=128):
    """Canonical v2 transformations, kept in the Spark plan, never persisted."""
    from pyspark.sql import functions as F
    keys = ['map_version', 'target_link_id', 'seg_idx', 'sample_id']
    known = lambda c: F.col(c).isNotNull() & ~F.isnan(c)
    def checked(value, valid, message):
        return F.when(valid,value).otherwise(F.raise_error(message))
    finite = lambda c: known(c) & (F.abs(F.col(c)) != float('inf'))
    pos = F.col('bin_idx')-50*F.col('seg_idx')-10
    ratio = F.round(F.col('ratio')*10)
    pieces = (raw.where('seg_mark = 1')
              .withColumn('start', F.when(known('T_cum') & known('T_diff'),
                  checked(F.col('T_cum')-F.col('T_diff'),
                          finite('T_cum') & finite('T_diff') & (F.col('T_diff')>=0),
                          'Nonfinite or negative piece timing')))
              .withColumn('valid', known('T_diff'))
              .withColumn('bin_pos',checked(pos,pos.between(0,49),'bin_pos outside 0..49 before cast').cast('byte')))
    grouped = pieces.groupBy(*keys).agg(
        F.min('start').alias('start'), F.max('t_ref').alias('t_ref'),
        F.array_sort(F.collect_list(F.struct(
            F.col('bin_pos').alias('b'), F.col('sub_idx').alias('s'),
            F.col('T_diff').cast('float').alias('T'),
            checked(ratio,finite('ratio') & ratio.between(1,10),
                    'Invalid ratio before cast').cast('byte').alias('R'),
            F.col('observed').cast('boolean').alias('O'), F.col('valid').alias('V')))).alias('pieces'))
    grouped = (grouped.where(F.col('start').isNotNull())
               .withColumn('enter',checked(F.col('t_ref')+F.col('start'),
                   finite('t_ref') & finite('start'),'Invalid segment reference time'))
               .withColumn('window', (F.floor(F.col('enter')/600)*600).cast('long'))
               .withColumn('dt', (F.col('enter')-F.col('window')).cast('float'))
               .withColumn('cell_id', F.xxhash64(F.concat_ws('|', F.col('map_version'),
                    F.col('target_link_id'), F.col('seg_idx').cast('string'), F.col('window').cast('string'))))
               .withColumn('day', F.from_unixtime('window','yyyyMMdd'))
               .withColumn('bucket', (((F.col('cell_id') % buckets)+buckets) % buckets).cast('int')))
    return grouped.select('day','bucket','cell_id','sample_id','dt',
        F.transform('pieces', lambda p:p['T']).alias('T_diff'),
        F.transform('pieces', lambda p:p['R']).alias('ratio_pct'),
        F.transform('pieces', lambda p:p['V']).alias('valid'),
        F.transform('pieces', lambda p:p['b']).alias('bin_pos'))


class MemoryStore:
    base = 'memory'
    def __init__(self, table):
        self.table = table
    def files(self, rel):
        return ['memory']
    def read(self, path):
        return self.table


def write_partition(rows, out, day, bucket, m_max=64, seed=20260921):
    """Consume sorted observations one cell at a time; write only final tensors."""
    import numpy as np
    import pyarrow as pa
    import zlib
    from experiments.trajectory_mlp_v1.data import CellDataset
    from experiments.trajectory_mlp_v1.prepared import file_record
    from experiments.trajectory_mlp_v1.tensor_corpus import TensorGroups
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = CellDataset.__new__(CellDataset)
    ds.m_max, ds.seed, ds.epoch = m_max, seed, 0
    ds._prepared, ds._tensors = {}, {}
    stats = dict(day=day,bucket=bucket,raw_rows=0,raw_cells=0,candidate_rows=0,
                 usable_rows=0,dropped_no_valid=0,dropped_tail=0,groups=0,
                 full_groups=0,tail_groups=0,selected_groups=0)
    # Fixed bounded UTF-8 storage makes streaming possible without a second scan.
    arrays = dict(x=np.zeros((m_max,50,3),np.float32),
                  bin_valid=np.zeros((m_max,50),bool),traj_valid=np.zeros(m_max,bool),
                  delta_t=np.zeros(m_max,np.float32),sample_ids=np.zeros(m_max,dtype='S512'))
    meta, offsets, previous = [], [0], None
    with (out/'groups.bin').open('wb') as stream:
        for cell, members in itertools.groupby(rows, key=lambda r:int(r['cell_id'])):
            if previous is not None and cell <= previous:
                raise ValueError('Observations must be sorted by cell_id')
            previous = cell
            records = list(members)
            table = pa.Table.from_pylist(records)
            for name in ('cell_id','sample_id','dt','T_diff','ratio_pct','valid','bin_pos'):
                if table[name].null_count:
                    raise ValueError('Null observation field: '+name)
            import pyarrow.compute as pc
            for name in ('T_diff','ratio_pct','valid','bin_pos'):
                if pc.list_flatten(table[name]).null_count:
                    raise ValueError('Null array element: '+name)
            if any(len(r['sample_id'].encode('utf-8')) > 512 for r in records):
                raise ValueError('sample_id exceeds 512 UTF-8 bytes; refusing truncation')
            loaded = ds._load_partition(MemoryStore(table),day,bucket)
            if loaded is None:
                stats['raw_rows'] += len(records)
                stats['raw_cells'] += 1
                stats['dropped_no_valid'] += len(records)
                continue
            for spec in ds._group_specs(loaded,day,bucket):
                item = ds._pack(loaded,spec)
                n = item['group_size']
                for name, value in arrays.items():
                    value.fill(b'' if name == 'sample_ids' else 0)
                    value[:n] = ([s.encode('utf-8') for s in item[name]]
                                 if name == 'sample_ids' else item[name])
                meta.append([cell,item['K'],item['K_raw'],int(item['group_id'].rsplit('/',1)[1]),n])
                stream.write(zlib.compress(b''.join(a.tobytes() for a in arrays.values()),1))
                offsets.append(stream.tell())
            for key in stats:
                if key not in ('day','bucket'):
                    stats[key] += loaded['partition_stats'][key]
    if stats['usable_rows'] != sum(r[4] for r in meta)+stats['dropped_tail']:
        raise ValueError('Trajectory accounting mismatch')
    index = dict(meta=np.asarray(meta,dtype=np.int64).reshape(-1,5),
                 offsets=np.asarray(offsets,dtype=np.int64))
    for name,a in index.items():
        np.save(out/(name+'.npy'),a,allow_pickle=False)
    receipt = dict(source=[], stats=stats,
        arrays={name:dict(file_record(out/(name+'.npy'),out),shape=list(a.shape),dtype=a.dtype.str)
                for name,a in index.items()},payload=file_record(out/'groups.bin',out),
        layout={name:dict(shape=list(a.shape),dtype=a.dtype.str) for name,a in arrays.items()})
    saved = TensorGroups(out,receipt,day,bucket,m_max)
    for i in sorted({0,len(saved)//2,len(saved)-1}) if len(saved) else []:
        saved[i]  # decode/check block layout before upload
    return receipt


class LocalFs:
    def exists(self, p): return Path(p).exists()
    def mkdir(self, p): Path(p).mkdir(parents=True,exist_ok=True)
    def acquire_lock(self, p):
        with open(p,'x'): pass
    def unlink(self, p): Path(p).unlink()
    def rename(self, a, b):
        # Local mode is a smoke-test helper; publication is serialized by lock.
        if Path(b).exists(): raise ValueError('Output exists: '+b)
        Path(a).rename(b)
    def put(self, a, b): shutil.copyfile(a,b)
    def write_json(self, p, value):
        with open(p,'x') as f: json.dump(value,f,indent=2)


def worker(pairs, config):
    from tools.prepare_tensors_yarn import Hdfs, join
    iterator = iter(pairs)
    first = next(iterator,None)
    if first is None: return
    day,bucket = first[0][:2]
    bucket = str(bucket)
    def records():
        for key,value in itertools.chain([first],iterator):
            if (key[0],str(key[1])) != (day,bucket):
                raise ValueError('Partition contains multiple day/bucket keys')
            if value is not None:
                yield value
    with tempfile.TemporaryDirectory(prefix='raw-tensors-') as temp:
        import torch
        torch.set_num_threads(1)
        rec = write_partition(records(),temp,day,bucket,config['m_max'],config['seed'])
        side = 'train' if day in config['train_days'] else 'val'
        rel = join('_parts','day='+day,'bucket='+bucket,'attempt='+uuid.uuid4().hex)
        destination = join(config['staging'],side,rel)
        fs = Hdfs() if config['hdfs'] else LocalFs()
        fs.mkdir(destination)
        for item in [*rec['arrays'].values(),rec['payload']]:
            local = Path(temp)/item['path']
            fs.put(local,join(destination,local.name))
            item['path'] = join(rel,local.name)
        fs.write_json(join(destination,'receipt.json'),rec)
        yield side,day+'/'+bucket,rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--train-days',nargs='+',required=True)
    p.add_argument('--val-days',nargs='+',required=True)
    p.add_argument('--buckets',type=int,default=128)
    p.add_argument('--m-max',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--shuffle-partitions',type=int,default=1024)
    p.add_argument('--master',default=None)
    a = p.parse_args()
    import re
    days = a.train_days+a.val_days
    if len(set(days)) != len(days) or any(not re.fullmatch(r'\d{8}',d) for d in days):
        p.error('Dates must be unique, disjoint YYYYMMDD values')
    if a.m_max<3 or not 1<=a.buckets<=128 or a.shuffle_partitions<1:
        p.error('Invalid sizes')
    inputs = [v.strip() for v in a.inputs.split(',') if v.strip()]
    if not inputs: p.error('At least one input is required')
    try:
        output = output_path(a.out,inputs)
    except ValueError as exc:
        p.error(str(exc))
    hdfs = output.startswith('hdfs://')
    if hdfs and any(not path.startswith('hdfs://') for path in inputs):
        p.error('HDFS builds require explicit hdfs:// input URIs')
    if not hdfs and not (a.master or '').startswith('local'):
        p.error('Local output is only supported with --master local[...]')
    from pyspark.sql import SparkSession
    from tools.prepare_tensors_yarn import JvmHdfs, publish
    builder = (SparkSession.builder.appName('raw-training-tensors')
               .config('spark.sql.session.timeZone','Asia/Shanghai')
               .config('spark.sql.shuffle.partitions',str(a.shuffle_partitions)))
    if a.master: builder = builder.master(a.master)
    if not hdfs:
        # Hosts may carry fs.defaultFS=hdfs://... even in local Spark mode.
        builder = builder.config('spark.hadoop.fs.defaultFS','file:///')
    spark = builder.getOrCreate()
    fs = JvmHdfs(spark) if hdfs else LocalFs()
    staging = output+'._building_'+uuid.uuid4().hex
    lock,locked = output+'._lock',False
    try:
        from tools.build_raw_training_tensors import check_worker as check
        print('Worker environment: '+json.dumps(
            spark.sparkContext.parallelize([0],1).map(check).collect()),flush=True)
        fs.mkdir(output.rsplit('/',1)[0])
        fs.acquire_lock(lock)
        locked = True
        if fs.exists(output): raise ValueError('Output exists; choose a fresh path')
        fs.mkdir(staging)
        for side in ('train','val'):
            fs.mkdir(staging+'/'+side)
            fs.write_json(staging+'/'+side+'/_TENSORS_BUILDING',{})
        config = dict(vars(a),staging=staging,hdfs=hdfs)
        fs.write_json(staging+'/build_config.json',config)
        raw = spark.read.parquet(*inputs)
        inventory = source_inventory(spark,raw)
        fs.write_json(staging+'/source_inventory.json',inventory)
        from pyspark.sql import functions as F
        obs = observations(raw,a.buckets).where(F.col('day').isin(days))
        day_index = {d:i for i,d in enumerate(sorted(days))}
        jobs = [dict(split='train' if d in a.train_days else 'val',day=d,bucket=str(b))
                for d in sorted(days) for b in range(a.buckets)]
        # Sentinels retain empty buckets without collecting observations on driver.
        sentinel = spark.sparkContext.parallelize([((j['day'],int(j['bucket']),0,''),None) for j in jobs])
        def keyed(row):
            value = row.asDict(recursive=True)
            return (value['day'],value['bucket'],value['cell_id'],value['sample_id']),value
        bucket_count = a.buckets
        shuffled = obs.rdd.map(keyed).union(sentinel).repartitionAndSortWithinPartitions(
            len(jobs),lambda key:day_index[key[0]]*bucket_count+key[1])
        from tools.build_raw_training_tensors import worker as convert
        results = shuffled.mapPartitions(lambda rows:convert(rows,config)).collect()
        # A missing source day must not be disguised by empty bucket sentinels.
        present_days = {key.split('/')[0] for _,key,r in results if r['stats']['raw_rows']}
        if present_days != set(days):
            raise ValueError('No timed target observations for days: '+str(set(days)-present_days))
        for side in ('train','val'):
            if not sum(r['stats']['groups'] for split,_,r in results if split == side):
                raise ValueError('No usable training groups in split: '+side)
        if source_inventory(spark,spark.read.parquet(*inputs)) != inventory:
            raise ValueError('Raw source inventory changed during build; refusing publication')
        publish(fs,staging,output,jobs,results,a.m_max,a.seed)
        print('Published raw-to-tensor corpus: '+output,flush=True)
    except BaseException as exc:
        import traceback
        detail = traceback.format_exc()
        print('Raw tensor build failed: '+repr(exc),flush=True)
        try:
            if fs.exists(staging):
                fs.write_json(staging+'/failure.json',dict(error=repr(exc),traceback=detail))
        except Exception as log_error:
            print('Cannot persist failure details: '+repr(log_error),flush=True)
        raise
    finally:
        try:
            if locked: fs.unlink(lock)
        except Exception as cleanup_error:
            print('Cannot remove build lock: '+repr(cleanup_error),flush=True)
        try:
            spark.stop()
        except Exception as cleanup_error:
            print('Cannot stop Spark: '+repr(cleanup_error),flush=True)


if __name__ == '__main__':
    main()
