"""Direct writer equivalence and raw time-boundary regression tests."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.build_raw_training_tensors import write_partition, observations, output_path
from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells
from experiments.trajectory_mlp_v1.tensor_corpus import TensorGroups
from experiments.trajectory_mlp_v1.tools.prepare_tensors import partition


def dataset(m=64):
    ds = CellDataset.__new__(CellDataset)
    ds.m_max, ds.seed, ds.epoch = m,20260921,0
    ds._prepared, ds._tensors = {},{}
    return ds


def sample(cell, i, invalid=False):
    return dict(cell_id=cell,sample_id=f'traj-{i:04d}',dt=float(i%600),
                T_diff=[1.,2.,float('nan') if invalid else 3.],
                ratio_pct=[4,6,10],valid=[True,True,not invalid],bin_pos=[0,0,4])


def test_direct_writer_matches_existing_converter_and_masks(tmp_path):
    rows = ([sample(-10,i,i==2) for i in range(130)]
            + [sample(20,i) for i in range(67)]
            + [dict(sample(30,0), valid=[False]*3)]
            + [sample(40,0),sample(40,1)])
    source, old, direct = tmp_path/'source',tmp_path/'old',tmp_path/'direct'
    part = source/'observations_v2/day=20260817/bucket=0'
    part.mkdir(parents=True)
    (old/'receipts').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),part/'part.parquet')
    before = partition(dataset(),source,old,'20260817','0')
    after = write_partition(iter(rows),direct,'20260817','0')
    assert before['stats'] == after['stats']
    a = TensorGroups(old,before,'20260817','0',64)
    b = TensorGroups(direct,after,'20260817','0',64)
    assert [b[i]['group_size'] for i in range(len(b))] == [64,64,64,3]
    for i in range(len(a)):
        for epoch in (0,1,1000000):
            aa,bb = collate_cells([a[i]],epoch=epoch),collate_cells([b[i]],epoch=epoch)
            for name in ('x','bin_valid','traj_valid','delta_t','mae_mask'):
                np.testing.assert_array_equal(aa[name],bb[name])
            assert aa['sample_ids'] == bb['sample_ids']
        assert a[i]['group_id'] == b[i]['group_id']
    assert not list(direct.rglob('*.parquet'))


def test_empty_bucket_and_invalid_inputs(tmp_path):
    rec = write_partition([],tmp_path/'empty','20260817','0')
    assert len(TensorGroups(tmp_path/'empty',rec,'20260817','0',64)) == 0
    cases = [([sample(1,0),sample(1,0)],'duplicate'),
             ([sample(2,0),sample(1,0)],'sorted'),
             ([dict(sample(1,0), sample_id='x'*513)],'512'),
             ([dict(sample(1,0), bin_pos=[0,0,50])],'bin_pos')]
    for i,(rows,msg) in enumerate(cases):
        with pytest.raises(ValueError,match=msg):
            write_partition(rows,tmp_path/str(i),'20260817','0')


def test_output_paths_are_separate_and_normalized():
    assert output_path('hdfs://host/out/../ready/',['hdfs://host/raw/*.parquet']) == 'hdfs://host/ready'
    for output in ('hdfs://host/','hdfs://host/raw/ready','hdfs://host','hdfs://host/ready?q=1'):
        with pytest.raises(ValueError):
            output_path(output,['hdfs://host/raw/*.parquet'])


@pytest.mark.skipif(__import__('os').environ.get('RUN_RAW_SPARK_TEST') != '1',
                    reason='Set RUN_RAW_SPARK_TEST=1 for local Spark integration')
def test_raw_transform_and_full_publication(tmp_path):
    from pyspark.sql import SparkSession
    import subprocess,sys,os
    spark = (SparkSession.builder.master('local[2]').appName('raw-tensor-test')
             .config('spark.ui.enabled','false')
             .config('spark.hadoop.fs.defaultFS','file:///')
             .config('spark.sql.session.timeZone','Asia/Shanghai').getOrCreate())
    # ref=2026-08-17 09:08:00 CST. seg0 starts 09:09; seg1 starts 09:10:20.
    ref = 1786928880.
    rows=[]
    for day_offset in (0,86400):
        for i in range(4):
            for seg,binidx,cum in ((0,10,65.),(1,60,145.)):
                rows.append(dict(map_version='map',target_link_id='A',sample_id=f'traj{i}#A#000#{day_offset}',
                    seg_idx=seg,seg_mark=1,bin_idx=binidx,sub_idx=0,t_ref=ref+day_offset,
                    T_cum=cum,T_diff=5.,ratio=1.,observed=True))
    raw = spark.createDataFrame(rows)
    raw.write.parquet(str(tmp_path/'raw'))
    result = observations(raw,2).collect()
    assert len(result)==16
    assert sorted(set(r.dt for r in result)) == [20.,540.]
    assert {r.day for r in result} == {'20260817','20260818'}
    assert all(r.bin_pos == [0] and r.ratio_pct == [10] for r in result)
    edge = dict(rows[0],sample_id='edge',seg_idx=1,bin_idx=60,T_cum=float('nan'),T_diff=float('nan'))
    fallback = dict(edge,bin_idx=61,T_cum=145.,T_diff=5.,ratio=0.4)
    unrelated = dict(fallback,seg_mark=0,bin_idx=9)
    checked = observations(spark.createDataFrame([edge,fallback,unrelated]),2).collect()
    assert len(checked) == 1
    assert checked[0].dt == 20.
    assert checked[0].bin_pos == [0,1]
    assert checked[0].valid == [False,True]
    assert checked[0].ratio_pct == [10,4]
    # 256 would wrap to bin 0 if the byte cast preceded validation.
    with pytest.raises(Exception,match='bin_pos outside'):
        observations(spark.createDataFrame([dict(rows[0],bin_idx=266)]),2).collect()
    # Persist ONLY a test oracle, never used by the direct builder.
    for day in ('20260817','20260818'):
        for bucket in (0,1):
            selected=[r.asDict() for r in result if r.day==day and r.bucket==bucket]
            if selected:
                part=tmp_path/'oracle'/'observations_v2'/f'day={day}'/f'bucket={bucket}'
                part.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist(selected),part/'part.parquet')
    spark.stop()
    env=dict(os.environ,SPARK_LOCAL_IP='127.0.0.1',PYSPARK_PYTHON=sys.executable)
    proc=subprocess.run([sys.executable,'tools/build_raw_training_tensors.py',
        '--inputs',str(tmp_path/'raw'),'--out',str(tmp_path/'ready'),
        '--train-days','20260817','--val-days','20260818','--buckets','2',
        '--shuffle-partitions','2','--master','local[2]'],env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    assert (tmp_path/'ready/_SUCCESS.json').exists()
    assert not list((tmp_path/'ready').rglob('*.parquet'))
    for side,day in [('train','20260817'),('val','20260818')]:
        old=list(CellDataset([str(tmp_path/'oracle')],[day]))
        new=list(CellDataset([str(tmp_path/'ready'/side)],[day]))
        old.sort(key=lambda x:x['group_id']);new.sort(key=lambda x:x['group_id'])
        assert len(old)==len(new)==2
        for a,b in zip(old,new):
            aa,bb=collate_cells([a]),collate_cells([b])
            for name in ('x','bin_valid','traj_valid','delta_t','mae_mask'):
                np.testing.assert_array_equal(aa[name],bb[name])
