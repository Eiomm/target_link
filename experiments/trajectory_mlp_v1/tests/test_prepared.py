import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from experiments.trajectory_mlp_v1.data import CellDataset,collate_cells
from experiments.trajectory_mlp_v1.prepared import FORMAT
from experiments.trajectory_mlp_v1.tools.prepare_final import build_partition


def make(tmp_path):
    src=tmp_path/'source';dst=tmp_path/'ready'
    path=src/'observations_v2/day=20260817/bucket=0';path.mkdir(parents=True)
    rows=[]
    for cid,n in [(0,130),(128,67),(256,2),(384,5)]:
        for i in range(n):
            rows.append(dict(cell_id=cid,sample_id=f'{cid}-{i}',dt=float(i),
                             T_diff=[1.,2.],ratio_pct=[10,10],valid=[i!=1,i!=1],bin_pos=[0,2]))
    np.random.default_rng(7).shuffle(rows)
    pq.write_table(pa.Table.from_pylist(rows),path/'part.parquet')
    d,b,r=build_partition((str(src),str(dst),'20260817','0',64,20260921))
    (dst/'_FINAL_SUCCESS.json').write_text(json.dumps(dict(format=FORMAT,m_max=64,data_seed=20260921,partitions={f'{d}/{b}':r})))
    return src,dst


def test_cached_matches_original_multiple_epochs_without_sorting(tmp_path,monkeypatch):
    src,dst=make(tmp_path)
    for epoch in [0,1,1000000]:
        old=list(CellDataset([str(src)],['20260817'],epoch=epoch))
        new_ds=CellDataset([str(dst)],['20260817'],epoch=epoch)
        def forbidden(*args,**kwargs):raise AssertionError('Prepared reader must not rebuild groups')
        monkeypatch.setattr(new_ds,'_group_specs',forbidden)
        new=list(new_ds)
        assert len(new)==len(old)
        for a,b in zip(old,new):
            assert a['group_id']==b['group_id'] and a['sample_ids']==b['sample_ids']
            assert a['partition_stats']==b['partition_stats']
            for k in ['x','bin_valid','delta_t']:np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(collate_cells(old,epoch=epoch)['mae_mask'],collate_cells(new,epoch=epoch)['mae_mask'])
    with pytest.raises(ValueError,match='protocol mismatch'):CellDataset([str(dst)],['20260817'],m_max=16)
    with pytest.raises(ValueError,match='protocol mismatch'):CellDataset([str(dst)],['20260817'],seed=3)
    index=next(dst.glob('group_indices/day=*/bucket=*/groups.npz'))
    with index.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):list(CellDataset([str(dst)],['20260817']))


def test_unpublished_is_rejected(tmp_path):
    (tmp_path/'_BUILDING').touch()
    with pytest.raises(ValueError,match='not published'):CellDataset([str(tmp_path)],['20260817'])
