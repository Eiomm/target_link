"""Contract checks for identity, real training grouping, masks and read-only input."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trajectory_mae.data_browser.data import Explorer, integer

CID = -9200017963794003461
DAY = '20260820'


def dump(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture()
def explorer(tmp_path):
    root = tmp_path/'data/cell_mlp_train/observations_v2'/f'day={DAY}'/'bucket=123'
    root.mkdir(parents=True)
    rows = []
    for cid, count in [(CID,78),(CID+128,66),(CID+256,2),(CID+384,2)]:
        for i in range(count):
            invalid=cid==CID+256
            rows.append(dict(cell_id=cid,sample_id=f'{i:04d}#{cid}',dt=float(i),n_pieces=2,
                             T_diff=[1.+i,2.],ratio_pct=[6,4],valid=[True,not invalid],
                             observed=[False,False],bin_pos=[3,3],map_version='2026082012',
                             target_link_id='90000287515851',seg_idx=0,window=1787231400))
    file=root/'part.parquet'
    pq.write_table(pa.Table.from_pylist(rows),file)
    report=tmp_path/'report';(report/'cell_parts').mkdir(parents=True)
    summary=[dict(day=DAY,cell_id=cid,window_start=1787231400,k_raw=k,k_usable=usable,min_present=1,max_present=1,min_covered_m=10.,max_covered_m=10.)
             for cid,k,usable in [(CID,78,78.),(CID+128,66,66.),(CID+256,2,0.),(CID+384,2,2.)]]
    pq.write_table(pa.Table.from_pylist(summary),report/'cell_parts/20260820_123.parquet')
    st=file.stat()
    dump(report/'input_manifest.json',dict(sources=[dict(day=DAY,bucket='123',files=[dict(path=str(file),bytes=st.st_size,mtime_ns=st.st_mtime_ns)])]))
    dump(report/'scan_totals.json',dict(rows=len(rows)))
    dump(report/'validation_summary.json',dict(status='passed',epoch=0))
    (report/'daily_totals.csv').write_text(f'day,observations\n{DAY},{len(rows)}\n')
    (report/'daily_training_retention.csv').write_text(f'day,retained_observations\n{DAY},142\n')
    (report/'window_10min.csv').write_text('window_start,local_time,observations\n1787231400,2026-08-20 21:10:00,148\n')
    return Explorer(tmp_path,report)


def test_identity_pagination_and_direct_lookup(explorer):
    page=explorer.cells(DAY,123,limit=1)
    assert page['total']==2
    assert page['rows'][0]['cell_id']==str(CID)
    second=explorer.cells(DAY,123,offset=1,limit=1)
    assert second['rows'][0]['cell_id']==str(CID+128)
    assert explorer.locate(str(CID),DAY)['matches'][0]['bucket']==123
    assert explorer.cells(DAY,123,window=1787230800)['total']==0


def test_real_training_groups_padding_and_membership(explorer):
    meta=explorer.cell(DAY,str(CID))
    assert meta['group_sizes']==[64,14]
    groups=[explorer.group(DAY,str(CID),i,0) for i in range(2)]
    assert [(g['group_size'],g['padding'],g['hidden_count']) for g in groups]==[(64,0,32),(14,50,7)]
    ids=[m['sample_id'] for g in groups for m in g['members']]
    assert len(ids)==len(set(ids))==78
    assert all(len(m['features'])==50 for g in groups for m in g['members'])


def test_epoch_changes_only_mask_not_members_or_values(explorer):
    first=explorer.group(DAY,str(CID),0,0)
    again=explorer.group(DAY,str(CID),0,0)
    later=explorer.group(DAY,str(CID),0,1)
    assert first==again
    assert [m['hidden'] for m in first['members']] != [m['hidden'] for m in later['members']]
    for a,b in zip(first['members'],later['members']):
        assert {k:v for k,v in a.items() if k!='hidden'}=={k:v for k,v in b.items() if k!='hidden'}


def test_piece_fold_ignores_observed_and_preserves_invalidity(explorer):
    g=explorer.group(DAY,str(CID),0,0)
    member=g['members'][0]
    assert member['features'][3]==[sum(p['time'] for p in member['pieces']),1.]
    assert member['features'][4]==[0.,0.]
    assert member['bin_valid'][3] is True
    assert member['bin_valid'][4] is False
    assert not any(p['observed'] for p in member['pieces'])
    meta=explorer.cell(DAY,str(CID+256))
    assert meta['group_count']==0 and meta['dropped_no_valid']==2
    assert meta['usable_count']==0


def test_tail_drop_and_no_group_error(explorer):
    meta=explorer.cell(DAY,str(CID+128))
    assert meta['group_sizes']==[64] and meta['dropped_tail']==2
    small=explorer.cell(DAY,str(CID+384))
    assert small['group_sizes']==[] and small['dropped_tail']==2
    with pytest.raises(LookupError): explorer.group(DAY,str(CID+384))
    with pytest.raises(LookupError): explorer.group(DAY,str(CID),2)


def test_inputs_remain_unchanged_and_json_is_finite(explorer):
    src=Path(explorer.manifest['sources'][0]['files'][0]['path'])
    before=(src.read_bytes(),src.stat().st_mtime_ns)
    group=explorer.group(DAY,str(CID))
    json.dumps(group,allow_nan=False)
    assert (src.read_bytes(),src.stat().st_mtime_ns)==before
    assert explorer.overview()['source_metadata_match']


@pytest.mark.parametrize('value',['../../etc/passwd','1.5','9007199254740992.0','NaN','9223372036854775808'])
def test_reject_invalid_or_out_of_range_identity(value):
    with pytest.raises(ValueError): integer(value,'cell_id',-(2**63),2**63-1)


def test_null_invalid_piece_is_safe_and_retains_known_ratio(explorer):
    src=Path(explorer.manifest['sources'][0]['files'][0]['path'])
    rows=pq.ParquetFile(src).read().to_pylist()
    rows[0].update(T_diff=[None,2.],valid=[False,True],bin_pos=[3,4])
    pq.write_table(pa.Table.from_pylist(rows),src)
    members=[m for i in range(2) for m in explorer.group(DAY,str(CID),i)['members']]
    member=next(m for m in members if m['sample_id']==f'0000#{CID}')
    assert member['pieces'][0]['time'] is None
    np.testing.assert_allclose(member['features'][3],[0.,.6])
    np.testing.assert_allclose(member['features'][4],[2.,.4])
    assert member['bin_valid'][3] is False
    assert member['bin_valid'][4] is True
    json.dumps(member,allow_nan=False)


@pytest.fixture()
def boundary_explorer(explorer):
    path=explorer.report/'cell_parts/20260820_123.parquet'
    counts=[0,2,3,63,64,65,66,78]
    rows=[dict(day=DAY,cell_id=CID+128*i,window_start=1787231400,
               k_raw=k,k_usable=float(k),min_present=1,max_present=1,
               min_covered_m=10.,max_covered_m=10.) for i,k in enumerate(counts)]
    pq.write_table(pa.Table.from_pylist(rows),path)
    return explorer


def test_inclusive_trajectory_range_and_pagination(boundary_explorer):
    e=boundary_explorer
    first=e.cells(DAY,123,min_k=3,max_k=64,limit=2)
    second=e.cells(DAY,123,min_k=3,max_k=64,limit=2,offset=2)
    assert first['total']==second['total']==3
    assert [r['k_usable'] for r in first['rows']+second['rows']]==[3,63,64]
    assert first['min_k']==3 and first['max_k']==64
    assert [r['k_usable'] for r in e.cells(DAY,123,min_k=64,max_k=64)['rows']]==[64]
    assert e.cells(DAY,123,min_k=3,max_k=64,window=1787230800)['total']==0


def test_one_sided_and_unbounded_trajectory_range(boundary_explorer):
    e=boundary_explorer
    assert [r['k_usable'] for r in e.cells(DAY,123,min_k='',max_k=2)['rows']]==[0,2]
    assert [r['k_usable'] for r in e.cells(DAY,123,min_k=64,max_k='')['rows']]==[64,65,66,78]
    assert e.cells(DAY,123,min_k='',max_k='')['total']==8
    assert e.cells(DAY,123,min_k=0,max_k=0)['total']==1
    assert e.cells(DAY,123)['total']==6  # Existing API callers keep default min=3.


@pytest.mark.parametrize('lower,upper',[(65,64),(-1,64),(3,-1),(3,'1.5'),('2.5',64),(3,1000000001)])
def test_invalid_trajectory_ranges(explorer,lower,upper):
    with pytest.raises(ValueError): explorer.cells(DAY,123,min_k=lower,max_k=upper)
