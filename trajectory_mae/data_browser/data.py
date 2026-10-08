"""Read-only, bounded exploration of the raw corpus and its existing census."""
from __future__ import annotations

import csv
import json
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from trajectory_mae.data import CellDataset, collate_cells

REPO = Path(__file__).resolve().parents[2]
DAYS = [f'202608{d:02d}' for d in range(17, 24)]
SHANGHAI = timezone(timedelta(hours=8))


def local_time(value):
    return datetime.fromtimestamp(int(value), SHANGHAI).strftime('%Y-%m-%d %H:%M')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key not in ('day', 'local_time') and value:
                try:
                    row[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
    return rows


def integer(value, name, low, high):
    if not re.fullmatch(r'-?\d+', str(value)):
        raise ValueError(f'{name} 必须是整数')
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f'{name} 必须在 {low}～{high} 之间')
    return result


class SelectedStore:
    """Feed a filtered raw cell through the unmodified training loader."""
    base = 'browser-selected-cell'

    def __init__(self, table):
        self.table = table

    def files(self, rel):
        return ['selected']

    def read(self, path):
        return self.table


class Explorer:
    def __init__(self, repo=REPO, report=None, data_seed=20260921, m_max=64):
        self.repo = Path(repo).resolve()
        self.report = Path(report or self.repo / 'outputs/trajectory_reports/seven_day_p0_20260817_23').resolve()
        self.data_seed = int(data_seed)
        self.m_max = int(m_max)
        self.lock = threading.RLock()  # One bounded parquet operation at a time.
        self.summary_cache = OrderedDict()
        self.cell_cache = OrderedDict()
        self.roots = [self.repo / 'data/cell_mlp_train', self.repo / 'data/cell_mlp_validation_20260823']
        self.dataset = CellDataset([str(p) for p in self.roots], DAYS, m_max=self.m_max, seed=self.data_seed)
        self.manifest = read_json(self.report / 'input_manifest.json')
        self.sources = {(s['day'], int(s['bucket'])): s for s in self.manifest['sources']}
        self.source_changes = []
        for spec in self.sources.values():
            for f in spec['files']:
                try:
                    st = Path(f['path']).stat()
                    if (st.st_size, st.st_mtime_ns) != (f['bytes'], f['mtime_ns']):
                        self.source_changes.append(f['path'])
                except OSError:
                    self.source_changes.append(f['path'])
        self.overview_data = self._overview()

    def _overview(self):
        totals = read_json(self.report / 'scan_totals.json')
        daily = read_csv(self.report / 'daily_totals.csv')
        retention = {r['day']: r for r in read_csv(self.report / 'daily_training_retention.csv')}
        for row in daily:
            row.update(retention[row['day']])
            row['split'] = 'validation' if row['day'] == '20260823' else 'train'
        validation = read_json(self.report / 'validation_summary.json')
        if sum(r['observations'] for r in daily) != totals['rows']:
            raise ValueError('日报总数与全量报告不一致')
        return dict(days=daily, totals=totals, windows=read_csv(self.report / 'window_10min.csv'),
                    data_seed=self.data_seed, m_max=self.m_max, batch_size=512,
                    validation_epoch=1000000, report_epoch=validation['epoch'],
                    report_validated=validation['status'] == 'passed',
                    source_metadata_match=not self.source_changes, changed_source_count=len(self.source_changes),
                    report_updated=datetime.fromtimestamp((self.report/'validation_summary.json').stat().st_mtime, SHANGHAI).isoformat(),
                    source_note='全量统计快照；使用文件路径、大小、修改时间核对原始数据。cell 明细直接读取当前 Parquet。')

    def overview(self):
        return self.overview_data

    def day(self, value):
        if value not in DAYS:
            raise ValueError('日期必须在 20260817～20260823 之间')
        return value

    def summary(self, day, bucket):
        key = (self.day(day), integer(bucket, 'bucket', 0, 127))
        with self.lock:
            if key not in self.summary_cache:
                path = self.report / 'cell_parts' / f'{day}_{key[1]:03d}.parquet'
                table = pq.ParquetFile(path).read(use_threads=False)
                table = table.take(pc.sort_indices(table, sort_keys=[('cell_id', 'ascending')]))
                self.summary_cache[key] = table
                while len(self.summary_cache) > 3:
                    self.summary_cache.popitem(last=False)
            self.summary_cache.move_to_end(key)
            return self.summary_cache[key]

    def cells(self, day='20260820', bucket=123, offset=0, limit=30, min_k=3, window=None, max_k=None):
        offset = integer(offset, 'offset', 0, 1000000000)
        limit = integer(limit, 'limit', 1, 100)
        min_k = 0 if min_k in (None, '') else integer(min_k, '最少轨迹数', 0, 1000000000)
        max_k = None if max_k in (None, '') else integer(max_k, '最多轨迹数', 0, 1000000000)
        if max_k is not None and min_k > max_k:
            raise ValueError('最少轨迹数不能大于最多轨迹数')
        bucket = integer(bucket, 'bucket', 0, 127)
        with self.lock:
            table = self.summary(day, bucket)
            mask = pc.greater_equal(table['k_usable'], min_k)
            if max_k is not None:
                mask = pc.and_(mask, pc.less_equal(table['k_usable'], max_k))
            if window not in (None, ''):
                win = integer(window, 'window', 0, 9999999999)
                mask = pc.and_(mask, pc.equal(table['window_start'], win))
            table = table.filter(mask)
            total = len(table)
            rows = table.slice(offset, limit).to_pylist()
        for r in rows:
            r['cell_id'] = str(r['cell_id'])  # Never send int64 identities as JS numbers.
            r['k_usable'] = int(r['k_usable'])
            r['groups'] = r['k_usable']//self.m_max + int(r['k_usable']%self.m_max >= 3)
            r['local_time'] = local_time(r['window_start'])
        return dict(rows=rows, total=total, offset=offset, limit=limit, day=day, bucket=bucket,
                    min_k=min_k, max_k=max_k,
                    scope='当前日期和 bucket 中满足筛选条件的全部 cell')

    def locate(self, cell_id, day=None):
        cid = integer(cell_id, 'cell_id', -(2**63), 2**63-1)
        bucket = cid % 128
        matches = []
        for d in ([self.day(day)] if day else DAYS):
            rows = self.summary(d, bucket).filter(pc.equal(self.summary(d, bucket)['cell_id'], cid)).to_pylist()
            for row in rows:
                matches.append(dict(day=d, bucket=bucket, cell_id=str(cid), local_time=local_time(row['window_start'])))
        return dict(matches=matches, source_metadata_match=not self.source_changes)

    def _cell(self, day, cid):
        day = self.day(day)
        cid = integer(cid, 'cell_id', -(2**63), 2**63-1)
        key = (day, cid)
        if key in self.cell_cache:
            self.cell_cache.move_to_end(key)
            return self.cell_cache[key]
        spec = self.sources.get((day, cid % 128))
        if spec is None:
            raise ValueError('没有找到对应的数据分区')
        files = [f['path'] for f in spec['files']]
        # Scanner materializes only selected rows; no Python list of the whole partition.
        dataset = pads.dataset(files, format='parquet')
        table = dataset.scanner(filter=pads.field('cell_id') == cid,
                                batch_size=8192, batch_readahead=1, fragment_readahead=1,
                                use_threads=False).to_table()
        if not len(table):
            raise LookupError('该日期未找到这个 cell；可使用跨日期查询')
        loaded = self.dataset._load_partition(SelectedStore(table), day, str(cid % 128))
        specs = [] if loaded is None else list(self.dataset._group_specs(loaded, day, str(cid % 128)))
        first = table.slice(0, 1).to_pylist()[0]
        meta = {k: str(first[k]) if k in ('map_version', 'target_link_id') else first[k]
                for k in ('map_version', 'target_link_id', 'seg_idx', 'window') if k in first}
        meta.update(cell_id=str(cid), day=day, bucket=cid % 128,
                    local_time=local_time(first['window']), raw_count=len(table),
                    usable_count=specs[0]['K'] if specs else (loaded['partition_stats']['usable_rows'] if loaded else 0),
                    group_count=len(specs), group_sizes=[s['group_size'] for s in specs],
                    data_seed=self.data_seed, m_max=self.m_max,
                    files=files)
        meta['dropped_no_valid'] = meta['raw_count'] - meta['usable_count']
        meta['dropped_tail'] = meta['usable_count'] - sum(meta['group_sizes'])
        value = (meta, loaded, specs, table)
        # Bound cache by both cell count and bytes, including duplicated flat columns.
        if table.nbytes <= 32*1024*1024:
            self.cell_cache[key] = value
            while len(self.cell_cache)>3 or sum(v[3].nbytes for v in self.cell_cache.values())>32*1024*1024:
                self.cell_cache.popitem(last=False)
        return value

    def cell(self, day, cell_id):
        with self.lock:
            return self._cell(day, cell_id)[0]

    def group(self, day, cell_id, index=0, epoch=0):
        index = integer(index, 'group', 0, 1000000000)
        epoch = integer(epoch, 'epoch', 0, 1000000000)
        with self.lock:
            meta, loaded, specs, raw = self._cell(day, cell_id)
            if index >= len(specs):
                raise LookupError('这个 group 不存在；该 cell 可能在训练时被丢弃')
            group = self.dataset._pack(loaded, specs[index])
            batch = collate_cells([group], m_max=self.m_max, epoch=epoch)
            raw_by_id = {r['sample_id']: r for r in raw.to_pylist()}
            members = []
            for slot, sid in enumerate(group['sample_ids']):
                r = raw_by_id[sid]
                members.append(dict(slot=slot, sample_id=sid, dt=float(group['delta_t'][slot]),
                                    hidden=bool(batch['mae_mask'][0, slot]),
                                    features=group['x'][slot, :, :2].tolist(),
                                    bin_valid=group['bin_valid'][slot].tolist(),
                                    pieces=[dict(bin=int(b), time=float(t) if t is not None and np.isfinite(t) else None,
                                                 ratio_pct=int(ratio), valid=bool(v), observed=bool(o))
                                            for b,t,ratio,v,o in zip(r['bin_pos'],r['T_diff'],r['ratio_pct'],r['valid'],r.get('observed',[False]*len(r['bin_pos']))) ]))
            return dict(cell=meta, group_index=index, group_id=group['group_id'], epoch=epoch,
                        members=members, group_size=len(members), padding=self.m_max-len(members),
                        hidden_count=int(batch['mae_mask'].sum()),
                        supervised_bins=int((batch['mae_mask'].unsqueeze(-1)&batch['bin_valid']).sum()),
                        features_kind='模型特征 [T_clean, ratio]，遮挡前；bin_valid 单独用于缺失处理和监督，不作为学习特征。')
