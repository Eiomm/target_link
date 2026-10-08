"""Validated sorted corpus with immutable, seed-specific group indices."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow as pa

FORMAT = 'trajectory_mlp_prepared_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path, root):
    path, root = Path(path), Path(root)
    # NFS may settle write attributes only after a flush/read; hash before stat.
    digest = sha256(path)
    st = path.stat()
    return dict(path=str(path.relative_to(root)), bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, sha256=digest)


def manifest(root, m_max, seed):
    root = Path(root)
    if root.name == 'observations_v2':
        root = root.parent
    marker = root/'_FINAL_SUCCESS.json'
    if (root/'_BUILDING').exists():
        raise ValueError(f'Prepared corpus is not published: {root}')
    if not marker.exists():
        return None
    result = json.loads(marker.read_text())
    if (result['format'], result['m_max'], result['data_seed']) != (FORMAT, m_max, seed):
        raise ValueError('Prepared index protocol mismatch: rebuild for requested m_max/data_seed')
    return root, result


class Groups:
    """Materialize only the selected group's small metadata dictionary."""
    def __init__(self, index, day, bucket, m_max):
        self.index, self.day, self.bucket, self.m = index, day, bucket, m_max
    def __len__(self):
        return len(self.index['cell_id'])
    def __getitem__(self, i):
        a = self.index
        cell, k, raw = int(a['cell_id'][i]), int(a['K'][i]), int(a['K_raw'][i])
        rows = a['rows'][a['ptr'][i]:a['ptr'][i+1]]
        return dict(cell_id=cell,K=k,K_raw=raw,group_size=len(rows),rows=rows,
                    group_id=f'groups-v1-m{self.m}/{self.day}/{self.bucket}/{cell}/{int(a["group_index"][i])}',
                    day=self.day,bucket=self.bucket,dropped_trajectories=raw-k,
                    dropped_no_valid=raw-k,dropped_tail=k%self.m if k%self.m<3 else 0)


def load(store, day, bucket, root, receipt, m_max):
    from .data import _flat, _offsets
    for key in ('observations','index'):
        rec = receipt[key]
        path = root/rec['path']
        st = path.stat()
        if st.st_size != rec['bytes']:
            raise ValueError(f'Prepared artifact changed: {path}')
    # Content hashes are authoritative on NFS; timestamps can settle after write.
    if sha256(root/receipt['observations']['path']) != receipt['observations']['sha256']:
        raise ValueError('Prepared observation checksum mismatch')
    data = (root/receipt['index']['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt['index']['sha256']:
        raise ValueError('Prepared group index checksum mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        index = {key:archive[key] for key in archive.files}
    table = store.read(str(root/receipt['observations']['path']))
    if len(table) != receipt['stats']['raw_rows']:
        raise ValueError('Prepared row count mismatch')
    table = table.take(pa.array(index['candidate_rows']))
    flat = {'T':_flat(table['T_diff']).to_numpy(zero_copy_only=False).astype(np.float64),
            'R':_flat(table['ratio_pct']).to_numpy(zero_copy_only=False).astype(np.float64),
            'V':_flat(table['valid']).to_numpy(zero_copy_only=False).astype(bool),
            'B':_flat(table['bin_pos']).to_numpy(zero_copy_only=False).astype(np.int64),
            'off':_offsets(table['T_diff'])}
    return dict(cid=table['cell_id'].to_numpy(),sid=np.asarray(table['sample_id'].to_pylist(),dtype=object),
                dt=table['dt'].to_numpy().astype(np.float32),flat=flat,
                partition_stats=dict(receipt['stats']),cached_groups=Groups(index,day,bucket,m_max))


def memory_status():
    candidates=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if 'memory' in controllers.split(','):
            bases=[p for p in Path('/sys/fs/cgroup').iterdir() if p.is_dir() and 'memory' in p.name.split(',')]
            limit_name,usage_name='memory.limit_in_bytes','memory.usage_in_bytes'
        elif controllers=='':
            bases=[Path('/sys/fs/cgroup')];limit_name,usage_name='memory.max','memory.current'
        else:continue
        for base in bases:
            start=base/relative.lstrip('/')
            if not start.exists():start=base
            for path in [start,*start.parents]:
                if not path.is_relative_to(base):break
                f=path/limit_name
                if not f.exists():continue
                value=f.read_text().strip()
                if value=='max':continue
                limit=int(value)
                if limit>=2**60:continue
                candidates.append(dict(path=str(path),limit=limit,used=int((path/usage_name).read_text())))
    if not candidates:
        raise RuntimeError('Cannot establish finite cgroup memory limit; refusing automatic full build')
    return min(candidates,key=lambda x:x['limit']-x['used'])
