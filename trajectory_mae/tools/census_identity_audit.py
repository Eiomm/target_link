"""Exact per-partition identity audit, bounded-memory and restartable."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time


def audit(path):
    import duckdb
    out=path.with_suffix('.identity.json')
    stat=path.stat()
    fingerprint=[stat.st_size,stat.st_mtime_ns]
    if out.exists():
        r=json.loads(out.read_text())
        if r['fingerprint']==fingerprint:return r
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='2GB'")
    c.read_parquet(str(path)).create_view('o')
    row=c.execute('''SELECT count(*),count(*)-count(DISTINCT(cell_id,sample_id)),
      count(*) FILTER(WHERE sample_id IS NULL OR cell_id IS NULL OR target_link_id IS NULL OR map_version IS NULL),
      count(*) FILTER(WHERE len(string_split(sample_id,'#'))<>4 OR split_part(sample_id,'#',2)<>target_link_id
        OR NOT regexp_full_match(split_part(sample_id,'#',1),'[0-9]+')) FROM o''').fetchone()
    r=dict(partition=path.stem,rows=row[0],duplicate_cell_samples=row[1],null_identity_rows=row[2],malformed_sample_ids=row[3],fingerprint=fingerprint)
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(r)+'\n');tmp.replace(out);c.close()
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--watch',action='store_true')
    a=p.parse_args();done=set();results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while True:
            paths=[f for f in sorted((a.out/'parts').glob('*.parquet')) if f.stem not in done and f.with_suffix('.json').exists()]
            for future in as_completed([pool.submit(audit,f) for f in paths]):
                r=future.result();done.add(r['partition']);results.append(r)
                if any(r[k] for k in ['duplicate_cell_samples','null_identity_rows','malformed_sample_ids']):raise ValueError(r)
                if len(done)%32==0:print('identity audited',len(done),flush=True)
            if len(done)==896:break
            if not a.watch:raise ValueError(f'Only {len(done)} partitions; requires 896')
            time.sleep(10)
    r=dict(status='passed',partitions=len(done),rows=sum(x['rows'] for x in results),duplicate_cell_samples=0,null_identity_rows=0,malformed_sample_ids=0)
    (a.out/'identity_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)

if __name__=='__main__':main()
