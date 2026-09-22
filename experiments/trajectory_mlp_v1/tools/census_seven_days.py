"""Exact seven-day observation census; restartable partition outputs, read-only sources.

Requires numpy, pyarrow and duckdb. Run scan first, then aggregate. No model training.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DAYS = [str(x) for x in range(20260817, 20260824)]
COLS = ['cell_id','sample_id','dt','T_diff','ratio_pct','valid','bin_pos',
        'map_version','target_link_id','seg_idx','window']


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def discover():
    partitions = []
    for day in DAYS:
        root = REPO / 'runtime' / ('cell_mlp_validation_20260823' if day == '20260823' else 'cell_mlp_train')
        base = root / 'observations_v2' / f'day={day}'
        buckets = sorted(base.glob('bucket=*'), key=lambda p: int(p.name.split('=')[1]))
        if [int(p.name.split('=')[1]) for p in buckets] != list(range(128)):
            raise ValueError(f'Incomplete bucket coverage: {day}')
        for bucket in buckets:
            files = sorted(bucket.glob('*.parquet'))
            if not files:
                raise ValueError(f'No files: {bucket}')
            partitions.append(dict(day=day, bucket=bucket.name.split('=')[1], files=[
                dict(path=str(p.resolve()), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files]))
    return partitions


def scan_partition(spec, out):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from census_partition_metrics import summarize_batch
    from census_reference_metrics import reference_metrics
    pa.set_cpu_count(1)
    day, bucket = spec['day'], spec['bucket']
    stem = f'{day}_{int(bucket):03d}'
    receipt = out / 'parts' / (stem + '.json')
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        prev = json.loads(receipt.read_text())
        compact_path = out / 'parts' / (stem + '.parquet')
        if not compact_path.exists() or pq.ParquetFile(compact_path).metadata.num_rows != prev['rows']:
            raise ValueError('Incomplete cached partition: '+stem)
        if prev['fingerprint'] != fingerprint:
            raise ValueError('Source metadata changed')
        if prev.get('data_seed') != 20260921:
            compact = pq.read_table(out / 'parts' / (stem + '.parquet'))
            prev.update(reference_metrics(compact, day=day, bucket=bucket, seed=20260921, epoch=0, m_max=64))
            prev['data_seed'] = 20260921
            dump(receipt, prev)
        return stem, prev['rows'], True
    start = time.time()
    tables = []
    expected = 0
    for file in spec['files']:
        src = pq.ParquetFile(file['path'])
        expected += src.metadata.num_rows
        for batch in src.iter_batches(batch_size=32768, columns=COLS, use_threads=False):
            tables.append(summarize_batch(batch))
    table = pa.concat_tables(tables)
    if len(table) != expected:
        raise AssertionError('Parquet metadata row count mismatch')
    stats = reference_metrics(table, day=day, bucket=bucket, seed=20260921, epoch=0, m_max=64)
    for name in ['n_present','n_valid']:
        hist = np.bincount(table[name].to_numpy(), minlength=51)
        stats[name + '_hist'] = hist.tolist()
    dest = out / 'parts' / (stem + '.parquet')
    temp = dest.with_suffix('.parquet.tmp')
    pq.write_table(table, temp, compression='zstd')
    temp.replace(dest)
    stats.update(day=day, bucket=bucket, data_seed=20260921, rows=len(table), fingerprint=fingerprint,
                 seconds=round(time.time()-start, 2), source_files=spec['files'])
    dump(receipt, stats)
    return stem, len(table), False


def scan(args):
    parts = discover()
    (args.out / 'parts').mkdir(parents=True, exist_ok=True)
    dump(args.out / 'input_manifest.json', dict(days=DAYS, partitions=len(parts),
         bytes=sum(f['bytes'] for p in parts for f in p['files']), sources=parts,
         source_fingerprint='file path/size/mtime, not content checksum', seed=20260921, epoch=0, m_max=64))
    selected = parts[:args.limit] if args.limit else parts
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_partition, p, args.out) for p in selected]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            stem, rows, cached = future.result()
            print(f'{i}/{len(selected)} {stem} rows={rows} cached={cached}', flush=True)
    print('SCAN COMPLETE', flush=True)


def aggregate(args):
    import duckdb
    import pyarrow.parquet as pq
    from datetime import datetime
    from zoneinfo import ZoneInfo
    parts = sorted((args.out / 'parts').glob('????????_???.json'))
    if len(parts) != (args.limit or 896):
        raise ValueError(f'Full census requires 896 receipts, found {len(parts)}')
    receipts = [json.loads(p.read_text()) for p in parts]
    expected_parts = discover()
    if args.limit:
        expected_parts = expected_parts[:args.limit]
    expected_keys = {(p['day'],p['bucket']) for p in expected_parts}
    if {(r['day'],r['bucket']) for r in receipts} != expected_keys:
        raise ValueError('Receipt partition set differs from requested complete input')
    by_key = {(r['day'],r['bucket']):r for r in receipts}
    for spec in expected_parts:
        r = by_key[(spec['day'],spec['bucket'])]
        compact = args.out/'parts'/f"{r['day']}_{int(r['bucket']):03d}.parquet"
        if pq.ParquetFile(compact).metadata.num_rows != r['rows']:
            raise ValueError('Compact rows do not match receipt')
        if r['fingerprint'] != hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest():
            raise ValueError('Source metadata changed since scan')
        if r.get('data_seed') != 20260921:
            raise ValueError('Mask/reference receipt data seed differs from training')
    c = duckdb.connect(str(args.out / 'census.duckdb'))
    c.execute("SET threads=4")
    c.execute("SET memory_limit='6GB'")
    c.execute("SET preserve_insertion_order=false")
    c.execute("SET temp_directory=?", [str(args.out / 'spill')])
    source = str(args.out / 'parts' / '*.parquet')
    c.read_parquet(source, filename=True).create_view("obs", replace=True)
    # Explicit window determines the time axis; receipt/file day is checked below.
    c.execute("CREATE OR REPLACE VIEW o AS SELECT *, \"window\" AS window_start, regexp_extract(filename,'([0-9]{8})_[0-9]{3}[.]parquet$',1) AS day, split_part(sample_id,'#',1) AS traj_id FROM obs")
    geometry_fingerprint = [(str(p.name), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted((args.out/'geometry').glob('*geometry.parquet'))]
    def export(name, query):
        start=time.time()
        query_hash = hashlib.sha256((query + json.dumps(geometry_fingerprint if name.startswith(('geometry_', 'exact_', 'city_')) else []) + json.dumps([(r['fingerprint'],r['data_seed']) for r in receipts])).encode()).hexdigest()
        result_file = args.out/(name+'.csv')
        result_receipt = args.out/(name+'.query.json')
        if result_file.exists() and result_receipt.exists():
            old = json.loads(result_receipt.read_text())
            if old['query_hash']==query_hash and old['bytes']==result_file.stat().st_size:
                print('CACHED '+name,flush=True)
                return
        print('QUERY '+name, flush=True)
        destination = str(args.out/(name+'.csv')).replace("'", "''")
        result = c.execute(f"COPY ({query}) TO '{destination}' (HEADER, DELIMITER ',')").fetchone()
        dump(result_receipt,dict(query_hash=query_hash,bytes=result_file.stat().st_size,rows=result[0]))
        print(f'DONE {name} rows={result[0]} seconds={time.time()-start:.1f}',flush=True)
    basic = '''count(*) observations, sum(n_pieces) pieces, sum(n_present) recorded_bins,
       sum(n_valid) valid_bins, sum(has_internal_gap::BIGINT) internal_gap_observations,
       sum(usable::BIGINT) usable_observations, sum(covered_m) covered_m,
       count(DISTINCT cell_id) cells, count(DISTINCT target_link_id) links,
       count(DISTINCT (map_version,target_link_id)) versioned_links,
       count(DISTINCT traj_id) distinct_traj_ids'''
    cell_parts = args.out/'cell_parts'
    if (args.out/'cell_summary_complete.json').exists():
        cell_manifest = json.loads((args.out/'cell_summary_complete.json').read_text())
        expected_stems = {p.stem for p in parts}
        if {p.stem for p in cell_parts.glob('????????_???.parquet')} != expected_stems:
            raise ValueError('Cell summary partition set mismatch')
        if cell_manifest['status']!='complete' or set(cell_manifest['source_fingerprints'])!=expected_stems or cell_manifest['partitions'] != len(parts) or cell_manifest['observations'] != sum(r['rows'] for r in receipts):
            raise ValueError('Cell summary manifest counts mismatch')
        for stem in expected_stems:
            stat = (args.out/'parts'/(stem+'.parquet')).stat()
            if cell_manifest['source_fingerprints'][stem] != [stat.st_size,stat.st_mtime_ns]:
                raise ValueError('Stale cell summary source: '+stem)
        old_cells = c.execute("SELECT table_type FROM information_schema.tables WHERE table_name='cells'").fetchone()
        if old_cells:
            c.execute('DROP '+('VIEW' if old_cells[0]=='VIEW' else 'TABLE')+' cells')
        c.read_parquet(str(cell_parts/'*.parquet')).create_view('cell_part_rows',replace=True)
        # Parquet exports HUGEINT sums as DOUBLE; these bounded per-cell
        # integer counts are exact, and must retain the public CSV integer type.
        c.execute('''CREATE OR REPLACE VIEW cells AS SELECT day,cell_id,window_start,
          cast(k_raw AS BIGINT) k_raw,cast(k_usable AS BIGINT) k_usable,
          min_present,max_present,min_covered_m,max_covered_m FROM cell_part_rows''')
        print('USING PARTITION CELL SUMMARIES',flush=True)
    else:
        print('BUILD CELL SUMMARY',flush=True)
        cell_query = '''CREATE OR REPLACE TABLE cells AS SELECT day,cell_id,any_value(window_start) AS window_start,
          count(*) AS k_raw,sum(usable::BIGINT) AS k_usable,
          min(n_present) AS min_present,max(n_present) AS max_present,
          min(covered_m) AS min_covered_m,max(covered_m) AS max_covered_m
          FROM o GROUP BY day,cell_id'''
        cell_hash = hashlib.sha256((cell_query+json.dumps([r['fingerprint'] for r in receipts])).encode()).hexdigest()
        cell_receipt = args.out/'cells.query.json'
        table_exists = c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='cells'").fetchone()[0]
        if not (table_exists and cell_receipt.exists() and json.loads(cell_receipt.read_text()).get('query_hash')==cell_hash):
            c.execute(cell_query)
            c.execute('CHECKPOINT')
            dump(cell_receipt,dict(query_hash=cell_hash))
    key_mode = (args.out/'key_summary_complete.json').exists()
    if key_mode:
        import pyarrow as pa
        key_manifest = json.loads((args.out/'key_summary_complete.json').read_text())
        if key_manifest['status']!='complete' or key_manifest['partitions']!=len(parts) or key_manifest['observations']!=sum(r['rows'] for r in receipts):
            raise ValueError('Key summary manifest mismatch')
        if set(key_manifest['source_fingerprints'])!={p.stem for p in parts}:
            raise ValueError('Key summary input partition set mismatch')
        for stem, fingerprint in key_manifest['source_fingerprints'].items():
            stat = (args.out/'parts'/(stem+'.parquet')).stat()
            if fingerprint != [stat.st_size,stat.st_mtime_ns]:
                raise ValueError('Stale key summary: '+stem)
        for name in ['traj','link','geometry','coverage']:
            paths = sorted((args.out/'key_parts'/name).glob('????????_???.parquet'))
            if {p.stem for p in paths}!={p.stem for p in parts}:
                raise ValueError('Incomplete key output: '+name)
            c.read_parquet([str(p) for p in paths],filename=True).create_view(name+'_keys',replace=True)
        stats = [json.loads(p.read_text()) for p in sorted((args.out/'key_parts'/'stats').glob('????????_???.json'))]
        if len(stats)!=len(parts) or sum(r['observations'] for r in stats)!=sum(r['rows'] for r in receipts):
            raise ValueError('Key scalar statistics mismatch')
        c.register('partition_stats',pa.Table.from_pylist(stats))
        print('MERGE WEIGHTED GEOMETRY KEYS',flush=True)
        merge_hash = hashlib.sha256(json.dumps(key_manifest,sort_keys=True).encode()).hexdigest()
        merge_receipt = args.out/'geometry_rows.query.json'
        merge_exists = c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='geometry_rows'").fetchone()[0]
        if not (merge_exists and merge_receipt.exists() and json.loads(merge_receipt.read_text()).get('query_hash')==merge_hash):
            c.execute("""CREATE OR REPLACE TABLE geometry_rows AS SELECT day,map_version,target_link_id,seg_idx,covered_m,
              sum(observations) observation_weight FROM geometry_keys
              GROUP BY day,map_version,target_link_id,seg_idx,covered_m""")
            c.execute('CHECKPOINT')
            dump(merge_receipt,dict(query_hash=merge_hash))
        c.execute('CREATE OR REPLACE VIEW geometry_input AS SELECT * FROM geometry_rows')
    else:
        c.execute('CREATE OR REPLACE VIEW geometry_input AS SELECT day,map_version,target_link_id,seg_idx,covered_m,1::BIGINT observation_weight FROM o')
    scalar_names = ['observations','pieces','recorded_bins','valid_bins','internal_gap_observations','usable_observations','covered_m']
    scalar_sums = ','.join(f'sum({name}) {name}' for name in scalar_names)
    original_daily_query = f'SELECT day,{basic} FROM o GROUP BY day ORDER BY day'
    original_daily_hash = hashlib.sha256((original_daily_query+'[]'+json.dumps([(r['fingerprint'],r['data_seed']) for r in receipts])).encode()).hexdigest()
    daily_receipt = args.out/'daily_totals.query.json'
    if key_mode:
        export('daily_totals',f"""WITH scalars AS (SELECT day,{scalar_sums} FROM partition_stats GROUP BY day),
          cc AS (SELECT day,count(*) cells FROM cells GROUP BY day),
          tc AS (SELECT day,count(DISTINCT traj_id) distinct_traj_ids FROM traj_keys GROUP BY day),
          lc AS (SELECT day,count(DISTINCT target_link_id) links,count(DISTINCT(map_version,target_link_id)) versioned_links FROM geometry_input GROUP BY day)
          SELECT scalars.*,cc.cells,lc.links,lc.versioned_links,tc.distinct_traj_ids FROM scalars
          JOIN cc USING(day) JOIN lc USING(day) JOIN tc USING(day) ORDER BY day""")
    elif daily_receipt.exists() and json.loads(daily_receipt.read_text()).get('query_hash') == original_daily_hash:
        export('daily_totals',original_daily_query)
    else:
        daily_basic = basic.replace('count(DISTINCT cell_id) cells', 'max(dc.cells) cells')
        export('daily_totals',f'''SELECT day,{daily_basic} FROM o
          LEFT JOIN (SELECT day,count(*) cells FROM cells GROUP BY day) dc USING(day)
          GROUP BY day ORDER BY day''')
    print('COUNT WEEKLY DISTINCT CELLS FROM SUMMARY',flush=True)
    if (args.out/'cell_summary_complete.json').exists():
        # The cell stage verifies pmod(cell_id,128)=bucket. Hence the sets
        # are disjoint between buckets, while repeated IDs across days are
        # still exactly deduplicated inside each seven-day bucket.
        weekly_cells = 0
        for bucket in range(128):
            bucket_paths = sorted(str(p) for p in cell_parts.glob(f'????????_{bucket:03d}.parquet'))
            if not bucket_paths:
                continue
            weekly_cells += c.execute('SELECT count(DISTINCT cell_id) FROM read_parquet(?)',[bucket_paths]).fetchone()[0]
            if bucket%16==15:
                print(f'WEEKLY CELL BUCKETS {bucket+1}/128',flush=True)
    else:
        weekly_cells = c.execute('SELECT count(DISTINCT cell_id) FROM cells').fetchone()[0]
    weekly_basic = basic.replace('count(DISTINCT cell_id) cells', f'{weekly_cells} cells')
    if key_mode:
        export('seven_day_totals',f"""SELECT {scalar_sums},{weekly_cells} cells,
          (SELECT count(DISTINCT target_link_id) FROM geometry_input) links,
          (SELECT count(DISTINCT(map_version,target_link_id)) FROM geometry_input) versioned_links,
          (SELECT count(DISTINCT traj_id) FROM traj_keys) distinct_traj_ids FROM partition_stats""")
    else:
        export('seven_day_totals',f'SELECT {weekly_basic} FROM o')
    window_files = []
    for day in DAYS:
        start_window = int(datetime.strptime(day,'%Y%m%d').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp())
        end_window = start_window + 86400
        window_name = 'window_10min_'+day
        if key_mode:
            export(window_name,f"""WITH tc AS (
              SELECT window_start,count(DISTINCT traj_id) distinct_traj_ids FROM traj_keys
              WHERE filename LIKE '%/{day}_%' GROUP BY window_start), lc AS (
              SELECT window_start,count(DISTINCT target_link_id) links FROM link_keys
              WHERE filename LIKE '%/{day}_%' GROUP BY window_start), cc AS (
              SELECT window_start,count(*) cells,sum(k_raw) observations FROM cells WHERE day='{day}' GROUP BY window_start)
              SELECT t.range window_start,
              strftime(to_timestamp(t.range) AT TIME ZONE 'Asia/Shanghai','%Y-%m-%d %H:%M:%S') local_time,
              coalesce(cc.observations,0) observations,coalesce(tc.distinct_traj_ids,0) distinct_traj_ids,
              coalesce(lc.links,0) links,coalesce(cc.cells,0) cells
              FROM range({start_window},{end_window},600) t LEFT JOIN tc ON tc.window_start=t.range
              LEFT JOIN lc ON lc.window_start=t.range LEFT JOIN cc ON cc.window_start=t.range ORDER BY t.range""")
        else:
            export(window_name, f"""WITH counts AS (
               SELECT window_start,count(*) observations,count(DISTINCT traj_id) distinct_traj_ids,
               count(DISTINCT target_link_id) links
               FROM o WHERE filename LIKE '%/{day}_%' GROUP BY window_start), cell_counts AS (
               SELECT window_start,count(*) cells FROM cells WHERE day='{day}' GROUP BY window_start)
               SELECT t.range AS window_start,
               strftime(to_timestamp(t.range) AT TIME ZONE 'Asia/Shanghai','%Y-%m-%d %H:%M:%S') local_time,
               coalesce(c.observations,0) observations,coalesce(c.distinct_traj_ids,0) distinct_traj_ids,
               coalesce(c.links,0) links,coalesce(cc.cells,0) cells
               FROM range({start_window},{end_window},600) t LEFT JOIN counts c ON c.window_start=t.range
               LEFT JOIN cell_counts cc ON cc.window_start=t.range
               ORDER BY t.range""")
        window_files.append(args.out/(window_name+'.csv'))
    window_rows = []
    for window_file in window_files:
        with window_file.open(newline='') as f:
            day_rows = list(csv.DictReader(f))
        if len(day_rows)!=144:
            raise ValueError('Daily window count mismatch')
        window_rows.extend(day_rows)
    window_dest = args.out/'window_10min.csv'
    with window_dest.with_suffix('.csv.tmp').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(window_rows[0]))
        writer.writeheader();writer.writerows(window_rows)
    window_dest.with_suffix('.csv.tmp').replace(window_dest)
    if key_mode:
        identity = json.loads((args.out/'identity_audit.json').read_text())
        if identity['status']!='passed' or identity['rows']!=sum(r['rows'] for r in receipts):
            raise ValueError('Missing matching identity audit')
        export('integrity',f"""SELECT sum(k_raw) observations,
          coalesce(sum(k_raw) FILTER(WHERE strftime(to_timestamp(window_start) AT TIME ZONE 'Asia/Shanghai','%Y%m%d')<>day),0) partition_day_mismatch,
          {identity['malformed_sample_ids']} malformed_sample_ids,
          coalesce(sum(k_raw) FILTER(WHERE day NOT BETWEEN '20260817' AND '20260823'),0) outside_days,
          coalesce(sum(k_raw) FILTER(WHERE window_start%600<>0),0) misaligned_windows FROM cells""")
    else:
        export('integrity', '''SELECT count(*) observations,
          count(*) FILTER(WHERE strftime(to_timestamp(window_start) AT TIME ZONE 'Asia/Shanghai','%Y%m%d')<>day) partition_day_mismatch,
          count(*) FILTER(WHERE len(string_split(sample_id,'#'))<>4 OR split_part(sample_id,'#',2)<>target_link_id) malformed_sample_ids,
          count(*) FILTER(WHERE day NOT BETWEEN '20260817' AND '20260823') outside_days,
          count(*) FILTER(WHERE window_start%600<>0) misaligned_windows
          FROM o''')
    with (args.out/'integrity.csv').open() as f:
        integrity = next(csv.DictReader(f))
    if any(int(v) for k,v in integrity.items() if k!='observations'):
        raise ValueError(f'Integrity checks failed: {integrity}')
    export('cell_k_distribution', '''SELECT day,kind,k,count(*) cells FROM (
      SELECT day,'raw' kind,k_raw k FROM cells UNION ALL SELECT day,'usable' kind,k_usable k FROM cells)
      GROUP BY day,kind,k ORDER BY day,kind,k''')
    export('daily_training_retention', '''SELECT day,count(*) cells,
      count(*) FILTER(WHERE k_usable>=3) trainable_cells,
      sum(k_raw) observations,sum(k_raw-k_usable) dropped_no_valid,
      sum(CASE WHEN k_usable%64<3 THEN k_usable%64 ELSE 0 END) dropped_small_tail,
      sum(k_usable-CASE WHEN k_usable%64<3 THEN k_usable%64 ELSE 0 END) retained_observations,
      sum(k_usable//64+CASE WHEN k_usable%64>=3 THEN 1 ELSE 0 END) training_groups FROM cells GROUP BY day ORDER BY day''')
    if key_mode:
        export('coverage_bins_distribution', 'SELECT day,n_present,n_valid,sum(observations) observations FROM coverage_keys GROUP BY day,n_present,n_valid ORDER BY day,n_present,n_valid')
    else:
        export('coverage_bins_distribution', 'SELECT day,n_present,n_valid,count(*) observations FROM o GROUP BY day,n_present,n_valid ORDER BY day,n_present,n_valid')
    # Distinct map/link table is useful for joining authoritative geometry later.
    export('observed_links', '''SELECT map_version,target_link_id,sum(observation_weight) observations,min(seg_idx) min_seg,max(seg_idx) max_seg,
      max(covered_m) max_observation_covered_m FROM geometry_input GROUP BY map_version,target_link_id ORDER BY map_version,target_link_id''')
    geometry_path = args.out/'geometry'/'geometry.parquet'
    if geometry_path.exists():
        c.read_parquet(str(geometry_path)).create_view('geo',replace=True)
        export('city_static_links', """SELECT map_version,count(*) links,min(length) min_length_m,
          median(length) median_length_m,avg(length) mean_length_m,max(length) max_length_m
          FROM geo GROUP BY map_version""")
        export('city_link_lengths', """SELECT CASE WHEN length<=50 THEN '01_0-50m'
          WHEN length<=100 THEN '02_50-100m' WHEN length<=200 THEN '03_100-200m'
          WHEN length<=500 THEN '04_200-500m' ELSE '05_over500m' END length_band,
          count(*) links FROM geo GROUP BY length_band ORDER BY length_band""")
        dynamic_path = args.out/'geometry'/'dynamic_geometry.parquet'
        if dynamic_path.exists():
            c.read_parquet(str(dynamic_path)).create_view('dynamic_geo',replace=True)
            c.execute("""CREATE OR REPLACE VIEW geometry_obs AS SELECT o.*,
              coalesce(d.link_length_m,CASE WHEN o.map_version=g.map_version THEN g.length END) AS geometry_link_length,
              least(500,coalesce(d.link_length_m,CASE WHEN o.map_version=g.map_version THEN g.length END)-500*o.seg_idx) AS geometry_segment_length,
              CASE WHEN d.link_length_m IS NOT NULL OR o.map_version=g.map_version THEN 'exact_version'
                WHEN g.target_link_id IS NULL THEN 'link_not_in_static_map'
                ELSE 'different_version_not_verified' END match_status,
              CASE WHEN d.link_length_m IS NOT NULL THEN 'raw_dynamic_same_version'
                WHEN o.map_version=g.map_version THEN 'static_same_version_fallback'
                ELSE 'unmatched' END geometry_source
              FROM geometry_input o LEFT JOIN dynamic_geo d ON o.map_version=d.map_version AND o.target_link_id=d.target_link_id
              LEFT JOIN geo g ON o.target_link_id=g.target_link_id""")
        else:
            c.execute("""CREATE OR REPLACE VIEW geometry_obs AS SELECT o.*,
              g.length AS geometry_link_length,
              least(500,g.length-500*o.seg_idx) AS geometry_segment_length,
              CASE WHEN g.target_link_id IS NULL THEN 'link_not_in_static_map'
                WHEN o.map_version=g.map_version THEN 'exact_version'
                ELSE 'different_version_not_verified' END match_status,
              CASE WHEN o.map_version=g.map_version THEN 'static_same_version_fallback' ELSE 'unmatched' END geometry_source
              FROM geometry_input o LEFT JOIN geo g ON o.target_link_id=g.target_link_id""")
        export('geometry_sources', """SELECT day,geometry_source,sum(observation_weight) observations
          FROM geometry_obs GROUP BY day,geometry_source ORDER BY day,geometry_source""")
        export('geometry_match', """SELECT day,match_status,sum(observation_weight) observations,
          count(DISTINCT (map_version,target_link_id)) versioned_links
          FROM geometry_obs GROUP BY day,match_status ORDER BY day,match_status""")
        export('exact_geometry_coverage', """SELECT day,
          CASE WHEN geometry_segment_length<=0 THEN '00_invalid_segment_geometry'
          WHEN covered_m/geometry_segment_length<0.25 THEN '01_below25pct'
          WHEN covered_m/geometry_segment_length<0.5 THEN '02_25-50pct'
          WHEN covered_m/geometry_segment_length<0.75 THEN '03_50-75pct'
          WHEN covered_m/geometry_segment_length<0.95 THEN '04_75-95pct'
          WHEN covered_m/geometry_segment_length<=1.05 THEN '05_95-105pct'
          ELSE '06_over105pct' END coverage_band,sum(observation_weight) observations
          FROM geometry_obs WHERE match_status='exact_version'
          GROUP BY day,coverage_band ORDER BY day,coverage_band""")
        export('geometry_segment_types', """SELECT day,
          CASE WHEN geometry_segment_length<=0 THEN 'invalid_segment_geometry'
            WHEN geometry_link_length<=500 THEN 'short_link_up_to_500m'
            WHEN geometry_segment_length<500 THEN 'last_partial_segment'
            ELSE 'full_500m_segment' END segment_type,
          sum(observation_weight) observations,count(DISTINCT (map_version,target_link_id,seg_idx)) versioned_segments
          FROM geometry_obs WHERE match_status='exact_version'
          GROUP BY day,segment_type ORDER BY day,segment_type""")
        export('geometry_absolute_difference', """SELECT day,
          CASE WHEN geometry_segment_length<50 THEN '01_below50m'
               WHEN geometry_segment_length<100 THEN '02_50-100m' ELSE '03_atleast100m' END length_band,
          CASE WHEN covered_m-geometry_segment_length < -10 THEN '01_below_minus10m'
               WHEN covered_m-geometry_segment_length < -5 THEN '02_minus10_to_minus5m'
               WHEN covered_m-geometry_segment_length <= 5 THEN '03_within_plus_minus5m'
               WHEN covered_m-geometry_segment_length <= 10 THEN '04_plus5_to_plus10m'
               ELSE '05_above_plus10m' END difference_band,
          sum(observation_weight) observations,min(covered_m-geometry_segment_length) min_difference_m,
          max(covered_m-geometry_segment_length) max_difference_m
          FROM geometry_obs WHERE match_status='exact_version' AND geometry_segment_length>0
          GROUP BY day,length_band,difference_band ORDER BY day,length_band,difference_band""")
        export('exact_segment_lengths', """WITH segments AS (
          SELECT DISTINCT map_version,target_link_id,seg_idx,geometry_segment_length
          FROM geometry_obs WHERE match_status='exact_version')
          SELECT geometry_segment_length length_m,count(*) segments FROM segments
          GROUP BY geometry_segment_length ORDER BY geometry_segment_length""")
    # Exact mask counts are computed with the training membership/RNG contract.
    support=[]
    numeric=['rows','retained_observations','dropped_no_valid','dropped_tail','groups','hidden_trajectories','supervised_bins','supported_supervised_bins']
    for day in DAYS:
        rows=[r for r in receipts if r['day']==day]
        support.append(dict(day=day,**{k:sum(r[k] for r in rows) for k in numeric}))
    with (args.out/'daily_mask_reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
    dump(args.out/'aggregation_complete.json',dict(status='partial_smoke' if args.limit else 'complete',partitions=len(parts),
        rows=sum(r['rows'] for r in receipts),seed=20260921,epoch=0,m_max=64,
        trajectory_identity='sample_id first # component; confirm against upstream identity contract',
        geometry_status='dynamic_same_version' if (args.out/'geometry'/'dynamic_geometry.parquet').exists() else 'static_same_version_subset'))
    c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['scan','aggregate'],required=True)
    p.add_argument('--out',type=Path,default=REPO/'experiments/trajectory_mlp_v1/reports/seven_day_p0_20260817_23')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    (scan if a.stage=='scan' else aggregate)(a)

if __name__=='__main__':main()
