"""Build published observation-v3 partitions on YARN and atomically publish HDFS output.

Each Spark task downloads one source day/bucket, invokes the canonical local
``build_partition`` with the executor's verified Python environment, then
uploads immutable artifacts under a unique attempt directory.  The driver is
only responsible for publication after every partition succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
import tempfile
import traceback
from urllib.parse import urlsplit
import uuid

# Make direct ``python tools/prepare_observation_v3_yarn.py`` invocation work
# as well as Spark's localized archive import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_tensors_yarn import Hdfs, JvmHdfs, join


FORMAT = 'trajectory_mlp_observation_v3'
_CHECKED_ENVIRONMENTS = set()


def worker_subprocess_env(job):
    """Use the archive's C++ runtime on old YARN nodes, only for conversion."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(job['code_root']).resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    library = Path(job['worker_python']).resolve().parent.parent / 'lib'
    if (library / 'libstdc++.so.6').is_file():
        env['LD_LIBRARY_PATH'] = str(library) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def worker_environment(job):
    """Validate the executor Python and localized v3 converter once per node."""
    code = Path(job['code_root']).resolve()
    converter = code / 'trajectory_mae/tools/prepare_observation_v3.py'
    if not converter.is_file():
        raise RuntimeError('Localized observation-v3 converter not found: ' + str(converter))
    key = (job['worker_python'], str(converter))
    if key not in _CHECKED_ENVIRONMENTS:
        check = (
            'import sys; assert sys.version_info >= (3, 9), '
            '"Observation-v3 converter requires Python >=3.9"; '
            'import numpy, pyarrow, torch; '
            'print("Python="+sys.version.split()[0]+" numpy="+numpy.__version__+'
            '" pyarrow="+pyarrow.__version__+" torch="+torch.__version__)'
        )
        try:
            result = subprocess.run([job['worker_python'], '-c', check], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=120, env=worker_subprocess_env(job))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError('Observation-v3 worker environment unavailable: %s' % exc) from exc
        if result.returncode:
            raise RuntimeError('Observation-v3 dependency check failed on executor:\n' +
                               result.stdout[-8000:])
        print('[OBSERVATION_V3_WORKER_ENV] ' + result.stdout.strip(), flush=True)
        _CHECKED_ENVIRONMENTS.add(key)
    return converter


def preflight(job):
    worker_environment(job)
    fs = Hdfs(job.get('hdfs_bin'))
    files = fs.inventory(join(job['source'], 'observations_v2', 'day=' + job['day'],
                              'bucket=' + job['bucket']))
    return dict(host=os.environ.get('HOSTNAME', 'unknown'), worker_python=job['worker_python'],
                source_files=len(files), day=job['day'], bucket=job['bucket'])


def dispatch_preflight(job):
    import os
    import sys
    sys.path.insert(0, os.path.abspath(job['code_root']))
    from tools.prepare_observation_v3_yarn import preflight as run
    return run(job)


def _run_local_build(job, source, output):
    """Invoke the canonical converter in an isolated Python subprocess."""
    payload = json.dumps([str(source), str(output), job['day'], job['bucket'],
                          job['m_max'], job['seed']])
    script = (
        'import json, sys; '
        'from trajectory_mae.tools.prepare_observation_v3 import build_partition; '
        'job = json.loads(sys.argv[1]); '
        'build_partition(tuple(job))'
    )
    env = worker_subprocess_env(job)
    result = subprocess.run([job['worker_python'], '-c', script, payload], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError('Observation-v3 worker %s/%s failed:\n%s' %
                           (job['day'], job['bucket'], result.stdout[-12000:]))


def convert_partition(job, fs=None):
    """Build one partition; retries never overwrite another attempt's output."""
    worker_environment(job)
    fs = fs or Hdfs(job.get('hdfs_bin'))
    day, bucket = job['day'], job['bucket']
    source_dir = join(job['source'], 'observations_v2', 'day=' + day, 'bucket=' + bucket)
    before = fs.inventory(source_dir)
    with tempfile.TemporaryDirectory(prefix='observation-v3-') as temp:
        temp = Path(temp)
        source, output = temp / 'source', temp / 'ready'
        local_part = source / 'observations_v2' / ('day=' + day) / ('bucket=' + bucket)
        local_part.mkdir(parents=True)
        for entry in before:
            fs.get(entry['uri'], local_part / entry['uri'].rsplit('/', 1)[-1])
        _run_local_build(job, source, output)
        receipt_path = output / 'receipts' / ('%s_%s.json' % (day, bucket))
        if not receipt_path.is_file():
            raise ValueError('Canonical v3 build did not write a receipt')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('format'), receipt.get('m_max'), receipt.get('data_seed')) != (
                FORMAT, job['m_max'], job['seed']):
            raise ValueError('Worker output protocol mismatch')
        if fs.inventory(source_dir) != before:
            raise ValueError('HDFS source changed during conversion: ' + source_dir)

        relative = join('_parts', 'day=' + day, 'bucket=' + bucket,
                        'attempt=' + uuid.uuid4().hex)
        remote = join(job['destination'], relative)
        fs.mkdir(remote)
        for name in ('observations', 'index', 'cells'):
            rec = receipt[name]
            local = output / rec['path']
            if not local.is_file():
                raise ValueError('Worker receipt artifact missing: ' + str(local))
            destination = join(remote, local.name)
            fs.put(local, destination)
            rec['path'] = join(relative, local.name)
        receipt['source_uri'] = source_dir
        receipt['source_inventory'] = before
        fs.write_json(join(remote, 'receipt.json'), receipt)
        return job['split'], day + '/' + bucket, receipt


def dispatch(job):
    import os
    import sys
    sys.path.insert(0, os.path.abspath(job['code_root']))
    from tools.prepare_observation_v3_yarn import convert_partition
    return convert_partition(job)


def publish(fs, staging, output, jobs, results, m_max, seed):
    expected = {(job['split'], job['day'] + '/' + job['bucket']) for job in jobs}
    actual = [(side, key) for side, key, _ in results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate partition results; refusing publication')
    for side in ('train', 'val'):
        parts = {key: rec for split, key, rec in sorted(results, key=lambda item: (item[0], item[1]))
                 if split == side}
        manifest = dict(
            format=FORMAT, m_max=m_max, data_seed=seed, backend='spark-yarn', partitions=parts,
            total_source_rows=sum(rec['stats']['raw_rows'] for rec in parts.values()),
            total_stored_rows=sum(rec['stored_rows'] for rec in parts.values()),
            total_groups=sum(rec['stats']['groups'] for rec in parts.values()),
            dropped_no_valid=sum(rec['stats']['dropped_no_valid'] for rec in parts.values()),
        )
        fs.write_json(join(staging, side, '_OBSERVATION_V3_SUCCESS.json'), manifest)
        fs.unlink(join(staging, side, '_BUILDING'))
    fs.write_json(join(staging, '_SUCCESS.json'),
                  dict(format=FORMAT, status='passed', partitions=len(results)))
    if fs.exists(output):
        raise ValueError('Output appeared during build; refusing to overwrite: ' + output)
    fs.rename(staging, output)


def _normalized_hdfs(parser, uri):
    value = urlsplit(uri)
    if not value.netloc or value.query or value.fragment:
        parser.error('Invalid HDFS URI: ' + uri)
    return 'hdfs://' + value.netloc + posixpath.normpath('/' + value.path.lstrip('/'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', required=True, help='HDFS corpus root containing observations_v2')
    parser.add_argument('--out', required=True, help='Fresh HDFS output root containing train/val')
    parser.add_argument('--train-days', nargs='+', default=['202608%02d' % day for day in range(17, 23)])
    parser.add_argument('--val-days', nargs='+', default=['20260823'])
    parser.add_argument('--buckets', type=int, default=128)
    parser.add_argument('--parallelism', type=int, default=20)
    parser.add_argument('--m-max', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--worker-python', required=True)
    parser.add_argument('--code-root', default='observation_v3_code')
    parser.add_argument('--hdfs-bin', default=None)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if (set(args.train_days) & set(args.val_days) or len(args.train_days) != len(set(args.train_days))
            or len(args.val_days) != len(set(args.val_days))):
        parser.error('Train/validation dates must be unique and disjoint')
    if args.parallelism < 1 or args.m_max < 3 or not 1 <= args.buckets <= 128:
        parser.error('Invalid sizes')
    if not args.source.startswith('hdfs://') or not args.out.startswith('hdfs://'):
        parser.error('This entrypoint requires HDFS input and output')
    if any(not re.fullmatch(r'[0-9]{8}', day) for day in args.train_days + args.val_days):
        parser.error('Dates must use YYYYMMDD')
    source, output = _normalized_hdfs(parser, args.source), _normalized_hdfs(parser, args.out)
    if source == output or source.startswith(output + '/') or output.startswith(source + '/'):
        parser.error('Input and output must be separate trees')

    spark = None
    fs = None
    locked = False
    staging_created = False
    phase = 'spark-initialization'
    lock, staging = output + '._lock', output + '._building_' + uuid.uuid4().hex
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName('prepare_observation_v3').getOrCreate()
        print('[OBSERVATION_V3_DRIVER] Spark initialized; applicationId=' +
              spark.sparkContext.applicationId, flush=True)
        fs = JvmHdfs(spark)
        jobs = []
        for side, days in (('train', args.train_days), ('val', args.val_days)):
            for day in days:
                for bucket in range(args.buckets):
                    jobs.append(dict(split=side, day=day, bucket=str(bucket), source=source,
                                     destination=join(staging, side), m_max=args.m_max,
                                     seed=args.seed, code_root=args.code_root,
                                     worker_python=args.worker_python, hdfs_bin=args.hdfs_bin))
        phase = 'executor-preflight'
        reports = spark.sparkContext.parallelize([jobs[0]], 1).map(dispatch_preflight).collect()
        print('[OBSERVATION_V3_PREFLIGHT_PASSED] ' + json.dumps(reports), flush=True)
        if args.preflight_only:
            print('Executor check completed; no conversion or HDFS output was created.', flush=True)
            return
        phase = 'output-initialization'
        fs.mkdir(output.rsplit('/', 1)[0])
        fs.acquire_lock(lock)
        locked = True
        if fs.exists(output):
            raise ValueError('Output exists; choose a new version: ' + output)
        fs.mkdir(staging)
        staging_created = True
        for side in ('train', 'val'):
            fs.mkdir(join(staging, side))
            fs.write_json(join(staging, side, '_BUILDING'), dict(format=FORMAT))
        fs.write_json(join(staging, 'build_config.json'), vars(args))
        print('Staging: %s; %d day/bucket tasks; parallelism=%d' %
              (staging, len(jobs), args.parallelism), flush=True)
        phase = 'partition-conversion'
        results = spark.sparkContext.parallelize(jobs, len(jobs)).map(dispatch).collect()
        phase = 'publication'
        publish(fs, staging, output, jobs, results, args.m_max, args.seed)
        print('Published observation-v3 corpus: ' + output, flush=True)
    except BaseException as exc:
        detail = traceback.format_exc()
        print('[OBSERVATION_V3_PREPARE_FAILED] phase=%s %s: %s' %
              (phase, type(exc).__name__, exc), file=sys.stderr, flush=True)
        if staging_created:
            print('Not published. Incomplete output retained at: ' + staging,
                  file=sys.stderr, flush=True)
            try:
                fs.write_json(join(staging, 'failure.json'),
                              dict(phase=phase, error=repr(exc), traceback=detail))
            except Exception as log_error:
                print('Cannot persist failure.json: ' + str(log_error),
                      file=sys.stderr, flush=True)
        raise
    finally:
        if locked:
            try:
                fs.unlink(lock)
            except Exception as cleanup_error:
                print('Lock cleanup failed: ' + str(cleanup_error), file=sys.stderr, flush=True)
        if spark is not None:
            try:
                spark.stop()
            except Exception as cleanup_error:
                print('Spark cleanup failed: ' + str(cleanup_error), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
