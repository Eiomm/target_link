"""Dispatch day/bucket tensor conversion to YARN workers; atomically publish HDFS output.

The Spark Python environment needs only stdlib + PySpark. Each executor invokes
an explicitly configured Python with NumPy, PyArrow and CPU-capable PyTorch to
reuse the exact canonical converter. No conversion runs on the submitting host.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
import posixpath
import re
from pathlib import Path
import subprocess
import tempfile
import uuid
from urllib.parse import urlsplit

FORMAT = 'trajectory_mlp_tensors_v1'
STORAGE = 'zlib-group-blocks-v1'
CHANNELS = ['piece_time_sum_seconds', 'piece_ratio_sum', 'zero']


def join(root, *parts):
    return '/'.join([str(root).rstrip('/')] + [str(p).strip('/') for p in parts])


def hadoop_shell_command():
    """Use the executor's localized Spark Hadoop jars, not a host hdfs command."""
    java_home = os.environ.get('JAVA_HOME')
    java = str(Path(java_home)/'bin/java') if java_home else shutil.which('java')
    if not java or not Path(java).is_file():
        raise RuntimeError('Executor Java executable unavailable; check JAVA_HOME')
    entries = []
    for key in ('HADOOP_CONF_DIR','SPARK_CONF_DIR'):
        if os.environ.get(key):
            entries.append(os.environ[key])
    roots = [Path.cwd()]
    try:
        from pyspark import SparkFiles
        localized = SparkFiles.getRootDirectory()
        if localized:
            roots.append(Path(localized))
    except (ImportError, RuntimeError, AssertionError):
        pass
    for root in roots:
        conf = root/'__spark_conf__'
        if conf.is_dir():
            entries.extend([str(conf),str(conf/'__hadoop_conf__')])
        for jars in root.glob('__spark_libs__*'):
            if jars.is_dir():
                entries.append(str(jars/'*'))
    spark_home = os.environ.get('SPARK_HOME')
    if spark_home and (Path(spark_home)/'jars').is_dir():
        entries.append(str(Path(spark_home)/'jars/*'))
    entries.extend(x for x in os.environ.get('CLASSPATH','').split(os.pathsep) if x)
    if not entries:
        raise RuntimeError('Executor Spark/Hadoop classpath unavailable; expected localized __spark_libs__ and __spark_conf__')
    return [java,'-cp',os.pathsep.join(dict.fromkeys(entries)),
            'org.apache.hadoop.fs.FsShell']


class Hdfs:
    """Executor transport. Spark supplies Java, Hadoop jars, config and tokens."""
    def __init__(self, binary=None):
        self.command = [binary,'dfs'] if binary else hadoop_shell_command()

    def _run(self, args):
        try:
            return subprocess.run(self.command+list(map(str,args)),text=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        except OSError as exc:
            raise RuntimeError('Cannot launch executor Hadoop transport %r: %s' %
                               (self.command[0],exc)) from exc

    def run(self, *args):
        result = self._run(args)
        if result.returncode:
            raise RuntimeError('Hadoop transport failed: %s\n%s' % (args,result.stderr[-8000:]))
        return result.stdout

    def exists(self, path):
        result = self._run(['-test','-e',path])
        if result.returncode == 0:
            return True
        if result.returncode != 1:
            raise RuntimeError('Cannot check HDFS path: %s: %s' % (path,result.stderr))
        return False

    def mkdir(self, path, exclusive=False):
        self.run(*(['-mkdir',path] if exclusive else ['-mkdir','-p',path]))

    def inventory(self, path):
        result = []
        base = urlsplit(path)
        for line in self.run('-ls',path).splitlines():
            fields = line.split(None,7)
            if len(fields) != 8 or not fields[0].startswith('-') or not fields[7].endswith('.parquet'):
                continue
            uri = fields[7]
            if uri.startswith('/'):
                uri = base.scheme + '://' + base.netloc + uri
            size, mtime = self.run('-stat','%b %Y',uri).strip().split()
            result.append(dict(uri=uri,bytes=int(size),mtime_ms=int(mtime)))
        if not result:
            raise ValueError('No input parquet files: ' + path)
        return sorted(result,key=lambda x:x['uri'])

    def get(self, remote, local):
        self.run('-get',remote,str(local))

    def put(self, local, remote):
        self.run('-put',str(local),remote)

    def rename(self, source, destination):
        self.run('-mv',source,destination)

    def unlink(self, path):
        self.run('-rm',path)

    def rmdir(self, path):
        self.run('-rmdir',path)

    def write_json(self, path, value):
        with tempfile.TemporaryDirectory(prefix='tensor-json-') as temp:
            local = Path(temp)/'manifest.json'
            local.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
            self.put(local,path)


class JvmHdfs:
    """Driver filesystem access through the initialized Spark JVM and its config."""
    def __init__(self, spark):
        self.sc = spark.sparkContext
        self.jvm = self.sc._jvm
        self.conf = self.sc._jsc.hadoopConfiguration()

    def _path_fs(self, uri):
        path = self.jvm.org.apache.hadoop.fs.Path(str(uri))
        return path,path.getFileSystem(self.conf)

    def exists(self, uri):
        path,fs = self._path_fs(uri)
        return fs.exists(path)

    def mkdir(self, uri):
        path,fs = self._path_fs(uri)
        if not fs.mkdirs(path):
            raise RuntimeError('Cannot create HDFS directory: '+uri)

    def acquire_lock(self, uri):
        path,fs = self._path_fs(uri)
        # create(overwrite=False) is atomic; checking exists then mkdir is not.
        stream = fs.create(path,False)
        stream.close()

    def unlink(self, uri):
        path,fs = self._path_fs(uri)
        if not fs.delete(path,False):
            raise RuntimeError('Cannot remove HDFS marker: '+uri)

    def rename(self, source, destination):
        src,_ = self._path_fs(source)
        dst,_ = self._path_fs(destination)
        context = self.jvm.org.apache.hadoop.fs.FileContext.getFileContext(src.toUri(),self.conf)
        option = self.jvm.org.apache.hadoop.fs.Options.Rename
        options = self.sc._gateway.new_array(option,1)
        options[0] = option.NONE
        context.rename(src,dst,options)

    def write_json(self, uri, value):
        path,fs = self._path_fs(uri)
        stream = fs.create(path,False)
        try:
            stream.write(bytearray((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf-8')))
        finally:
            stream.close()


_CHECKED_ENVIRONMENTS = set()


def worker_environment(job):
    code = Path(job['code_root']).resolve()
    converter = code/'experiments/trajectory_mlp_v1/tools/prepare_tensors.py'
    if not converter.is_file():
        raise RuntimeError('Localized converter not found: '+str(converter))
    key = (job['worker_python'],str(converter))
    if key not in _CHECKED_ENVIRONMENTS:
        check = ('import sys; '
                 'assert sys.version_info >= (3,9), "Converter requires Python >=3.9"; '
                 'import numpy, pyarrow, torch; '
                 'print("Python="+sys.version.split()[0]+" numpy="+numpy.__version__+'
                 '" pyarrow="+pyarrow.__version__+" torch="+torch.__version__)')
        try:
            result = subprocess.run([job['worker_python'],'-c',check],text=True,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError('Converter environment unavailable on executor: %s. '
                               'Distribute TENSOR_ENV_ARCHIVE or configure a verified worker Python. %s' %
                               (job['worker_python'],exc)) from exc
        if result.returncode:
            raise RuntimeError('Converter dependency check failed on executor:\n'+result.stdout[-8000:])
        print('[TENSOR_WORKER_ENV] '+result.stdout.strip(),flush=True)
        _CHECKED_ENVIRONMENTS.add(key)
    return converter


def preflight(job):
    """Run inside a real Spark worker; imports dependencies and checks HDFS reads."""
    worker_environment(job)
    fs = Hdfs(job.get('hdfs_bin'))
    files = fs.inventory(join(job['source'],'observations_v2','day='+job['day'],'bucket='+job['bucket']))
    return dict(host=os.environ.get('HOSTNAME','unknown'),worker_python=job['worker_python'],
                source_files=len(files),day=job['day'],bucket=job['bucket'])


def dispatch_preflight(job):
    # Keep the bootstrap self-contained: Spark serializes this __main__ function
    # before the executor has imported our package.
    import os
    import sys
    sys.path.insert(0, os.path.abspath(job['code_root']))
    from tools.prepare_tensors_yarn import preflight
    return preflight(job)


def convert_partition(job, fs=None):
    """One attempt owns its own HDFS path, so retries never overwrite each other."""
    converter = worker_environment(job)
    fs = fs or Hdfs(job.get('hdfs_bin'))
    day,bucket = job['day'],job['bucket']
    source_dir = join(job['source'],'observations_v2','day='+day,'bucket='+bucket)
    before = fs.inventory(source_dir)
    with tempfile.TemporaryDirectory(prefix='tensor-convert-') as temp:
        temp = Path(temp)
        source,out = temp/'source',temp/'ready'
        part = source/'observations_v2'/('day='+day)/('bucket='+bucket)
        part.mkdir(parents=True)
        for entry in before:
            fs.get(entry['uri'],part/entry['uri'].rsplit('/',1)[-1])
        # The child only sees executor-local temporary files. It does not write
        # to the shared repository, and it does not run any model training.
        cmd = [job['worker_python'],str(converter),'--source',str(source),'--out',str(out),
               '--days',day,'--m-max',str(job['m_max']),'--seed',str(job['seed']),'--workers','1']
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode:
            raise RuntimeError('Tensor worker %s/%s failed:\n%s' % (day,bucket,result.stdout[-12000:]))
        manifest = json.loads((out/'_TENSORS_SUCCESS.json').read_text())
        if (manifest['format'],manifest['storage'],manifest['m_max'],manifest['data_seed']) != (
                FORMAT,STORAGE,job['m_max'],job['seed']):
            raise ValueError('Worker output protocol mismatch')
        key = day+'/'+bucket
        if set(manifest['partitions']) != {key}:
            raise ValueError('Worker produced unexpected partitions')
        receipt = manifest['partitions'][key]
        if fs.inventory(source_dir) != before:
            raise ValueError('HDFS source changed during conversion: ' + source_dir)
        relative = join('_parts','day='+day,'bucket='+bucket,'attempt='+uuid.uuid4().hex)
        remote = join(job['destination'],relative)
        fs.mkdir(remote)
        for rec in [*receipt['arrays'].values(),receipt['payload']]:
            local = out/rec['path']
            fs.put(local,join(remote,local.name))
            rec['path'] = join(relative,local.name)
        receipt['source_uri'] = source_dir
        receipt['source_inventory'] = before
        receipt['builder_sha256'] = manifest['builder_sha256']
        receipt['reader_sha256'] = manifest['reader_sha256']
        # Uploaded records can be inspected even if the Spark driver later fails.
        fs.write_json(join(remote,'receipt.json'),receipt)
        return job['split'],key,receipt


def dispatch(job):
    import os
    import sys
    sys.path.insert(0, os.path.abspath(job['code_root']))
    from tools.prepare_tensors_yarn import convert_partition
    return convert_partition(job)


def publish(fs, staging, output, jobs, results, m_max, seed):
    expected = {(j['split'],j['day']+'/'+j['bucket']) for j in jobs}
    actual = [(side,key) for side,key,_ in results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate partition results; refusing publication')
    for side in sorted({j['split'] for j in jobs}):
        parts = {key:rec for split,key,rec in sorted(results,key=lambda x:(x[0],x[1])) if split == side}
        manifest = dict(format=FORMAT,storage=STORAGE,channels=CHANNELS,m_max=m_max,data_seed=seed,
                        backend='spark-yarn',partitions=parts,
                        total_groups=sum(r['stats']['groups'] for r in parts.values()),
                        total_rows=sum(r['stats']['raw_rows'] for r in parts.values()),
                        output_bytes=sum(a['bytes'] for r in parts.values()
                                         for a in [*r['arrays'].values(),r['payload']]))
        fs.write_json(join(staging,side,'_TENSORS_SUCCESS.json'),manifest)
        fs.unlink(join(staging,side,'_TENSORS_BUILDING'))
    fs.write_json(join(staging,'_SUCCESS.json'),dict(format=FORMAT,status='passed',partitions=len(results)))
    if fs.exists(output):
        raise ValueError('Output appeared during build; refusing to overwrite: ' + output)
    fs.rename(staging,output)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--source',required=True,help='HDFS corpus root containing observations_v2')
    p.add_argument('--out',required=True,help='Fresh HDFS output root containing train/val')
    p.add_argument('--train-days',nargs='+',default=['202608%02d'%d for d in range(17,23)])
    p.add_argument('--val-days',nargs='+',default=['20260823'])
    p.add_argument('--buckets',type=int,default=128)
    p.add_argument('--parallelism',type=int,default=20)
    p.add_argument('--m-max',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--worker-python',required=True)
    p.add_argument('--code-root',default='tensor_code')
    p.add_argument('--hdfs-bin',default=None,help='Optional executor Hadoop CLI override; default uses Spark Hadoop jars')
    p.add_argument('--preflight-only',action='store_true',help='Check actual executor dependencies and source access; do not convert or write output')
    a = p.parse_args()
    if set(a.train_days)&set(a.val_days) or len(a.train_days)!=len(set(a.train_days)) or len(a.val_days)!=len(set(a.val_days)):
        p.error('Train/validation dates must be unique and disjoint')
    if a.parallelism<1 or a.m_max<3 or not 1<=a.buckets<=128:
        p.error('Invalid sizes')
    if not a.source.startswith('hdfs://') or not a.out.startswith('hdfs://'):
        p.error('This entrypoint requires HDFS input and output')
    def normalized(uri):
        value = urlsplit(uri)
        if not value.netloc or value.query or value.fragment:
            p.error('Invalid HDFS URI: '+uri)
        return 'hdfs://'+value.netloc+posixpath.normpath('/'+value.path.lstrip('/'))
    source,output = normalized(a.source),normalized(a.out)
    if any(not re.fullmatch(r'[0-9]{8}', day) for day in a.train_days+a.val_days):
        p.error('Dates must use YYYYMMDD')
    if source == output or source.startswith(output+'/') or output.startswith(source+'/'):
        p.error('Input and output must be separate trees')
    spark = None
    fs = None
    locked = False
    staging_created = False
    phase = 'spark-initialization'
    lock = output+'._lock'
    staging = output+'._building_'+uuid.uuid4().hex
    try:
        # Initialize Spark first, matching the established cluster jobs. Driver
        # HDFS access never assumes the hdfs executable exists in the container.
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName('prepare_training_tensors_v1').getOrCreate()
        print('[TENSOR_DRIVER] Spark initialized; applicationId='+spark.sparkContext.applicationId,flush=True)
        fs = JvmHdfs(spark)
        jobs = []
        for side,days in [('train',a.train_days),('val',a.val_days)]:
            for day in days:
                for bucket in range(a.buckets):
                    jobs.append(dict(split=side,day=day,bucket=str(bucket),source=source,
                                     destination=join(staging,side),m_max=a.m_max,seed=a.seed,
                                     code_root=a.code_root,worker_python=a.worker_python,hdfs_bin=a.hdfs_bin))
        phase = 'executor-preflight'
        # This checks an actual executor before any output is created. Each later
        # conversion task also checks its own environment, including newly allocated nodes.
        reports = spark.sparkContext.parallelize([jobs[0]],1).map(dispatch_preflight).collect()
        print('[TENSOR_PREFLIGHT_PASSED] '+json.dumps(reports),flush=True)
        if a.preflight_only:
            print('Executor check completed; no conversion or HDFS output was created.',flush=True)
            return
        phase = 'output-initialization'
        fs.mkdir(output.rsplit('/',1)[0])
        fs.acquire_lock(lock)
        locked = True
        if fs.exists(output):
            raise ValueError('Output exists; choose a new version: '+output)
        fs.mkdir(staging)
        staging_created = True
        for side in ('train','val'):
            fs.mkdir(join(staging,side))
            fs.write_json(join(staging,side,'_TENSORS_BUILDING'),dict(format=FORMAT))
        fs.write_json(join(staging,'build_config.json'),vars(a))
        print('Staging: %s; %d day/bucket tasks; parallelism=%d' % (staging,len(jobs),a.parallelism),flush=True)
        phase = 'partition-conversion'
        results = spark.sparkContext.parallelize(jobs,len(jobs)).map(dispatch).collect()
        phase = 'publication'
        publish(fs,staging,output,jobs,results,a.m_max,a.seed)
        print('Published training tensors: '+output,flush=True)
    except BaseException as exc:
        detail = traceback.format_exc()
        print('[TENSOR_PREPARE_FAILED] phase=%s %s: %s' % (phase,type(exc).__name__,exc),file=sys.stderr,flush=True)
        if staging_created:
            print('Not published. Incomplete output retained at: '+staging,file=sys.stderr,flush=True)
            try:
                fs.write_json(join(staging,'failure.json'),dict(phase=phase,error=repr(exc),traceback=detail))
            except Exception as log_error:
                print('Cannot persist failure.json: '+str(log_error),file=sys.stderr,flush=True)
        raise
    finally:
        if locked:
            try:
                fs.unlink(lock)
            except Exception as cleanup_error:
                print('Lock cleanup failed: '+str(cleanup_error),file=sys.stderr,flush=True)
        if spark is not None:
            try:
                spark.stop()
            except Exception as cleanup_error:
                print('Spark cleanup failed: '+str(cleanup_error),file=sys.stderr,flush=True)


if __name__ == '__main__':
    main()
