"""YARN adapter tests with local transport; never contact a real cluster."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.prepare_tensors_yarn import convert_partition, publish, FORMAT, STORAGE
from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells
from experiments.trajectory_mlp_v1.tensor_corpus import verify

ROOT = Path(__file__).resolve().parents[1]


class LocalTransport:
    def __init__(self, root):
        self.root = root
    def path(self, uri):
        return self.root/urlsplit(uri).path.lstrip('/')
    def exists(self, uri):
        return self.path(uri).exists()
    def mkdir(self, uri, exclusive=False):
        self.path(uri).mkdir(parents=not exclusive, exist_ok=not exclusive)
    def inventory(self, uri):
        return [dict(uri=uri+'/'+p.name,bytes=p.stat().st_size,mtime_ms=p.stat().st_mtime_ns//1000000)
                for p in sorted(self.path(uri).glob('*.parquet'))]
    def get(self, uri, local):
        shutil.copyfile(self.path(uri),local)
    def put(self, local, uri):
        shutil.copyfile(local,self.path(uri))
    def write_json(self, uri, value):
        self.path(uri).write_text(json.dumps(value))
    def unlink(self, uri):
        self.path(uri).unlink()
    def rename(self, source, dest):
        self.path(source).rename(self.path(dest))


def setup(tmp_path):
    fs = LocalTransport(tmp_path)
    source,staging,output = 'hdfs://test/source','hdfs://test/staging','hdfs://test/published'
    part = fs.path(source+'/observations_v2/day=20260817/bucket=0')
    part.mkdir(parents=True)
    rows = [dict(cell_id=0,sample_id='sample-'+str(i),dt=float(i),T_diff=[1.,2.,3.],
                 ratio_pct=[4,6,10],valid=[True,True,i!=0],bin_pos=[0,0,3]) for i in range(5)]
    pq.write_table(pa.Table.from_pylist(rows),part/'part.parquet')
    fs.mkdir(staging+'/train')
    fs.write_json(staging+'/train/_TENSORS_BUILDING',{})
    job = dict(split='train',day='20260817',bucket='0',source=source,destination=staging+'/train',
               m_max=64,seed=20260921,code_root=str(ROOT),worker_python=sys.executable,hdfs_bin='unused')
    return fs,job,staging,output


def test_worker_retry_and_atomic_publication_keep_training_contract(tmp_path):
    fs,job,staging,output = setup(tmp_path)
    first = convert_partition(job,fs)
    retry = convert_partition(job,fs)
    assert first[2]['payload']['path'] != retry[2]['payload']['path']
    assert first[2]['payload']['sha256'] == retry[2]['payload']['sha256']
    assert not fs.exists(output)
    publish(fs,staging,output,[job],[retry],64,20260921)
    manifest = verify(fs.path(output+'/train'))
    assert manifest['format'] == FORMAT and manifest['storage'] == STORAGE
    assert manifest['total_groups'] == 1
    old = list(CellDataset([str(fs.path(job['source']))],['20260817']))
    new = list(CellDataset([str(fs.path(output+'/train'))],['20260817']))
    a,b = collate_cells(old,epoch=8),collate_cells(new,epoch=8)
    for name in ('x','bin_valid','traj_valid','delta_t','mae_mask'):
        np.testing.assert_array_equal(a[name],b[name])
    assert a['sample_ids'] == b['sample_ids']
    assert not fs.exists(staging)


def test_source_mutation_is_rejected_before_upload(tmp_path):
    fs,job,staging,output = setup(tmp_path)
    original = fs.inventory
    calls = []
    def changed(uri):
        entries = original(uri)
        calls.append(1)
        if len(calls)>1:
            entries[0]['mtime_ms'] += 1
        return entries
    fs.inventory = changed
    with pytest.raises(ValueError,match='source changed'):
        convert_partition(job,fs)
    assert not fs.exists(staging+'/train/_parts')
    assert not fs.exists(output)


@pytest.mark.parametrize('results',[[], [('train','20260817/0',{}),('train','20260817/0',{})]])
def test_incomplete_or_duplicate_results_cannot_publish(tmp_path,results):
    fs,job,staging,output = setup(tmp_path)
    with pytest.raises(ValueError,match='Incomplete or duplicate'):
        publish(fs,staging,output,[job],results,64,20260921)
    assert not fs.exists(output)


def test_submission_dry_run_is_cluster_only(tmp_path):
    import os
    env = dict(os.environ,MODE='dry',SPARK_SUBMIT='/not-executed/spark-submit')
    result = subprocess.run(['bash',str(ROOT/'scripts/submit_training_tensors_yarn.sh')],
                            env=env,text=True,capture_output=True,check=True)
    assert '--master yarn --deploy-mode cluster' in result.stdout
    assert '--executor-cores 1' in result.stdout
    assert '--worker-python' in result.stdout
    assert 'no YARN submission or data conversion' in result.stdout


def test_missing_worker_python_is_explicit_and_precedes_input_read(tmp_path):
    from tools.prepare_tensors_yarn import worker_environment
    with pytest.raises(RuntimeError,match='Converter environment unavailable on executor'):
        worker_environment(dict(code_root=str(ROOT),worker_python=str(tmp_path/'missing-python')))


def test_hadoop_transport_uses_localized_jars_without_hdfs_binary(tmp_path,monkeypatch):
    from tools.prepare_tensors_yarn import hadoop_shell_command
    monkeypatch.chdir(tmp_path)
    java = tmp_path/'jdk/bin/java'
    java.parent.mkdir(parents=True)
    java.touch()
    (tmp_path/'__spark_libs__').mkdir()
    (tmp_path/'__spark_conf__').mkdir()
    monkeypatch.setenv('JAVA_HOME',str(tmp_path/'jdk'))
    for name in ('SPARK_HOME','HADOOP_CONF_DIR','SPARK_CONF_DIR','CLASSPATH'):
        monkeypatch.delenv(name,raising=False)
    command = hadoop_shell_command()
    assert command[0] == str(java)
    assert command[-1] == 'org.apache.hadoop.fs.FsShell'
    assert str(tmp_path/'__spark_libs__/*') in command[2]
    assert str(tmp_path/'__spark_conf__') in command[2]


def test_driver_initializes_spark_before_io_and_preflight_failure_writes_nothing(monkeypatch):
    import types
    import tools.prepare_tensors_yarn as module
    events = []
    class RDD:
        def map(self,fn):
            assert fn is module.dispatch_preflight
            events.append('executor-preflight')
            raise RuntimeError('simulated worker dependency failure')
    class Context:
        applicationId = 'test-application'
        def parallelize(self,items,n):
            return RDD()
    class Spark:
        sparkContext = Context()
        def stop(self): events.append('spark-stop')
    class Builder:
        def appName(self,name): return self
        def getOrCreate(self):
            events.append('spark-start')
            return Spark()
    fake_sql = types.ModuleType('pyspark.sql')
    fake_sql.SparkSession = types.SimpleNamespace(builder=Builder())
    monkeypatch.setitem(sys.modules,'pyspark.sql',fake_sql)
    def filesystem(spark):
        assert events == ['spark-start']
        events.append('filesystem-constructed')
        return object()  # no filesystem methods may be called before preflight passes
    monkeypatch.setattr(module,'JvmHdfs',filesystem)
    monkeypatch.setattr(sys,'argv',['prepare','--source','hdfs://test/source','--out','hdfs://test/out',
                                   '--worker-python','unused','--preflight-only'])
    with pytest.raises(RuntimeError,match='simulated worker dependency failure'):
        module.main()
    assert events == ['spark-start','filesystem-constructed','executor-preflight','spark-stop']


def test_driver_preflight_success_does_not_convert_or_write(monkeypatch,capsys):
    import types
    import tools.prepare_tensors_yarn as module
    calls = []
    class RDD:
        def map(self,fn):
            assert fn is module.dispatch_preflight
            calls.append('preflight')
            return self
        def collect(self): return [{'host':'executor','source_files':1}]
    context = types.SimpleNamespace(applicationId='test',parallelize=lambda items,n:RDD())
    spark = types.SimpleNamespace(sparkContext=context,stop=lambda:calls.append('stop'))
    class Builder:
        def appName(self,name): return self
        def getOrCreate(self): return spark
    fake_sql = types.ModuleType('pyspark.sql')
    fake_sql.SparkSession = types.SimpleNamespace(builder=Builder())
    monkeypatch.setitem(sys.modules,'pyspark.sql',fake_sql)
    monkeypatch.setattr(module,'JvmHdfs',lambda spark:object())
    monkeypatch.setattr(sys,'argv',['prepare','--source','hdfs://test/source','--out','hdfs://test/out',
                                   '--worker-python','unused','--preflight-only'])
    module.main()
    assert calls == ['preflight','stop']
    assert 'TENSOR_PREFLIGHT_PASSED' in capsys.readouterr().out


def test_live_jvm_filesystem_and_executor_transport(tmp_path,monkeypatch):
    """Opt-in actual JVM integration, using only local files and no cluster access."""
    import os
    if os.environ.get('RUN_SPARK_INTEGRATION') != '1':
        pytest.skip('Set RUN_SPARK_INTEGRATION=1 for actual Spark/JVM filesystem check')
    from pyspark.sql import SparkSession
    from tools.prepare_tensors_yarn import JvmHdfs,Hdfs
    spark = (SparkSession.builder.master('local[1]').appName('tensor-filesystem-test')
             .config('spark.ui.enabled','false').config('spark.driver.host','127.0.0.1').getOrCreate())
    try:
        fs = JvmHdfs(spark)
        staging = (tmp_path/'staging').as_uri()
        final = (tmp_path/'final').as_uri()
        fs.mkdir(staging)
        lock = (tmp_path/'lock').as_uri()
        fs.acquire_lock(lock)
        with pytest.raises(Exception): fs.acquire_lock(lock)
        fs.write_json(staging+'/test.json',{'message':'张量'})
        assert json.loads((tmp_path/'staging/test.json').read_text()) == {'message':'张量'}
        fs.rename(staging,final)
        assert fs.exists(final) and not fs.exists(staging)
        # FileContext Rename.NONE must reject an existing destination, not nest it.
        fs.mkdir(staging)
        with pytest.raises(Exception): fs.rename(staging,final)
        fs.unlink(lock)
        pq.write_table(pa.table({'value':[1]}),tmp_path/'final/input.parquet')
        conf = tmp_path/'hadoop-conf'
        conf.mkdir()
        for name in ('core-site.xml','hdfs-site.xml'):
            (conf/name).write_text('<configuration/>')
        monkeypatch.setenv('HADOOP_CONF_DIR',str(conf))
        shell = Hdfs()
        assert len(shell.inventory(final)) == 1
        shell.get(final+'/input.parquet',tmp_path/'download.parquet')
        shell.put(tmp_path/'download.parquet',final+'/copy.parquet')
        assert (tmp_path/'download.parquet').read_bytes() == (tmp_path/'final/copy.parquet').read_bytes()
    finally:
        spark.stop()


def test_distributed_code_archive_can_run_the_converter(tmp_path):
    """Exercise the exact generated code archive, not the checkout's import path."""
    import os
    launcher = tmp_path/'spark-submit'
    launcher.write_text('#!'+sys.executable+'\n'+'''
import os,sys,zipfile
if sys.argv[1:] == ['--version']:
    print('version 3.2.4')
else:
    archive_conf = next(x for x in sys.argv if x.startswith('spark.yarn.dist.archives='))
    code = next(x.split('#')[0] for x in archive_conf.split('=',1)[1].split(',') if x.endswith('#tensor_code'))
    modules = sys.argv[sys.argv.index('--py-files')+1]
    assert modules != code
    assert open(modules,'rb').read() == open(code,'rb').read()
    with zipfile.ZipFile(code) as archive:
        archive.extractall(os.environ['EXTRACTED_CODE'])
''')
    launcher.chmod(0o755)
    extracted = tmp_path/'extracted'
    env = dict(os.environ,MODE='preflight',SPARK_SUBMIT=str(launcher),HADOOP_USER_NAME='test',
               TENSOR_ENV_ARCHIVE='hdfs://test/env.tar.gz',EXTRACTED_CODE=str(extracted))
    subprocess.run(['bash',str(ROOT/'scripts/submit_training_tensors_yarn.sh')],env=env,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    # Simulate a serialized __main__ entrypoint in a worker with no checkout or
    # ZIP on sys.path. Both entrypoints must bootstrap from the extracted code.
    probe = '''
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1])
tree = ast.parse((root/'tools/prepare_tensors_yarn.py').read_text())
for name in ('dispatch_preflight', 'dispatch'):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<worker>', 'exec'), namespace)
    try:
        namespace[name]({'code_root': str(root), 'worker_python': '/missing/tensor/python'})
    except RuntimeError as error:
        assert 'Converter environment unavailable on executor' in str(error), str(error)
    else:
        raise AssertionError('Expected missing converter environment')
    sys.path.remove(str(root))
    for key in list(sys.modules):
        if key == 'tools' or key.startswith('tools.'):
            del sys.modules[key]
'''
    subprocess.run([sys.executable,'-I','-c',probe,str(extracted)],cwd=tmp_path,
                   check=True,capture_output=True,text=True)
    fs,job,staging,output = setup(tmp_path/'transport')
    job['code_root'] = str(extracted)
    result = convert_partition(job,fs)
    publish(fs,staging,output,[job],[result],64,20260921)
    assert verify(fs.path(output+'/train'))['total_groups'] == 1
