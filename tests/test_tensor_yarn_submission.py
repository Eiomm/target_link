"""Contract tests for the YARN tensor submission wrapper."""
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_training_tensors_yarn.sh"


def run_submit(**overrides):
    env = dict(os.environ)
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT)], env=env, text=True,
                          capture_output=True, check=False)


def fake_spark(tmp_path):
    launcher = tmp_path / "spark-submit"
    launcher.write_text("#!/usr/bin/env bash\nif [[ \"$1\" == --version ]]; then echo 'version 3.2.4'; fi\nprintf '%s\\n' \"$@\" > \"$SPARK_LOG\"\n")
    launcher.chmod(0o755)
    return launcher


def test_yarn_refuses_without_a_worker_tensor_environment(tmp_path):
    result = run_submit(MODE="yarn", SPARK_SUBMIT=str(tmp_path / "not-used"))
    assert result.returncode == 2
    assert "requires TENSOR_ENV_ARCHIVE (preferred) or an explicit TENSOR_PYTHON" in result.stderr


def test_archive_distributes_converter_and_code_but_keeps_scheduler_python():
    result = run_submit(
        MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit",
        TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor.tgz",
        TENSOR_ENV_PYTHON="./tensor_env/python/bin/python",
    )
    assert result.returncode == 0, result.stderr
    assert "spark.yarn.dist.archives=" in result.stdout
    assert "hdfs://cluster/envs/tensor.tgz#tensor_env" in result.stdout
    assert "tensor_code.zip#tensor_code" in result.stdout
    assert result.stdout.count("spark.pyspark.python=./minipy3/minipy3/bin/python") == 1
    assert result.stdout.count("spark.pyspark.driver.python=./minipy3/minipy3/bin/python") == 1
    assert "--worker-python ./tensor_env/python/bin/python" in result.stdout
    assert "--py-files" in result.stdout and "tensor_modules.zip" in result.stdout
    assert "spark.yarn.appMasterEnv.PYTHONUNBUFFERED=1" in result.stdout
    assert "spark.executorEnv.PYTHONUNBUFFERED=1" in result.stdout


def test_explicit_tensor_python_only_controls_converter():
    result = run_submit(
        MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit",
        TENSOR_PYTHON="/opt/tensor/bin/python",
    )
    assert result.returncode == 0, result.stderr
    assert "spark.pyspark.python=./minipy3/minipy3/bin/python" in result.stdout
    assert "spark.pyspark.driver.python=./minipy3/minipy3/bin/python" in result.stdout
    assert "--worker-python /opt/tensor/bin/python" in result.stdout
    assert "minipy3" in result.stdout


def test_custom_spark_python_overrides_only_the_scheduler_environment():
    result = run_submit(
        MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit",
        TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor.tgz",
        SPARK_PYTHON_ARCHIVE="hdfs://cluster/envs/spark-python.tgz",
        SPARK_PYTHON_REL="./spark_py/bin/python",
    )
    assert result.returncode == 0, result.stderr
    assert "hdfs://cluster/envs/spark-python.tgz#spark_python" in result.stdout
    assert "spark.pyspark.python=./spark_py/bin/python" in result.stdout
    assert "spark.pyspark.driver.python=./spark_py/bin/python" in result.stdout
    assert "--worker-python ./tensor_env/bin/python" in result.stdout


def test_preflight_submits_a_preflight_task_and_check_only_checks_launcher(tmp_path):
    launcher = fake_spark(tmp_path)
    log = tmp_path / "spark.log"
    base = dict(
        SPARK_SUBMIT=str(launcher), SPARK_LOG=str(log), HADOOP_USER_NAME="tester",
        TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor.tgz",
    )
    preflight = run_submit(MODE="preflight", **base)
    assert preflight.returncode == 0, preflight.stderr
    preflight_args = log.read_text().splitlines()
    modules = preflight_args[preflight_args.index('--py-files') + 1]
    archives = next(arg.split('=', 1)[1] for arg in preflight_args
                    if arg.startswith('spark.yarn.dist.archives='))
    assert Path(modules).name == 'tensor_modules.zip'
    assert modules not in [entry.split('#')[0] for entry in archives.split(',')]
    assert "--preflight-only" in preflight_args
    assert preflight_args[preflight_args.index("--num-executors") + 1] == "1"
    log.unlink()
    check = run_submit(MODE="check", **base)
    assert check.returncode == 0, check.stderr
    assert log.read_text().splitlines() == ["--version"]


def test_dry_without_environment_remains_inspectable_but_warns():
    result = run_submit(MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit")
    assert result.returncode == 0, result.stderr
    assert "Dry run warning: no worker tensor environment was supplied" in result.stdout


def test_check_without_environment_warns_without_submitting_a_job(tmp_path):
    launcher = fake_spark(tmp_path)
    result = run_submit(MODE="check", SPARK_SUBMIT=str(launcher), SPARK_LOG=str(tmp_path / "spark.log"))
    assert result.returncode == 0, result.stderr
    assert "no worker tensor environment was supplied" in result.stdout


def test_default_output_name_uses_the_selected_tensor_shape_and_seed():
    result = run_submit(
        MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit",
        M_MAX="31", DATA_SEED="77",
    )
    assert result.returncode == 0, result.stderr
    assert "training_tensors_v1_m31_seed77" in result.stdout
