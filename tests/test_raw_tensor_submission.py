"""Contract tests for the direct raw-to-tensor YARN submission wrapper."""
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_raw_training_tensors_yarn.sh"


def run_submit(**overrides):
    env = dict(os.environ)
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT)], env=env, text=True,
                          capture_output=True, check=False)


def fake_launcher(tmp_path):
    launcher = tmp_path / "spark-submit"
    launcher.write_text("#!/usr/bin/env bash\nif [[ \"$1\" == --version ]]; then echo 'version 3.5.1'; exit 0; fi\nprintf '%s\\n' \"$@\" > \"$SPARK_LOG\"\n")
    launcher.chmod(0o755)
    return launcher


def fake_hdfs(tmp_path):
    hdfs = tmp_path / "hdfs"
    hdfs.write_text("#!/usr/bin/env bash\nexit 1\n")
    hdfs.chmod(0o755)
    return hdfs


def test_dry_run_needs_no_credentials_or_tensor_environment():
    result = run_submit(MODE="dry", HADOOP_USER_NAME="", SPARK_SUBMIT="/not/run")
    assert result.returncode == 0, result.stderr
    assert "--master yarn --deploy-mode cluster" in result.stdout
    assert "tools/build_raw_training_tensors.py" in result.stdout
    assert "--inputs" in result.stdout
    assert "--shuffle-partitions 1024" in result.stdout
    assert "--num-executors 20" in result.stdout
    assert "spark.speculation=false" in result.stdout
    assert "no HDFS checks, YARN submission, or data conversion" in result.stdout


def test_dry_command_uses_one_tensor_environment_for_driver_and_workers():
    result = run_submit(
        MODE="dry", SPARK_SUBMIT="/not/run",
        TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor-python.tgz",
        TENSOR_ENV_PYTHON="./tensor_env/python/bin/python",
    )
    assert result.returncode == 0, result.stderr
    assert "hdfs://cluster/envs/tensor-python.tgz#tensor_env" in result.stdout
    assert "spark.pyspark.python=./tensor_env/python/bin/python" in result.stdout
    assert "spark.pyspark.driver.python=./tensor_env/python/bin/python" in result.stdout
    assert "minipy3" not in result.stdout


def test_day_partitions_are_defaulted_and_must_not_overlap():
    result = run_submit(MODE="dry", SPARK_SUBMIT="/not/run")
    assert result.returncode == 0, result.stderr
    for day in range(17, 24):
        assert f"event_hour=202608{day}\\*" in result.stdout
    overlap = run_submit(MODE="dry", SPARK_SUBMIT="/not/run",
                         TRAIN_DAYS="20260817", VAL_DAYS="20260817")
    assert overlap.returncode == 2
    assert "overlap" in overlap.stderr

    selected = run_submit(MODE="dry", SPARK_SUBMIT="/not/run",
                          TRAIN_DAYS="20260819", VAL_DAYS="20260823")
    assert selected.returncode == 0, selected.stderr
    assert "event_hour=20260819\\*" in selected.stdout
    assert "event_hour=20260823\\*" in selected.stdout
    assert "event_hour=20260817\\*" not in selected.stdout


def test_cluster_sizes_are_bounded():
    too_small = run_submit(MODE="dry", SPARK_SUBMIT="/not/run", M_MAX="2")
    assert too_small.returncode == 2
    assert "at least 3" in too_small.stderr
    too_many_buckets = run_submit(MODE="dry", SPARK_SUBMIT="/not/run", BUCKETS="129")
    assert too_many_buckets.returncode == 2
    assert "at most 128" in too_many_buckets.stderr


def test_yarn_requires_tensor_archive_and_refuses_existing_destination(tmp_path):
    launcher, hdfs = fake_launcher(tmp_path), fake_hdfs(tmp_path)
    no_env = run_submit(MODE="yarn", HADOOP_USER_NAME="tester",
                        SPARK_SUBMIT=str(launcher), HDFS_BIN=str(hdfs))
    assert no_env.returncode == 2
    assert "requires TENSOR_ENV_ARCHIVE" in no_env.stderr

    existing_hdfs = tmp_path / "existing-hdfs"
    existing_hdfs.write_text("#!/usr/bin/env bash\nexit 0\n")
    existing_hdfs.chmod(0o755)
    blocked = run_submit(MODE="yarn", HADOOP_USER_NAME="tester",
                         SPARK_SUBMIT=str(launcher), HDFS_BIN=str(existing_hdfs),
                         TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor.tgz")
    assert blocked.returncode == 2
    assert "refusing overwrite" in blocked.stderr


def test_yarn_submits_only_after_fresh_destination_check(tmp_path):
    launcher, hdfs = fake_launcher(tmp_path), fake_hdfs(tmp_path)
    log = tmp_path / "spark.log"
    result = run_submit(MODE="yarn", HADOOP_USER_NAME="tester", SPARK_LOG=str(log),
                        SPARK_SUBMIT=str(launcher), HDFS_BIN=str(hdfs),
                        TENSOR_ENV_ARCHIVE="hdfs://cluster/envs/tensor.tgz")
    assert result.returncode == 0, result.stderr
    args = log.read_text().splitlines()
    assert args[args.index("--py-files") + 1].endswith("raw_tensor_modules.zip")
    assert "--train-days" in args and "--val-days" in args
    assert args[args.index("--m-max") + 1] == "64"
