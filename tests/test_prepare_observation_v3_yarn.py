"""Local transport contracts for the observation-v3 YARN adapter."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import pytest

from tools.prepare_observation_v3_yarn import FORMAT, convert_partition, publish, worker_subprocess_env


ROOT = Path(__file__).resolve().parents[1]


def test_converter_uses_packaged_runtime_without_mutating_scheduler_environment(tmp_path, monkeypatch):
    prefix = tmp_path / 'env'
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'lib').mkdir()
    (prefix / 'bin/python').touch()
    (prefix / 'lib/libstdc++.so.6').touch()
    monkeypatch.setenv('LD_LIBRARY_PATH', '/scheduler/lib')
    job = dict(worker_python=str(prefix / 'bin/python'), code_root=str(ROOT))
    env = worker_subprocess_env(job)
    assert env['LD_LIBRARY_PATH'] == str(prefix / 'lib') + ':/scheduler/lib'
    assert env['PYTHONPATH'].split(':')[0] == str(ROOT)
    assert __import__('os').environ['LD_LIBRARY_PATH'] == '/scheduler/lib'


class FakeHdfs:
    def __init__(self, root):
        self.root = root

    def path(self, uri):
        return self.root / urlsplit(uri).path.lstrip("/")

    def exists(self, uri):
        return self.path(uri).exists()

    def mkdir(self, uri, exclusive=False):
        self.path(uri).mkdir(parents=not exclusive, exist_ok=not exclusive)

    def inventory(self, uri):
        return [dict(uri=uri + "/" + path.name, bytes=path.stat().st_size,
                     mtime_ms=path.stat().st_mtime_ns // 1_000_000)
                for path in sorted(self.path(uri).glob("*.parquet"))]

    def get(self, uri, local):
        shutil.copyfile(self.path(uri), local)

    def put(self, local, uri):
        shutil.copyfile(local, self.path(uri))

    def write_json(self, uri, value):
        self.path(uri).write_text(json.dumps(value))

    def unlink(self, uri):
        self.path(uri).unlink()

    def rename(self, source, destination):
        self.path(source).rename(self.path(destination))


def _setup(tmp_path):
    fs = FakeHdfs(tmp_path)
    source, staging, output = "hdfs://test/source", "hdfs://test/staging", "hdfs://test/output"
    source_part = fs.path(source + "/observations_v2/day=20260817/bucket=0")
    source_part.mkdir(parents=True)
    (source_part / "part.parquet").write_bytes(b"source")
    fs.mkdir(staging + "/train")
    fs.mkdir(staging + "/val")
    fs.write_json(staging + "/train/_BUILDING", {})
    fs.write_json(staging + "/val/_BUILDING", {})
    job = dict(split="train", day="20260817", bucket="0", source=source,
               destination=staging + "/train", m_max=4, seed=99,
               code_root=str(ROOT), worker_python=sys.executable, hdfs_bin="unused")
    return fs, job, staging, output


def _fake_local_build(job, source, output):
    """Write exactly the local receipt shape produced by the canonical builder."""
    paths = {
        "observations": output / "observations_v3/day=20260817/bucket=0/part-00000.parquet",
        "index": output / "group_indices/day=20260817/bucket=0/groups.npz",
        "cells": output / "group_indices/day=20260817/bucket=0/cells.npz",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((key + " payload").encode())
    receipt = dict(format=FORMAT, m_max=4, data_seed=99, stored_rows=3,
                   stats=dict(day="20260817", bucket="0", raw_rows=3, raw_cells=1,
                              candidate_rows=3, usable_rows=3, dropped_no_valid=0,
                              dropped_tail=0, groups=1, full_groups=0, tail_groups=1,
                              selected_groups=0),
                   **{key: dict(path=str(path.relative_to(output)), bytes=path.stat().st_size,
                               sha256="test-" + key) for key, path in paths.items()})
    receipt_path = output / "receipts/20260817_0.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))


def test_attempts_are_unique_relative_and_publish_only_after_complete_results(tmp_path, monkeypatch):
    fs, job, staging, output = _setup(tmp_path)
    monkeypatch.setattr("tools.prepare_observation_v3_yarn.worker_environment", lambda job: ROOT)
    monkeypatch.setattr("tools.prepare_observation_v3_yarn._run_local_build", _fake_local_build)
    first = convert_partition(job, fs)
    retry = convert_partition(job, fs)
    assert first[2]["observations"]["path"] != retry[2]["observations"]["path"]
    assert retry[2]["source_inventory"] == fs.inventory(
        "hdfs://test/source/observations_v2/day=20260817/bucket=0")
    for record in (retry[2]["observations"], retry[2]["index"], retry[2]["cells"]):
        assert not record["path"].startswith("/")
        assert record["path"].startswith("_parts/day=20260817/bucket=0/attempt=")
        assert fs.exists(job["destination"] + "/" + record["path"])

    publish(fs, staging, output, [job], [retry], 4, 99)
    marker = json.loads(fs.path(output + "/train/_OBSERVATION_V3_SUCCESS.json").read_text())
    assert marker["format"] == FORMAT and marker["total_groups"] == 1
    assert not fs.exists(staging)


def test_source_mutation_duplicates_and_existing_output_cannot_publish(tmp_path, monkeypatch):
    fs, job, staging, output = _setup(tmp_path)
    monkeypatch.setattr("tools.prepare_observation_v3_yarn.worker_environment", lambda job: ROOT)
    monkeypatch.setattr("tools.prepare_observation_v3_yarn._run_local_build", _fake_local_build)
    original_inventory = fs.inventory
    calls = []

    def changing_inventory(uri):
        result = original_inventory(uri)
        calls.append(uri)
        if len(calls) > 1:
            result[0] = dict(result[0], mtime_ms=result[0]["mtime_ms"] + 1)
        return result

    monkeypatch.setattr(fs, "inventory", changing_inventory)
    with pytest.raises(ValueError, match="source changed"):
        convert_partition(job, fs)
    assert not fs.exists(staging + "/train/_parts")

    # A duplicate result cannot advance publication, even if its files exist.
    monkeypatch.setattr(fs, "inventory", original_inventory)
    result = convert_partition(job, fs)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        publish(fs, staging, output, [job], [result, result], 4, 99)
    assert not fs.exists(output)

    fs.mkdir(output)
    sentinel = fs.path(output + "/keep.txt")
    sentinel.write_text("do not overwrite")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish(fs, staging, output, [job], [result], 4, 99)
    assert sentinel.read_text() == "do not overwrite"


def test_submission_dry_run_packages_v3_worker_and_uses_yarn_cluster(tmp_path):
    environment = dict(__import__("os").environ, MODE="dry", SPARK_SUBMIT="/not-executed/spark-submit",
                       SCHEDULER_PYTHON="./scheduler_env/minipy3/bin/python",
                       V3_PYTHON="./v3_env/bin/python",
                       HADOOP_USER_PASSWORD="test-password-must-not-be-printed")
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_observation_v3_yarn.sh")],
                            env=environment, text=True, capture_output=True, check=True)
    assert "--master yarn --deploy-mode cluster" in result.stdout
    assert "prepare_observation_v3_yarn.py" in (ROOT / "scripts/submit_observation_v3_yarn.sh").read_text()
    assert "prepare_observation_v3.py" in (ROOT / "scripts/submit_observation_v3_yarn.sh").read_text()
    assert "#observation_v3_code" in result.stdout
    assert "spark.pyspark.python=./scheduler_env/minipy3/bin/python" in result.stdout
    assert "--worker-python ./v3_env/bin/python" in result.stdout
    assert "test-password-must-not-be-printed" not in result.stdout + result.stderr
    assert "Dry run: no submission or conversion." in result.stdout
