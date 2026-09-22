import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "submit_cell_mlp_job.sh"
PYTHON = Path("/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python")


def _tensor_root(root, *, missing=None, m_max=64, seed=20260921):
    root.mkdir(parents=True)
    partitions = {}
    missing = set(missing or ())
    for day in [*(f"202608{d:02d}" for d in range(17, 23)), "20260823"]:
        for bucket in range(128):
            key = f"{day}/{bucket}"
            if key in missing:
                continue
            rel = f"files/{day}-{bucket}.bin"
            path = root / rel
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"x")
            rec = {"path": rel, "bytes": 1}
            partitions[key] = {"arrays": {"meta": rec}, "payload": rec}
    marker = {
        "format": "trajectory_mlp_tensors_v1",
        "m_max": m_max,
        "data_seed": seed,
        "channels": ["piece_time_sum_seconds", "piece_ratio_sum", "zero"],
        "storage": "zlib-group-blocks-v1",
        "partitions": partitions,
    }
    (root / "_TENSORS_SUCCESS.json").write_text(json.dumps(marker))


def _run(tmp_path, data, val, *, dry_run="1"):
    out = tmp_path / "out"
    env = os.environ.copy()
    env.update(
        DATA=str(data),
        VAL_DATA=str(val),
        OUT=str(out),
        DRY_RUN=dry_run,
        PYTHON=str(PYTHON if PYTHON.exists() else Path(os.sys.executable)),
    )
    return subprocess.run(["bash", str(SCRIPT)], cwd=REPO, env=env,
                          text=True, capture_output=True)


def test_tensor_roots_pass_dry_run_without_gpu(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    _tensor_root(train)
    _tensor_root(val)
    result = _run(tmp_path, train, val)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout


def test_tensor_root_requires_all_requested_partitions(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    _tensor_root(train, missing={"20260820/127"})
    _tensor_root(val)
    result = _run(tmp_path, train, val)
    assert result.returncode != 0
    assert "expected 128 partitions" in result.stdout
