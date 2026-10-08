#!/usr/bin/env bash
set -euo pipefail
BROWSER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON=/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python
BROWSER_PYTHON="${DATA_BROWSER_PYTHON:-$DEFAULT_PYTHON}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"$BROWSER_PYTHON" -c 'import importlib.util; assert all(importlib.util.find_spec(m) for m in ("numpy", "pyarrow", "torch"))' || {
  echo '当前 Python 缺少训练环境依赖。请将 DATA_BROWSER_PYTHON 指向已有训练环境的 Python。' >&2
  exit 1
}
exec "$BROWSER_PYTHON" "$BROWSER_DIR/server.py" "$@"
