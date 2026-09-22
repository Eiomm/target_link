#!/usr/bin/env bash
# Thirty-second bucket time encoding. Run on an allocated GPU node.
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo '本脚本固定使用30秒分桶时间编码，无需附加参数。' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TIME_ENCODING=bucket30
export BATCH_SIZE=512
export EPOCHS=10
exec bash "$SCRIPT_DIR/train_traj_mae_raw.sh" bucket30
