#!/usr/bin/env bash
# 平台直接选择本脚本，无需填写命令行参数。
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo '本脚本无需附加参数；训练配置直接修改下方设置。' >&2
  exit 2
fi

# 训练配置：需要调整时直接修改这里。
export TIME_ENCODING=bucket30
export BATCH_SIZE=512
export EPOCHS=10
export SEED=42
export WORKERS=0
export M_MAX=64
# 仅检查数据时，将 0 改为 1；检查后改回 0 才会训练。
export DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train.sh"
