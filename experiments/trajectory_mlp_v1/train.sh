#!/usr/bin/env bash
# Run on an allocated GPU node; this does not submit a YARN job.
# Usage: bash train.sh [seconds|bucket30]
# Check data only: DRY_RUN=1 bash train.sh seconds
# Overrides: DATA, VAL_DATA, OUT, PYTHON, ENV_ROOT, BATCH_SIZE, WORKERS,
#            EPOCHS, SEED, M_MAX, CUDA_VISIBLE_DEVICES.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 1 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash $0 [seconds|bucket30]"
  echo "Default: seconds, batch=512, workers=0, epochs=10, seed=42, M=64."
  echo "DRY_RUN=1 checks data paths only; no training or GPU probe."
  echo "Run on an allocated GPU node. This script does not submit to YARN."
  [[ $# -le 1 ]] && exit 0
  exit 2
fi
export TIME_ENCODING="${1:-${TIME_ENCODING:-seconds}}"
case "$TIME_ENCODING" in
  seconds|bucket30) ;;
  *) echo 'TIME_ENCODING must be seconds or bucket30' >&2; exit 2 ;;
esac

# Explicitly select the raw corpus, even if runtime/final is later published.
export DATA="${DATA:-$REPO/runtime/cell_mlp_train}"
export VAL_DATA="${VAL_DATA:-$REPO/runtime/cell_mlp_validation_20260823}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export WORKERS="${WORKERS:-0}"
export EPOCHS="${EPOCHS:-10}"
export SEED="${SEED:-42}"
export M_MAX="${M_MAX:-64}"
export OUT="${OUT:-$REPO/runtime/traj_mae_v4_raw_${TIME_ENCODING}_seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$}"
for name in WORKERS SEED; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "$name must be a nonnegative integer" >&2; exit 2; }
done
for name in EPOCHS M_MAX; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
(( M_MAX >= 3 )) || { echo 'M_MAX must be >=3' >&2; exit 2; }

echo "当前模型：MAE v4；时间编码=$TIME_ENCODING；已知ratio；3维输入；参数低于1000万"
echo "原始数据模式：每轮会重新排序和构建固定分组，请留意加载时间。"
echo "结构：4层encoder、2层decoder、256维、8个注意力头。"
echo "输出目录：$OUT"
exec bash "$REPO/scripts/submit_cell_mlp_job.sh"
