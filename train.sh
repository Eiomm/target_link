#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
DATA="${DATA:-$REPO/data/cell_mlp_train}"
VAL_DATA="${VAL_DATA:-$REPO/data/cell_mlp_validation_20260823}"
OUT="${OUT:-$REPO/runtime/$(date +%F)/traj_mae_v6_$(date +%Y%m%d_%H%M%S)_$$}"
cd "$REPO"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code.txt"' EXIT
export PYTHONUNBUFFERED=1
cmd=("$PYTHON" -u -m trajectory_mae.run train --config "$REPO/train.toml"
  --data "$DATA" --val-data "$VAL_DATA" --out "$OUT/artifacts")
for pair in BATCH_SIZE:batch-size EPOCHS:epochs WORKERS:workers SEED:seed M_MAX:m-max TIME_ENCODING:time-encoding; do
  name="${pair%%:*}"
  if [[ -v "$name" ]]; then cmd+=("--${pair#*:}" "${!name}"); fi
done
case "${DRY_RUN:-0}" in
  0) ;;
  1) cmd+=(--dry-run) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;;
esac
cmd+=("$@")
printf '%q ' "${cmd[@]}" > "$OUT/command.sh"
printf '\n' >> "$OUT/command.sh"
echo "训练数据：$DATA；验证数据：$VAL_DATA；结果：$OUT"
"${cmd[@]}"
