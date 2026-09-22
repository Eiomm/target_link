#!/usr/bin/env bash
# Cluster platform entrypoint: bash scripts/submit_cell_mlp_job.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
PYTHON="${PYTHON:-$ENV_ROOT/bin/python}"
if [[ -f "$REPO/runtime/final/_SUCCESS.json" ]]; then
  DEFAULT_DATA="$REPO/runtime/final/train"
  DEFAULT_VAL_DATA="$REPO/runtime/final/val"
else
  DEFAULT_DATA="$REPO/runtime/cell_mlp_train"
  DEFAULT_VAL_DATA="$REPO/runtime/cell_mlp_validation_20260823"
fi
DATA_SEED="${DATA_SEED:-20260921}"
OUT="${OUT:-$REPO/runtime/cell_mlp_l20_$(date +%Y%m%d_%H%M%S)_$$}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
M_MAX="${M_MAX:-64}"
TENSOR_DEFAULT="$REPO/runtime/training_tensors_v1_m64_seed20260921"
if [[ "$M_MAX" == 64 && "$DATA_SEED" == 20260921 \
      && "${DATA+x}" != x && "${VAL_DATA+x}" != x \
      && -f "$TENSOR_DEFAULT/train/_TENSORS_SUCCESS.json" \
      && -f "$TENSOR_DEFAULT/val/_TENSORS_SUCCESS.json" \
      && ! -e "$TENSOR_DEFAULT/train/_TENSORS_BUILDING" \
      && ! -e "$TENSOR_DEFAULT/val/_TENSORS_BUILDING" ]]; then
  DATA="$TENSOR_DEFAULT/train"
  VAL_DATA="$TENSOR_DEFAULT/val"
fi
DATA="${DATA:-$DEFAULT_DATA}"
VAL_DATA="${VAL_DATA:-$DEFAULT_VAL_DATA}"
TIME_ENCODING="${TIME_ENCODING:-seconds}"
[[ "$TIME_ENCODING" == seconds || "$TIME_ENCODING" == bucket30 ]] || { echo 'Invalid TIME_ENCODING'; exit 2; }
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN must be 0 or 1'; exit 2; }
[[ "$BATCH_SIZE" == auto || "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid BATCH_SIZE'; exit 2; }
cd "$REPO"
# Reserve one run directory; never append into another experiment.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
heartbeat_pid=''
finish() {
  rc=$?
  trap - EXIT
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; fi
  echo "[$(date -Is)] 运行结束，退出码=$rc（0表示成功），输出目录=$OUT"
  printf '%s\n' "$rc" > "$OUT/exit_code.txt"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "====================【启动整轨迹 MLP 实验】===================="
echo "时间编码：$TIME_ENCODING；训练：8月17–22日；验证：8月23日"
echo "每批 $BATCH_SIZE 个group；每组最多 $M_MAX 条轨迹；共 $EPOCHS 轮；读取进程 $WORKERS；随机种子 $SEED"
echo "训练数据：$DATA"
echo "验证数据：$VAL_DATA"
echo "数据种子：$DATA_SEED"
echo "输出目录：$OUT（console.log日志；artifacts内保存模型与趋势图）"
echo "先计算验证集均值基线，再逐轮训练和验证。原始数据首个分区加载可能较慢。"
"$PYTHON" - "$DATA" "$VAL_DATA" "$M_MAX" "$DATA_SEED" "$DRY_RUN" <<'PY'
import json
import sys
from pathlib import Path
from experiments.trajectory_mlp_v1.run import file_manifest

train_days = [f'202608{d:02d}' for d in range(17, 23)]
val_days = ['20260823']
m_max, data_seed = int(sys.argv[3]), int(sys.argv[4])
for root, days, label in [(sys.argv[1], train_days, '训练'),
                          (sys.argv[2], val_days, '验证')]:
    root_path = Path(root)
    marker = root_path / '_TENSORS_SUCCESS.json'
    if (root_path / '_TENSORS_BUILDING').exists():
        raise SystemExit(f'Tensor data not published: {root_path}')
    if marker.exists():
        info = json.loads(marker.read_text())
        if (info.get('m_max'), info.get('data_seed')) != (m_max, data_seed):
            raise SystemExit(
                f'Tensor corpus protocol mismatch at {root_path}: '
                f'expected m_max={m_max}, data_seed={data_seed}')
    try:
        manifest = file_manifest([str(root_path)], days)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f'{label} data check failed: {exc}') from exc
    by_day = {day: set() for day in days}
    for entry in manifest['files']:
        by_day.setdefault(entry['day'], set()).add(entry['bucket'])
    expected = {f'bucket={i}' for i in range(128)}
    for day in days:
        if by_day.get(day) != expected:
            raise SystemExit(
                f'Incomplete partition set: {root_path} day={day}; '
                f'expected 128 partitions, found {len(by_day.get(day, set()))}')
    kind = 'tensor artifacts' if marker.exists() else 'Parquet files'
    print(f'[数据目录检查] {label}：{len(days)}天、每天128个分区的{kind}均已通过', flush=True)
if sys.argv[5] != '1':
    import torch
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable: run on GPU node')
    print('[训练GPU]',torch.cuda.get_device_name(0),flush=True)
PY
if [[ "$DRY_RUN" == 1 ]]; then
  echo '[DRY RUN] Paths checked; no GPU probe or training launched.'
  exit 0
fi
(
  while sleep 60; do
    echo "[$(date -Is)] 进程仍在运行；GPU状态：型号、利用率、已用显存、总显存"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  done
) &
heartbeat_pid=$!
if [[ "$BATCH_SIZE" == auto ]]; then
  echo "====================【探测 GPU 安全 batch size】===================="
  "$PYTHON" -u experiments/trajectory_mlp_v1/tools/probe_batch.py --out "$OUT/batch_probe.json" --m-max "$M_MAX" --time-encoding "$TIME_ENCODING"
  BATCH_SIZE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "$OUT/batch_probe.json")"
fi
echo "====================【进入基线与训练流程】===================="
echo "本次batch=$BATCH_SIZE（只有auto模式执行显存探测）"
cmd=("$PYTHON" -u experiments/trajectory_mlp_v1/run.py train
  --data "$DATA" --val-data "$VAL_DATA" --out "$OUT/artifacts"
  --device cuda --batch-size "$BATCH_SIZE" --workers "$WORKERS"
  --epochs "$EPOCHS" --seed "$SEED" --m-max "$M_MAX" --data-seed "$DATA_SEED" --time-encoding "$TIME_ENCODING" --log-every 20)
printf '%q ' "${cmd[@]}" > "$OUT/command.sh"
printf '\n' >> "$OUT/command.sh"
"${cmd[@]}"
