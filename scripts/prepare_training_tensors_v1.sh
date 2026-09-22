#!/usr/bin/env bash
# Prepare validation then training, preserving all source observations.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_SEED="${DATA_SEED:-20260921}"
TENSOR_OUT="${TENSOR_OUT:-runtime/training_tensors_v1_m64_seed${DATA_SEED}}"
BUILD_WORKERS="${BUILD_WORKERS:-2}"
mkdir -p "$TENSOR_OUT"
"$PYTHON_BIN" -u experiments/trajectory_mlp_v1/tools/prepare_tensors.py \
  --source runtime/cell_mlp_validation_20260823 --out "$TENSOR_OUT/val" \
  --days 20260823 --m-max 64 --seed "$DATA_SEED" --workers "$BUILD_WORKERS" \
  2>&1 | tee -a "$TENSOR_OUT/val.log"
"$PYTHON_BIN" -u experiments/trajectory_mlp_v1/tools/prepare_tensors.py \
  --source runtime/cell_mlp_train --out "$TENSOR_OUT/train" \
  --days 20260817 20260818 20260819 20260820 20260821 20260822 \
  --m-max 64 --seed "$DATA_SEED" --workers "$BUILD_WORKERS" \
  2>&1 | tee -a "$TENSOR_OUT/train.log"
