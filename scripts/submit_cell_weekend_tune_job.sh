#!/usr/bin/env bash
# Stage A: tune/fix the recipe without looking at Sunday.
# Train on all 128 cell-hash buckets for Monday-Friday; validate on Saturday.
# Platform command:
#   bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_cell_weekend_tune_job.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

parts_for_days() {
  local result="" day bucket
  for day in "$@"; do
    for ((bucket=0; bucket<128; bucket++)); do
      result+="${day}:${bucket} "
    done
  done
  printf '%s' "${result% }"
}

export TRAIN_PARTS="$(parts_for_days 20260817 20260818 20260819 20260820 20260821)"
export VAL_PARTS="$(parts_for_days 20260822)"
export DATA="$REPO/runtime/cell_weekend_tune_20260817_22"
export OBS_DIR=observations_v3
export GROUPS_DIR=training_groups_k3

# Sample every day/hash-bucket instead of stopping after a partition prefix:
# 5 days x 128 buckets x 4096 groups = at most 2.62M groups/epoch.
export EPOCHS=10
export MAX_BATCHES=-1
export BATCH_SIZE=32
export WORKERS=4
export DEVICE=cuda
export OUT="$REPO/runtime/cell_weekend_tune_d256_$(date +%m%d_%H%M)"
export EXTRA="--d-model 256 --heads 8 --traj-layers 4 --level2-layers 4 --dropout 0.1 --lr 2e-4 --groups-per-partition 4096 --probe-batches 20"

exec bash "$REPO/scripts/submit_cell_train_job.sh"
