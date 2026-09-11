#!/usr/bin/env bash
# Stage B: final run after Stage A has frozen architecture, LR and epoch count.
# Train on all 128 cell-hash buckets for Monday-Saturday; evaluate Sunday once.
# Do not tune the recipe from this Sunday result.
# Platform command:
#   bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_cell_weekend_final_job.sh
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

export TRAIN_PARTS="$(parts_for_days 20260817 20260818 20260819 20260820 20260821 20260822)"
export VAL_PARTS="$(parts_for_days 20260823)"
export DATA="$REPO/runtime/cell_weekend_final_20260817_23"
# Sunday final must wait for the rebuilt, fully validated HDFS corpus. Pointing
# this at v2/v3 could silently trigger a fresh download of an untrusted copy.
export OBS_DIR=observations_v4
export GROUPS_DIR=training_groups_k3

# These must match the recipe selected in Stage A. Sunday is the held-out test.
export EPOCHS=10
export MAX_BATCHES=-1
export BATCH_SIZE=32
export WORKERS=4
export DEVICE=cuda
export OUT="$REPO/runtime/cell_weekend_final_d256_$(date +%m%d_%H%M)"
export EXTRA="--d-model 256 --heads 8 --traj-layers 4 --level2-layers 4 --dropout 0.1 --lr 2e-4 --groups-per-partition 4096 --probe-batches 20 --val-last-only"

exec bash "$REPO/scripts/submit_cell_train_job.sh"
