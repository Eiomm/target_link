#!/usr/bin/env bash
# Main A100 run for the 500m Cell MAE with the per-bin group channel.
# Platform command:
#   bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_cell_main_job.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Six training days and one held-out day, using four representative hash
# buckets per day. K is the full cell K stored in each group, so K-bucketed
# validation remains meaningful even though only selected hash buckets are read.
export TRAIN_PARTS="20260817:64 20260817:65 20260817:66 20260817:67 20260818:64 20260818:65 20260818:66 20260818:67 20260819:64 20260819:65 20260819:66 20260819:67 20260820:64 20260820:65 20260820:66 20260820:67 20260821:64 20260821:65 20260821:66 20260821:67 20260822:64 20260822:65 20260822:66 20260822:67"
export VAL_PARTS="20260823:64 20260823:65 20260823:66 20260823:67"
export DATA="$REPO/runtime/cell_main_k3_b64_67"
export OBS_DIR=observations_v2
export GROUPS_DIR=training_groups_k3

export EPOCHS=20
export MAX_BATCHES=5000
export BATCH_SIZE=32
export WORKERS=4
export DEVICE=cuda
export OUT="$REPO/runtime/cell_main_group_bins_d256_$(date +%m%d_%H%M)"

# Explicitly pin the larger model even if train_cells.py defaults change later.
export EXTRA="--d-model 256 --heads 8 --traj-layers 4 --level2-layers 4 --dropout 0.1 --lr 2e-4"

exec bash "$REPO/scripts/submit_cell_train_job.sh"
