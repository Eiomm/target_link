#!/usr/bin/env bash
# A100 单机 smoke:cell 级 whole-trajectory MAE 的"数据 -> 训练"全链路连通性验证。
# 不追指标,只回答四件事:语料能拉到、契约没破、GPU 看得见、一个 step 能反传。
#
# 平台启动命令(只填这一行,参数全部写死在本脚本里):
#   bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_cell_smoke_job.sh
#   资源: 1×GPU 单机(A100)
#
# 数据量:4 个 train 分片 + 1 个 val 分片 ≈ 4×(83+28) + 111 ≈ 660 MB,首次从 HDFS
# 拉一次,NFS 上留着,重跑同一 $DATA 不会再拉。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 数据:拿一部分,不动全量 -------------------------------------------------
# bucket = xxhash64(cell_id) % 128 的存储分桶，不是 10min window；每个 bucket
# 都混有当天多个 window。train 取 4 个 hash bucket，val 取次日相同 hash bucket。
export TRAIN_PARTS="20260821:64 20260821:65 20260821:66 20260821:67"
export VAL_PARTS="20260822:64"
export DATA="$REPO/runtime/cell_smoke_20260821_64_67"
export OBS_DIR="observations_v2"     # 必须:v1 的 observations 没有 bin_pos
export GROUPS_DIR="training_groups_k3"

# --- 训练:够跑通就行 --------------------------------------------------------
export EPOCHS=2
export MAX_BATCHES=200
export BATCH_SIZE=32
export WORKERS=4
export DEVICE=cuda
export OUT="$REPO/runtime/cell_train_smoke_a100_$(date +%m%d_%H%M)"

exec bash "$REPO/scripts/submit_cell_train_job.sh"
