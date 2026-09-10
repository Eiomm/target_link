#!/usr/bin/env bash
# A100 平台连通性验证(不依赖 build 出料):用仓库里已有的 smoke 语料跑 4 个 batch。
#
# 平台启动命令:
#   bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/legacy/scripts/submit_train_smoke_job.sh
#   资源:1×GPU 单机(A100)
#
# 验证的是「平台这条路」而不是模型效果:镜像里有没有 hadoop 客户端/conda 环境、
# GPU 是否可见、torch/pyarrow 能不能 import、WindowDataset 能不能从 NFS 流式读、
# 训练循环能不能在 cuda 上跑起来。这些如果留到真语料落盘后再发现,要白等一小时。
#
# 语料 = runtime/windows_smoke_20260909_083001/corpus(§9 集群回归用的那一份):
#   anchors [600,1440) stride 60 W 600;train 2 anchors [600,720),val 2 anchors [1320,1440)
#   —— val_start(1320) = train_end(720) + W,正好卡着 legacy/tools/train_windows.py 的下界。
# 因为语料在本地,FETCH=0:完全不碰 HDFS。
#
# 真语料训练走 legacy/scripts/submit_train_windows_job.sh,不要改这个脚本。
set -euo pipefail

# legacy/scripts/ -> repo root; implementation is also archived (legacy/README.md)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # sibling entrypoints moved here too

export DATA="$REPO/runtime/windows_smoke_20260909_083001/corpus"
export FETCH=0
export ANCHOR_START=600
export ANCHOR_END=1440
export TRAIN_END=720
export VAL_START=1320
export VAL_END=1440
export EPOCHS=1
export MAX_BATCHES=4
export BATCH_SIZE=8
export OUT="$REPO/runtime/windows_train_smoke_a100_$(date +%m%d_%H%M)"

exec bash "$SCRIPTS/submit_train_windows_job.sh"
