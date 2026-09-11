# 全量 Cell 语料本地规范化

## 目的

HDFS `observations_v2` 的行集与字段可用，但 parquet 物理行序不能作为训练契约。
当前 `CellCorpusDataset` 用 `searchsorted(cell_id)` 定位一个 cell 的连续 observation，
因此全量训练前需要一次性生成本地有序副本，不能在每个 epoch 内重复排序。

本流程不修改 HDFS 数据。它逐个下载 `(day, bucket)`，按
`(cell_id, sample_id)` 对完整 observation 行排序，精确核对
`training_groups_k3` 的 K 和全部 sample membership，然后原子发布到 NFS。

## 全量准备

在服务器仓库根目录运行：

```bash
bash scripts/prepare_full_cell_corpus_local.sh
```

默认划分：

- train：20260817–20260822，每天 128 buckets；
- val：20260823，128 buckets；
- 输出：`runtime/cell_trainready_v2_20260817_23`；
- 输入：HDFS `corpus_v1/observations_v2` 与 `training_groups_k3`。

每个完成的 bucket 都有 `_TRAINREADY.json`，根目录全部 896 个分区完成后才会写
`_TRAINREADY_SUCCESS`。脚本可以断点续跑，已有成功标记的 bucket 会跳过；发现无标记的
不完整目标时会停止，绝不覆盖。

建议先用两个分区（一天 train、一天 val、各一个 bucket）验证服务器环境：

```bash
TRAIN_DAYS=20260821 VAL_DAYS=20260822 \
BUCKET_START=64 BUCKET_END=64 \
OUT="$PWD/runtime/cell_trainready_v2_smoke" \
bash scripts/prepare_full_cell_corpus_local.sh
```

## 使用规范化语料训练

完成后让现有训练入口禁止回读 HDFS：

```bash
DATA="$PWD/runtime/cell_trainready_v2_20260817_23" \
OBS_DIR=observations_v2 \
GROUPS_DIR=training_groups_k3 \
FETCH=0 FORCE_FETCH=0 \
bash scripts/submit_cell_train_job.sh
```

首次可增加 `DRY_RUN=1`，只执行现有训练 preflight，不启动训练：

```bash
DATA="$PWD/runtime/cell_trainready_v2_20260817_23" \
OBS_DIR=observations_v2 GROUPS_DIR=training_groups_k3 \
FETCH=0 DRY_RUN=1 \
bash scripts/submit_cell_train_job.sh
```

## 磁盘与失败语义

流程同时保存有序 observations 和原始 groups，本地最终空间应按 HDFS 逻辑大小而不是
三副本大小估算。任一 bucket 的下载、排序、校验或写回失败，都不会产生成功标记；已完成
bucket 不受影响。临时数据位于输出目录的 `.download_tmp/`。
