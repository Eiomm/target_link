#!/usr/bin/env bash
# Luban k8s-job entrypoint for the v1 window MAE (GPU). Pattern: submit_pretrain_job.sh.
#
# Platform form (Web 训练任务 -> 创建单机任务 / luban-client):
#   启动命令: bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_train_windows_job.sh
#   资源:     1×GPU 单机 (选 A100)。模型很小(d_model=128/2 层),显存不是瓶颈,
#             A100 的价值在吞吐和 fp32/tf32 的稳定;batch_size 可以往上调。
#   镜像:     本实验环境快照(需含 hadoop 客户端 — 首次要把语料从 HDFS 拉到 NFS)
#   日志:     stdout 不重定向,平台日志页实时可见;metrics/ckpt 落 NFS 的 $OUT
#
# 语料由 build_windows_spark 在 YARN 上产出、存在 HDFS;而 WindowDataset 只读本地路径,
# 所以本脚本会先 `hdfs dfs -get` 到 NFS(已存在则跳过,除非 FORCE_FETCH=1)。
#
# 单日 split(北京 2026-08-21,W=S=600 不重叠):
#   train 96 anchors [1787241600, 1787299200)
#   val   48 anchors [1787299800, 1787328000)   # val_start = train_end + W
#
# Overridable: DAY / DATA / OUT / EPOCHS / MAX_BATCHES / BATCH_SIZE / WORKERS /
#   DEVICE / CONFIG / FETCH / FORCE_FETCH / DRY_RUN / EXTRA
set -euo pipefail

REPO="/nfs/dataset-ofs-494-1/project/user/junao/target_link"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
DAY="${DAY:-20260821}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
HDFS_CORPUS="${HDFS_CORPUS:-$HDFS_BASE/windows/day${DAY}_w600_s600}"
DATA="${DATA:-$REPO/data/windows_day${DAY}}"
OUT="${OUT:-$REPO/runtime/windows_train_day${DAY}_$(date +%m%d_%H%M)}"
CONFIG="${CONFIG:-$REPO/configs/pretrain_windows.yaml}"
FETCH="${FETCH:-1}"            # 0 = 完全不碰 HDFS,语料必须已在 $DATA
FORCE_FETCH="${FORCE_FETCH:-0}"  # 1 = 即使本地有也重拉
DRY_RUN="${DRY_RUN:-0}"

ANCHOR_START="${ANCHOR_START:-1787241600}"
ANCHOR_END="${ANCHOR_END:-1787328000}"
TRAIN_END="${TRAIN_END:-1787299200}"
VAL_START="${VAL_START:-1787299800}"
VAL_END="${VAL_END:-1787328000}"

EPOCHS="${EPOCHS:-2}"
# 一天全量 ~6e7 个 snapshot,跑满一个 epoch 不现实;首次看效果用 max-batches 截断
MAX_BATCHES="${MAX_BATCHES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"

export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

cd "$REPO"
set +u; source "$ENV_ROOT/bin/activate"; set -u
PYTHON="$(command -v python)"

echo "repo=$REPO python=$PYTHON"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU VISIBLE')"
echo "corpus=$DATA out=$OUT"
echo "split: train[$TRAIN_END) val[$VAL_START,$VAL_END) epochs=$EPOCHS max_batches=$MAX_BATCHES batch=$BATCH_SIZE workers=$WORKERS device=$DEVICE"

# --- 语料:本地缺就一次性从 HDFS 拉(WindowDataset 不读 hdfs://) --------------
# Only the tables training reads: window_stream.py:30 globs window_curves, and
# the meta is mandatory. window_members is 150 GB and never read here -- pulling
# the whole corpus directory would triple the transfer for nothing.
NEED_TABLES="${NEED_TABLES:-window_curves snapshots window_meta.json.d}"
mkdir -p "$DATA"
# Fetch each table into a temp path, then move it into place and drop a marker.
# The marker alone would not be enough: `hdfs dfs -get` into an existing
# destination merges into it, so a retry after an interrupted copy could leave a
# short table behind a marker from an earlier attempt. Staging + mv means the
# final path only ever holds a complete table, and a killed job leaves neither.
# (`hdfs dfs -get` has no -f; only -put does.)
if [[ "$FETCH" == "1" ]]; then
  for t in $NEED_TABLES; do
    marker="$DATA/.fetched_$t"
    if [[ "$FORCE_FETCH" == "1" || ! -e "$marker" ]]; then
      echo "fetching $HDFS_CORPUS/$t -> $DATA/ ..."
      rm -rf "$DATA/.tmp_$t"
      if hdfs dfs -get "$HDFS_CORPUS/$t" "$DATA/.tmp_$t"; then
        rm -rf "${DATA:?}/$t"
        mv "$DATA/.tmp_$t" "$DATA/$t"
        touch "$marker"
      fi
    fi
  done
  missing=""
  for t in $NEED_TABLES; do
    [[ -e "$DATA/.fetched_$t" ]] || missing="$missing $t"
  done
else
  # FETCH=0: the caller supplies the corpus (e.g. a local smoke corpus with no
  # fetch markers); require the tables to be present, not to carry markers.
  missing=""
  for t in $NEED_TABLES; do
    [[ -e "$DATA/$t" ]] || missing="$missing $t"
  done
fi
[[ -z "$missing" ]] || { echo "corpus incomplete under $DATA (missing:$missing); FETCH=$FETCH" >&2; exit 1; }
echo "corpus size: $(du -sh "$DATA" | cut -f1)  tables:$NEED_TABLES"

# --- preflight:语料契约 + 依赖 + split 边界 ---------------------------------
"$PYTHON" - "$DATA" "$CONFIG" "$ANCHOR_START" "$ANCHOR_END" "$TRAIN_END" "$VAL_START" "$VAL_END" <<'PY'
import glob, importlib, json, os, sys
data, config, a0, a1, train_end, val_start, val_end = sys.argv[1:8]
a0, a1, train_end, val_start, val_end = map(int, (a0, a1, train_end, val_start, val_end))
for m in ("numpy", "torch", "yaml", "pyarrow"):
    mod = importlib.import_module(m)
    print("import_ok %s %s" % (m, getattr(mod, "__version__", "?")))
hits = glob.glob(os.path.join(data, "window_meta.json.d/part-*.txt"))
if len(hits) != 1:
    sys.exit("require exactly one corpus meta file under %s (got %d)" % (data, len(hits)))
meta = json.load(open(hits[0]))
print("corpus format=%s anchors=[%s,%s) W=%s S=%s unit=%s max_bins=%s"
      % (meta.get("format"), meta.get("anchor_start"), meta.get("anchor_end"),
         meta.get("lookback_seconds"), meta.get("stride_seconds"),
         meta.get("snapshot_unit"), meta.get("max_bins")))
if meta.get("format") != "target_link_windows_v2":
    sys.exit("unexpected corpus format %r (need target_link_windows_v2)" % meta.get("format"))
if meta.get("anchor_start") != a0 or meta.get("anchor_end") != a1:
    sys.exit("corpus anchor range %s..%s != expected %s..%s"
             % (meta.get("anchor_start"), meta.get("anchor_end"), a0, a1))
w = int(meta.get("lookback_seconds"))
if not (a0 < train_end <= a1) or not (a0 <= val_start < val_end <= a1):
    sys.exit("split boundaries outside the corpus anchor range")
if val_start < train_end + w or val_end <= val_start:
    sys.exit("require val_start >= train_end + W and val_end > val_start")
print("split_ok train<[%d] val[%d,%d) W=%d" % (train_end, val_start, val_end, w))
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; exiting after preflight."; exit 0
fi

# --- train ----------------------------------------------------------------
echo "Launching window MAE training..."
exec "$PYTHON" -u tools/train_windows.py --config "$CONFIG" \
  --data "$DATA" --out "$OUT" \
  --train-end "$TRAIN_END" --val-start "$VAL_START" --val-end "$VAL_END" \
  --epochs "$EPOCHS" --max-batches "$MAX_BATCHES" --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" --device "$DEVICE" ${EXTRA:-}
