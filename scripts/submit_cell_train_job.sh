#!/usr/bin/env bash
# Fetch selected Cell-MAE partitions, validate the local corpus, then train.
#
# Typical platform entrypoint:
#   bash scripts/submit_cell_train_job.sh
#
# Wrapper scripts set the experiment-specific TRAIN_PARTS, VAL_PARTS, DATA,
# OUT and model arguments. This file owns only the shared fetch/check/launch
# workflow. The observations directory must contain ragged bin_pos values;
# tools/check_cell_corpus.py enforces the complete reader contract.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
HDFS_CORPUS="${HDFS_CORPUS:-$HDFS_BASE/corpus_v1}"
OBS_DIR="${OBS_DIR:-observations_v2}"
GROUPS_DIR="${GROUPS_DIR:-training_groups_k3}"

TRAIN_PARTS="${TRAIN_PARTS:-20260821:64 20260821:65 20260821:66 20260821:67}"
VAL_PARTS="${VAL_PARTS:-20260822:64}"
DATA="${DATA:-$REPO/runtime/cell_train_day20260821}"
OUT="${OUT:-$REPO/runtime/cell_train_day20260821_$(date +%m%d_%H%M)}"

FETCH="${FETCH:-1}"
FORCE_FETCH="${FORCE_FETCH:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-1}"
MAX_BATCHES="${MAX_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
M_MAX="${M_MAX:-16}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"

export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

for flag in FETCH FORCE_FETCH DRY_RUN; do
  value="${!flag}"
  [[ "$value" == "0" || "$value" == "1" ]] || {
    echo "$flag must be 0 or 1 (got $value)" >&2
    exit 2
  }
done

cd "$REPO"
set +u
source "$ENV_ROOT/bin/activate"
set -u
PYTHON="$(command -v python)"

echo "repo=$REPO python=$PYTHON"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU VISIBLE')"
echo "corpus=$HDFS_CORPUS/$OBS_DIR data=$DATA out=$OUT"
echo "train_parts=[$TRAIN_PARTS] val_parts=[$VAL_PARTS]"
echo "epochs=$EPOCHS max_batches=$MAX_BATCHES batch=$BATCH_SIZE workers=$WORKERS device=$DEVICE"

fetch_part() {  # side table day bucket
  local side="$1" table="$2" day="$3" bucket="$4"
  [[ "$side" == "train" || "$side" == "val" ]] || return 2
  [[ "$table" =~ ^[A-Za-z0-9_-]+$ && "$day" =~ ^[0-9]{8}$ \
      && "$bucket" =~ ^[0-9]+$ ]] || {
    echo "invalid fetch target: $side/$table/day=$day/bucket=$bucket" >&2
    return 2
  }
  local rel="day=${day}/bucket=${bucket}"
  local dst="$DATA/$side/$table/$rel"
  local staged="${dst}.tmp"
  local marker="$DATA/$side/.fetched_${table}_${day}_${bucket}"

  if [[ "$FORCE_FETCH" != "1" && -f "$marker" && -d "$dst" ]] \
      && compgen -G "$dst/*.parquet" >/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  rm -rf -- "$staged"
  echo "fetching $side/$table/$rel ..."
  if ! hdfs dfs -get "$HDFS_CORPUS/$table/$rel" "$staged"; then
    echo "FETCH FAILED: $HDFS_CORPUS/$table/$rel" >&2
    rm -rf -- "$staged"
    return 1
  fi
  rm -rf -- "$dst"
  mv "$staged" "$dst"
  touch "$marker"
}

if [[ "$FETCH" == "1" ]]; then
  command -v hdfs >/dev/null || {
    echo "no hdfs client; set FETCH=0 only with a complete local DATA corpus" >&2
    exit 1
  }
  for spec in $TRAIN_PARTS; do
    fetch_part train "$OBS_DIR" "${spec%%:*}" "${spec##*:}"
    fetch_part train "$GROUPS_DIR" "${spec%%:*}" "${spec##*:}"
  done
  for spec in $VAL_PARTS; do
    fetch_part val "$OBS_DIR" "${spec%%:*}" "${spec##*:}"
    fetch_part val "$GROUPS_DIR" "${spec%%:*}" "${spec##*:}"
  done
fi

for side in train val; do
  for table in "$OBS_DIR" "$GROUPS_DIR"; do
    [[ -d "$DATA/$side/$table" ]] || {
      echo "missing $DATA/$side/$table (FETCH=$FETCH)" >&2
      exit 1
    }
  done
  echo "$side size: $(du -sh "$DATA/$side" | cut -f1)"
done

"$PYTHON" tools/check_cell_corpus.py \
  --data "$DATA" --obs-dir "$OBS_DIR" --groups-dir "$GROUPS_DIR" --m-max "$M_MAX"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; preflight passed, training skipped."
  exit 0
fi

extra_args=()
if [[ -n "$EXTRA" ]]; then
  read -r -a extra_args <<< "$EXTRA"
fi
echo "Launching cell MAE training..."
exec "$PYTHON" -u tools/train_cells.py \
  --data "$DATA/train" --val-data "$DATA/val" --out "$OUT" \
  --obs-dir "$OBS_DIR" --groups-dir "$GROUPS_DIR" --m-max "$M_MAX" \
  --epochs "$EPOCHS" --max-batches "$MAX_BATCHES" --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" --device "$DEVICE" "${extra_args[@]}"
