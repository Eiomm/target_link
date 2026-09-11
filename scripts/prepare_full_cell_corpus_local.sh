#!/usr/bin/env bash
# Download observations_v2 + training_groups_k3 one bucket at a time, normalize
# observations locally once, and build an all-partition train/val corpus on NFS.
# Safe to rerun: completed buckets have explicit markers and are skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HDFS_CORPUS="${HDFS_CORPUS:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link/corpus_v1}"
OBS_DIR="${OBS_DIR:-observations_v2}"
GROUPS_DIR="${GROUPS_DIR:-training_groups_k3}"
OUT="${OUT:-$REPO/runtime/cell_trainready_v2_20260817_23}"
TRAIN_DAYS="${TRAIN_DAYS:-20260817 20260818 20260819 20260820 20260821 20260822}"
VAL_DAYS="${VAL_DAYS:-20260823}"
BUCKET_START="${BUCKET_START:-0}"
BUCKET_END="${BUCKET_END:-127}"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
ROW_GROUP_SIZE="${ROW_GROUP_SIZE:-131072}"
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

[[ "$OBS_DIR" == "observations_v2" ]] || {
  echo "This workflow is pinned to accepted-row-set observations_v2 (got $OBS_DIR)" >&2
  exit 2
}
[[ "$BUCKET_START" =~ ^[0-9]+$ && "$BUCKET_END" =~ ^[0-9]+$ ]] || {
  echo "BUCKET_START/BUCKET_END must be integers" >&2; exit 2;
}
(( BUCKET_START >= 0 && BUCKET_END < 128 && BUCKET_START <= BUCKET_END )) || {
  echo "bucket range must satisfy 0 <= start <= end < 128" >&2; exit 2;
}
command -v hdfs >/dev/null || { echo "hdfs CLI is required" >&2; exit 2; }

cd "$REPO"
set +u
source "$ENV_ROOT/bin/activate"
set -u
PYTHON="$(command -v python)"
"$PYTHON" -c 'import numpy, pyarrow; print("python/pyarrow OK", pyarrow.__version__)'

mkdir -p "$OUT/.ready" "$OUT/.download_tmp"
active_work=""
cleanup() {
  if [[ -n "$active_work" && "$active_work" == "$OUT/.download_tmp/"* ]]; then
    rm -rf -- "$active_work"
  fi
}
trap cleanup EXIT INT TERM

prepare_one() {
  local side="$1" day="$2" bucket="$3"
  local rel="day=${day}/bucket=${bucket}"
  local obs_out="$OUT/$side/$OBS_DIR/$rel"
  local groups_out="$OUT/$side/$GROUPS_DIR/$rel"
  local ready="$OUT/.ready/${side}_${day}_${bucket}"

  if [[ -f "$ready" && -f "$obs_out/_TRAINREADY.json" && -d "$groups_out" ]]; then
    echo "SKIP ready $side $rel"
    return 0
  fi
  if [[ -e "$obs_out" ]]; then
    echo "Incomplete existing bucket; refusing to overwrite:" >&2
    echo "  $obs_out" >&2
    echo "  $groups_out" >&2
    echo "Move those two paths aside, then rerun." >&2
    return 3
  fi

  active_work="$(mktemp -d "$OUT/.download_tmp/${side}_${day}_${bucket}.XXXXXX")"
  echo "DOWNLOAD $side $rel"
  hdfs dfs -get "$HDFS_CORPUS/$OBS_DIR/$rel" "$active_work/observations"
  local groups_input="$groups_out"
  local groups_was_downloaded=0
  if [[ ! -d "$groups_out" ]]; then
    hdfs dfs -get "$HDFS_CORPUS/$GROUPS_DIR/$rel" "$active_work/groups"
    groups_input="$active_work/groups"
    groups_was_downloaded=1
  fi

  echo "NORMALIZE $side $rel"
  "$PYTHON" tools/normalize_cell_partition.py \
    --observations "$active_work/observations" \
    --groups "$groups_input" \
    --out "$active_work/normalized" \
    --row-group-size "$ROW_GROUP_SIZE"

  mkdir -p "$(dirname "$obs_out")" "$(dirname "$groups_out")"
  if [[ "$groups_was_downloaded" == "1" ]]; then
    mv "$active_work/groups" "$groups_out"
  fi
  mv "$active_work/normalized" "$obs_out"
  touch "$ready"
  rm -rf -- "$active_work"
  active_work=""
  echo "READY $side $rel"
}

for day in $TRAIN_DAYS; do
  for ((bucket=BUCKET_START; bucket<=BUCKET_END; bucket++)); do
    prepare_one train "$day" "$bucket"
  done
done
for day in $VAL_DAYS; do
  for ((bucket=BUCKET_START; bucket<=BUCKET_END; bucket++)); do
    prepare_one val "$day" "$bucket"
  done
done

read -r -a train_day_array <<< "$TRAIN_DAYS"
read -r -a val_day_array <<< "$VAL_DAYS"
expected=$(( (${#train_day_array[@]} + ${#val_day_array[@]}) * (BUCKET_END - BUCKET_START + 1) ))
actual="$(find "$OUT/.ready" -maxdepth 1 -type f | wc -l | tr -d ' ')"
if [[ "$TRAIN_DAYS" == "20260817 20260818 20260819 20260820 20260821 20260822" \
      && "$VAL_DAYS" == "20260823" && "$BUCKET_START" == "0" && "$BUCKET_END" == "127" ]]; then
  expected=896
fi
[[ "$actual" == "$expected" ]] || {
  echo "ready marker count $actual, expected $expected" >&2; exit 4;
}
touch "$OUT/_TRAINREADY_SUCCESS"
echo "COMPLETE: $OUT ($actual partitions)"
du -sh "$OUT"
