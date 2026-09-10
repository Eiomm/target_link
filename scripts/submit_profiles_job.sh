#!/usr/bin/env bash
# Luban k8s-job entrypoint (CPU, no GPU needed) for one day's profiles build.
# Pattern: legacy/scripts/submit_pretrain_job.sh (same activation/preflight style).
#
# Per-day flow: hdfs get the day's ingest outputs (bins_shards ~10-15GB/day)
# -> run tools/build_profiles.py UNCHANGED on local paths (streaming numpy,
# peak RSS ~10-15GB/day -> size the task >=32GB RAM) -> copy the small npz
# products to NFS for training (+ optional hdfs archive).
#
# Platform form (Web 训练任务 -> 创建单机任务, CPU-only resource):
#   启动命令: DAY=20260820 bash /nfs/.../target_link/scripts/submit_profiles_job.sh
#   资源:     CPU 任务, 内存 >=32GB
#   镜像:     须含 hadoop 客户端 (hdfs 命令) — the experiment-env snapshot has it
#   日志:     stdout 不重定向; 产物落 NFS runtime/profiles/
#
# Overridable: DAY / HDFS_BASE / TMP_DIR / CONFIG / DRY_RUN / SKIP_PUT
#   TMP_DIR: where the day's bins land (default /tmp — container local disk;
#            point it at an NFS dir if local disk is tight; DRY_RUN prints df)
set -euo pipefail

REPO="/nfs/dataset-ofs-494-1/project/user/junao/target_link"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
DAY="${DAY:?set DAY=YYYYMMDD}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link/processed_spark}"
CONFIG="${CONFIG:-$REPO/configs/profiles_job.yaml}"
TMP_DIR="${TMP_DIR:-/tmp/profiles_$DAY}"
OUT_NFS="${OUT_NFS:-$REPO/runtime/profiles/day$DAY}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PUT="${SKIP_PUT:-0}"      # 1 = don't archive products back to hdfs

cd "$REPO"
set +u; source "$ENV_ROOT/bin/activate"; set -u
PYTHON="$(command -v python)"
SRC="$HDFS_BASE/day$DAY"

echo "day=$DAY src=$SRC tmp=$TMP_DIR out=$OUT_NFS"
echo "python=$PYTHON  hdfs=$(command -v hdfs || echo MISSING)"
df -h "$TMP_DIR/.." 2>/dev/null || df -h /tmp

# --- preflight: ingest outputs exist, day complete --------------------------
[[ -n "$(hdfs dfs -ls "$SRC/samples.parquet" 2>/dev/null)" ]] \
  || { echo "ingest output missing: $SRC/samples.parquet (run submit_ingest_yarn first)" >&2; exit 1; }
N_SHARDS=$(hdfs dfs -ls "$SRC/bins_shards" 2>/dev/null | grep -c "part-.*parquet" || true)
[[ "$N_SHARDS" -gt 0 ]] || { echo "no bins shards under $SRC/bins_shards" >&2; exit 1; }
echo "shards on hdfs: $N_SHARDS"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; would: get $SRC -> $TMP_DIR, build profiles, copy to $OUT_NFS"; exit 0
fi

# --- pull the day's ingest outputs (local disk; pyarrow-hdfs is unreliable) --
mkdir -p "$TMP_DIR" "$OUT_NFS"
[[ -f "$TMP_DIR/.fetch_done" ]] || {
  echo "fetching $SRC -> $TMP_DIR ..."
  hdfs dfs -get "$SRC/samples.parquet" "$TMP_DIR/samples.parquet"
  hdfs dfs -get "$SRC/bins_shards" "$TMP_DIR/bins_shards"
  touch "$TMP_DIR/.fetch_done"
}
echo "local shards: $(ls "$TMP_DIR/bins_shards"/part-*.parquet | wc -l)"

# --- profiles: the SAME streaming tool, path-overridden ----------------------
"$PYTHON" -u tools/build_profiles.py --config "$CONFIG" \
    --input-dir "$TMP_DIR" --output-dir "$OUT_NFS"

# --- archive products back to hdfs (npz ~0.2-1.5GB/day) ----------------------
if [[ "$SKIP_PUT" != "1" ]]; then
  hdfs dfs -mkdir -p "$SRC/profiles_l200"
  hdfs dfs -put -f "$OUT_NFS"/profiles_l200.npz "$SRC/profiles_l200/" || \
      echo "WARN: hdfs put failed; NFS copies remain at $OUT_NFS"
fi
echo "done: $(ls -lh "$OUT_NFS")"
# hygiene only (k8s containers die with the job anyway): guarded cleanup —
# rm ONLY on a path that strictly matches /tmp/profiles_*; anything else
# (empty var, NFS path, typo) is never deleted
if [[ "${KEEP_TMP:-0}" != "1" ]]; then
  case "$TMP_DIR" in
    /tmp/profiles_*) rm -rf "$TMP_DIR" ;;
    *) echo "[profiles_job] TMP_DIR=$TMP_DIR not under /tmp/profiles_* — skipping cleanup" ;;
  esac
fi
