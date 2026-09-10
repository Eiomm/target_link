#!/usr/bin/env bash
# Luban k8s-job entrypoint for the cell-level whole-trajectory MAE (GPU).
#
# Platform form (Web 训练任务 -> 创建单机任务 / luban-client):
#   启动命令: bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/scripts/submit_cell_train_job.sh
#   资源:     1×GPU 单机 (选 A100)。主规格为 d_model=256、4+4 层、8 heads；
#             仍是约 700 万参数的中小模型，A100 的主要价值在吞吐。
#   镜像:     本实验环境快照(需含 hadoop 客户端 — 本脚本要把语料从 HDFS 拉到 NFS)
#   日志:     stdout 不重定向,平台日志页实时可见;metrics/ckpt 落 NFS 的 $OUT
#
# 语料在 HDFS 上(yarn 的 build_corpus 产出),而 CellCorpusDataset 只读本地路径,
# 所以这里先 `hdfs dfs -get` 到 NFS。**只拉选中的 (day,bucket) 分片** —— 一天 128
# 个 bucket,一个 bucket ≈ 83MB observations + 28MB training_groups,拉 4+1 个就是
# 整个 smoke 的数据量,这也是"拿一部分数据"的实现方式。
#
# 两个 split 是两个独立的 corpus root:$DATA/train 与 $DATA/val。train_cells.py
# 按目录读,切分因此是显式的、不藏在时间过滤里。
#
# 必须用 observations_v2:只有它带 ragged `bin_pos`(段内 10m bin 位置,0..49)。
# 老的 observations/ 没有这一列,reader 会直接报缺列。
#
# Overridable: TRAIN_PARTS / VAL_PARTS / OBS_DIR / GROUPS_DIR / DATA / OUT / EPOCHS /
#   MAX_BATCHES / BATCH_SIZE / WORKERS / DEVICE / FETCH / FORCE_FETCH / DRY_RUN / EXTRA
set -euo pipefail

REPO="/nfs/dataset-ofs-494-1/project/user/junao/target_link"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
HDFS_CORPUS="${HDFS_CORPUS:-$HDFS_BASE/corpus_v1}"
OBS_DIR="${OBS_DIR:-observations_v2}"
GROUPS_DIR="${GROUPS_DIR:-training_groups_k3}"

# bucket 是 xxhash64(cell_id) % 128 的存储分桶，不是时间窗口。默认 train 取
# 2026-08-21 的 4 个 hash bucket，val 取次日相同 bucket；每个 bucket 含多个 window。
TRAIN_PARTS="${TRAIN_PARTS:-20260821:64 20260821:65 20260821:66 20260821:67}"
VAL_PARTS="${VAL_PARTS:-20260822:64}"

DATA="${DATA:-$REPO/runtime/cell_train_day20260821}"
OUT="${OUT:-$REPO/runtime/cell_train_day20260821_$(date +%m%d_%H%M)}"
FETCH="${FETCH:-1}"              # 0 = 完全不碰 HDFS,语料必须已在 $DATA
FORCE_FETCH="${FORCE_FETCH:-0}"  # 1 = 即使本地有也重拉
DRY_RUN="${DRY_RUN:-0}"

EPOCHS="${EPOCHS:-1}"
# 一个 bucket ≈ 10 万个 training group,跑满不现实;首次看效果用 max-batches 截断
MAX_BATCHES="${MAX_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
M_MAX="${M_MAX:-16}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"

export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

cd "$REPO"
set +u; source "$ENV_ROOT/bin/activate"; set -u
PYTHON="$(command -v python)"

echo "repo=$REPO python=$PYTHON"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'NO GPU VISIBLE')"
echo "corpus=$HDFS_CORPUS/$OBS_DIR  data=$DATA out=$OUT"
echo "train_parts=[$TRAIN_PARTS] val_parts=[$VAL_PARTS]"
echo "epochs=$EPOCHS max_batches=$MAX_BATCHES batch=$BATCH_SIZE workers=$WORKERS device=$DEVICE"

# --- 语料:把选中的分片一次性拉到 NFS -----------------------------------------
# 临时目录 + mv: `hdfs dfs -get` 往已存在的目标里是"合并",一次中断的重试会在
# 早先的 marker 后面留半张表;先 staged 再 mv 的话最终路径只可能是完整的。
# (`hdfs dfs -get` 没有 -f,只有 -put 有。)
fetch_part() {  # side table day bucket
  local side="$1" table="$2" day="$3" bucket="$4"
  local rel="day=${day}/bucket=${bucket}"
  local dst="$DATA/$side/$table/$rel"
  local marker="$DATA/$side/.fetched_${table}_${day}_${bucket}"
  if [[ "$FORCE_FETCH" != "1" && -e "$marker" ]]; then return 0; fi
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst.tmp"
  echo "fetching $side/$table/$rel ..."
  if ! hdfs dfs -get "$HDFS_CORPUS/$table/$rel" "$dst.tmp"; then
    echo "FETCH FAILED: $HDFS_CORPUS/$table/$rel" >&2
    return 1
  fi
  rm -rf "$dst"
  mv "$dst.tmp" "$dst"
  touch "$marker"
}

if [[ "$FETCH" == "1" ]]; then
  command -v hdfs >/dev/null || { echo "no hdfs client on this pod; set FETCH=0 with a preloaded \$DATA" >&2; exit 1; }
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
    [[ -d "$DATA/$side/$table" ]] || { echo "missing $DATA/$side/$table (FETCH=$FETCH)" >&2; exit 1; }
  done
  echo "$side size: $(du -sh "$DATA/$side" | cut -f1)"
done

# --- preflight:语料契约 + 依赖 -------------------------------------------------
"$PYTHON" - "$DATA" "$OBS_DIR" "$GROUPS_DIR" "$M_MAX" <<'PY'
import glob, importlib, os, sys
data, obs_dir, grp_dir, m_max = sys.argv[1:5]
for m in ("numpy", "torch", "pyarrow"):
    mod = importlib.import_module(m)
    print("import_ok %s %s" % (m, getattr(mod, "__version__", "?")))
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Scalars and ragged (list) columns separately: pyarrow prints a list column as
# "list<element: X>", so the check is on the value type.
OBS_SCALAR = {"cell_id": "int64", "sample_id": "string", "dt": "float", "n_pieces": "int16"}
OBS_LIST = {"T_diff": "float", "ratio_pct": "int8", "observed": "bool",
            "valid": "bool", "bin_pos": "int8"}
GRP_SCALAR = {"group_id": "string", "cell_id": "int64", "K": "int32",
              "group_size": "int64", "window": "int64"}
GRP_LIST = {"sample_ids": "string"}
JOIN_GROUPS = 2000      # per-group sample_id membership is checked on this many
M_MAX = int(sys.argv[4])


def schema_check(path, table, scalar, lst):
    s = pq.ParquetFile(path).schema_arrow
    for col, want in scalar.items():
        if col not in s.names:
            sys.exit("%s: %s lacks column %s" % (path, table, col))
        got = str(s.field(col).type)
        if got != want:
            sys.exit("%s: %s.%s is %s, want %s" % (path, table, col, got, want))
    for col, want in lst.items():
        if col not in s.names:
            sys.exit("%s: %s lacks column %s" % (path, table, col))
        t = s.field(col).type
        if not pa.types.is_list(t) or str(t.value_type) != want:
            sys.exit("%s: %s.%s is %s, want list<%s>" % (path, table, col, t, want))


for side in ("train", "val"):
    o_days = sorted(os.listdir("%s/%s/%s" % (data, side, obs_dir)))
    g_days = sorted(os.listdir("%s/%s/%s" % (data, side, grp_dir)))
    if o_days != g_days:
        sys.exit("%s: days differ between %s %s and %s %s" % (side, obs_dir, o_days, grp_dir, g_days))
    n_obs = n_grp = 0
    for day in o_days:
        o_b = sorted(os.listdir("%s/%s/%s/%s" % (data, side, obs_dir, day)))
        g_b = sorted(os.listdir("%s/%s/%s/%s" % (data, side, grp_dir, day)))
        if o_b != g_b:
            sys.exit("%s/%s: buckets differ: %s vs %s" % (side, day, o_b, g_b))
        for bucket in o_b:
            of = glob.glob("%s/%s/%s/%s/%s/*.parquet" % (data, side, obs_dir, day, bucket))
            gf = glob.glob("%s/%s/%s/%s/%s/*.parquet" % (data, side, grp_dir, day, bucket))
            if len(of) != len(gf):
                sys.exit("%s/%s/%s: file counts differ" % (side, day, bucket))
            n_obs += len(of); n_grp += len(gf)
            schema_check(of[0], obs_dir, OBS_SCALAR, OBS_LIST)
            schema_check(gf[0], grp_dir, GRP_SCALAR, GRP_LIST)
            o = pq.read_table(of[0], columns=["cell_id", "sample_id", "dt", "T_diff",
                                              "valid", "bin_pos"])
            g = pq.read_table(gf[0], columns=["group_id", "cell_id", "K", "group_size",
                                              "sample_ids"])
            cid = o["cell_id"].to_numpy(zero_copy_only=False)
            # the reader locates a cell's rows with one searchsorted, so the
            # partition must be written cell_id-sorted
            if cid.size > 1 and bool((np.diff(cid) < 0).any()):
                sys.exit("%s: cell_id is not sorted; CellCorpusDataset requires it" % of[0])
            bp = pc.list_flatten(o["bin_pos"]).to_numpy(zero_copy_only=False)
            if bp.size and (bp.min() < 0 or bp.max() > 49):
                sys.exit("%s: bin_pos outside [0,49]: [%d,%d]"
                         % (of[0], int(bp.min()), int(bp.max())))
            # valid=1 means "T_diff is known": if it were not, the reader's
            # bincount would sum a NaN straight into that bin
            v = pc.list_flatten(o["valid"]).to_numpy(zero_copy_only=False)
            td = pc.list_flatten(o["T_diff"]).to_numpy(zero_copy_only=False)
            bad = int((v & ~np.isfinite(td)).sum())
            if bad:
                sys.exit("%s: %d pieces are valid=1 with a nonfinite T_diff" % (of[0], bad))
            dt = o["dt"].to_numpy(zero_copy_only=False)
            # dt is a float32 of a double difference: a few rows round up to
            # exactly 600.0 (6 in 1.2M on bucket 64 of 20260822). The reader
            # clamps those just inside the window; only a value PAST 600 by more
            # than the float32 spacing is corruption.
            edge = int((dt >= 600.0).sum()) if dt.size else 0
            if dt.size and (dt.min() < 0 or dt.max() > 600.0):
                sys.exit("%s: dt outside [0,600]" % of[0])
            gs = g["group_size"].to_numpy(zero_copy_only=False)
            if gs.size and (gs.min() < 1 or gs.max() > M_MAX):
                sys.exit("%s: group_size outside 1..%d" % (gf[0], M_MAX))
            # K is the number of observations of that cell IN THIS PARTITION,
            # and a cell's groups must partition those observations exactly
            cells, counts = np.unique(cid, return_counts=True)
            n_of = dict(zip(cells.tolist(), counts.tolist()))
            gcell = g["cell_id"].to_numpy(zero_copy_only=False)
            K = g["K"].to_numpy(zero_copy_only=False)
            for i in range(len(gcell)):
                if n_of.get(int(gcell[i])) != int(K[i]):
                    sys.exit("%s: group %s has K=%d but the partition holds %d "
                             "observations of cell %s"
                             % (gf[0], g["group_id"][i].as_py(), K[i],
                                n_of.get(int(gcell[i]), 0), gcell[i]))
            uniq, inv = np.unique(gcell, return_inverse=True)
            tot = np.bincount(inv, weights=gs)
            exp = np.array([n_of.get(int(c), 0) for c in uniq], dtype=np.float64)
            if not np.array_equal(tot, exp):
                i = int(np.flatnonzero(tot != exp)[0])
                sys.exit("%s: cell %d is covered by group_size %d but has %d observations"
                         % (gf[0], uniq[i], tot[i], exp[i]))
            # sample_id is unique WITHIN a cell, not across the partition: one
            # pass can enter two different segments of a link in the same
            # 10-minute window. The reader keys its per-cell lookup by
            # sample_id, so a duplicate inside one cell would drop a row.
            dup = o.num_rows - len(set(zip(cid.tolist(), o["sample_id"].to_pylist())))
            if dup:
                sys.exit("%s: %d duplicate (cell_id, sample_id) rows" % (of[0], dup))
            # per-group membership, on a bounded prefix: the cheap vectorised
            # checks above already force every group's rows to exist
            lo = np.searchsorted(cid, gcell, "left")
            hi = np.searchsorted(cid, gcell, "right")
            for i in range(min(JOIN_GROUPS, len(gcell))):
                members = set(o["sample_id"][int(lo[i]):int(hi[i])].to_pylist())
                sids = g["sample_ids"][i].as_py()
                miss = [s for s in sids if s not in members]
                if miss:
                    sys.exit("%s: group %s references %d samples absent from cell %s"
                             % (gf[0], g["group_id"][i].as_py(), len(miss), gcell[i]))
            print("ok %s/%s/%s obs=%d groups=%d cells=%d bin_pos[%d,%d] "
                  "group_size[%d,%d] dt_edge=%d"
                  % (side, day, bucket, o.num_rows, g.num_rows, len(cells),
                     int(bp.min()), int(bp.max()), int(gs.min()), int(gs.max()), edge))
    print("contract_ok %s: %d obs files, %d group files" % (side, n_obs, n_grp))
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; exiting after preflight."; exit 0
fi

# --- train ----------------------------------------------------------------
echo "Launching cell MAE training..."
exec "$PYTHON" -u tools/train_cells.py \
  --data "$DATA/train" --val-data "$DATA/val" --out "$OUT" \
  --obs-dir "$OBS_DIR" --groups-dir "$GROUPS_DIR" --m-max "$M_MAX" \
  --epochs "$EPOCHS" --max-batches "$MAX_BATCHES" --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" --device "$DEVICE" ${EXTRA:-}
