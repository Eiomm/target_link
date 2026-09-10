#!/usr/bin/env bash
# Luban k8s-job entrypoint for stage-1 pretraining (pattern: rpgpt/gpt_geohash_pretrain/
# pt_qwen3_nat_prefix_ar.sh, trimmed: no staging/tokenizer stages — the corpus is
# produced upstream by build_profiles -> build_pretrain_corpus and lives on NFS).
#
# Platform form (Web 训练任务 -> 创建单机任务 / luban-client):
#   启动命令: bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/legacy/scripts/submit_pretrain_job.sh
#   资源:     1×GPU 单机 (model is 1.1M params; no accelerate/multi-GPU needed)
#   镜像:     本实验环境快照或任意 CUDA<=11.4 基础镜像 — the python env is the
#             NFS conda qwen12 (torch verified on 470.129.06 driver machines by RPGPT)
#   日志:     stdout 不重定向(平台日志页实时可见); metrics/ckpt 落 NFS output.dir
#
# Overridable: CONFIG / SEEDS / CORPUS / EXTRA (train_pretrain args) / DRY_RUN=1.
set -euo pipefail

REPO="/nfs/dataset-ofs-494-1/project/user/junao/target_link"
ENV_ROOT="${ENV_ROOT:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12}"
CONFIG="${CONFIG:-$REPO/legacy/configs/pretrain.yaml}"
CORPUS="${CORPUS:-}"          # default: read from the yaml
SEEDS="${SEEDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO"
set +u; source "$ENV_ROOT/bin/activate"; set -u
PYTHON="$(command -v python)"
echo "repo=$REPO env=$ENV_ROOT python=$PYTHON config=$CONFIG seeds=$SEEDS"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"

echo "Running dependency preflight..."
"$PYTHON" - "$CONFIG" "${CORPUS:-}" <<'PY'
import importlib, sys, yaml
for m in ("numpy", "torch", "yaml"):
    mod = importlib.import_module(m)
    print(f"import_ok {m} {getattr(mod, '__version__', '?')}")
cfg = yaml.safe_load(open(sys.argv[1]))
corpus = sys.argv[2] or cfg["data"]["corpus"]
import os
if not os.path.isfile(corpus):
    sys.exit(f"corpus missing: {corpus} — run the profiles/corpus pipeline first")
print(f"corpus_ok {corpus} ({os.path.getsize(corpus)/1e6:.0f} MB)")
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; exiting after preflight."
  exit 0
fi

extra=()
[[ -n "$CORPUS" ]] && extra+=(--corpus "$CORPUS")
echo "Launching stage-1 pretraining..."
exec "$PYTHON" -u legacy/tools/train_pretrain.py --config "$CONFIG" --seeds "$SEEDS" \
     "${extra[@]}" ${EXTRA:-}
