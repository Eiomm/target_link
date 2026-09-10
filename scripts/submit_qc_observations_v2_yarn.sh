#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link/corpus_v1}"
OUT="${OUT:-$ROOT/_qc/observations_v2_acceptance.json.d}"
cd "$REPO"
exec /usr/local/spark-current/bin/spark-submit \
  --master yarn --deploy-mode cluster \
  --queue "${QUEUE:-root.xinsi_yanfaerzu_default}" \
  --name tl_qc_observations_v2 \
  --conf spark.yarn.dist.archives=hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz#minipy3 \
  --conf spark.pyspark.python=./minipy3/minipy3/bin/python \
  --driver-memory 8g --executor-cores 2 --executor-memory 8g \
  --conf spark.executor.memoryOverhead=4096 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.minExecutors=2 \
  --conf spark.dynamicAllocation.maxExecutors=40 \
  --conf spark.sql.adaptive.enabled=true \
  tools/qc_observations_v2.py --root "$ROOT" --out "$OUT"
