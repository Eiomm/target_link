#!/usr/bin/env bash
# Rebuild only the observations table with the corrected physical sort order.
# The row set and IDs are unchanged, so cells/ and training_groups_k3/ remain
# compatible and do not need to be rebuilt.
set -euo pipefail

# v3 completed as a Spark job but failed the persisted-order acceptance check
# on large partitions. Keep this entrypoint only for controlled reproduction;
# an accidental rerun would create another corpus that the reader must reject.
if [[ "${ALLOW_BROKEN_V3_REPRO:-0}" != "1" ]]; then
  echo "observations_v3 is rejected; set ALLOW_BROKEN_V3_REPRO=1 only for a controlled reproduction" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODE=yarn
export STAGES=obs
export OBS_DIR=observations_v3
export GROUPS_DIR=training_groups_k3
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

exec bash "$REPO/scripts/submit_build_corpus_yarn.sh"
