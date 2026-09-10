#!/usr/bin/env bash
# Rebuild only the policy-dependent groups with K_min=3. observations_v2 and
# the existing K_min=4 training_groups are preserved.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODE=yarn
export STAGES=groups
export OBS_DIR=observations_v2
export GROUPS_DIR=training_groups_k3
export K_MIN=3
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"

exec bash "$REPO/scripts/submit_build_corpus_yarn.sh"
