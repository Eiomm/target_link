#!/usr/bin/env bash
# Build the 24h causal-window corpus on YARN (Beijing 2026-08-21, W=600/S=600).
#
#   MODE=yarn bash scripts/submit_windows_24h.sh   # submit
#   MODE=dry  bash scripts/submit_windows_24h.sh   # print the spark-submit only
#
# Every knob is exported with a default here, so the script runs with no
# external environment; it only sets the 24h parameters and hands over to
# scripts/submit_windows_yarn.sh (single source of truth for the spark-submit
# flags).
#
# Input is the adapted events table written by scripts/submit_adapt_yarn.sh
# (25h = warm-up hour 2026082023 + the 24 partitions of 20260821).
#
# Anchors [1787241600, 1787328000) = 144 windows, 600s aligned, disjoint.
# Train/val split for tools/train_windows.py (train_end/val_start/val_end):
#   train 96 anchors = [1787241600, 1787299200)
#   val   48 anchors = [1787299800, 1787328000)
# The 600s gap equals W, so no window is shared across the split.
#
# MAX_PASSES=64 is a safety net, not a sampling decision: the pass rate is
# ~28 passes per link per hour (~5 per 600s window), so 64 clips only true
# outliers -- and it keeps every snapshot under the reader's hard limit of 512
# sub-curves (that limit errors out, it does not truncate, so an uncapped hot
# link can make the whole corpus unreadable after a multi-hour build). The cap
# is auditable: the snapshots table carries n_passes_before_cap / n_passes_kept
# per link and anchor. Set MAX_PASSES=0 to keep every pass.
set -euo pipefail
# legacy/scripts/ -> repo root: only these entrypoints moved, tools/ and the
# rest of the code stayed in place (legacy/README.md)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # sibling entrypoints moved here too
cd "$REPO"

DAY="${DAY:-20260821}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"

export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"
export INPUT_GLOB="${INPUT_GLOB:-${HDFS_BASE}/events/day${DAY}/events}"
export OUT_DIR="${OUT_DIR:-${HDFS_BASE}/windows/day${DAY}_w600_s600}"
export ANCHOR_START="${ANCHOR_START:-1787241600}"
export ANCHOR_END="${ANCHOR_END:-1787328000}"
export LOOKBACK_SECONDS="${LOOKBACK_SECONDS:-600}"
export STRIDE_SECONDS="${STRIDE_SECONDS:-600}"
export MAX_PASSES="${MAX_PASSES:-64}"
export SUB_LENGTH_M="${SUB_LENGTH_M:-200}"
export MAX_BINS="${MAX_BINS:-21}"
export CURVES_PARTITIONS="${CURVES_PARTITIONS:-800}"
export SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-8000}"
export MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
export MODE="${MODE:-yarn}"

exec bash "$SCRIPTS/submit_windows_yarn.sh"
