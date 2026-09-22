#!/usr/bin/env bash
# Compatibility entrypoint; the experiment owns the training launcher.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO/experiments/trajectory_mlp_v1/train.sh" "$@"
