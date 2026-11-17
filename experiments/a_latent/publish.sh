#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Run separately only after the dedicated repository exists and publication is authorized.
export CUDA_VISIBLE_DEVICES=''
exec "$PY" "$CODE/roboicl/publish_epochs.py" --run "$RUN_DIR" \
  --state-dir "$BASE/hf-publish" --repo "$HF_REPO"
