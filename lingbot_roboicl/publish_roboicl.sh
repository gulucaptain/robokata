#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/data/haoyuzhao/envs/video/zero-wam-ig10k/hf-cache}"
export HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/tmp"
exec .venv/bin/python roboicl/publish_epochs.py "$@"
