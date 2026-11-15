#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/tmp/cache" WANDB_DIR="$ROOT/tmp/wandb"
export HF_HUB_CACHE="$ROOT/tmp/hub"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$HF_HUB_CACHE"
: "${CUDA_VISIBLE_DEVICES:?Set explicitly to GPUs reserved for this experiment}"
CONFIG=${1:-configs/demo-top.json}
: "${2:?Usage: scripts/train.sh CONFIG OUTPUT [--resume]}"
OUTPUT=$2
shift 2
WORLD=$("$ROOT/.venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["world_size"])' "$CONFIG")
export PYTHONPATH="$ROOT/vendor/openpi/src:$ROOT/vendor/openpi/packages/openpi-client/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" scripts/preflight.py --config "$CONFIG"
exec "$ROOT/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" train.py --config "$CONFIG" --output "$OUTPUT" "$@"
