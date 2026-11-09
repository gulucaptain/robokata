#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
export TMPDIR="$root/tmp" HF_HOME="$root/cache/hf" HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor" TRITON_CACHE_DIR="$root/cache/triton"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ ${1:-} = inspect ]]; then
    CUDA_VISIBLE_DEVICES="" exec .venv/bin/python "$root/roboicl/run.py" "$@"
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to checked GPU IDs}"
: "${NGPU:?Set NGPU to the number of selected GPUs}"
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "$root/roboicl/run.py" "$@"
