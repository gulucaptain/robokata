#!/usr/bin/env bash
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-/data/haoyuzhao/models/weights/lingbot-va/.venv/bin/python}
BASE=${A_LATENT_BASE:-/data/haoyuzhao/exps/train/robokata-a-latent-20261116}
RUN_DIR=${RUN_DIR:-$BASE/run}
WORK=${WORK:-/data/haoyuzhao/envs/video/zero-wam-ig10k/cache/roboicl-full-v1}
MODEL=${MODEL:-/data/haoyuzhao/envs/video/zero-wam-ig10k/weights/zero-wam-pretrain}
EMPTY=${EMPTY:-/home/zhaohaoyu/code/zero-wam-ig10k/Zero-WAM/wan_va/assets/empty_text_emb.pt}
HF_REPO=${HF_REPO:-gulucaptain/RoboKata-A-latent-21860}
RECIPE=${RECIPE:-$CODE/experiments/a_latent/recipe.json}
export WORK MODEL EMPTY
export PYTHONPATH="$CODE/backend:$CODE/roboicl:$CODE:/data/haoyuzhao/exps/train/robokata-1111/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$BASE/tmp"
export TORCHINDUCTOR_CACHE_DIR="$BASE/torchinductor-cache"
export TRITON_CACHE_DIR="$BASE/triton-cache"
export IG_RUNTIME_CACHE_ROOT="$BASE/runtime"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
