#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ $# == 0 ]] || { echo 'This fixed recipe accepts no extra flags; resume needs a separate explicit command.' >&2; exit 1; }
: "${TRAIN_GPUS:?Set four allocated GPU IDs after checking current ownership}"
IFS=, read -ra devices <<< "$TRAIN_GPUS"
[[ ${#devices[@]} == 4 && $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -eq 4 ]] || { echo 'Expected four distinct GPUs; batch 16 each, global 64.' >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo 'Use a new run directory for common-pretrain initialization.' >&2; exit 1; }
# CPU-only validation before any GPU initialization; no automatic publisher.
CUDA_VISIBLE_DEVICES='' "$PY" "$CODE/experiments/a_latent/preflight.py" --run-dir "$RUN_DIR" --recipe "$RECIPE"
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
exec "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CODE/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" --empty-text "$EMPTY" \
  --run-dir "$RUN_DIR" --demo-latents --lora-rank 98 --demo-dropout 0.3 \
  --batch-per-gpu 16 --epochs 20 --learning-rate 1e-4 --warmup-steps 100 \
  --seed 1 --num-workers 2 --save-epoch-freq 1 --cuda-cache-gib 120
