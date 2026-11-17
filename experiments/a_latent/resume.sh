#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ $# == 0 ]] || { echo 'Use the training CLI for custom resume options.' >&2; exit 1; }
: "${TRAIN_GPUS:?Set the same four allocated GPUs for resuming this run}"
IFS=, read -ra devices <<< "$TRAIN_GPUS"
[[ ${#devices[@]} == 4 && $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -eq 4 ]] || { echo 'Resume requires four distinct GPUs.' >&2; exit 1; }
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/latest.pt}
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
exec "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CODE/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" --empty-text "$EMPTY" \
  --run-dir "$RUN_DIR" --demo-latents --lora-rank 98 --demo-dropout 0.3 \
  --batch-per-gpu 16 --epochs 20 --learning-rate 1e-4 --warmup-steps 100 \
  --seed 1 --num-workers 2 --save-epoch-freq 1 --cuda-cache-gib 120 \
  --resume-from "$CHECKPOINT"
