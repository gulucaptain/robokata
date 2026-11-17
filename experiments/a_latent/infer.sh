#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${INFER_GPU:?Set an allocated inference GPU}"
: "${CHECKPOINT:?Set a local .pt or published adapter.safetensors path}"
: "${OUTPUT_JSON:?Set a new output JSON path}"
args=(--work "$WORK" --model "$MODEL" --empty-text "$EMPTY" --checkpoint "$CHECKPOINT"
      --pair-index "${PAIR_INDEX:-0}" --seed "${SEED:-1}" --denoise-steps "${DENOISE_STEPS:-50}" --output "$OUTPUT_JSON")
if [[ $CHECKPOINT == *.safetensors ]]; then
  config=${ADAPTER_CONFIG:-$(dirname "$CHECKPOINT")/adapter_config.json}
  if [[ ! -f $config && -z ${ADAPTER_CONFIG:-} ]]; then
    config=$(dirname "$(dirname "$CHECKPOINT")")/adapter_config.json
  fi
  args+=(--adapter-config "$config")
fi
export CUDA_VISIBLE_DEVICES=$INFER_GPU
exec "$PY" "$CODE/roboicl/infer_checkpoint.py" "${args[@]}"
