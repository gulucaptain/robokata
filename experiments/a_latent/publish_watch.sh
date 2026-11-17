#!/usr/bin/env bash
# Install as BASE/publisher/supervise.sh. It never imports or changes training.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$HERE/env.sh" ]]; then
  source "$HERE/env.sh"
  EXPORTER="$CODE/roboicl/publish_epochs.py"
else
  # Also supports an isolated publisher-only installation under BASE/publisher.
  BASE=$(cd "$HERE/.." && pwd)
  PY=${PY:-/data/haoyuzhao/models/weights/lingbot-va/.venv/bin/python}
  RUN_DIR=${RUN_DIR:-$BASE/run}
  HF_REPO=${HF_REPO:-gulucaptain/RoboKata-A-latent-21860}
  EXPORTER="$HERE/publish_epochs.py"
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$BASE/tmp"
export HF_HUB_CACHE="$BASE/hf-cache"
export HF_XET_CACHE="$BASE/hf-xet-cache"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export PYTHONPATH="$(dirname "$EXPORTER")${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$BASE/hf-publish"
while true; do
  echo "Publisher start $(date -u +%FT%TZ)"
  set +e
  nice -n 10 ionice -c 2 -n 7 "$PY" -u "$EXPORTER" \
    --run "$RUN_DIR" --state-dir "$BASE/hf-publish" \
    --repo "$HF_REPO"
  status=$?
  set -e
  if [[ $status == 0 ]]; then
    echo 'All 20 epochs published and verified; supervisor complete.'
    exit 0
  fi
  echo "Publisher exited with status $status; retrying in 60 seconds."
  sleep 60
done
