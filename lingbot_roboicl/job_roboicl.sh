#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 ABS_WORK ABS_RUN [EPOCHS] [--resume]" >&2
    exit 2
fi
work=$1 run=$2 epochs=20
shift 2
if [[ ${1:-} =~ ^[1-9][0-9]*$ ]]; then
    epochs=$1
    shift
fi
if [[ "$work" != /* || "$run" != /* || $# -gt 1 || ( $# -eq 1 && "$1" != --resume ) ]]; then
    echo "Use absolute paths and optionally pass --resume" >&2
    exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to your allocated GPUs}"
: "${NGPU:?Set NGPU to the number of selected GPUs}"
accumulation=${ACCUMULATION:-16}
mkdir -p "$root/logs/jobs" "$(dirname "$run")"
exec 9>"$run.lock"
flock -n 9 || { echo "This run already has an active job" >&2; exit 1; }
log_dir=$(mktemp -d "$root/logs/jobs/$(basename "$run").XXXXXXXX")
echo "Job logs: $log_dir"
updates=$(PYTHONPATH="$root/roboicl" .venv/bin/python -c '
import json, sys
from pathlib import Path
from epoch_state import epoch_updates
work, world, accumulation, epochs = sys.argv[1:]
assert int(epochs) > 0
plan = json.loads((Path(work) / "plan.json").read_text())
print(epoch_updates(len(plan["pairs"]["train"]), int(world), int(accumulation)))
' "$work" "$NGPU" "$accumulation" "$epochs")
first_gpu=${CUDA_VISIBLE_DEVICES%%,*}
upload_pid=
cleanup() {
    code=$?
    if [[ -n "$upload_pid" ]]; then
        kill "$upload_pid" 2>/dev/null || true
        wait "$upload_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$code" > "$log_dir/exit"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
./publish_roboicl.sh --run "$run" --epochs "$epochs" --steps-per-epoch "$updates" \
    --wait-seconds "${UPLOAD_WAIT_SECONDS:-86400}" > "$log_dir/upload.log" 2>&1 &
upload_pid=$!
./run_roboicl.sh train --work "$work" --output "$run" --epochs "$epochs" \
    --accumulation "$accumulation" --save-interval "${RESTART_INTERVAL:-50}" \
    "$@" 2>&1 | tee "$log_dir/train.log"
checkpoint="$run/checkpoints/checkpoint_step_$((updates * epochs))"
if [[ ! -f "$run/inference.ok" ]]; then
    output="$run/inference/$(basename "$log_dir")"
    CUDA_VISIBLE_DEVICES="$first_gpu" NGPU=1 ./run_roboicl.sh infer \
        --work "$work" --checkpoint "$checkpoint" --output "$output" \
        2>&1 | tee "$log_dir/infer.log"
    .venv/bin/python - "$output" <<'PY'
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np

output = Path(sys.argv[1])
actions = np.load(output / "actions.npy", allow_pickle=False)
assert actions.shape == (16, 10) and np.isfinite(actions).all()
with imageio.get_reader(output / "demo.mp4") as video:
    assert video.get_data(0).size > 0
PY
    printf '%s\n' "$output" > "$run/inference.ok.tmp"
    mv "$run/inference.ok.tmp" "$run/inference.ok"
fi
upload_status=0
wait "$upload_pid" || upload_status=$?
upload_pid=
if (( upload_status )); then
    echo "Epoch upload failed; see $log_dir/upload.log" >&2
    exit "$upload_status"
fi
echo "Training, held-out inference and epoch uploads completed"
