# 11.11 checkpoint inference

The repository root is the mixed25k A+B1+Foresight Indexing implementation used
by the 11.11 run. Demo input is **top/external only** (including ALOHA head),
while query input is **top/external + left wrist**. The old A/B1 instructions
are archived in `docs/README-legacy-a-b1.md`; `baseline/` is a historical baseline.

Weights: <https://huggingface.co/gulucaptain/RoboKata-1111>.
Select an epoch whose `epoch-NNN/adapter.safetensors` and `adapter.json` both
exist. Training publishes and verifies one adapter per completed epoch.

## Use the existing 11.11 caches on server T

Run on a GPU allocated to inference. The training run uses GPUs 4–7; check
current occupancy before choosing a device. A new checkout does not need to
modify the running training snapshot.

```bash
ROOT=/absolute/path/to/robokata
BASE=/data/haoyuzhao/exps/train/robokata-1111
PY=/data/haoyuzhao/models/weights/lingbot-va/.venv/bin/python
export PYTHONPATH="$BASE/deps:$ROOT/backend"
WORK="$BASE/cache"
MODEL=/data/haoyuzhao/envs/video/zero-wam-ig10k/weights/zero-wam-pretrain
EMPTY_TEXT=/home/zhaohaoyu/code/zero-wam-ig10k/Zero-WAM/wan_va/assets/empty_text_emb.pt
export ADAPTER_DIR=/absolute/path/to/downloaded-1111-adapters
export EPOCH=003
OUTPUT_DIR=/absolute/path/to/new-evaluation-results
mkdir -p "$OUTPUT_DIR"
```

Download the adapter with its integrity receipt and model contract:

```bash
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
epoch = os.environ["EPOCH"]
if len(epoch) != 3 or not epoch.isdigit() or not 1 <= int(epoch) <= 20:
    raise ValueError("EPOCH must be 001 through 020")
snapshot_download(
    "gulucaptain/RoboKata-1111",
    local_dir=os.environ["ADAPTER_DIR"],
    allow_patterns=["adapter_config.json", f"epoch-{epoch}/*"],
)
PY
```

Run one held-out pair (index 0–1199). `INFER_GPU` is a physical GPU you have
checked and allocated. Each command loads one model and emits a new JSON file.

```bash
CUDA_VISIBLE_DEVICES="${INFER_GPU:?Set an allocated inference GPU}" \
"$PY" "$ROOT/roboicl/infer_checkpoint.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --demo-features "$WORK/demo-dino.pt" \
  --checkpoint "$ADAPTER_DIR/epoch-$EPOCH/adapter.safetensors" \
  --adapter-config "$ADAPTER_DIR/adapter_config.json" \
  --pair-index 0 --seed 1 --denoise-steps 50 \
  --output "$OUTPUT_DIR/epoch-$EPOCH-pair-0000-seed-1.json"
```

For a local training checkpoint, replace `--checkpoint` with the absolute
`run/checkpoints/epoch_003.pt` path and **omit** `--adapter-config`. The `.pt`
loader also requires the original training base-model path. Published
Safetensors are the portable checkpoint format. Use only trusted `.pt` files.

## What the output measures

The JSON records the pair, optimizer step, seed, denoising steps and a finite
`[16,10]` action chunk in the original `roboicl_world_hand10_absolute`
representation, at 50 Hz (0.32 seconds). It conditions on the recorded initial
query observation and state. This entry point does **not** execute actions in a
simulator or on a robot, and does not compute closed-loop task success.

Compare epochs on the same frozen 1,200-pair evaluation split, pair indices,
seeds and 50 denoising steps. Keep Franka-demo and ALOHA-demo results separate.
The previous validation split overlaps this version's training set and must
not be used to claim held-out performance. An offline action-error analysis
needs aligned recorded query actions; it cannot establish task success.

`--disable-demo-attention` disables the extra demo attention and foresight
matching while retaining the VAE demonstration context. It is an A-off
ablation, **not** a no-demonstration baseline. The inference script resets the
noise seed after module construction for this comparison.

## Cache and version requirements

Inference checks the adapter receipt's size/SHA-256, camera schema, demo feature
hash, VAE receipt, plan, normalization and selected source identities. Use the
original verified `WORK` on T. Copying or regenerating caches on another
machine requires the preparation and verification flow in the root README;
receipt timestamps and absolute recording paths make a raw cache copy
insufficient. Do not bypass these checks to combine older caches or statistics
with 11.11 weights.

`experiments/1111/` contains the actual launch configuration, normalization,
sampling contract and validation receipts. `version.json` describes cache
generation GPUs 0–3; the training GPUs are 4–7. The source file hashes in
`validation/training-source-sha256.json` identify the exact training snapshot;
release documentation and provenance files were added afterward. Training
modules and `roboicl/infer_checkpoint.py` match that snapshot.

Validation for this source release: all 95 original files matched server
SHA-256 values, Python syntax checks and inference CLI import passed. The run
completed its first CUDA optimizer update and its first checkpoint was read
back with finite model/optimizer tensors. A full CUDA inference or closed-loop
evaluation was not run as part of publishing this source release.
