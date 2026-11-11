# RoboKata mixed 25k

**11.11 training release.** Epoch weights are published to
[gulucaptain/RoboKata-1111](https://huggingface.co/gulucaptain/RoboKata-1111).
For checkpoint download and evaluation commands, see
[the 11.11 inference guide](docs/EVALUATION-1111.md).
The actual run settings and normalization are in [experiments/1111](experiments/1111).

This independent variant trains A+B1+Foresight Indexing from Zero-WAM pretrain.
It uses all 12,000 RoboKata v2.1 official pairs and 13,000 selected old training
pairs, yielding 19,000 Franka-to-Franka and 6,000 ALOHA-to-Franka pairs.
Each namespaced recording has its original path and SHA-256. Old and new
same-name RGB recordings have separate cache identities.

Demo VAE inputs use external/head camera 0 at 224×224. ALOHA HDF5 decoding
reads only head-camera JPEGs. Demo DINO uses 8 frames × 4 cells = 32 tokens
of dimension 780. Queries use external + left wrist at 224×448.
B1 and key regression supervise both query views. Foresight matching compares
the query external-camera cells to the external-only demo cells.

The independent evaluation split contains 1,200 frozen pairs. The old validation
split overlaps the new training data and is excluded. Inference predicts the
first 16 recorded-episode actions. Closed-loop evaluation needs simulator integration.

## Environment

Use the existing PyTorch 2.9 training environment and requirements-server.txt,
including h5py 3.15.1. ROOT is this directory. Set these paths for the experiment.

```bash
ROOT=/absolute/path/to/RoboKata-mixed25k
PY=/path/to/training/python
export PYTHONPATH="$ROOT/backend"
DATA=/data/haoyuzhao/exps/train/robokata-dataset-v21-20261111/data
MIXED=/data/haoyuzhao/exps/train/robokata-dataset-v21-20261111/train25000
WORK=/absolute/path/to/new/mixed25k-cache
RUN=/absolute/path/to/new/mixed25k-run
MODEL=/data/haoyuzhao/envs/video/zero-wam-ig10k/weights/zero-wam-pretrain
DINO=/data/haoyuzhao/models/dinov2-base
EMPTY_TEXT=/home/zhaohaoyu/code/zero-wam-ig10k/Zero-WAM/wan_va/assets/empty_text_emb.pt
```

## Data and caches

```bash
"$PY" "$ROOT/roboicl/prepare.py" plan --root "$DATA" --mixed-manifest "$MIXED" \
  --work "$WORK" --dual-view --demo-top-view
"$PY" "$ROOT/roboicl/check_data.py" --work "$WORK" --workers 8 --output "$WORK/data-check.json"
"$PY" "$ROOT/roboicl/prepare.py" stats --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" cache --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/prepare.py" verify --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/demo_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/demo-dino.pt" --device cuda:0 --bf16
"$PY" "$ROOT/roboicl/semantic_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/query-dino.pt" --device cuda:0 --bf16
"$PY" "$ROOT/roboicl/semantic_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/query-dino.pt" --verify
```

VAE extraction uses the GPU exposed by the caller, or --device cpu --threads 16.
VAE workers accept --shards N --shard I and lock individual cache files.
Both DINO extractors accept --shards N --shard I. Merge using the same command
with --shards N and no --shard, keeping the same encoder and precision.

Full caches include 6,380 demo latents and 6,000 query latents. Only training
queries supply normalization statistics and B1 targets. The source check covers
all 6,680 source records, NPZ headers/actions, 50 Hz timing, ALOHA JPEG decoding,
and representative NPZ camera decoding. Source hashes are checked by statistics
fitting and cache extraction. VAE verification checks source identity, role,
encoder identity, shape, dtype, finite values and action cadence.

## Train

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:?Set the four allocated GPUs}" \
"$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$RUN" \
  --demo-features "$WORK/demo-dino.pt" --demo-dropout 0.3 \
  --semantic-features "$WORK/query-dino.pt" --semantic-weight 0.5 \
  --foresight-indexing --key-weight 0.5 --batch-per-gpu 16 \
  --epochs 20 --save-epoch-freq 1 --warmup-steps 0 --learning-rate 5e-5 \
  --seed 1 --num-workers 2 --cuda-cache-gib 120
```

The optimizer, scheduler implementation, model adaptations, BF16 precision and
activation checkpointing are inherited from the current dual-view experiment.
The command preserves four ranks, per-GPU batch 16, effective batch 64, 20 epochs,
learning rate 5e-5, warmup 0, demo dropout 0.3, and auxiliary weights 0.5.

The inherited sampler groups exact temporal shapes and drops incomplete batches.
This manifest yields 368 updates per epoch, 23,552 presentations per epoch and
7,360 updates over 20 epochs. The manifest contains 25,000 pairs. The data-check
report records the retained embodiment counts at seed 1, epoch 0.
Resume with the same caches, world size, batch and options using --resume-from.
A CPU --dry-run is available after the full caches have been built and verified.

## Infer and publish

```bash
"$PY" "$ROOT/roboicl/infer_checkpoint.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --demo-features "$WORK/demo-dino.pt" \
  --checkpoint "$RUN/checkpoints/latest.pt" --pair-index 0 --output "$OUTPUT_JSON"
"$PY" "$ROOT/roboicl/publish_epochs.py" --run "$RUN" \
  --state-dir "$UPLOAD_STATE" --repo "$HF_REPO"
```

Inference selects a Franka or ALOHA external-only demo and dual-view Franka
query from frozen evaluation. It validates cache identities and the raw query
hash. B1 target features are used during training only.
Run the publisher alongside training with a dedicated state directory and
destination repository. It uploads each completed epoch and verifies remote
size and SHA-256. Safetensors inference additionally requires --adapter-config
and the matching adapter.json receipt beside the epoch adapter.
