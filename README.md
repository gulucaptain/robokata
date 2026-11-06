# RoboKata · Robot-ICL + A

This source snapshot reproduces the active Version A training recipe. It contains the RoboICL preparation, DINOv2 cache, training and recorded-episode inference entry points, with the exact Zero-WAM training backend used by the run. The backend was adapted from The-Imitator-Game commit `3c3404a` and Zero-WAM commit `08e2c4a`. The backend already includes the A attention patch.

Use a Python 3.10 CUDA environment with PyTorch 2.9 and the dependencies listed in `requirements-server.txt`. Activate that environment and set `PYTHONPATH` to this repository's `backend/` directory. Inputs and outputs go outside the source directory. Supply the verified RoboICL release, base Zero-WAM weights, the empty text embedding and DINOv2 weights separately.

```bash
ROOT="$(pwd -P)"
PY=python
export PYTHONPATH="$ROOT/backend"
# Set these to existing absolute paths, outside RoboKata.
# DATA, WORK, MODEL, EMPTY_TEXT, DINO, DEMO_CACHE, RUN
```

## Prepare

Reuse the original verified `WORK` and `DEMO_CACHE` when continuing an existing run. To prepare a new work directory, run:

```bash
"$PY" "$ROOT/roboicl/prepare.py" plan --root "$DATA" --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" stats --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" cache --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/prepare.py" verify --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/demo_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$DEMO_CACHE" --device cuda:0 --bf16
```

For parallel extraction, pass `--shards N --shard I` to each cache worker. VAE workers write independent files and `verify` follows the last worker. For the DINOv2 cache, merge by running `demo_features.py` with `--shards N` and no `--shard` after all parts finish. Use the same encoder and precision for all parts and the merge.

## Train

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$RUN" --demo-features "$DEMO_CACHE" \
  --epochs 20 --batch-per-gpu 16 --warmup-steps 0 --learning-rate 5e-5 \
  --cuda-cache-gib 120 --initialize-from "$BASELINE_CHECKPOINT"
```

The launcher accepts the same arguments as `roboicl/train.py` plus `--cuda-cache-gib`. The CUDA cache flag creates a reusable allocator pool and suppresses periodic cache purges. Set it only on GPUs allocated to the run. The default is zero. For an existing Version A run, use `--resume-from "$RUN/checkpoints/latest.pt"` in place of `--initialize-from` and keep the original seed, rank count, batch, schedule and cache. `--initialize-from` loads a compatible Robot-ICL baseline adapter and starts a new optimizer. Each completed epoch gets a separate checkpoint and `latest.pt` is a rolling restart point.

## Infer

```bash
CUDA_VISIBLE_DEVICES="${INFER_GPU:-0}" "$PY" "$ROOT/roboicl/infer_checkpoint.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --checkpoint "$RUN/checkpoints/epoch_001.pt" --demo-features "$DEMO_CACHE" \
  --pair-index 0 --output "$OUTPUT_JSON"
```

The inference entry point generates the first 16 actions for a held-out recorded pair. Add `--disable-demo-attention` to compare A-off with the same checkpoint, pair and seed. This is a recorded-episode inference entry point.

`roboicl/publish_epochs.py` is a separate, opt-in publisher for epoch adapters. It requires a Hugging Face login and an existing public model repository. Running the training launcher does not invoke it.

## Original baseline

See [baseline/README.md](baseline/README.md) for the first Robot-ICL run and its separate, unmodified backend.
