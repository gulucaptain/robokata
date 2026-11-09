# RoboICL on LingBot-VA

Server T deployment root

```text
/data/haoyuzhao/models/weights/lingbot-va
```

## Install from this source bundle

On a Linux CUDA host with Python 3.10 and `uv`, choose a fresh absolute
deployment directory. The installer fetches the pinned LingBot-VA source
recorded in `SOURCE_REV`, applies the training and inference integration,
and installs the environment used for the run.

```bash
bash bootstrap.sh /absolute/path/to/new/deployment
cd /absolute/path/to/new/deployment
./fetch-weights.sh
.venv/bin/python roboicl/test_workflow.py
```

The RoboICL dataset, Wan VAE cache, model weights, credentials and run outputs
are external to this bundle. Point `WORK` at a verified RoboICL cache with
`plan.json`, `stats.json`, `verified-cache.json` and `latents/{demo,query}/`.
For uploads from a different host, set `HF_HOME` to the local Hugging Face
login directory. The handoff script is specific to the existing server T run.

The adapter reads the verified RoboICL Wan VAE cache in place. It preserves the
published directed demo-to-query pairs and the existing held-out initial-state
family split. It checks the cache receipt, statistics and VAE fingerprints
before running.

The demo contributes 128 visual context tokens to LingBot-VA's cross-attention
layers. The tokens come from eight evenly spaced latent frames, each pooled to
an 8×8 grid before the model's 2×2 patch embedding. The query contributes a
clean first latent and video and action targets. The 10D absolute world-frame
hand and gripper action is mapped to channels 0–8 and 28. The first action
group is masked, with later groups carrying 16 actions at 50 Hz. Measured state
and task text are outside this baseline's conditioning inputs.

## Inspect

```bash
cd /data/haoyuzhao/models/weights/lingbot-va
WORK=/data/haoyuzhao/envs/video/zero-wam-ig10k/cache/roboicl-full-v1
./run_roboicl.sh inspect --work "$WORK"
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

## Train

Set `TRAIN_GPUS` and `TRAIN_GPU_COUNT` after checking the shared GPU state.
Start with a fresh output directory. This command runs one update on
two query latents per rank and saves a model checkpoint. The model is
initialized from the official LingBot-VA base weights. The default uses
rank-8, alpha-16 LoRA on the query, key, value and output projections of each
cross-attention block. The frozen base runs in BF16. The default learning rate
is 1e-4. Use `--lora-rank`, `--lora-alpha` and `--learning-rate` to override
these settings. Pass `--lora-rank 0` for full fine-tuning.

```bash
RUN="$PWD/runs/roboicl-smoke-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NGPU="$TRAIN_GPU_COUNT" \
  ./run_roboicl.sh train --work "$WORK" --output "$RUN" \
  --max-frames 2 --steps 1 --save-interval 1 --warmup-steps 0 --workers 0
```

Use a new run directory and omit `--max-frames` for full trajectories. Set
`--accumulation` to control effective batch size. LoRA checkpoints contain only
`roboicl_lora.safetensors` under each `checkpoint_step_N` directory. Inference
loads the pinned base and merges the adapter before using it. Existing
full-model checkpoints can also be passed to the inference command.
The legacy step-only command above saves inference weights. Use the epoch job
below for optimizer-boundary restart checkpoints.

## Infer on a held-out pair

Set `INFER_GPU` to a GPU with available memory. The model receives the held-out
demo and the query's first latent. The output contains `demo.mp4`, a
`[16, 10]` action array in original units, and the selected pair metadata.
Use a distinct output directory and the same seed with `--demo-index` to
compare against another held-out demonstration.

```bash
CUDA_VISIBLE_DEVICES="$INFER_GPU" NGPU=1 ./run_roboicl.sh infer \
  --work "$WORK" --checkpoint "$RUN/checkpoints/checkpoint_step_1" \
  --output "$PWD/runs/roboicl-infer-$(date +%Y%m%d-%H%M%S)" --pair 0
```

A one-update checkpoint is for a mechanical smoke test. Evaluation of
demonstration use requires training and comparing matched and swapped
demonstrations on the held-out set.

## Run the full epoch job

Run inside tmux with GPUs assigned under the server's sharing rules.
The default accumulation is 16 microsteps per rank with microbatch 1.
Four ranks therefore use effective batch 64 and take 341 optimizer updates
per epoch. The default 20 epochs total 6,820 optimizer updates. Each epoch shuffles the 21,860 pairs and uses 21,824 of them.
The job starts a CPU uploader for the public model repo
`gulucaptain/lingbot-va-roboicl` and runs one held-out inference after training.
It checks that the action array is finite and the generated video decodes.

```bash
RUN="$PWD/runs/roboicl-lora-001"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NGPU=4 ./job_roboicl.sh "$WORK" "$RUN"
```

Each epoch and every 50 updates by default, a completed checkpoint is
published atomically to the run directory. The checkpoint contains the
LoRA adapter and per-rank optimizer, scheduler and RNG state. Uploads
include only the adapters and portable metadata for epoch checkpoints.
Training, upload and inference logs and the exit code are in the
job log directory printed at startup.

To restart after interruption, use the same run directory, epoch count,
GPU count, accumulation and restart interval. The latest completed checkpoint
is selected automatically and unfinished staging directories are ignored.
Updates after the last completed checkpoint are replayed. A completed
inference is reused if the job is restarted to retry an upload.

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NGPU=4 ./job_roboicl.sh "$WORK" "$RUN" --resume
```

For the server T allocator configuration used in the active run, set
`ROBOICL_RESERVE_GIB=121`. This is an optional per-rank cache target.
Use `RESTART_INTERVAL` to change the periodic checkpoint interval before
the initial run. Use `UPLOAD_WAIT_SECONDS` to change how long the uploader
waits for each epoch checkpoint. To retry uploads independently, use
the exact updates-per-epoch from the run's `roboicl.json`.

```bash
./publish_roboicl.sh --check
./publish_roboicl.sh --run "$RUN" --epochs 20 --steps-per-epoch 341
```

## Server T GPU 4–7 handoff

This handoff is specific to the existing RoboTwin run
`rank106-window16-e200`. It checks the old trainer fingerprint and live GPU
worker identities, requests a checkpoint and graceful exit through rank 0,
and verifies that the checkpoint advanced and the launcher exited cleanly.
It starts the 20-epoch job as soon as all four GPUs are empty and at least
95% of their memory is free. An unexpected GPU process aborts the handoff.
The local lock coordinates this handoff script on the host.

Use a fresh run directory. First run the read-only preflight. Use a tmux
session for the actual handoff and add `--execute` to the same command.

```bash
RUN="$PWD/runs/roboicl-lora-4to7-001"
.venv/bin/python roboicl/handoff.py --run "$RUN"
.venv/bin/python roboicl/handoff.py --run "$RUN" --execute
```

Epoch jobs save a local restart checkpoint after the first optimizer update
with finite gradients. Learning-rate warmup starts at a nonzero value.
Later restart checkpoints and public epoch uploads follow the usual schedule.
