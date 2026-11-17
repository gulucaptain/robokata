# A_latent: action-to-demonstration latent reading

This experiment trains an action readout over the same demonstration video
input patch embeddings used by Zero-WAM's video branch. Semantic foresight (B)
and foresight indexing (FI) are disabled; original latent MCP remains enabled.

Weights and per-epoch integrity receipts:
[gulucaptain/RoboKata-A-latent-21860](https://huggingface.co/gulucaptain/RoboKata-A-latent-21860).
The base is `Robbyant-Research/zero-wam-pretrain`, revision
`7040c4195df216c900334ef62d5fdcf05c0601aa`.

## Model and training recipe

| Setting | Value |
| --- | --- |
| Initialization | Shared Zero-WAM pretrain; no A/B/FI warm-start |
| Dataset | Original Franka-only RoboICL train3000 |
| Training / validation pairs | 21,860 / 260 |
| Demo / query RGB size | 320x480 / 224x288, external camera only |
| Topology | Four GPUs, 16 trajectories per GPU, global batch 64 |
| Epochs / optimizer updates | 20 / 6,600 (330 per epoch) |
| LoRA rank / alpha | 98 / 196 |
| Trainable parameters | 471,342,625 (460,113,920 in LoRA factors) |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01 |
| Learning rate | 1e-4; 100 warmup updates; cosine decay to 1e-6 |
| Seed / gradient clipping | 1 / 1.0 |
| Demo readout dropout | 0.3 |
| Noise conditioning probability | 0.5 (original A-only recipe) |
| Precision / checkpointing | BF16 / activation checkpointing |

Three gated cross-attention modules at zero-based blocks 7, 15 and 23 use
width 256 and four heads. Gates start at zero. K/V input width is 3072, rather
than DINO's 780. Training reuses `icl_hs` before any backbone block; inference
uses the same video patch embedding. No pooling, additional positional features
or extra temporal subsampling is applied to this readout. Dropout zeros only
the additional action readout, preserving video ICL.

The target trainable budget is the rank-106 A+B+FI model's 470,597,413 parameters.
A_latent rank 98 is 0.158% above that target, within the enforced 0.2% tolerance;
rank 97 would be 0.839% below. State/action heads and demo attention are also
trained; the backbone is frozen. Parameter counts and budget differences are
saved in `trainable_parameters.json`. Other DINO modes retain default rank 106.
Report rank and actual parameter count when comparing these variants.

## Data and environment

Use Python 3.10 and the CUDA PyTorch 2.9 environment in
`requirements-server.txt`. Install the CUDA-specific dependencies for the host;
the requirements file records the reference server environment. Put model
weights, data, caches and run outputs outside this source checkout.

The original split contains 3,000 Franka recordings, with 450 training families
and 50 held-out families. There are 2,700 training query recordings and 2,700
training demonstration recordings. Three held-out families have no surviving
non-self pair. Preparation excludes 3,000 self-pairs and 4,880 cross-split pairs.
This is distinct from mixed25k and its 19,000-pair Franka subset.

A_latent checks counts, Franka identities, pair uniqueness, task alignment and
family separation. It rejects mixed data rather than silently filtering it.
Normalization is fitted only on unique training query recordings. Reuse the
same verified plan, VAE cache and statistics as the A_DINO control.

With exact-shape batching, each epoch presents 21,120 of the 21,860 training
pairs and drops 740; the retained subset is shuffled across epochs. Changing
batch size or topology changes the sampling/update schedule.

The original A/B1 cache plan SHA-256 is
`9218a390ea2a6e627f81ceef58f8fc0006ffa7171bea85115241312aac397ca1`.
Paths and hashes in `experiments/a_latent/recipe.json` describe this reference
cache. Moving/rebuilding data requires regenerating the plan, statistics and
verification receipt; the trainer does not bypass their identity checks.
Use `roboicl/prepare.py` with split seed `20261105`, validation fraction `0.10`,
and the original single-camera defaults if preparing the verified train3000
release again. Do not use `--mixed-manifest`, `--dual-view` or `--demo-top-view`.
No DINO cache is needed for A_latent.

## Train and resume

Reference-server launch:

```bash
export A_LATENT_BASE=/absolute/path/to/new-experiment
export TRAIN_GPUS=4,5,6,7  # Explicitly allocated GPUs only.
bash experiments/a_latent/train.sh
```

`env.sh` supplies the reference server's model/cache paths. Override `PY`,
`WORK`, `MODEL` and `EMPTY` for another installation. The fixed CPU preflight
pins the reference plan hash and model geometry; provide a customized
`--recipe` JSON to preflight when relocating an otherwise verified cache.
The generic CLI below accepts verified paths without assuming server locations:

```bash
export PYTHONPATH="$ROOT/backend:$ROOT/roboicl"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=4 "$ROOT/roboicl/train_entry.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY" --run-dir "$RUN_DIR" \
  --demo-latents --lora-rank 98 --demo-dropout 0.3 \
  --batch-per-gpu 16 --epochs 20 --learning-rate 1e-4 --warmup-steps 100 \
  --seed 1 --num-workers 2 --save-epoch-freq 1
```

On the reference H200 server, `train.sh` additionally uses
`--cuda-cache-gib 120`. This retains a reusable allocator cache, reserving it
at distributed initialization and restoring it after model/optimizer loading.
It is not an exclusive allocation or a live-tensor requirement. On GPUs that
cannot support this reservation, use the generic command without that flag.
Temporary/compiler caches are placed under the experiment directory on the
reference server. No script stops existing jobs or chooses GPUs automatically.

To resume on the same four-rank topology:

```bash
export TRAIN_GPUS=4,5,6,7
bash experiments/a_latent/resume.sh
```

Alternatively append `--resume-from "$RUN_DIR/checkpoints/latest.pt"` to the
same generic training command. Resume restores weights, optimizer, scheduler,
RNG and sampler cursor. Do not use `--initialize-from` for continuation.
Rank-106 A/B/FI adapters cannot directly initialize rank-98 tensors. Valid
checkpoints survive an incomplete atomic save, but disk capacity must cover
both the previous checkpoint and the temporary replacement. Each full training
checkpoint is about 5.66 GB; each published adapter is about 1.89 GB. Historical
optimizer checkpoints require separate retention planning; publication does
not delete them automatically.

## Download weights and infer

Download one epoch, including its receipt and configuration:

```bash
export ADAPTER_DIR=/absolute/path/to/adapters
export EPOCH=006  # Choose an epoch present in the Hugging Face repository.
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'gulucaptain/RoboKata-A-latent-21860',
    local_dir=os.environ['ADAPTER_DIR'],
    allow_patterns=['adapter_config.json', f"epoch-{os.environ['EPOCH']}/*"],
)
PY
```

Generate an action chunk for a held-out pair:

```bash
export PYTHONPATH="$ROOT/backend:$ROOT/roboicl"
CUDA_VISIBLE_DEVICES="$INFER_GPU" "$PY" "$ROOT/roboicl/infer_checkpoint.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY" \
  --checkpoint "$ADAPTER_DIR/epoch-$EPOCH/adapter.safetensors" \
  --adapter-config "$ADAPTER_DIR/epoch-$EPOCH/adapter_config.json" \
  --pair-index 0 --seed 1 --denoise-steps 50 --output "$OUTPUT_JSON"
```

`experiments/a_latent/infer.sh` is an equivalent wrapper. The checkpoint
identifies latent reading and rank 98 automatically; omit `--demo-features`.
For a local `.pt` checkpoint omit `--adapter-config`. The loader validates
size/SHA-256, demo schema, plan/statistics and cache receipt identity. A local
training checkpoint also requires the original base-model path. Published
adapters contain all trainable tensors but require the frozen base and the
matching integration code.

The result is a finite `[16,10]` recorded-episode action chunk in absolute
world-hand coordinates at 50 Hz. This entry point does not execute a robot
or simulator and does not measure closed-loop task success. `--disable-demo-attention`
disables the extra readout while preserving video ICL.

Training logs record the actual demonstration token count and its ratio to
32 DINO tokens. This count varies with demonstration length; summarize the
evaluation demonstrations before reporting the paper's token reduction factor.

## Automatic epoch publication

The dedicated repository is
[gulucaptain/RoboKata-A-latent-21860](https://huggingface.co/gulucaptain/RoboKata-A-latent-21860).
With existing write credentials and publication authorization:

```bash
bash experiments/a_latent/publish_watch.sh
```

Run this in a separate persistent session. It uses CPU only, two compute/upload
threads, nice 10 and low I/O priority. No training code or process is changed.
Each completed epoch receives `adapter.safetensors`, `adapter.json` and
`adapter_config.json`. Root files contain the model card and the same contract.
The publisher checks the weight manifest/shapes, remote file size and LFS
SHA-256, and receipt/configuration contents. Only verified epochs enter
`hf-publish/verified-uploads.json`. Restart rechecks previously published epochs
and rejects conflicting weights, repository identities or run contracts.
Network failures are retried. Optimizer/RNG state and frozen base weights are
not uploaded.

## Validation and provenance

```bash
PYTHONPATH="$ROOT/backend:$ROOT/roboicl" \
  "$PY" -m unittest discover -s "$ROOT/validation" -p test_a_latent.py -v
```

Twelve CPU tests cover readout routing, zero gates/dropout, gradients, original
MCP forward/backward, LoRA budget, serialization and upload restart checks.
The native small-model test replaces only CUDA sparse kernels with CPU dense
SDPA while retaining actual mask predicates and backbone/MCP blocks. The source
also passed server PyTorch 2.9.0+cu126 CPU checks and actual four-H200 training
updates with batch 16. Full-size inference and closed-loop evaluation have not
been validated as part of this source release.

`validation/a_latent_source_sha256.json` records 15 core source hashes checked
against the running training/inference snapshot and isolated epoch publisher
on 2026-11-17. Launch wrappers are reusable versions; one-shot process handoff
scripts and dynamic server logs are not part of the release.
