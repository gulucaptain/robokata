# π0.5 with RGB Demo conditioning on RoboICL3000

Training and inference implementation for
[gulucaptain/pi05-icl3000-demo](https://huggingface.co/gulucaptain/pi05-icl3000-demo).
This directory preserves the **107 source files hashed by the published run's
contract**, without changing the training implementation. The reference run is
`pi05-lora-cached-sdpa-b256-e50-20261114-02-f33d99c88aeb`; its published epoch-008
checkpoint contains 9,640 optimizer updates / 8 complete Query-frame epochs.
The configured budget was 50 epochs; this release does not claim completion of
that budget or a measured robot task success rate.

## Model and inputs

- Initialization: official OpenPI `gs://openpi-assets/checkpoints/pi05_base`,
  converted to BF16 PyTorch. See `reproduction/verification.json` for the base
  identity and the CPU comparison against the official JAX checkpoint.
- Independent top RGB Demo video sampled at 5 Hz, up to 60 frames; current
  Query top RGB and measured world-hand10 state. No task text or Demo
  actions/states are used. Keep the complete Demo within the 60-frame cap.
- Native π0.5 discrete state tokens and time-conditioned adaRMSNorm are retained.
  Output: 50 world-hand10 commands at 50 Hz, internally padded to 32 dimensions.
- LoRA rank 232, alpha 464: 396,166,816 trainable parameters, including the
  additional Demo, projector, action and time modules. Frozen SigLIP Demo
  features are cached; Query RGB remains encoded online. SDPA and the original
  training/checkpoint logic are included in the vendored OpenPI implementation.
- Reference training: 4 ranks × batch 256 = global batch 1,024, 1,205 updates
  per epoch, LR peak 2.5e-5 / end 2.5e-6, 1,000 warmup updates.

## Environment

Use Python 3.11 and a dedicated environment. The upstream dependency lock and
project files are in `vendor/openpi`; observed production versions are in
`environment-lock.json`. One installation route is:

```bash
# Run from baselines/pi05-demo. Requires uv and a compatible CUDA driver.
(cd vendor/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev)
ln -s vendor/openpi/.venv .venv
uv pip install --python .venv/bin/python h5py==3.13.0 safetensors==0.5.3 huggingface-hub==0.32.3
.venv/bin/python vendor/openpi/examples/segmented_demo/prepare_transformers.py --apply
```

The Transformers replacement is required; apply it only inside the dedicated
environment. The installation recipe has not been rerun on a clean machine;
the CPU tests below were run in the experiment's existing environment.

## Base checkpoint and data

`scripts/prepare_base.py --output /absolute/path/to/pi05-base-pytorch` downloads
the official JAX checkpoint and runs the included converter. It is an explicit
large download; training never invokes it automatically. Inference requires
the base matching the adapter's SHA256:

```text
abecb1fa2c618368b5aecbc0e10c546f64927077d88fea6afc7c564371781270
```

`data.py --help` describes conversion of source recordings and a frozen split
plan into `data-top`. Training expects `manifest.json`, `assets.json`, and
per-record `rgb.npy` / `supervision.npz`; see `data.py` for the exact schema.
Original recordings, prepared datasets, caches and model weights are not in
Git. The reference split/provenance is available beside the HF run's contract.
`reproduction/contract.json` preserves the actual server paths for provenance;
use `configs/reproduce.example.json` with your own absolute paths for a new run.

## Training and resume

Update the example configuration's dataset, normalization, tokenizer, base
checkpoint and cache paths. It defaults to no HF upload. To publish your own
run, set `hf_repo`, enable `upload_each_epoch`, and authenticate Hugging Face
outside the configuration file.

The reference config requires the completed Demo cache. The cache tools are
`scripts/build_demo_cache.py` and `scripts/finalize_demo_cache.py`; the original
cache launcher uses server-specific GPU assignments. Review and adapt those
assignments to your allocation before running it. Set `demo_cache` to null to
use the included RGB path instead; that is not the exact cached reference run.

```bash
# Choose four GPUs allocated to this experiment.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/reproduce.example.json "$PWD/runs/my-run"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/reproduce.example.json "$PWD/runs/my-run" --resume
```

Epoch checkpoints include model, optimizer, RNG and sampler state locally.
The optional publisher uploads all trainable parameters, tokenizer,
normalization, inference configuration, source and SHA receipts; optimizer
states remain local. Exact resume requires the original run contract and
local training state, not just the HF adapter.

## Inference with the published checkpoint

Download the checkpoint files without optimizer states:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'gulucaptain/pi05-icl3000-demo', local_dir='hf-model',
    allow_patterns=['runs/pi05-lora-cached-sdpa-b256-e50-20261114-02-f33d99c88aeb/checkpoints/epoch-008/*'],
)
PY
```

```python
from policy import Policy

checkpoint = 'hf-model/runs/pi05-lora-cached-sdpa-b256-e50-20261114-02-f33d99c88aeb/checkpoints/epoch-008'
policy = Policy(checkpoint, device='cuda',
                base_checkpoint='/absolute/path/to/pi05-base-pytorch/model.safetensors')
policy.set_demo(demo_rgb, timestamps_s, duration_s)
actions = policy.predict(query_rgb, measured_state10, num_steps=10)
```

`demo_rgb`: uint8 `[T,224,224,3]`, with a native 50 Hz clock starting at zero;
`duration_s` is the exclusive end (e.g. `T/50`). `query_rgb`: uint8
`[224,224,3]`; `measured_state10`: finite float32 `[10]` in the training
world-hand10 coordinate convention. The returned array is `[50,10]` in
denormalized action units. This is an action predictor; robot transport and
closed-loop execution are not included.

For the CLI, supply Demo NPZ keys `rgb`, `timestamp_s`, `duration_s`, and
Query NPZ keys `rgb`, `state`:

```bash
.venv/bin/python policy.py --checkpoint /absolute/path/to/epoch-008 \
  --base-checkpoint /absolute/path/to/pi05-base-pytorch/model.safetensors \
  --demo demo.npz --input query.npz --output actions.npz --device cuda
```

`Policy` checks the checkpoint SHA, base SHA, assets and every frozen source
hash. Keep these source files unchanged when using the published checkpoint.
The HF checkpoint contains trainable parameters only, so the base is required.

## Validation and provenance

All 18 CPU tests passed in the production environment: tiny-model
forward/backward, Demo conditioning, LoRA gradients and zero initialization,
adapter/optimizer roundtrip, cache/SDPA equivalence, prefetch ordering and
failure propagation, and publisher integrity/idempotence/failure handling.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -v
```

The vendored OpenPI license is retained in `vendor/openpi/LICENSE`. Upstream
code is locally adapted for this experiment; see `reproduction/contract.json`
for exact file hashes. Upstream model, Gemma and dataset terms still apply.
