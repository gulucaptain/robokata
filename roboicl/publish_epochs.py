"""Publish completed RoboICL epoch adapters to a public model repo."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from prepare import read_json, sha256, write_json


BASE = "Robbyant-Research/zero-wam-pretrain"
BASE_REVISION = "7040c4195df216c900334ef62d5fdcf05c0601aa"
CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# Robot-ICL + A

Version A adds three zero-initialized, gated cross-attention layers that
allow Zero-WAM's action branch to read tokens from a robot demonstration.
A frozen DINOv2 ViT-B/14 encodes eight RGB samples from the external camera.
Each sample contributes a 2×2 grid of pooled features with time and space
indices. Video ICL and the original MCP training objective are retained.

Training warm-starts from the Robot-ICL baseline at step 3,698, which uses
[Zero-WAM pretraining weights](https://huggingface.co/Robbyant-Research/zero-wam-pretrain)
(Apache-2.0, revision `7040c4195df216c900334ef62d5fdcf05c0601aa`).
The RoboICL fine-tuning and demo input use robot data. The inherited Zero-WAM
weights were pretrained with human demonstrations.

## Files

- `epoch-NNN/adapter.safetensors` contains the trainable tensors, including
  LoRA, state/action heads, MCP adapters and A's attention layers.
- `epoch-NNN/adapter.json` records the optimizer step, parameter count and
  file checksum.

To run inference, load the base Zero-WAM model and attach these tensors
with the matching Robot-ICL + A integration.

Training is configured for 20 epochs with 330 optimizer steps per epoch,
four GPUs, and a global batch of 64. Epoch snapshots are published after
the corresponding training checkpoint is saved. Closed-loop performance
has not been measured for these snapshots.
"""

B1_CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# Robot-ICL + A+B1

B1 replaces latent MCP denoising with a training-only semantic prediction
objective on frozen DINOv2 targets at future query latent anchors.
Inference uses the Version A architecture with the matching Zero-WAM base
model and Robot-ICL + A integration.

Each epoch folder contains the adapter tensors in Safetensors and an
upload receipt with their SHA-256. Closed-loop performance has not been
measured for these snapshots.
"""


INDEXED_CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# RoboICL + A+B1 + Foresight Indexing

These Safetensors contain RoboICL trainable adapter weights. See
`adapter_config.json` for the base revision, camera encoding, feature
schemas, and indexing settings. Each checkpoint has a SHA-256 receipt.
Load the indicated Zero-WAM base and the matching RoboKata integration.
The semantic future heads are training-only. The frame key heads, demo
attention, and indexing parameters are used at inference.
"""


LATENT_CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# RoboICL + A_latent

The action branch directly reads the demonstration's video input patch tokens
through three gated cross-attention modules. Original video/action flow matching
and latent MCP are retained. Semantic foresight and foresight indexing are off.
Use the matching A_latent integration and the verified VAE cache; no DINO cache
is needed. See adapter_config.json for the model and data contract.

The reference experiment starts from the shared Zero-WAM pretrain, uses
21,860 Franka-only training pairs, 20 epochs, batch 16 per GPU, LoRA rank 98,
and 471,342,625 trainable parameters (0.158% above the A+B1+FI reference).
The active recipe uses LR 1e-4, 100 warmup optimizer steps, cosine decay to
1e-6, seed 1 and demo dropout 0.3.
Each epoch folder contains adapter.safetensors (all trainable weights,
including heads and MCP adapters), adapter.json (step/count/SHA-256), and
adapter_config.json. Load the stated base revision with the matching A_latent
code. Optimizer and RNG state remain in the original training checkpoints.
Published adapters are not full standalone base models. Closed-loop success
has not been measured by the checkpoint exporter.
"""


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def adapter_contract(settings):
    config = {
        "base_model": BASE,
        "base_model_revision": BASE_REVISION,
        "lora_rank": settings["ig_lora_rank"],
        "lora_alpha": 2 * settings["ig_lora_rank"],
        "demo_attention": settings["ig_demo_attention"],
        "demo_dropout": settings["ig_demo_dropout"],
        "action_representation": settings["ig_action_representation"],
        "action_channels": settings["ig_roboicl_channels"],
        "actions_per_latent": settings["ig_action_per_frame"],
        "video_snr_shift": settings["snr_shift"],
        "action_snr_shift": settings["action_snr_shift"],
        "icl_rope_h": settings["icl_rope_h"],
        "enable_mcp": settings["enable_mcp"],
        "effective_batch": settings["ig_effective_batch"],
        "semantic_ifp": settings.get("ig_semantic_ifp"),
    }

    if settings.get("ig_demo_source") == "latent":
        if config["semantic_ifp"] or settings.get("ig_foresight_indexing"):
            raise ValueError("A_latent must be trained without B/FI")
        config.update(
            demo_source="latent",
            video_encoding=settings["ig_roboicl_encoding"],
            latent_receipt_sha256=settings["ig_roboicl_cache_sha256"],
            video_contract=settings.get("ig_roboicl_video_contract"),
            normalization=settings.get("ig_normalization"),
            data_contract=settings.get("ig_single_embodiment_data"),
            training={key: settings.get(key) for key in (
                "ig_total_epochs", "batch_size", "world_size", "num_steps",
                "learning_rate", "warmup_steps", "seed", "noisy_img_prob",
                "beta1", "beta2", "weight_decay", "max_norm", "ig_initialization",
                "ig_trainable_parameter_target", "ig_trainable_parameter_tolerance",
            )},
        )
    if settings.get("ig_foresight_indexing"):
        config.update(
            foresight_indexing=settings["ig_foresight_indexing"],
            video_encoding=settings.get("ig_roboicl_encoding", "wan_causal_external_rgb_bf16_mu_v1"),
            latent_receipt_sha256=settings["ig_roboicl_cache_sha256"],
            demo_features_sha256=settings["ig_demo_cache_sha256"],
            query_features_sha256=settings["ig_semantic_ifp_cache_sha256"],
        )
    return config


def export_epoch(checkpoint_path, weights_path, epoch, updates, config):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if ckpt["global_step"] != epoch * updates or ckpt["progress"]["epoch"] != epoch:
        raise ValueError(f"Checkpoint is outside the expected epoch {epoch} boundary")
    if adapter_contract(ckpt["native_config"]) != config:
        raise ValueError(f"Epoch {epoch} uses a different model contract")
    weights = ckpt["adapter_state_dict"]
    shapes = ckpt["adaptation"]["trainable_shapes"]
    if weights.keys() != shapes.keys() or any(list(weights[k].shape) != shapes[k] for k in weights):
        raise ValueError("Adapter tensors do not match the checkpoint parameter manifest")
    save_file({k: v.contiguous() for k, v in weights.items()}, weights_path)
    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(shapes):
            raise ValueError("Exported Safetensors keys do not match")
    return {
        "epoch": epoch,
        "global_step": ckpt["global_step"],
        "trainable_parameters": sum(v.numel() for v in weights.values()),
        "size_bytes": weights_path.stat().st_size,
        "sha256": sha256(weights_path),
    }


def remote_weights(api, repo, path):
    files = api.get_paths_info(repo, [path], repo_type="model")
    if not files:
        return None
    entry = files[0]
    return (entry.size, entry.lfs.sha256) if entry.lfs else None


def bind_upload_state(state_dir, repo, config, updates, epochs):
    """Prevent a reused state directory from silently skipping another run."""
    path = state_dir / "upload-contract.json"
    identity = dict(repo=repo, adapter_config=config, updates_per_epoch=updates, epochs=epochs)
    if path.exists():
        if read_json(path) != identity:
            raise ValueError("Upload state belongs to a different repository or training run")
    else:
        previous = state_dir / "verified-uploads.json"
        if previous.exists() and read_json(previous):
            raise ValueError("Existing upload state lacks a run identity; use a fresh state directory")
        write_json(path, identity)


def verify_remote_epoch(api, repo, folder, receipt, config):
    expected = (receipt["size_bytes"], receipt["sha256"])
    if remote_weights(api, repo, f"{folder}/adapter.safetensors") != expected:
        raise ValueError(f"Remote weight checksum differs: {folder}")
    for filename, value in (("adapter.json", receipt), ("adapter_config.json", config)):
        path = api.hf_hub_download(repo, f"{folder}/{filename}", repo_type="model")
        if read_json(Path(path)) != value:
            raise ValueError(f"Remote metadata differs: {folder}/{filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = read_json(args.run / "config.json")
    config = adapter_contract(settings)
    version = "A+B1" if config["semantic_ifp"] else "A"
    card = B1_CARD if config["semantic_ifp"] else CARD
    updates = read_json(args.run / "sampling_contract.json")["updates_per_epoch"]
    epochs = settings["ig_total_epochs"]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "verified-uploads.json"
    state = read_json(state_path) if state_path.exists() else {}
    api = HfApi()
    if api.whoami()["name"] != args.repo.split("/")[0] or api.model_info(args.repo, token=False).private:
        raise RuntimeError("Expected a public model repository owned by the signed-in user")
    indexed = bool(config.get("foresight_indexing"))
    portable = indexed or config.get("demo_source") == "latent"
    if portable:
        bind_upload_state(args.state_dir, args.repo, config, updates, epochs)
        version, card = ("A+B1 + Foresight Indexing", INDEXED_CARD) if indexed else ("A_latent", LATENT_CARD)
        files = set(api.list_repo_files(args.repo, repo_type="model"))
        if "adapter_config.json" in files:
            remote = api.hf_hub_download(args.repo, "adapter_config.json", repo_type="model")
            if read_json(Path(remote)) != config:
                raise ValueError("The remote adapter configuration belongs to another run")
        elif files - {".gitattributes", "README.md"}:
            raise ValueError("Use a dedicated model repository for this run")
        else:
            api.create_commit(
                repo_id=args.repo,
                operations=[
                    CommitOperationAdd("README.md", card.encode()),
                    CommitOperationAdd("adapter_config.json", json_bytes(config)),
                ],
                commit_message=f"Document Robot-ICL + {version} adapter release",
                num_threads=2,
            )
    elif not api.get_paths_info(args.repo, ["README.md"], repo_type="model"):
        api.create_commit(
            repo_id=args.repo,
            operations=[CommitOperationAdd("README.md", card.encode())],
            commit_message=f"Document Robot-ICL + {version} adapter release",
            num_threads=2,
        )
    print(f"Watching {epochs} epochs in {args.run}; {updates} updates per epoch", flush=True)
    frequency = settings.get("save_epoch_freq", 1)
    selected = sorted({*range(frequency, epochs + 1, frequency), epochs})
    for epoch in selected:
        key = f"{epoch:03d}"
        if key in state:
            if portable:
                # The local state is a receipt, not proof of current remote state.
                receipt = {k: v for k, v in state[key].items() if k != "path"}
                verify_remote_epoch(api, args.repo, f"epoch-{key}", receipt, config)
            continue
        filename = "final.pt" if epoch == epochs and epoch % frequency else f"epoch_{key}.pt"
        source = args.run / "checkpoints" / filename
        while not source.is_file():
            time.sleep(15)
        folder = f"epoch-{key}"
        destination = f"{folder}/adapter.safetensors"
        with TemporaryDirectory(dir=args.state_dir) as temporary:
            weights = Path(temporary) / "adapter.safetensors"
            receipt = export_epoch(source, weights, epoch, updates, config)
            expected = (receipt["size_bytes"], receipt["sha256"])
            attempt = 0
            while True:
                try:
                    current = remote_weights(api, args.repo, destination)
                    if portable and current is not None and current != expected:
                        raise ValueError(f"Existing indexed adapter conflicts: {destination}")
                    # Portable publication commits metadata with the weights,
                    # including recovery after a remote commit/local-state gap.
                    if current != expected or portable:
                        print(f"Uploading epoch {epoch} at step {receipt['global_step']}", flush=True)
                        api.create_commit(
                            repo_id=args.repo,
                            operations=[
                                CommitOperationAdd(destination, weights),
                                CommitOperationAdd(f"{folder}/adapter.json", json_bytes(receipt)),
                                *([CommitOperationAdd(f"{folder}/adapter_config.json", json_bytes(config))] if portable else []),
                            ],
                            commit_message=f"Publish Robot-ICL + {version} epoch {epoch}",
                            num_threads=2,
                        )
                    if remote_weights(api, args.repo, destination) != expected:
                        raise RuntimeError("Hugging Face file size or checksum does not match")
                    if portable:
                        verify_remote_epoch(api, args.repo, folder, receipt, config)
                    break
                except ValueError:
                    raise
                except Exception as error:
                    attempt += 1
                    print(f"Epoch {epoch} upload attempt {attempt}: {type(error).__name__}", flush=True)
                    time.sleep(min(30 * attempt, 300))
            state[key] = {**receipt, "path": destination}
            write_json(state_path, state)
            print(f"Verified epoch {epoch}: {destination}", flush=True)
    print("All epoch uploads verified", flush=True)


if __name__ == "__main__":
    main()
