"""Cross-attention LoRA for the RoboICL visual context."""

import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from safetensors import safe_open
from safetensors.torch import save_file

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
ADAPTER_FILE = "roboicl_lora.safetensors"
BASE_REVISION = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5), generator=generator)

    def forward(self, inputs):
        hidden = nn.functional.linear(inputs, self.lora_a.to(inputs.dtype))
        return self.base(inputs) + self.scale * nn.functional.linear(hidden, self.lora_b.to(inputs.dtype))


def targets(model):
    for index, block in enumerate(model.blocks):
        for path in TARGETS:
            prefix = f"blocks.{index}.attn2.{path}"
            parent_path, _, name = prefix.rpartition(".")
            parent = model.get_submodule(parent_path)
            yield prefix, parent, name, getattr(parent, name)


def install_lora(model, rank, alpha, seed):
    model.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    for _, parent, name, linear in targets(model):
        setattr(parent, name, LoRALinear(linear, rank, alpha, generator))


def save_lora(model, directory, rank, alpha, is_main):
    state = {}
    for prefix, _, _, layer in targets(model):
        for key in ("lora_a", "lora_b"):
            weight = getattr(layer, key).detach()
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            if is_main:
                state[f"{prefix}.{key}"] = weight.cpu().contiguous()
    if is_main:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ADAPTER_FILE
        temporary = path.with_suffix(".tmp")
        save_file(state, temporary, metadata={
            "rank": str(rank), "alpha": str(alpha), "base_revision": BASE_REVISION,
        })
        temporary.replace(path)


@torch.no_grad()
def merge_lora(model, path):
    layers = list(targets(model))
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata["base_revision"] != BASE_REVISION:
            raise ValueError("RoboICL LoRA was trained against another base revision")
        rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
        expected = {f"{prefix}.{key}" for prefix, _, _, _ in layers
                    for key in ("lora_a", "lora_b")}
        if set(checkpoint.keys()) != expected:
            raise ValueError("RoboICL LoRA checkpoint targets do not match the model")
        for prefix, _, _, layer in layers:
            a = checkpoint.get_tensor(f"{prefix}.lora_a")
            b = checkpoint.get_tensor(f"{prefix}.lora_b")
            if a.shape != (rank, layer.in_features) or b.shape != (layer.out_features, rank):
                raise ValueError(f"RoboICL LoRA shape mismatch at {prefix}")
            merged = torch.addmm(layer.weight.float(), b.float(), a.float(), alpha=alpha / rank)
            layer.weight.copy_(merged)


@torch.no_grad()
def load_lora(model, path, rank, alpha):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if (metadata["base_revision"], int(metadata["rank"]), float(metadata["alpha"])) != (
            BASE_REVISION, rank, alpha,
        ):
            raise ValueError("Restart LoRA configuration differs from the checkpoint")
        for prefix, _, _, layer in targets(model):
            for key in ("lora_a", "lora_b"):
                getattr(layer, key).copy_(checkpoint.get_tensor(f"{prefix}.{key}"))
