"""Deterministic RoboICL epoch sampling and optimizer-boundary restart state."""

import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.data import Sampler

from lora import save_lora


def epoch_updates(size, world, accumulation):
    updates = size // (world * accumulation)
    if not updates:
        raise ValueError("Training split is smaller than one global batch")
    return updates


class EpochSampler(Sampler):
    def __init__(self, size, world, rank, accumulation, seed):
        self.size = size
        self.world = world
        self.rank = rank
        self.per_rank = epoch_updates(size, world, accumulation) * accumulation
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.offset = 0

    def seek(self, step, accumulation):
        epoch, offset = divmod(step * accumulation, self.per_rank)
        self.set_epoch(epoch)
        self.offset = offset

    def __len__(self):
        return self.per_rank - self.offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)
        return iter(indices[self.rank:self.per_rank * self.world:self.world][self.offset:].tolist())


def latest_checkpoint(run):
    candidates = []
    for path in (run / "checkpoints").glob("checkpoint_step_*"):
        if (path / "restart.json").is_file():
            candidates.append((int(path.name.removeprefix("checkpoint_step_")), path))
    if not candidates:
        raise FileNotFoundError(f"No completed restart checkpoint under {run}")
    return max(candidates)[1]


def local_optimizer_state(optimizer):
    state = optimizer.state_dict()
    state["state"] = {key: values.copy() for key, values in state["state"].items()}
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, DTensor):
                values[key] = {"shard": value.to_local().detach().cpu()}
            elif isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu()
    return state


def restore_optimizer(optimizer, state):
    for saved, current in zip(state["param_groups"], optimizer.param_groups):
        for param_id, parameter in zip(saved["params"], current["params"]):
            for key, value in state["state"].get(param_id, {}).items():
                if isinstance(value, dict):
                    local = value["shard"].to(parameter.device)
                    state["state"][param_id][key] = DTensor.from_local(
                        local, parameter.device_mesh, parameter.placements,
                        shape=parameter.shape, stride=parameter.stride(),
                    )
    optimizer.load_state_dict(state)


def save_restart(trainer):
    config = trainer.config
    target = trainer.save_dir / f"checkpoint_step_{trainer.step}"
    temporary = [None]
    if config.rank == 0:
        temporary[0] = tempfile.mkdtemp(prefix=".restart-", dir=trainer.save_dir)
    dist.broadcast_object_list(temporary, src=0)
    stage = Path(temporary[0])
    save_lora(trainer.transformer, stage, config.lora_rank, config.lora_alpha, config.rank == 0)
    torch.save({
        "optimizer": local_optimizer_state(trainer.optimizer),
        "scheduler": trainer.lr_scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(trainer.device),
    }, stage / f"rank-{config.rank}.pt")
    dist.barrier()
    if config.rank == 0:
        (stage / "restart.json").write_text(json.dumps({
            "step": trainer.step,
            "epoch": trainer.step // config.epoch_updates,
            "world_size": config.world_size,
        }))
        stage.rename(target)
    dist.barrier()


def restore_restart(trainer, directory):
    config = trainer.config
    progress = json.loads((directory / "restart.json").read_text())
    if progress["world_size"] != config.world_size:
        raise ValueError("Resume requires the original number of ranks")
    state = torch.load(directory / f"rank-{config.rank}.pt", weights_only=True)
    restore_optimizer(trainer.optimizer, state["optimizer"])
    trainer.lr_scheduler.load_state_dict(state["scheduler"])
    trainer.step = progress["step"]
    trainer.train_loader.sampler.seek(trainer.step, config.gradient_accumulation_steps)
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], trainer.device)


def reserve_training_memory(device, gib):
    """Grow the training allocator's reusable cache without keeping a live tensor."""
    target = gib * (1 << 30)
    reserved = torch.cuda.memory_reserved(device)
    needs = torch.tensor(int(reserved < target), device=device)
    dist.all_reduce(needs, op=dist.ReduceOp.MAX)
    if not needs.item():
        return
    extra = max(0, target - reserved)
    free, _ = torch.cuda.mem_get_info(device)
    ready = torch.tensor(int(free >= extra + (1 << 30)), device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        raise RuntimeError(f"Cannot reserve a {gib} GiB allocator pool on every training GPU")
    if extra:
        pool = torch.empty(extra, dtype=torch.uint8, device=device)
        del pool
    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    reached = torch.tensor(int(reserved >= target), device=device)
    dist.all_reduce(reached, op=dist.ReduceOp.MIN)
    if not reached.item():
        raise RuntimeError(f"Training allocator failed to retain {gib} GiB on every rank")
    print(f"rank {dist.get_rank()} training allocator reserved {reserved / (1 << 30):.2f} GiB", flush=True)
