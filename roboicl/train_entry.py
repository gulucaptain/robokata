"""Launch the RoboICL trainer with an optional reusable CUDA memory pool."""

import argparse
import sys

import torch
import torch.distributed as dist

import train
from examples.baselines.zero_wam.zero_wam import training
from examples.baselines.zero_wam.zero_wam.wan_va.distributed import util


def configure_cuda_pool(target_gib):
    native_config = training.native_config
    init_distributed = util.init_distributed
    resume = training.resume
    gib = 1024**3

    def native_config_with_pool(config, manifest):
        settings = native_config(config, manifest)
        settings.ig_cuda_cache_floor_gib = target_gib
        return settings

    def reserve_pool(device, rank, stage):
        free, _ = torch.cuda.mem_get_info(device)
        extra = max(0, target_gib * gib - torch.cuda.memory_reserved(device))
        ready = torch.tensor(int(not extra or free >= extra + gib), device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if not ready.item():
            raise RuntimeError("Insufficient memory for the requested CUDA cache on a rank")
        if extra:
            pool = torch.empty(extra, dtype=torch.uint8, device=device)
            del pool
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        reserved = torch.cuda.memory_reserved(device) / gib
        print(f"Rank {rank}: CUDA pool reserved {reserved:.2f} GiB for reuse ({stage})", flush=True)

    def init_with_pool(world_size, local_rank, rank):
        init_distributed(world_size, local_rank, rank)
        reserve_pool(torch.device("cuda", local_rank), rank, "distributed initialization")

    def resume_with_pool(trainer):
        resume(trainer)
        # Loading libraries may clear unused cache. Restore the floor after
        # model/optimizer construction, immediately before training starts.
        reserve_pool(trainer.device, trainer.config.rank, "model ready")

    training.native_config = native_config_with_pool
    util.init_distributed = init_with_pool
    training.resume = resume_with_pool


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-cache-gib", type=int, default=0)
    options, remaining = parser.parse_known_args()
    if options.cuda_cache_gib < 0:
        parser.error("--cuda-cache-gib must be nonnegative")
    if options.cuda_cache_gib:
        configure_cuda_pool(options.cuda_cache_gib)
    sys.argv[1:] = remaining
    train.main()


if __name__ == "__main__":
    main()
