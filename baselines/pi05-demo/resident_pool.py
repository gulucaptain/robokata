"""Reusable per-rank CUDA pool for the explicitly authorized training run."""
import json
import torch

def reserve_pool(gib,local_rank):
    if not gib:return
    device=torch.device(f'cuda:{local_rank}');torch.cuda.set_device(device)
    target=int(gib)*(1<<30)
    extra=max(0,target-torch.cuda.memory_reserved(device))
    if extra:
        buffer=torch.empty(extra,dtype=torch.uint8,device=device)
        del buffer
    torch.cuda.synchronize(device)
    reserved=torch.cuda.memory_reserved(device)
    if reserved<target:raise RuntimeError('Requested resident CUDA pool was not retained')
    print(json.dumps(dict(stage='cuda_pool_ready',local_rank=local_rank,reserved_gib=reserved/(1<<30))),flush=True)
