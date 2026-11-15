"""Bounded validation on authorized GPU0 only; no optimizer or DDP initialization."""
import argparse
import dataclasses
from pathlib import Path
import sys
import time
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src'),str(ROOT/'scripts')]
import numpy as np
import torch
from data import read,write,Frames
from demo_cache import identity,load_patches
from build_demo_cache import load_vision
from fast_execution import sdpa_attention
from transformers.models.gemma.modeling_gemma import eager_attention_forward

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Validation is authorized on physical GPU0 only')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info();budget=min(8*2**30,free-4*2**30)
    if budget<4*2**30:raise RuntimeError('Insufficient free memory; leave existing jobs untouched')
    torch.cuda.set_per_process_memory_fraction(budget/total)
    cfg=read(a.config);dataset=Frames(cfg);row=dataset[0];record=dataset.by_id[row['demo_id']]
    cached,clock,indices=load_patches(cfg['demo_cache'],record,identity(cfg))
    model=load_vision(cfg['base_checkpoint']).to('cuda')
    pixels=torch.from_numpy(row['demo_rgb'][:,0].copy()).float().permute(0,3,1,2)/127.5-1
    valid=torch.from_numpy(row['demo_valid'][:,0]);pixels=torch.where(valid[:,None,None,None],pixels,0)
    with torch.inference_mode():
        # Warmup the same native path; include transfers in each timed path.
        model(pixels[:4].to('cuda'));torch.cuda.synchronize()
        online=[];transfer=[];maximum_error=0.
        for _ in range(3):
            start=time.perf_counter()
            patches=torch.cat([model(x.to('cuda')).last_hidden_state.bfloat16() for x in pixels.split(4)])
            torch.cuda.synchronize();online.append(time.perf_counter()-start)
            start=time.perf_counter();fast=cached.to('cuda');torch.cuda.synchronize();transfer.append(time.perf_counter()-start)
            maximum_error=max(maximum_error,(fast[:,0].float()-patches[:len(fast)].float()).abs().max().item())
        if maximum_error!=0:raise ValueError(f'Production native/cache mismatch {maximum_error}')
        # Check native joint attention shape (8 query heads, 1 KV head, dim256),
        # preserving additive masks. No attention weights are needed by training.
        module=type('Attention',(),dict(num_key_value_groups=8,training=False))()
        q=torch.randn(2,8,128,256,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(2,1,128,256,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        mask=torch.zeros(2,1,128,128,device='cuda',dtype=torch.float32)
        mask[:,:,0,:]=torch.finfo(torch.float32).min
        expected,_=eager_attention_forward(module,q,k,v,mask,256**-.5)
        actual,_=sdpa_attention(module,q,k,v,mask,256**-.5)
        torch.testing.assert_close(expected[:,1:].float(),actual[:,1:].float(),atol=.02,rtol=.02)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            sdpa_attention(module,q,k,v,mask,256**-.5);torch.cuda.synchronize()
        kernels=[x.key for x in profiler.key_averages() if 'attention' in x.key]
    result=dict(passed=True,physical_gpu=0,cache_max_abs_error=maximum_error,online_demo_seconds=online,
                cached_transfer_seconds=transfer,attention_operators=kernels,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                scope='vision_and_attention_only_not_full_training_throughput')
    write(ROOT/'verification/gpu0-cache-benchmark.json',result);print(result)

if __name__=='__main__':main()
