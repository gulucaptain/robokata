"""Four independent GPU shards, bounded allocator, atomic/resumable Demo cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
import numpy as np
import torch
import safetensors
import safetensors.torch
from transformers import CONFIG_MAPPING, SiglipVisionModel
from data import read, write, sha, demo_indices
from demo_cache import identity, file_receipt, manifest_digest

def load_vision(base):
    config=CONFIG_MAPPING['paligemma']().vision_config
    config.intermediate_size=4304
    config.projection_dim=2048
    config.projector_hidden_act='gelu_fast'
    # Match the existing training tower (including its default attention backend).
    model=SiglipVisionModel(config).to(torch.bfloat16)
    for name,p in model.named_parameters():
        if any(x in name for x in ('embeddings.patch_embedding.weight','embeddings.patch_embedding.bias','embeddings.position_embedding.weight')):
            p.data=p.data.float()
    prefix='paligemma_with_expert.paligemma.model.vision_tower.'
    with safetensors.safe_open(str(base),framework='pt',device='cpu') as f:
        state={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    model.load_state_dict(state,strict=True)
    return model.eval().requires_grad_(False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--max-gib',type=float,default=10);p.add_argument('--batch',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    if a.batch!=4:raise ValueError('Preserve the native four-frame SigLIP numerical path')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3'):
        raise RuntimeError('Cache worker must expose exactly one authorized physical GPU 0–3')
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    # Include context/library headroom; never take another job's memory.
    budget=min(a.max_gib-2,free/2**30-4)
    if budget<4:raise RuntimeError('Insufficient shared free VRAM; leave other jobs untouched')
    torch.cuda.set_per_process_memory_fraction(budget*2**30/total,0)
    cfg=read(a.config);signature=identity(cfg);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=read(Path(cfg['data'])/'manifest.json');by={r['id']:r for r in manifest['rows']}
    selected=sorted({pair['demo'] for pairs in manifest['pairs'].values() for pair in pairs})
    selected=selected[a.shard::a.shards]
    print(json.dumps(dict(stage='start',shard=a.shard,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        records=len(selected),allocator_cap_gib=budget,process_budget_gib=a.max_gib)),flush=True)
    if sha(cfg['base_checkpoint'])!=signature['base_sha256']:raise ValueError('Official base SHA mismatch')
    model=load_vision(cfg['base_checkpoint']).to('cuda')
    print(json.dumps(dict(stage='vision_loaded',attention=model.config._attn_implementation,
                         dtype=str(next(model.parameters()).dtype))),flush=True)
    receipts=[];start=time.monotonic();verified=False
    for number,key in enumerate(selected):
        row=by[key];folder=out/'records'/key;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            _,marker=file_receipt(out,row,signature);receipts.append((key,marker['sha256']));continue
        if shutil.disk_usage(out).free<40*2**30:raise RuntimeError('Preserve 40GiB disk headroom for active training')
        raw=Path(cfg['data'])/'records'/key
        with np.load(raw/'supervision.npz') as labels:indices,timestamps=demo_indices(labels['timestamp_s'],row['seconds'])
        rgb=np.load(raw/'rgb.npy',mmap_mode='r')[indices]
        # Same CPU FP32 normalization and BF16 patch boundary as current training.
        pixels=torch.from_numpy(np.array(rgb[:,0],copy=True)).float().permute(0,3,1,2)/127.5-1
        chunks=[]
        with torch.inference_mode():
            for chunk in pixels.split(a.batch):
                count=len(chunk)
                # Native batches start on four-frame boundaries in each cap60 Demo.
                # Invalid native frames are normalized zeros, not black RGB (-1).
                if count<4:
                    chunk=torch.cat((chunk,chunk.new_zeros((4-count,*chunk.shape[1:])))).contiguous(memory_format=torch.channels_last)
                chunks.append(model(chunk.to('cuda')).last_hidden_state[:count].to(torch.bfloat16).cpu())
            patches=torch.cat(chunks)[:,None].contiguous()
            if not verified:
                native_rgb=np.zeros((1,60,1,224,224,3),np.uint8);native_rgb[0,:len(rgb)]=rgb
                native=torch.from_numpy(native_rgb).float().permute(0,1,2,5,3,4)/127.5-1
                native_valid=torch.arange(60,device='cuda')[None,:,None]<len(rgb)
                native=torch.where(native_valid[...,None,None,None],native.to('cuda'),0).flatten(0,2)
                reference=torch.cat([model(x).last_hidden_state.cpu() for x in native.split(4)])[:len(patches)]
                check=patches[:,0].float();reference=reference.float()
                delta=(check-reference).abs();rmse=delta.square().mean().sqrt().item()
                relative_rmse=rmse/max(reference.square().mean().sqrt().item(),1e-9)
                if not torch.isfinite(patches).all() or delta.max().item()!=0:
                    raise ValueError(f'Cache/native microbatch mismatch: relative_rmse={relative_rmse}')
                write(out/f'equivalence-shard-{a.shard}.json',dict(relative_rmse=relative_rmse,max_abs=delta.max().item(),
                    reference_batch=4,cache_batch=a.batch,identity=signature))
                verified=True
        temp=folder/f'patches-{os.getpid()}.tmp';file=folder/'patches.safetensors'
        safetensors.torch.save_file(dict(patches=patches,timestamps=torch.from_numpy(timestamps),indices=torch.from_numpy(indices)),str(temp),
                                   metadata={'identity':json.dumps(signature,sort_keys=True)})
        digest=sha(temp);temp.replace(file)
        marker=dict(identity=signature,rgb_sha256=row['rgb_sha256'],sha256=digest,bytes=file.stat().st_size,
                    shape=list(patches.shape),frames=len(indices))
        write(folder/'complete.json',marker);receipts.append((key,digest))
        if number%10==0 or number==len(selected)-1:
            status=dict(stage='encoding',shard=a.shard,done=number+1,total=len(selected),elapsed_s=time.monotonic()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write(out/f'progress-{a.shard}.json',status);print(json.dumps(status),flush=True)
    write(out/f'shard-{a.shard}-complete.json',dict(identity=signature,records=receipts,digest=manifest_digest(receipts)))
    print(json.dumps(dict(stage='complete',shard=a.shard,records=len(receipts))),flush=True)

if __name__=='__main__':main()
