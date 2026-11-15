"""CPU readiness check; does not allocate model weights or start optimization."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
from data import read,sha,Frames
from model import Pi05DemoConfig
from openpi.models.tokenizer import PaligemmaTokenizer
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
subprocess.run([sys.executable,str(ROOT/'vendor/openpi/examples/segmented_demo/prepare_transformers.py')],check=True)
assert cfg['global_batch']==cfg['world_size']*cfg['per_gpu_batch']
base=Path(cfg['base_checkpoint']);identity=read(base.parent/'base_identity.json')
assert identity['source']=='gs://openpi-assets/checkpoints/pi05_base' and identity['model_sha256']==sha(base)
assert base.name=='model.safetensors'
model=Pi05DemoConfig(demo_frames=cfg['demo_frames']);assert model.pi05 and model.discrete_state_input
if cfg.get('demo_cache'):
    from cached_data import CachedFrames
    frames=CachedFrames(cfg)
else:frames=Frames(cfg)
print(dict(passed=True,training_frames=len(frames),updates_per_epoch=len(frames)//cfg['global_batch'],
    demo_frames=cfg['demo_frames'],conditioning='demo_top_query_top_state',gpu_memory_tested=False))

if cfg.get('upload_each_epoch',False):
    from huggingface_hub import HfApi
    from publisher import check_access
    print(check_access(HfApi(token=True),cfg['hf_repo']))

import torch
with torch.device('meta'):
    lora_model=Pi05DemoConfig(lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],trainable_parameter_budget=cfg['trainable_parameter_budget']).create_pytorch()
trainable=sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
assert trainable==cfg['expected_trainable_parameters']<=cfg['trainable_parameter_budget']
print(dict(tuning_mode='lora',rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],trainable_parameters=trainable))
