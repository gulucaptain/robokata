"""Production LoRA parameter audit on meta device: zero GPU allocation."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
import torch
from model import Pi05DemoConfig
from data import read,write
cfg=read(root/'configs/demo-top.json')
with torch.device('meta'):model=Pi05DemoConfig(lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],trainable_parameter_budget=cfg['trainable_parameter_budget']).create_pytorch()
trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
report=dict(rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],target_modules=len(model.lora_targets),
    total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(trainable.values()),
    lora_parameters=sum(v for n,v in trainable.items() if n.endswith(('.lora_a','.lora_b'))),
    extra_trainable_parameters=sum(v for n,v in trainable.items() if not n.endswith(('.lora_a','.lora_b'))),
    frozen_vision=not any(p.requires_grad for p in model.paligemma_with_expert.paligemma.model.vision_tower.parameters()),
    physical_batch=cfg['per_gpu_batch'],global_batch=cfg['global_batch'],epochs=cfg['epochs'],steps=cfg['steps'],device='meta',gpu_allocated=False)
assert len(model.lora_targets)==252
assert 390000000<=report['trainable_parameters']<=400000000
(root/'verification').mkdir(exist_ok=True);write(root/'verification/lora-inventory.json',report);print(json.dumps(report,indent=2))
