"""Inspect full production architecture on meta device, without weights or GPU allocation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
import torch
from model import Pi05DemoConfig
with torch.device('meta'):model=Pi05DemoConfig().create_pytorch()
report=dict(pi05=model.pi05,has_pi0_state_proj=hasattr(model,'state_proj'),
    total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    demo_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('demo_pool.','demo_time_embedding','demo_view_embedding','source_embedding.'))),
    frozen_siglip=not any(p.requires_grad for p in model.paligemma_with_expert.paligemma.model.vision_tower.parameters()),
    device='meta',weights_loaded=False,gpu_tested=False)
print(json.dumps(report,indent=2));(ROOT/'model-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
