"""Load complete official weights into production Demo pi05 on CPU, no forward/optimizer."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
import torch
from data import read,sha,write
from model import Pi05DemoConfig
cfg=read(root/'configs/demo-top.json');base=Path(cfg['base_checkpoint']);identity=read(base.parent/'base_identity.json')
assert identity['source']=='gs://openpi-assets/checkpoints/pi05_base' and identity['model_sha256']==sha(base)
torch.set_num_threads(4)
model=Pi05DemoConfig(demo_frames=cfg['demo_frames'],checkpoint_strategy=cfg['checkpoint_strategy']).create_pytorch()
model.load_base_checkpoint(base)
assert model.pi05 and not hasattr(model,'state_proj')
assert model.paligemma_with_expert.gemma_expert.config.use_adarms
report=dict(passed=True,source=identity['source'],model_sha256=identity['model_sha256'],bytes=base.stat().st_size,
    production_model_load_passed=True,new_demo_parameters_randomly_initialized=True,device='cpu',forward_executed=False,optimizer_executed=False,
    total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    time_mlp_finite=bool(torch.isfinite(model.time_mlp_in.weight).all()))
write(root/'verification/base-verified.json',report);print(json.dumps(report,indent=2))
