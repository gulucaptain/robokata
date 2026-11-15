"""Production-shape official base + adapter persistence check, strictly CPU only."""
from pathlib import Path
import sys,json,tempfile
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
import torch
import safetensors
from data import read,write,sha
from model import Pi05DemoConfig
assert not torch.cuda.is_available(),'Run with CUDA_VISIBLE_DEVICES empty'
torch.set_num_threads(2);cfg=read(root/'configs/demo-top.json');base=Path(cfg['base_checkpoint']);identity=read(base.parent/'base_identity.json')
model=Pi05DemoConfig(lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],trainable_parameter_budget=cfg['trainable_parameter_budget']).create_pytorch()
model.load_base_checkpoint(base);assert model._base_sha256==identity['model_sha256']
count=sum(p.numel() for p in model.parameters() if p.requires_grad);assert count==cfg['expected_trainable_parameters']
with tempfile.TemporaryDirectory(dir=root/'tmp') as temp:
    checkpoint=Path(temp)/'model.safetensors';model.save_training_checkpoint(checkpoint)
    with safetensors.safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        assert set(f.keys())==model.trainable_names();metadata=f.metadata()
    parameters=dict(model.named_parameters());name='paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.lora_a'
    expected=parameters[name].detach().clone()
    with torch.no_grad():parameters[name].zero_()
    model.load_training_checkpoint(checkpoint)
    torch.testing.assert_close(parameters[name],expected,rtol=0,atol=0)
    report=dict(passed=True,trainable_parameters=count,rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],target_modules=len(model.lora_targets),
        base_source=identity['source'],base_sha256=model._base_sha256,adapter_bytes=checkpoint.stat().st_size,
        production_base_load_passed=True,production_adapter_roundtrip_passed=True,metadata=metadata,
        device='cpu',gpu_allocated=False,production_forward_executed=False,production_optimizer_executed=False,active_training_untouched=True)
write(root/'verification/production-lora-verified.json',report);print(json.dumps(report,indent=2))
