"""LoRA with native base parameter names and trainable-only checkpoint identity."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import safetensors
import safetensors.torch
from model import Pi05DemoPytorch
from data import sha

FORMAT='pi05_demo_lora_v1'
TARGETS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
ROOTS=('paligemma_with_expert.paligemma.model.language_model.layers.',
       'paligemma_with_expert.gemma_expert.model.layers.')
EXTRA=('demo_pool.','demo_time_embedding','demo_view_embedding','source_embedding.',
       'action_in_proj.','action_out_proj.','time_mlp_in.','time_mlp_out.',
       'paligemma_with_expert.paligemma.model.multi_modal_projector.')

class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.weight=base.weight;self.bias=base.bias
        self.in_features=base.in_features;self.out_features=base.out_features
        self.rank=rank;self.scale=alpha/rank
        self.lora_a=nn.Parameter(torch.empty(rank,self.in_features,device=self.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(self.out_features,rank,device=self.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:self.bias.requires_grad_(False)
    def forward(self,x):
        result=F.linear(x,self.weight,self.bias)
        # FP32 trainable masters/moments; matmuls follow the native stream dtype.
        low=F.linear(x,self.lora_a.to(x.dtype))
        delta=F.linear(low,self.lora_b.to(x.dtype))
        return result+delta.to(result.dtype)*self.scale

class Pi05DemoLoRAPytorch(Pi05DemoPytorch):
    def __init__(self,config):
        super().__init__(config);self.requires_grad_(False);self.lora_targets=[];self._base_sha256=None
        targets=[(n,m) for n,m in self.named_modules() if isinstance(m,nn.Linear) and n.startswith(ROOTS) and n.rsplit('.',1)[-1] in TARGETS]
        for name,base in targets:
            parent,leaf=name.rsplit('.',1);setattr(self.get_submodule(parent),leaf,LoRALinear(base,config.lora_rank,config.lora_alpha));self.lora_targets.append(name)
        if not self.lora_targets:raise ValueError('No native language/action expert LoRA targets found')
        for name,p in self.named_parameters():
            if name.endswith(('.lora_a','.lora_b')) or name.startswith(EXTRA):p.requires_grad_(True)
        count=sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count>config.trainable_parameter_budget:raise ValueError(f'Trainable budget exceeded: {count}')
    def trainable_names(self):return {n for n,p in self.named_parameters() if p.requires_grad}
    def adapter_metadata(self):
        names=sorted(self.trainable_names())
        return dict(format=FORMAT,trainable_parameters=str(sum(p.numel() for p in self.parameters() if p.requires_grad)),rank=str(self.config.lora_rank),alpha=str(self.config.lora_alpha),base_sha256=self._base_sha256 or '',
            trainable_names_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())
    def load_base_checkpoint(self,path):
        missing,unexpected=safetensors.torch.load_model(self,str(path),strict=False)
        added=('demo_pool.','demo_time_embedding','demo_view_embedding','source_embedding.','phase_code')
        bad=[n for n in missing if not (n.startswith(added) or n.endswith(('.lora_a','.lora_b')))]
        if unexpected or bad:raise ValueError(f'Incompatible native pi05 base: missing={bad}, unexpected={unexpected}')
        if not any(n.endswith('.lora_a') for n in missing):raise ValueError('Expected an unadapted official base')
        self._base_sha256=sha(path)
    def save_training_checkpoint(self,path):
        if not self._base_sha256:raise ValueError('Official base must be loaded before saving adapter')
        state={n:p.detach().cpu().contiguous() for n,p in self.named_parameters() if p.requires_grad}
        safetensors.torch.save_file(state,str(path),metadata=self.adapter_metadata())
    def load_training_checkpoint(self,path):
        if not self._base_sha256:raise ValueError('Load the matching official base before adapter/resume')
        with safetensors.safe_open(str(path),framework='pt',device='cpu') as f:
            metadata=f.metadata()
            if metadata!=self.adapter_metadata():raise ValueError('Adapter format/rank/targets/base identity mismatch')
            if set(f.keys())!=self.trainable_names():raise ValueError('Adapter must contain all and only trainable parameters')
        tensors=safetensors.torch.load_file(str(path),device='cpu');current=dict(self.named_parameters())
        for n,t in tensors.items():
            if t.shape!=current[n].shape or not torch.isfinite(t).all():raise ValueError(f'Invalid adapter tensor: {n}')
        result=self.load_state_dict(tensors,strict=False)
        if result.unexpected_keys or any(n in self.trainable_names() for n in result.missing_keys):raise ValueError('Incomplete adapter load')
