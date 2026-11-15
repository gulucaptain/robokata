import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT/'vendor/openpi'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'packages/openpi-client/src'))
from data import read,write,sha,Frames
from runtime import FrameLoader,RemainingBatches
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.config import TrainConfig, DataConfig
from openpi.training.optimizer import AdamW, CosineDecaySchedule


from model import Pi05DemoConfig
from inputs import observation_from_samples
from publishing import EpochPublisher
from cached_data import CachedFrames


class Loader(FrameLoader):
    dynamic_graph = True

    def __init__(self,cfg,rank,world):
        self.dataset = CachedFrames(cfg) if cfg.get('demo_cache') else Frames(cfg)
        self.prefetch_depth=cfg.get('prefetch_batches',0)
        self.raw_query=cfg.get('query_uint8_transfer',False)
        self.pin_batches=cfg.get('pin_batches',False)
        self.trim_prompt=cfg.get('trim_prompt',False)
        self.rank,self.world,self.global_batch = rank,world,cfg['global_batch']
        self.gradient_receipt = None
        assert world*cfg['per_gpu_batch'] == self.global_batch
        self.sampler = DistributedSampler(self.dataset,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed'],drop_last=True)
        self.batches = RemainingBatches(self.sampler,cfg['per_gpu_batch'],drop_last=True)
        self.epoch,self.cursor = 0,0
        self.loader = DataLoader(self.dataset,batch_sampler=self.batches,num_workers=0,collate_fn=self.collate,
                                 generator=torch.Generator().manual_seed(cfg['seed']+rank))

    def collate(self,samples):
        def stack(k):return torch.from_numpy(np.stack([s[k] for s in samples]))
        obs=observation_from_samples(samples,raw_query=self.raw_query,trim_prompt=self.trim_prompt)
        if self.pin_batches:
            import jax
            with torch.cuda.device(int(os.environ.get('LOCAL_RANK',0))):
                obs=jax.tree.map(lambda x:x.pin_memory() if isinstance(x,torch.Tensor) else x,obs)
        return obs,stack('actions'),stack('valid')

    def verify_gradients(self,model):
        if self.gradient_receipt is not None:return
        raw = model.module if hasattr(model,'module') else model
        vision=raw.paligemma_with_expert.paligemma.model.vision_tower
        assert not any(p.requires_grad or p.grad is not None for p in vision.parameters())
        assert raw.pi05 and not hasattr(raw,'state_proj')
        params = dict(raw.named_parameters())
        names = ['action_out_proj.weight',
            'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.lora_b',
            'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.lora_b',
            'demo_pool.queries','demo_pool.key.weight','time_mlp_in.weight']
        values = {n:float(params[n].grad.float().norm()) if params[n].grad is not None else None for n in names}
        assert all(v is not None and np.isfinite(v) and v>0 for v in values.values()),values
        self.gradient_receipt = values


def main():
    p=argparse.ArgumentParser(description='Demo-conditioned pi05 ICL3000 fine tuning; explicit launch required')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    cfg=read(a.config);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    assert world==cfg['world_size'];torch.set_num_threads(cfg.get('cpu_threads',4))
    assert cfg['global_batch']==world*cfg['per_gpu_batch']
    if cfg.get('cuda_pool_gib',0):
        from resident_pool import reserve_pool
        reserve_pool(cfg['cuda_pool_gib'],int(os.environ.get('LOCAL_RANK',0)))
    base=Path(cfg['base_checkpoint']);assert base.is_file() and base.name=='model.safetensors','Prepare official pi05_base first'
    identity=read(base.parent/'base_identity.json')
    assert identity['source']=='gs://openpi-assets/checkpoints/pi05_base' and identity['model_sha256']==sha(base)
    model_cfg=Pi05DemoConfig(demo_frames=cfg['demo_frames'],demo_encode_batch=cfg['demo_encode_batch'],
        query_encode_batch=cfg['query_encode_batch'],checkpoint_strategy=cfg['checkpoint_strategy'],pool_valid_only=cfg.get('pool_valid_only',False),
        joint_attention_backend=cfg.get('joint_attention_backend','eager'),skip_empty_demo_blocks=cfg.get('skip_empty_demo_blocks',False),
        lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],trainable_parameter_budget=cfg['trainable_parameter_budget'])
    loader=Loader(cfg,rank,world);output=a.output.resolve()
    epoch_steps=len(loader.batches)
    assert epoch_steps>0
    steps=a.steps or (cfg['epochs']*epoch_steps if cfg.get('epochs') else cfg['steps'])
    assert steps>0
    paths=[ROOT/'scripts/train_pytorch.py',*[p for p in PROJECT.glob('*.py') if not p.name.startswith('._')],
           *[p for p in (ROOT/'src/openpi').rglob('*.py') if not p.name.startswith('._')],
           *[p for p in (ROOT/'packages/openpi-client/src').rglob('*.py') if not p.name.startswith('._')],
           ROOT/'pyproject.toml',ROOT/'uv.lock',ROOT/'packages/openpi-client/pyproject.toml',PROJECT/'environment-lock.json',
           *[p for p in (PROJECT/'scripts').glob('*.py') if not p.name.startswith('._')],PROJECT/'scripts/train.sh']
    loader.contract=dict(model='demo_pi05_lora',config=cfg,model_config=dataclasses.asdict(model_cfg),
        source_sha256={str(f.relative_to(PROJECT)):sha(f) for f in paths},
        updates_per_epoch=epoch_steps,total_updates=steps,assets_sha256=sha(cfg['assets']),base_sha256=sha(cfg['base_checkpoint']),tokenizer_sha256=sha(cfg['tokenizer']))
    loader.contract=json.loads(json.dumps(loader.contract))
    if cfg.get('demo_cache'):loader.contract['demo_cache_index_sha256']=sha(Path(cfg['demo_cache'])/'index.json')
    native=TrainConfig(name=output.parent.name,exp_name=output.name,checkpoint_base_dir=str(output.parent.parent),model=model_cfg,
        batch_size=cfg['global_batch'],num_train_steps=steps,seed=cfg['seed'],num_workers=0,
        save_interval=epoch_steps,log_interval=10,wandb_enabled=False,resume=a.resume,ema_decay=None,
        pytorch_weight_path=str(Path(cfg['base_checkpoint']).parent),optimizer=AdamW(),
        lr_schedule=CosineDecaySchedule(warmup_steps=cfg['warmup'],peak_lr=cfg['lr_peak'],decay_steps=steps,decay_lr=cfg['lr_end']))
    assert native.checkpoint_dir==output
    spec=importlib.util.spec_from_file_location('openpi_native_trainer',ROOT/'scripts/train_pytorch.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer);trainer.init_logging()
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        if (output/'contract.json').exists():assert read(output/'contract.json')==loader.contract
        write(output/'contract.json',loader.contract)
        print(json.dumps(dict(updates_per_epoch=epoch_steps,total_updates=steps,global_batch=cfg['global_batch'],conditioning='demo_rgb_query_rgb_state_tokens',frozen_siglip=True)))
    publisher=None
    if rank==0 and cfg.get('upload_each_epoch',False):
        publisher=EpochPublisher(PROJECT,output,cfg['hf_repo']).start()
    training_ok=False
    try:
        trainer.train_loop(native,data_loader=(loader,DataConfig()))
        training_ok=True
    finally:
        if publisher is not None:publisher.finish(training_ok)


if __name__=='__main__':main()
