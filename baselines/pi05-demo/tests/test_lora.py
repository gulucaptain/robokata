"""CPU LoRA invariants and adapter/optimizer roundtrip; no production GPU use."""
from pathlib import Path
import sys,tempfile,unittest,importlib.util
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
from model import Pi05DemoConfig
from lora import LoRALinear
from inputs import observation_from_samples
from openpi.models_pytorch import gemma_pytorch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from transformers import PaliGemmaConfig,SiglipVisionConfig,GemmaConfig
import safetensors,safetensors.torch

def tiny_paligemma():
    return PaliGemmaConfig(vision_config=SiglipVisionConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,image_size=224,patch_size=14))

class LoraTests(unittest.TestCase):
    def test_zero_initialization_and_first_second_gradient(self):
        torch.manual_seed(12)
        for dtype in (torch.float32,torch.bfloat16):
            base=nn.Linear(16,24).to(dtype);linear=LoRALinear(base,8,16);x=torch.randn(4,16,dtype=dtype)
            torch.testing.assert_close(linear(x),base(x),rtol=0,atol=0)
            optimizer=torch.optim.AdamW([p for p in linear.parameters() if p.requires_grad],lr=1e-3)
            linear(x).float().square().mean().backward()
            self.assertIsNone(linear.weight.grad);self.assertEqual(linear.lora_a.grad.norm(),0);self.assertGreater(linear.lora_b.grad.norm(),0)
            self.assertEqual(linear.lora_b.grad.dtype,torch.float32)
            optimizer.step();optimizer.zero_grad();linear(x).float().square().mean().backward();self.assertGreater(linear.lora_a.grad.norm(),0)
    def test_native_base_adapter_and_optimizer_roundtrip(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        cfg=Pi05DemoConfig(paligemma_variant='dummy',action_expert_variant='dummy',dtype='float32',demo_frames=6,
            lora_rank=8,lora_alpha=16,checkpoint_strategy='layer',query_encode_batch=2)
        with patch.object(gemma_pytorch,'CONFIG_MAPPING',{'paligemma':tiny_paligemma,'gemma':GemmaConfig}):
            native=PI0Pytorch(Pi0Config(pi05=True,paligemma_variant='dummy',action_expert_variant='dummy',dtype='float32',pytorch_compile_mode=None))
            model=cfg.create_pytorch();restored=cfg.create_pytorch()
        self.assertEqual(len(model.lora_targets),56)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp=Path(tmp);base=tmp/'base.safetensors';adapter=tmp/'model.safetensors'
            safetensors.torch.save_model(native,str(base));del native
            model.load_base_checkpoint(base);restored.load_base_checkpoint(base)
            model.gradient_checkpointing_enable()
            rng=np.random.default_rng(5);sample=dict(rgb=rng.integers(0,256,(1,224,224,3),np.uint8),state=np.zeros(32,np.float32),
                tokens=np.arange(12,dtype=np.int64),mask=np.ones(12,bool),demo_rgb=rng.integers(0,256,(6,1,224,224,3),np.uint8),
                demo_valid=np.ones((6,1),bool),demo_timestamps=np.arange(6,dtype=np.float32)/5)
            obs=observation_from_samples([sample]);actions=torch.randn(1,50,32);noise=torch.randn_like(actions);time=torch.full((1,),.5)
            valid=torch.zeros_like(actions,dtype=torch.bool);valid[:,:,:10]=True
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            model.train();loss=model(obs,actions,noise=noise,time=time,action_valid=valid).mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name,p in model.named_parameters():
                if not p.requires_grad:self.assertIsNone(p.grad,name)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            from openpi.training.config import TrainConfig,DataConfig
            spec=importlib.util.spec_from_file_location('test_native_trainer',ROOT/'vendor/openpi/scripts/train_pytorch.py')
            trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
            trainer.log_memory_usage=lambda *args:None
            native_config=TrainConfig(name='test',exp_name='adapter',checkpoint_base_dir=str(tmp),model=cfg,num_train_steps=1,save_interval=1,wandb_enabled=False)
            native_config.checkpoint_dir.mkdir(parents=True)
            trainer.save_checkpoint(model,optimizer,1,native_config,True,DataConfig())
            adapter=native_config.checkpoint_dir/'1/model.safetensors'

            with safetensors.safe_open(str(adapter),framework='pt') as f:
                self.assertEqual(set(f.keys()),model.trainable_names());self.assertEqual(f.metadata()['format'],'pi05_demo_lora_v1')
                self.assertFalse(any(n.endswith('q_proj.weight') for n in f.keys()))
            resumed_optimizer=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4)
            self.assertEqual(trainer.load_checkpoint(restored,resumed_optimizer,native_config.checkpoint_dir,torch.device('cpu')),1)
            self.assertEqual(len(resumed_optimizer.state),len(optimizer.state))
            model.eval();restored.eval()
            with torch.inference_mode():
                expected=model(obs,actions,noise=noise,time=time,action_valid=valid)
                actual=restored(obs,actions,noise=noise,time=time,action_valid=valid)
                torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)
                generated=restored.sample_actions('cpu',obs,noise=noise,num_steps=1)
                self.assertTrue(torch.isfinite(generated).all())
            restored._base_sha256='wrong-base'
            with self.assertRaises(ValueError):restored.load_training_checkpoint(adapter)

if __name__=='__main__':unittest.main(verbosity=2)
