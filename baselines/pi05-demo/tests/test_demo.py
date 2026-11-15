"""CPU correctness checks with tiny real Gemma/SigLIP layers; no optimizer/GPU."""
import dataclasses
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
from data import action_window,normalize,denormalize,demo_indices,validate_pairs
from model import Pi05DemoConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import tempfile
import safetensors.torch
from inputs import observation_from_samples
from openpi.models_pytorch import gemma_pytorch
from transformers import PaliGemmaConfig,SiglipVisionConfig,GemmaConfig

def tiny_paligemma():
    return PaliGemmaConfig(vision_config=SiglipVisionConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,
        intermediate_size=64,image_size=224,patch_size=14))

class DemoTests(unittest.TestCase):
    def test_normalization_tail(self):
        stats=dict(q01=[-1]*10,q99=[1]*10);x=np.linspace(-2,2,30).reshape(3,10).astype(np.float32)
        np.testing.assert_allclose(denormalize(normalize(x,stats),stats),x,atol=1e-6)
        action,valid=action_window(x,2,stats)
        self.assertEqual(valid.sum(),10);self.assertEqual(action.shape,(50,32));self.assertTrue((action[:,10:]==0).all())
    def test_sampling_full_video(self):
        clock=np.arange(565)/50;ids,times=demo_indices(clock,565/50)
        self.assertEqual(len(ids),57);self.assertEqual(ids[-1],560);np.testing.assert_allclose(times,clock[ids],atol=1e-6)
    def test_pair_leakage_rejected(self):
        rows=[dict(id='d',family='f1',split='train',task_id='t'),dict(id='q',family='f2',split='train',task_id='t')]
        pairs={'train':[dict(demo='d',query='q',task_id='t')]};validate_pairs(rows,pairs)
        with self.assertRaises(AssertionError):validate_pairs(rows,{'train':[dict(demo='q',query='q',task_id='t')]})
        rows[1]['split']='validation'
        with self.assertRaises(AssertionError):validate_pairs(rows,pairs)
    def test_real_pi05_demo_forward_backward_and_cached_inference(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        cfg=Pi05DemoConfig(paligemma_variant='dummy',action_expert_variant='dummy',dtype='float32',
            demo_frames=10,query_encode_batch=4,checkpoint_strategy='layer',lora_rank=8,lora_alpha=16)
        # Keep the real decoder/compression/adaRMS kernels; shrink only architecture widths for CPU.
        with patch.object(gemma_pytorch,'CONFIG_MAPPING',{'paligemma':tiny_paligemma,'gemma':GemmaConfig}):model=cfg.create_pytorch()
        # Exercise strict native-base compatibility: only new Demo modules may be absent.
        with patch.object(gemma_pytorch,'CONFIG_MAPPING',{'paligemma':tiny_paligemma,'gemma':GemmaConfig}):
            native=PI0Pytorch(Pi0Config(pi05=True,paligemma_variant='dummy',action_expert_variant='dummy',dtype='float32',pytorch_compile_mode=None))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            file=str(Path(temp)/'model.safetensors');safetensors.torch.save_model(native,file)
            model.load_base_checkpoint(file)
            torch.testing.assert_close(model.time_mlp_in.weight,native.time_mlp_in.weight)
        del native
        self.assertTrue(model.pi05);self.assertFalse(hasattr(model,'state_proj'))
        self.assertTrue(model.paligemma_with_expert.gemma_expert.config.use_adarms)
        rng=np.random.default_rng(7)
        sample=dict(rgb=rng.integers(0,256,(1,224,224,3),np.uint8),state=np.zeros(32,np.float32),
            tokens=np.arange(8,dtype=np.int64),mask=np.ones(8,bool),
            demo_rgb=rng.integers(0,256,(6,1,224,224,3),np.uint8),
            demo_valid=np.array([[1],[1],[1],[1],[1],[0]],bool),demo_timestamps=np.arange(6,dtype=np.float32)/5)
        obs=observation_from_samples([sample]);actions=torch.randn(1,50,32);valid=torch.zeros_like(actions,dtype=torch.bool);valid[:,:,:10]=True
        noise=torch.randn_like(actions);time=torch.full((1,),.5)
        model.gradient_checkpointing_enable()
        model.train();loss=model(obs,actions,noise=noise,time=time,action_valid=valid).mean()
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for name in ('demo_pool.queries','demo_pool.key.weight','time_mlp_in.weight','action_out_proj.weight',
            'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.lora_b',
            'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.lora_b'):
            grad=dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name);self.assertGreater(grad.norm().item(),0,name)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.paligemma_with_expert.paligemma.model.vision_tower.parameters()))
        model.eval();model.zero_grad(set_to_none=True)
        with torch.inference_mode():
            base=model(obs,actions,noise=noise,time=time,action_valid=valid)
            changed=dict(sample);changed['demo_rgb']=np.zeros_like(sample['demo_rgb'])
            other=model(observation_from_samples([changed]),actions,noise=noise,time=time,action_valid=valid)
            self.assertGreater((base-other).abs().max().item(),1e-8,'Demo must influence output')
            padded=dict(sample);padded['demo_rgb']=sample['demo_rgb'].copy();padded['demo_rgb'][-1]=255
            same=model(observation_from_samples([padded]),actions,noise=noise,time=time,action_valid=valid)
            torch.testing.assert_close(base,same,atol=1e-6,rtol=1e-6)
            # Compare one-step cached generation with the velocity from a joint forward at t=1.
            velocity=[]
            handle=model.action_out_proj.register_forward_hook(lambda m,args,out:velocity.append(out.detach().clone()))
            model(obs,actions,noise=noise,time=torch.ones(1),action_valid=valid);handle.remove()
            generated=model.sample_actions('cpu',obs,noise=noise,num_steps=1)
            torch.testing.assert_close(generated[...,:10],(noise-velocity[-1])[...,:10],atol=3e-5,rtol=3e-5)
            self.assertTrue((generated[...,10:]==0).all())
            self.assertEqual(tuple(generated.shape),(1,50,32));self.assertTrue(torch.isfinite(generated).all())

if __name__=='__main__':unittest.main(verbosity=2)
