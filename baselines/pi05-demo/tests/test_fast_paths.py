"""Output/gradient equivalence of caching and SDPA; prefetch failure/cursor safety."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
import numpy as np
import torch
import safetensors.torch
from fast_execution import prefetch,sdpa_attention
from inputs import observation_from_samples
from model import Pi05DemoConfig
from openpi.models_pytorch import gemma_pytorch
from transformers.models.gemma.modeling_gemma import eager_attention_forward
from transformers import PaliGemmaConfig,SiglipVisionConfig,GemmaConfig

class FastPathTests(unittest.TestCase):
    def test_cache_hash_and_identity_reject_corruption(self):
        from demo_cache import file_receipt
        from data import sha,write
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);folder=root/'records'/'demo';folder.mkdir(parents=True)
            file=folder/'patches.safetensors'
            safetensors.torch.save_file({'patches':torch.zeros(5,1,2,4,dtype=torch.bfloat16)},str(file))
            signature={'format':'test'};record={'id':'demo','rgb_sha256':'rgb-id'}
            write(folder/'complete.json',dict(identity=signature,rgb_sha256='rgb-id',sha256=sha(file),
                 bytes=file.stat().st_size,shape=[5,1,2,4]))
            file_receipt(root,record,signature)
            with self.assertRaises(ValueError):file_receipt(root,record,{'format':'wrong'})
            content=bytearray(file.read_bytes());content[-1]^=1;file.write_bytes(content)
            with self.assertRaises(ValueError):file_receipt(root,record,signature)

    def test_prefetch_keeps_saved_cursor_at_consumed_batches(self):
        from runtime import FrameLoader
        class Sampler:
            def set_epoch(self,epoch):pass
        class Batches:
            start=0
        class Source:
            def __init__(self,owner):self.owner=owner
            def __iter__(self):yield from range(self.owner.batches.start,5)
        class Loader(FrameLoader):
            def __init__(self):
                self.dataset=type('Dataset',(),{'epoch':0})();self.sampler=Sampler();self.batches=Batches()
                self.epoch=0;self.cursor=0;self.prefetch_depth=1;self.loader=Source(self)
        loader=Loader();stream=iter(loader);self.assertEqual(next(stream),0);self.assertEqual(next(stream),1)
        self.assertEqual(loader.cursor,2);stream.close()
        restored=Loader();restored.cursor=2;self.assertEqual(list(restored),[2,3,4])

    def test_prefetch_order_exception_and_close(self):
        self.assertEqual(list(prefetch(range(9))),list(range(9)))
        def failed():
            yield 1
            raise ValueError('loader-error')
        stream=prefetch(failed());self.assertEqual(next(stream),1)
        with self.assertRaisesRegex(ValueError,'loader-error'):next(stream)
        stream=prefetch(range(1000));self.assertEqual(next(stream),0);stream.close()

    def test_sdpa_grouped_mask_output_and_gradient(self):
        torch.manual_seed(2);torch.set_num_threads(2)
        module=type('Attention',(),dict(num_key_value_groups=2,training=True))()
        q=torch.randn(2,4,9,8,requires_grad=True);k=torch.randn(2,2,9,8,requires_grad=True);v=torch.randn(2,2,9,8,requires_grad=True)
        # Includes entirely masked rows with the baseline's finite additive sentinel.
        mask=torch.where(torch.rand(2,1,9,9)>.4,0.,torch.finfo(torch.float32).min);mask[:,:,0,:]=torch.finfo(torch.float32).min
        expected,_=eager_attention_forward(module,q,k,v,mask,8**-.5)
        # Padded query outputs cannot feed supervised tokens; compare the valid
        # graph and explicitly require zero/finite outputs on unused rows.
        expected=expected.clone();expected[:,0]=0
        actual,_=sdpa_attention(module,q,k,v,mask,8**-.5)
        torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-6)
        g1=torch.autograd.grad(expected.square().mean(),(q,k,v),retain_graph=True)
        g2=torch.autograd.grad(actual.square().mean(),(q,k,v))
        for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_demo_cache_sdpa_and_raw_query_match_loss_and_gradients(self):
        from test_lora import tiny_paligemma
        torch.manual_seed(13);torch.set_num_threads(2)
        cfg=Pi05DemoConfig(paligemma_variant='dummy',action_expert_variant='dummy',dtype='float32',demo_frames=12,
                          lora_rank=8,lora_alpha=16,checkpoint_strategy='layer',query_encode_batch=4,pool_valid_only=True)
        with patch.object(gemma_pytorch,'CONFIG_MAPPING',{'paligemma':tiny_paligemma,'gemma':GemmaConfig}):model=cfg.create_pytorch()
        # Nonzero adapters ensure gradients through the trunk are exercised too.
        for name,p in model.named_parameters():
            if name.endswith('lora_b'):torch.nn.init.normal_(p,std=.001)
        rng=np.random.default_rng(7)
        sample=dict(rgb=rng.integers(0,256,(1,224,224,3),np.uint8),state=np.zeros(32,np.float32),
                    tokens=np.arange(16,dtype=np.int64),mask=np.array([True]*8+[False]*8),
                    demo_rgb=rng.integers(0,256,(12,1,224,224,3),np.uint8),
                    demo_valid=np.array([[True]]*5+[[False]]*7),demo_timestamps=np.arange(12,dtype=np.float32)/5)
        obs=observation_from_samples([sample]);actions=torch.randn(1,50,32);noise=torch.randn_like(actions);t=torch.tensor([.5])
        valid=torch.zeros_like(actions,dtype=torch.bool);valid[:,:,:10]=True
        model.eval()
        with torch.no_grad():
            rgb=torch.where(obs.demo_valid[...,None,None,None],obs.demo_rgb,0)
            patches=torch.cat([model.paligemma_with_expert.embed_image_patches(x) for x in rgb.flatten(0,2).split(4)]).reshape(1,12,1,256,32).bfloat16()
        patches[:,5:]=0
        cached=dict(sample);cached.pop('demo_rgb');cached['demo_patches']=patches[0]
        cached_obs=observation_from_samples([cached],raw_query=True,trim_prompt=True)
        self.assertEqual(cached_obs.images['base_0_rgb'].dtype,torch.uint8)
        self.assertEqual(cached_obs.tokenized_prompt.shape[-1],8)
        model.train();model.gradient_checkpointing_enable()
        # Fix random augmentation across both paths.
        torch.manual_seed(31);expected=model(obs,actions,noise=noise,time=t,action_valid=valid).mean();expected.backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None};model.zero_grad(set_to_none=True)
        model.paligemma_with_expert.joint_attention_backend='sdpa'
        calls=[]
        from transformers.models.gemma import modeling_gemma
        def tracked(*args,**kwargs):
            calls.append(1)
            return sdpa_attention(*args,**kwargs)
        modeling_gemma.ALL_ATTENTION_FUNCTIONS.register('pi05_demo_sdpa',tracked)
        torch.manual_seed(31);actual=model(cached_obs,actions,noise=noise,time=t,action_valid=valid).mean();actual.backward()
        modeling_gemma.ALL_ATTENTION_FUNCTIONS.register('pi05_demo_sdpa',sdpa_attention)
        self.assertGreater(len(calls),0,'Compressed decoder must actually execute SDPA')
        torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-5)
        for n,p in model.named_parameters():
            if n in grads:torch.testing.assert_close(grads[n],p.grad,atol=2e-5,rtol=2e-3,msg=n)
        # Skipping only empty native blocks keeps the original feature grouping.
        model.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():
            calls=[]
            hook=model.paligemma_with_expert.paligemma.model.vision_tower.register_forward_hook(lambda *args:calls.append(1))
            baseline=model._demo_tokens(obs,train=False)
            self.assertEqual(len(calls),3);calls.clear()
            object.__setattr__(model.config,'skip_empty_demo_blocks',True)
            skipped=model._demo_tokens(obs,train=False)
            self.assertEqual(len(calls),2);hook.remove()
            for a,b in zip(baseline,skipped):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main(verbosity=2)
