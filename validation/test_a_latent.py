"""CPU routing/gradient/serialization checks; CUDA kernels are not exercised.

Run: PYTHONPATH=backend:roboicl python -m unittest discover -s validation -p test_a_latent.py -v
"""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'roboicl')]
from demo_attention import (DemoAttention, SPEC, attach_demo_attention,
                            attach_foresight_indexing, latent_spec_for_views)
from examples.baselines.zero_wam.zero_wam.wan_va.modules import icl_model as backend
from publish_epochs import adapter_contract, export_epoch, bind_upload_state, verify_remote_epoch
from infer_checkpoint import load_checkpoint
from single_embodiment import audit_plan
import train
from examples.baselines.zero_wam.zero_wam.lora import adapt_transformer


def grid(frames, height, width, batch):
    return torch.stack(torch.meshgrid(torch.arange(frames), torch.arange(height),
                                     torch.arange(width), indexing='ij')).flatten(1)[None].repeat(batch, 1, 1)


def tiny_model(hidden_dim=16):
    model = backend.WanICLTransformer3DModel(
        num_attention_heads=1, attention_head_dim=hidden_dim, in_channels=2, out_channels=2,
        action_dim=4, text_dim=8, freq_dim=8, ffn_dim=32, num_layers=24,
        action_inner_dim=hidden_dim, action_ffn_dim=32, enable_mcp=True,
        num_mcp_modules=1, mcp_hidden_collect_layers=(7, 15, 23))
    attach_demo_attention(model, latent_spec_for_views(1))
    return model


def training_inputs(batch=2):
    latent = torch.randn(batch, 2, 2, 4, 4)
    action = torch.randn(batch, 4, 2, 1, 1)
    return dict(
        latent_dict=dict(latent=latent, noisy_latents=latent + .1, timesteps=torch.ones(batch, 2),
                         cond_timesteps=torch.zeros(batch, 2), grid_id=grid(2, 2, 2, batch)),
        action_dict=dict(latent=action, noisy_latents=action + .1, timesteps=torch.ones(batch, 2),
                         cond_timesteps=torch.zeros(batch, 2), grid_id=grid(2, 1, 1, batch)),
        icl_latent_dict=dict(latent=torch.randn(batch, 2, 3, 4, 4, requires_grad=True),
                             timesteps=torch.zeros(batch, 3), grid_id=grid(3, 2, 2, batch)),
        text_emb=torch.randn(batch, 2, 8), encoder_seq_ids=torch.zeros(2, dtype=torch.int),
        chunk_size=1, max_frame_chunk_size=1, window_size=4,
        demo_keep=torch.tensor([True, False])[:batch, None, None])


class CaptureBlock(torch.nn.Module):
    """Isolate demo routing from unrelated CUDA backbone attention kernels."""
    def __init__(self, source):
        super().__init__()
        if hasattr(source, 'demo_attention'):
            self.demo_attention = source.demo_attention
        self.seen = None

    def forward(self, video, action, *args, demo_features=None, **kwargs):
        if hasattr(self, 'demo_attention') and demo_features is not None:
            self.seen = demo_features
            action = self.demo_attention(action, demo_features)
        return video, action


class LatentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)

    def test_single_embodiment_data_contract(self):
        records = {f'FR2P_{i:04d}': dict(family=f'family_{i // 6}', task_id='task')
                   for i in range(3000)}
        train_ids, val_ids = list(records)[:2700], list(records)[2700:]
        def rows(ids, count):
            return [dict(demo=ids[i % len(ids)],
                         query=ids[(i % len(ids) + 1 + i // len(ids)) % len(ids)],
                         task_id='task') for i in range(count)]
        plan = dict(records=records, pairs=dict(train=rows(train_ids, 21860), validation=rows(val_ids, 260)),
                    split_seed=20261105, validation_families=[f'family_{i}' for i in range(450, 500)],
                    self_pairs_excluded=3000, cross_split_excluded=4880)
        report = audit_plan(plan)
        self.assertEqual(report['train_pairs'], 21860)
        self.assertEqual(report['train_query_recordings'], 2700)
        records[train_ids[0]]['kind'] = 'aloha'
        with self.assertRaisesRegex(ValueError, 'Non-Franka'):
            audit_plan(plan)
        records[train_ids[0]]['kind'] = 'franka'
        records[val_ids[0]]['family'] = records[train_ids[0]]['family']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit_plan(plan)

    def test_reject_mixed25k_data(self):
        with self.assertRaisesRegex(ValueError, '21860 train / 260 validation'):
            audit_plan(dict(records={}, pairs=dict(train=[{}] * 25000, validation=[{}] * 1200)))

    def test_full_architecture_lora_parameter_budget(self):
        # Meta tensors count the production architecture without allocating
        # model weights or accessing any server/GPU.
        with torch.device('meta'):
            model = backend.WanICLTransformer3DModel()
            report = adapt_transformer(model, dict(train_scope='joint', ig_lora_rank=train.LATENT_LORA_RANK))
            attach_demo_attention(model, latent_spec_for_views(1))
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(actual, 471342625)
        self.assertEqual(report['rank'], 98)
        self.assertLess(abs(actual / train.LATENT_PARAMETER_TARGET - 1), train.LATENT_PARAMETER_TOLERANCE)
        self.assertFalse(model.patch_embedding_mlp.weight.requires_grad)
        self.assertFalse(model.blocks[0].attn1.to_q.base.weight.requires_grad)
        self.assertTrue(model.blocks[0].attn1.to_q.lora_a.requires_grad)
        self.assertTrue(model.blocks[7].demo_attention.key_value.weight.requires_grad)

    def test_gate_dropout_and_dino_compatibility(self):
        module = DemoAttention(16, 256, 4, input_dim=16)
        action, demo = torch.randn(2, 3, 16), torch.randn(2, 12, 16)
        torch.testing.assert_close(module(action, demo), action, rtol=0, atol=0)
        module.gate.data.fill_(.5)
        torch.testing.assert_close(module(action, demo * 0), action, rtol=0, atol=0)
        self.assertFalse(torch.equal(module(action, demo), action))
        dino = tiny_model()
        attach_demo_attention(dino, SPEC)
        self.assertEqual(dino.blocks[7].demo_attention.key_value.in_features, 780)
        self.assertEqual(dino.demo_source, 'dino')

    def test_training_reads_video_input_and_backpropagates(self):
        model, inputs = tiny_model(), training_inputs()
        model.blocks = torch.nn.ModuleList([CaptureBlock(b) for b in model.blocks])
        for i in (7, 15, 23):
            model.blocks[i].demo_attention.gate.data.fill_(.5)
        expected = model._training_embed(inputs['icl_latent_dict']['latent'],
                                         inputs['icl_latent_dict']['timesteps'], 'video')[0]
        # The production mask builder is unrelated to this routing check.
        with patch.object(model, '_build_training_masks'):
            video, actions, mcp = model.forward_train(inputs)
        for i in (7, 15, 23):
            torch.testing.assert_close(model.blocks[i].seen, expected * inputs['demo_keep'])
        self.assertEqual(actions.shape, (2, 2, 4))
        self.assertEqual(video.shape, (2, 32, 2))
        self.assertEqual(mcp, [])  # No future streams in this small fixture.
        self.assertTrue(hasattr(model, 'mcp_blocks'))
        self.assertFalse(hasattr(model, 'semantic_ifp'))
        self.assertFalse(hasattr(model, 'foresight_indexing'))
        actions.square().mean().backward()
        gradient = inputs['icl_latent_dict']['latent'].grad
        self.assertGreater(gradient[0].abs().sum().item(), 0)
        self.assertEqual(gradient[1].abs().sum().item(), 0)
        self.assertGreater(model.patch_embedding_mlp.weight.grad.abs().sum().item(), 0)

    def test_native_backbone_and_mcp_cpu_backward(self):
        from types import SimpleNamespace
        model, inputs = tiny_model(hidden_dim=24), training_inputs()
        inputs['mcp_latent_dicts'] = [dict(noisy_latents=inputs['latent_dict']['latent'],
                                         timesteps=torch.ones(2, 2), grid_id=grid(2, 2, 2, 2))]
        for i in (7, 15, 23):
            model.blocks[i].demo_attention.gate.data.fill_(.5)

        def mask_builder(predicate, batch, heads, qlen, klen, **kwargs):
            return SimpleNamespace(predicate=predicate, batch=batch)

        def dense_attention(query, key, value, *, cross_attention, block_mask):
            # Use the actual mask predicates and all native backbone/MCP blocks;
            # only replace CUDA sparse kernels with CPU dense SDPA.
            b = torch.arange(block_mask.batch)[:, None, None, None]
            q = torch.arange(query.shape[1])[None, None, :, None]
            k = torch.arange(key.shape[1])[None, None, None, :]
            mask = block_mask.predicate(b, 0, q, k)
            return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
                                                  value.transpose(1, 2), attn_mask=mask).transpose(1, 2)

        with patch.object(backend, '_compiled_create_block_mask', mask_builder), \
             patch.object(backend.ICLAttentionBackend, 'apply', side_effect=dense_attention):
            video, actions, mcp = model.forward_train(inputs)
            self.assertEqual(len(mcp), 1)
            self.assertTrue(all(torch.isfinite(v).all() for v in (video, actions, *mcp)))
            loss = video.square().mean() + actions.square().mean() + mcp[0].square().mean()
            loss.backward()
        self.assertTrue(torch.isfinite(model.blocks[7].demo_attention.key_value.weight.grad).all())
        self.assertGreater(model.blocks[7].demo_attention.key_value.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.mcp_projections[0].weight.grad)

    def test_stream_reads_same_embedding_and_can_disable(self):
        model = tiny_model()
        model.blocks = torch.nn.ModuleList([CaptureBlock(b) for b in model.blocks])
        inputs = training_inputs(batch=1)
        stream = dict(action_res_lst=dict(noisy_latents=inputs['action_dict']['latent'],
                                          timesteps=torch.zeros(1, 2)),
                      action_grid_id=grid(2, 1, 1, 1), text_emb=inputs['text_emb'],
                      demo_latents=inputs['icl_latent_dict']['latent'])
        expected = model._training_embed(stream['demo_latents'], torch.zeros(1, 3), 'video')[0]
        with patch.object(backend.ICLAttentionBackend, 'build_self_mask'), \
             patch.object(backend.ICLAttentionBackend, 'build_cross_mask'), \
             patch.object(model, '_set_masks'):
            output = model._forward_stream(stream, 'action', 0, 'pos', False)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(model.blocks[7].seen, expected)
            for b in model.blocks:
                b.seen = None
            del stream['demo_latents']
            model._forward_stream(stream, 'action', 0, 'pos', False)
            self.assertIsNone(model.blocks[7].seen)

    def test_weights_roundtrip_and_indexing_rejection(self):
        model, other = tiny_model(), tiny_model()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, other.state_dict()[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'does not support'):
            attach_foresight_indexing(model, 'unused')

    def test_published_contract_and_safetensors_roundtrip(self):
        import tempfile
        model = tiny_model()
        settings = dict(ig_lora_rank=4, ig_demo_attention=latent_spec_for_views(1), ig_demo_dropout=.3,
                        ig_action_representation='roboicl_world_hand10_absolute',
                        ig_roboicl_channels=[*range(9), 28], ig_action_per_frame=16, snr_shift=5.,
                        action_snr_shift=1., icl_rope_h=24, enable_mcp=True, ig_effective_batch=16,
                        ig_demo_source='latent', ig_roboicl_encoding='fixture', ig_roboicl_cache_sha256='receipt')
        contract = adapter_contract(settings)
        with tempfile.TemporaryDirectory() as tmp:
            import json
            folder = Path(tmp)
            weights = {k: p.detach().clone() for k, p in model.named_parameters()}
            checkpoint = dict(global_step=1, progress=dict(epoch=1), native_config=settings,
                              adapter_state_dict=weights,
                              adaptation=dict(trainable_shapes={k:list(v.shape) for k,v in weights.items()}))
            torch.save(checkpoint, folder/'epoch.pt')
            receipt = export_epoch(folder/'epoch.pt', folder/'adapter.safetensors', 1, 1, contract)
            (folder/'adapter.json').write_text(json.dumps(receipt))
            (folder/'adapter_config.json').write_text(json.dumps(contract))
            loaded, tensors, step = load_checkpoint(folder/'adapter.safetensors', folder/'adapter_config.json',
                                                    folder, dict(encoding='fixture'))
            self.assertEqual(loaded['ig_demo_source'], 'latent')
            self.assertEqual(step, 1)
            self.assertEqual(set(tensors), set(weights))
            for k in weights:
                torch.testing.assert_close(weights[k], tensors[k], rtol=0, atol=0)
        settings['ig_semantic_ifp'] = dict(weight=.5)
        with self.assertRaisesRegex(ValueError, 'without B/FI'):
            adapter_contract(settings)

    def test_cli_rejects_mixed_inputs_and_checkpoint_modes(self):
        import contextlib
        common = ['train.py', '--work', '/fixture', '--model', '/fixture/model',
                  '--empty-text', '/fixture/text.pt', '--run-dir', '/fixture/run',
                  '--epochs', '1', '--demo-latents']
        for extra in (['--demo-features', '/fixture/dino.pt'],
                      ['--semantic-features', '/fixture/query.pt'],
                      ['--foresight-indexing']):
            with patch.object(sys, 'argv', common + extra), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                train.main()
            self.assertEqual(raised.exception.code, 2)

        plan = dict(pairs=dict(train=[{}] * 16))
        previous = dict(model_path='/fixture/model', ig_roboicl_cache_sha256='receipt',
                        ig_demo_attention=SPEC, ig_semantic_ifp=dict(weight=.5),
                        ig_foresight_indexing=dict(key_weight=.5))
        for option in ('--initialize-from', '--resume-from'):
            with patch.object(sys, 'argv', common + [option, '/fixture/full.pt']), \
                 patch.object(train, 'verified_plan', return_value=(plan, {})), \
                 patch.object(train, 'sha256', return_value='receipt'), \
                 patch.object(train.torch, 'load', return_value=dict(native_config=previous)), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                train.main()
            self.assertEqual(raised.exception.code, 2)

    def test_upload_state_binds_repository_and_recipe(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bind_upload_state(folder, 'owner/latent', {'rank':98}, 330, 20)
            bind_upload_state(folder, 'owner/latent', {'rank':98}, 330, 20)
            with self.assertRaisesRegex(ValueError, 'different repository'):
                bind_upload_state(folder, 'owner/other', {'rank':98}, 330, 20)
            with self.assertRaisesRegex(ValueError, 'different repository'):
                bind_upload_state(folder, 'owner/latent', {'rank':106}, 330, 20)

    def test_upload_resume_rechecks_remote_metadata(self):
        import tempfile
        from unittest.mock import Mock
        api = Mock()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = dict(size_bytes=12, sha256='hash', epoch=1)
            config = dict(lora_rank=98)
            import json
            (folder/'adapter.json').write_text(json.dumps(receipt))
            (folder/'adapter_config.json').write_text(json.dumps(config))
            api.hf_hub_download.side_effect = lambda repo, name, **kwargs: str(folder/Path(name).name)
            with patch('publish_epochs.remote_weights', return_value=(12, 'hash')):
                verify_remote_epoch(api, 'owner/latent', 'epoch-001', receipt, config)
                (folder/'adapter_config.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'metadata differs'):
                    verify_remote_epoch(api, 'owner/latent', 'epoch-001', receipt, config)



if __name__ == '__main__':
    unittest.main()
