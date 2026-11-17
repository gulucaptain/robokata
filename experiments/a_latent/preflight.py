"""Validate the fixed A_latent recipe on CPU; never train or publish."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'roboicl')]
import torch
from easydict import EasyDict
import train
import train_entry
from examples.baselines.zero_wam.zero_wam.wan_va.modules.icl_model import WanICLTransformer3DModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, default=Path(__file__).with_name('recipe.json'))
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    work = Path(os.environ.get('WORK', recipe['work']))
    model = Path(os.environ.get('MODEL', recipe['model']))
    empty_text = os.environ.get('EMPTY', recipe['empty_text'])
    if hashlib.sha256((work / 'plan.json').read_bytes()).hexdigest() != recipe['plan_sha256']:
        raise ValueError('The fixed recipe requires the original A/B1 cache plan')
    contract = json.loads((work / 'verified-cache.json').read_text())['contract']
    for key in ('demo_resize_hw', 'query_resize_hw'):
        if contract[key] != recipe[key]:
            raise ValueError(f'Unexpected camera geometry: {key}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run this CPU check with CUDA_VISIBLE_DEVICES empty')
    os.environ['WORLD_SIZE'] = str(recipe['world_size'])
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    sys.argv = ['train.py', '--work', str(work), '--model', str(model),
                '--empty-text', empty_text, '--run-dir', str(args.run_dir),
                '--demo-latents', '--lora-rank', str(recipe['lora_rank']),
                '--batch-per-gpu', str(recipe['batch_per_gpu']), '--epochs', str(recipe['epochs']),
                '--learning-rate', str(recipe['learning_rate']), '--warmup-steps', str(recipe['warmup_steps']),
                '--demo-dropout', str(recipe['demo_dropout']), '--seed', str(recipe['seed']),
                '--num-workers', '2', '--save-epoch-freq', '1', '--cuda-cache-gib', '120', '--dry-run']
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        train_entry.main()
    settings = json.loads(captured.getvalue())
    required = dict(batch_size=1, ig_effective_batch=64, ig_total_epochs=20, num_steps=6600,
                    learning_rate=1e-4, warmup_steps=100, ig_lora_rank=98, noisy_img_prob=.5,
                    ig_demo_dropout=.3, seed=1, gradient_accumulation_steps=1, enable_mcp=True,
                    ig_initialization=None, ig_resume=None, ig_cuda_cache_floor_gib=120)
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ValueError(f'Wrong effective setting: {key}={settings.get(key)!r}, expected {expected!r}')
    if settings.get('ig_semantic_ifp') or settings.get('ig_foresight_indexing'):
        raise ValueError('B/FI must be absent')
    base_config = json.loads((model / 'transformer/config.json').read_text())
    with torch.device('meta'):
        transformer = WanICLTransformer3DModel.from_config(base_config)
        report = train.training.adapt_transformer(transformer, EasyDict(settings))
    if report['trainable_parameters'] != recipe['trainable_parameters']:
        raise ValueError('Actual base-model architecture changes the parameter budget')
    # The native Trainer is constructed with batch_size=1, then the bridge
    # changes it to 16 when setting up the loader. Validate a real cached batch
    # rather than interpreting the staging dry-run value as the runtime batch.
    plan = train.load_plan(work)
    dataset = train.FullRoboICLDataset(work / 'plan.json', empty_text, 1, 16,
                                     work / 'stats.json', 'episode', 'zero_history_supervised_v1')
    sampler = train.SameLengthSampler(train.length_buckets(plan), 4, 0, 16, 1)
    from examples.baselines.zero_wam.zero_wam.dataset import collate
    batch = collate([dataset[item] for item in next(iter(sampler))])
    if batch['latents'].shape[0] != 16 or batch['icl_latents'].shape[0] != 16:
        raise ValueError('The actual loader did not produce batch 16')
    batch_shapes = {key: list(batch[key].shape) for key in ('latents', 'icl_latents', 'actions', 'ig_states')}
    print(json.dumps(dict(status='passed_cpu_only', runtime_batch_per_gpu=16,
                          cached_batch_shapes=batch_shapes, recipe=recipe, native_config=settings,
                          trainable_parameters=report['trainable_parameters'], parameter_budget=report['parameter_budget'],
                          trainable_tensors=len(report['trainable_shapes']),
                          torch_version=torch.__version__, cuda_allocated=False), indent=2))


if __name__ == '__main__':
    main()
