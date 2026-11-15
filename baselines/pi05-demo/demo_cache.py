"""Immutable, verified raw BF16 SigLIP patches; never cache trainable pooling."""
import hashlib
import json
from pathlib import Path

import torch
import safetensors
from data import read, sha

FORMAT = 'pi05_frozen_demo_siglip_bf16_v2'

def identity(config):
    base = Path(config['base_checkpoint'])
    info = read(base.parent / 'base_identity.json')
    return dict(format=FORMAT, base_sha256=info['model_sha256'],
                manifest_sha256=sha(Path(config['data'])/'manifest.json'),
                sample_hz=5, normalization='float32_rgb_div127.5_minus1',
                native_batch=4, memory_format='channels_last_including_padded_tail',matmul_precision='high',
                boundary='vision_tower.last_hidden_state_before_trainable_projector')

def file_receipt(root, record, signature, *, verify_hash=True):
    root=Path(root); marker=read(root/'records'/record['id']/'complete.json')
    if marker['identity'] != signature or marker['rgb_sha256'] != record['rgb_sha256']:
        raise ValueError('Demo cache identity mismatch: '+record['id'])
    file=root/'records'/record['id']/'patches.safetensors'
    if file.stat().st_size != marker['bytes'] or (verify_hash and sha(file)!=marker['sha256']):
        raise ValueError('Demo cache checksum mismatch: '+record['id'])
    with safetensors.safe_open(str(file),framework='pt',device='cpu') as f:
        p=f.get_slice('patches')
        if p.get_dtype()!='BF16' or p.get_shape()!=marker['shape']:
            raise ValueError('Demo cache shape/dtype mismatch')
    return file, marker

def load_patches(root, record, signature):
    file, marker=file_receipt(root,record,signature)
    with safetensors.safe_open(str(file),framework='pt',device='cpu') as f:
        return f.get_tensor('patches'), f.get_tensor('timestamps'), f.get_tensor('indices')

def manifest_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
