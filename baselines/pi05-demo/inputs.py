"""Shared training/inference tensor layout; no task text or Demo action inputs."""
import numpy as np
import torch
from openpi.models.model import Observation

def observation_from_samples(samples,device='cpu',*,raw_query=False,trim_prompt=False):
    def stack(key):
        if isinstance(samples[0][key],torch.Tensor):return torch.stack([s[key] for s in samples]).to(device)
        return torch.as_tensor(np.stack([s[key] for s in samples]),device=device)
    rgb=stack('rgb').permute(0,1,4,2,3)
    if not raw_query:rgb=rgb.float()/127.5-1
    tokens=stack('tokens');token_mask=stack('mask')
    if trim_prompt:
        columns=token_mask.any(dim=0).nonzero().flatten()
        if not len(columns):raise ValueError('Expected pi05 state/prompt tokens')
        length=int(columns[-1])+1
        tokens=tokens[:,:length];token_mask=token_mask[:,:length]
    assert rgb.shape[1]==1
    return Observation(images={'base_0_rgb':rgb[:,0]},
        image_masks={'base_0_rgb':torch.ones(len(samples),device=device,dtype=torch.bool)},
        state=stack('state'),tokenized_prompt=tokens,tokenized_prompt_mask=token_mask,
        demo_rgb=None if 'demo_patches' in samples[0] else stack('demo_rgb').float().permute(0,1,2,5,3,4)/127.5-1,
        demo_patches=stack('demo_patches') if 'demo_patches' in samples[0] else None,
        demo_valid=stack('demo_valid'),demo_timestamps=stack('demo_timestamps'),
        source_id=torch.zeros(len(samples),device=device,dtype=torch.long))
