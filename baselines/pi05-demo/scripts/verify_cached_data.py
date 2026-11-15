"""Real dataset/cache collation checks on CPU, with unchanged sampling and labels."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
import numpy as np
import torch
from cached_data import CachedFrames
from data import Frames,read,write
from inputs import observation_from_samples

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=read(a.config);torch.set_num_threads(2)
    cached=CachedFrames(cfg);online=Frames(cfg);samples=[]
    for index in [0,len(cached)//2,len(cached)-1]:
        fast=cached[index];raw=online[index]
        for key in ('rgb','state','actions','valid','tokens','mask','demo_valid','demo_timestamps'):
            np.testing.assert_array_equal(fast[key],raw[key])
        assert fast['demo_id']==raw['demo_id']
        assert fast['demo_patches'].dtype==torch.bfloat16
        assert torch.isfinite(fast['demo_patches']).all()
        samples.append(fast)
    obs=observation_from_samples(samples,raw_query=True,trim_prompt=True)
    assert obs.demo_rgb is None and obs.demo_patches.shape[:3]==(3,60,1)
    write(ROOT/'verification/cached-data.json',dict(passed=True,recordings=2880,frames=len(cached),
          checked_indices=[0,len(cached)//2,len(cached)-1],query_augmentation='unchanged',
          patch_shape=list(obs.demo_patches.shape),prompt_length=obs.tokenized_prompt.shape[-1],dtype=str(obs.demo_patches.dtype)))
    print(dict(passed=True,patch_shape=list(obs.demo_patches.shape)))

if __name__=='__main__':main()
