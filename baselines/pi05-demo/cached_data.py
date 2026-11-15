"""Same Query frame sampling and augmentation; immutable frozen Demo patches."""
import functools
from pathlib import Path
import numpy as np
import torch
from data import Frames,read,sha
from demo_cache import identity,load_patches

class CachedFrames(Frames):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self.cache=Path(cfg['demo_cache']);self.signature=identity(cfg)
        index=read(self.cache/'index.json')
        if index['identity']!=self.signature:raise ValueError('Completed Demo cache differs from dataset/base')
        wanted={d for ds in self.candidates.values() for d in ds}
        if not wanted.issubset(index['records']):raise ValueError('Demo cache is incomplete for this split')
        self.cache_receipts=index['records']
        self.verified=set()

    @functools.lru_cache(maxsize=32)
    def patches(self,key):
        # Hash once per recording/process, not once per cache eviction/batch.
        from demo_cache import file_receipt
        file,marker=file_receipt(self.cache,self.by_id[key],self.signature,verify_hash=key not in self.verified)
        if marker['sha256']!=self.cache_receipts[key]:raise ValueError('Cache receipt changed')
        self.verified.add(key)
        import safetensors
        with safetensors.safe_open(str(file),framework='pt',device='cpu') as f:
            return f.get_tensor('patches'),f.get_tensor('timestamps'),f.get_tensor('indices')

    def demo_payload(self,key,indices,times):
        patches,clock,frames=self.patches(key)
        if not np.array_equal(frames.numpy(),indices) or not np.array_equal(clock.numpy(),times):
            raise ValueError('Cached Demo sampling differs from the full-video contract')
        result=patches.new_zeros((self.demo_frames,*patches.shape[1:]));result[:len(patches)]=patches
        valid=np.zeros((self.demo_frames,1),bool);valid[:len(patches)]=True
        timestamps=np.zeros(self.demo_frames,np.float32);timestamps[:len(patches)]=times
        return dict(demo_patches=result,demo_valid=valid,demo_timestamps=timestamps)
