"""Demo top video + current Query top/state -> 50 world-hand10 commands."""
import argparse
from pathlib import Path
import numpy as np
import torch
import safetensors.torch
from train import PROJECT,PaligemmaTokenizer
from model import Pi05DemoConfig
from inputs import observation_from_samples
from data import read,sha,normalize,denormalize,demo_indices

class Policy:
    def __init__(self,checkpoint,device='cuda',base_checkpoint=None):
        checkpoint=Path(checkpoint);receipt=read(checkpoint/'complete.json');contract=receipt['contract']
        assert contract['model']=='demo_pi05_lora'
        assert sha(checkpoint/'model.safetensors')==receipt['model_sha256']
        for name,digest in contract['source_sha256'].items():assert sha(PROJECT/name)==digest,name
        self.cfg=contract['config'];self.device=device
        assets=checkpoint/'assets.json' if (checkpoint/'assets.json').is_file() else Path(self.cfg['assets'])
        tokenizer=checkpoint/'tokenizer.model' if (checkpoint/'tokenizer.model').is_file() else Path(self.cfg['tokenizer'])
        assert sha(assets)==contract['assets_sha256'] and sha(tokenizer)==contract['tokenizer_sha256']
        self.assets=read(assets);self.tokenizer=PaligemmaTokenizer(200,tokenizer_path=tokenizer)
        fields=contract['model_config'];fields['query_image_keys']=tuple(fields['query_image_keys']);fields['action_contracts']=tuple(fields['action_contracts'])
        self.model=Pi05DemoConfig(**fields).create_pytorch()
        base=Path(base_checkpoint or self.cfg['base_checkpoint'])
        assert sha(base)==contract['base_sha256'],'Use the matching official pi05_base'
        self.model.load_base_checkpoint(base)
        self.model.load_training_checkpoint(checkpoint/'model.safetensors')
        self.model.to(device).eval()
        self.demo=None

    def set_demo(self,rgb,timestamps,duration):
        """Full top-view video uint8 [T,224,224,3], native 50Hz clock from zero.

        duration is exclusive end (e.g. T/50); whole video is sampled at 5Hz.
        Demo state/actions and task labels are not consumed.
        """
        rgb=np.asarray(rgb);assert rgb.dtype==np.uint8 and rgb.ndim==4 and rgb.shape[1:]==(224,224,3)
        assert len(rgb)==len(timestamps)
        indices,times=demo_indices(timestamps,duration);cap=self.cfg['demo_frames']
        assert len(indices)<=cap,'Demo cap cannot truncate the video'
        images=np.zeros((cap,1,224,224,3),np.uint8);images[:len(indices),0]=rgb[indices]
        valid=np.zeros((cap,1),bool);valid[:len(indices)]=True
        clock=np.zeros(cap,np.float32);clock[:len(indices)]=times
        self.demo=dict(demo_rgb=images,demo_valid=valid,demo_timestamps=clock)

    @torch.inference_mode()
    def predict(self,rgb,state,*,num_steps=10):
        assert self.demo is not None,'Call set_demo first'
        rgb=np.asarray(rgb);state=np.asarray(state,np.float32)
        assert rgb.dtype==np.uint8 and rgb.shape==(224,224,3)
        assert state.shape==(10,) and np.isfinite(state).all() and num_steps>0
        state=np.pad(normalize(state,self.assets['normalization']['state']),(0,22))
        tokens,mask=self.tokenizer.tokenize('',state);assert mask.sum()<200,'State token overflow'
        obs=observation_from_samples([dict(rgb=rgb[None],state=state,tokens=tokens,mask=mask,**self.demo)],self.device)
        predicted=self.model.sample_actions(self.device,obs,num_steps=num_steps)[0,:,:10].float().cpu().numpy()
        action=denormalize(predicted,self.assets['normalization']['action']);assert np.isfinite(action).all()
        return action

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True)
    p.add_argument('--demo',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--base-checkpoint');p.add_argument('--denoise-steps',type=int,default=10);a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    policy=Policy(a.checkpoint,a.device,base_checkpoint=a.base_checkpoint)
    with np.load(a.demo,allow_pickle=False) as d:policy.set_demo(d['rgb'],d['timestamp_s'],float(d['duration_s']))
    with np.load(a.input,allow_pickle=False) as d:action=policy.predict(d['rgb'],d['state'],num_steps=a.denoise_steps)
    np.savez(output,action=action,action_hz=np.int32(50))
if __name__=='__main__':main()
