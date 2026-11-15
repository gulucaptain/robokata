"""One production-architecture loss/backward comparison on GPU0; no updates/save."""
import argparse
from pathlib import Path
import sys,os,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
import torch,jax
from data import Frames,read,write
from cached_data import CachedFrames
from inputs import observation_from_samples
from model import Pi05DemoConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 validation only')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info();budget=min(16*2**30,free-4*2**30)
    if budget<16*2**30:raise RuntimeError('Need 16GiB validation capacity while preserving other jobs')
    torch.cuda.set_per_process_memory_fraction(budget/total)
    cfg=read(a.config);raw=Frames(cfg)[0];cached=CachedFrames(cfg)[0]
    torch.manual_seed(12)
    model=Pi05DemoConfig(lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],
        checkpoint_strategy='layer',demo_frames=60,query_encode_batch=4,pool_valid_only=True).create_pytorch()
    model.load_base_checkpoint(cfg['base_checkpoint']);model=model.to('cuda').train();model.gradient_checkpointing_enable()
    selected=['action_out_proj.weight','demo_pool.key.weight','time_mlp_in.weight',
        'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.lora_b',
        'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.lora_b']
    params=dict(model.named_parameters());results=[];reference=None
    from fast_execution import sdpa_attention
    from transformers.models.gemma import modeling_gemma
    calls=[]
    def tracked(*args,**kwargs):
        calls.append(1)
        return sdpa_attention(*args,**kwargs)
    modeling_gemma.ALL_ATTENTION_FUNCTIONS.register('pi05_demo_sdpa',tracked)
    noise=torch.randn(1,50,32,device='cuda');t=torch.tensor([.5],device='cuda')
    action=torch.from_numpy(raw['actions'])[None].to('cuda');valid=torch.from_numpy(raw['valid'])[None].to('cuda')
    paths=[('online_eager',False,False,'eager',raw),('cached_eager_full',True,False,'eager',cached),
           ('cached_eager_trimmed',True,True,'eager',cached),('cached_sdpa_trimmed',True,True,'sdpa',cached),
           ('cached_sdpa_full',True,False,'sdpa',cached)]
    for name,fast,trim,backend,sample in paths:
        calls.clear()
        obs=observation_from_samples([sample],raw_query=fast,trim_prompt=trim)
        with torch.cuda.device(0):obs=jax.tree.map(lambda x:x.pin_memory(),obs)
        obs=jax.tree.map(lambda x:x.to('cuda',non_blocking=True),obs)
        model.paligemma_with_expert.joint_attention_backend=backend
        torch.manual_seed(71);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        loss=model(obs,action,noise=noise,time=t,action_valid=valid).mean();loss.backward();torch.cuda.synchronize()
        row=dict(path=name,loss=float(loss),seconds=time.perf_counter()-start,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,prompt_length=obs.tokenized_prompt.shape[-1],sdpa_calls=len(calls))
        if backend=='sdpa' and not calls:raise RuntimeError('Compressed production decoder did not execute SDPA')
        gradients={name:params[name].grad.detach().float().cpu() for name in selected}
        if reference is None:reference=gradients
        else:
            row['gradient_relative_l2']={name:float((gradients[name]-reference[name]).norm()/reference[name].norm().clamp_min(1e-9)) for name in selected}
            row['within_5percent_gradient']=max(row['gradient_relative_l2'].values())<=.05
        results.append(row);print(row,flush=True);model.zero_grad(set_to_none=True)
        del obs,loss
    relative_loss={row['path']:abs(results[0]['loss']-row['loss'])/max(abs(results[0]['loss']),1e-9) for row in results[1:]}
    # Report every path separately; select only validated paths in the final config.
    result=dict(physical_gpu=0,batch=1,optimizer_updates=0,relative_loss_error=relative_loss,
                results=results,scope='production_architecture_batch1_not_DDP_batch256_throughput')
    write(ROOT/'verification/production-gpu0.json',result)

if __name__=='__main__':main()
