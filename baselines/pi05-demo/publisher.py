"""CPU-only publication of atomically completed pi05 checkpoints; no optimizer upload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from data import read,sha

MODEL='demo_pi05_lora'

def canonical(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_bytes(canonical(value));temp.replace(path)

def check_access(api,repo):
    owner=api.whoami()['name'];info=api.repo_info(repo,repo_type='model')
    if owner!=repo.split('/')[0]:raise ValueError('HF owner does not match authenticated account')
    return dict(account=owner,repo_id=info.id,private=info.private)

def expected_steps(contract):
    interval,total=contract['updates_per_epoch'],contract['total_updates']
    if not isinstance(interval,int) or not isinstance(total,int) or min(interval,total)<1:raise ValueError('Invalid update budget')
    return sorted(set([*range(interval,total+1,interval),total]))

def checkpoint_payload(root,run,step):
    root,run=Path(root).resolve(),Path(run).resolve()
    raw=(run/'contract.json').read_bytes();contract=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if contract['model']!=MODEL or step not in expected_steps(contract):raise ValueError('Not a scheduled checkpoint')
    checkpoint=run/str(step)
    if not checkpoint.is_dir() or checkpoint.is_symlink():raise ValueError('Require immutable completed checkpoint directory')
    marker=read(checkpoint/'complete.json')
    if marker['contract']!=contract or marker['step']!=step:raise ValueError('Checkpoint contract/step differs')
    model=checkpoint/'model.safetensors'
    if sha(model)!=marker['model_sha256']:raise ValueError('Local checkpoint weights changed')
    cfg=contract['config'];assets=Path(cfg['assets']);tokenizer=Path(cfg['tokenizer'])
    with model.open('rb') as f:
        size=int.from_bytes(f.read(8),'little')
        if not 0<size<30000000:raise ValueError('Invalid adapter header')
        header=json.loads(f.read(size));meta=header.get('__metadata__',{})
    wanted=dict(format='pi05_demo_lora_v1',rank=str(cfg['lora_rank']),alpha=str(cfg['lora_alpha']),
        base_sha256=contract['base_sha256'],trainable_parameters=str(cfg['expected_trainable_parameters']))
    if any(meta.get(k)!=v for k,v in wanted.items()):raise ValueError('Adapter header identity differs from run contract')
    import math
    tensors={n:t for n,t in header.items() if n!='__metadata__'}
    if not tensors or sum(math.prod(t['shape']) for t in tensors.values())!=cfg['expected_trainable_parameters']:
        raise ValueError('Adapter tensor count differs from trainable budget')
    if model.stat().st_size!=8+size+max(t['data_offsets'][1] for t in tensors.values()):raise ValueError('Truncated adapter')

    if sha(assets)!=contract['assets_sha256'] or sha(tokenizer)!=contract['tokenizer_sha256']:raise ValueError('Inference assets changed')
    manifest=Path(cfg['data'])/'manifest.json'
    if sha(manifest)!=read(assets)['manifest_sha256']:raise ValueError('Dataset manifest changed')
    prefix=f'runs/{run.name}-{digest[:12]}'
    epoch,remainder=divmod(step,contract['updates_per_epoch'])
    label=f'epoch-{epoch:03d}' if remainder==0 else f'final-step-{step}'
    destination=f'{prefix}/checkpoints/{label}'
    receipt=dict(format='pi05_demo_lora_v1',step=step,completed_epochs=epoch,epoch_boundary=remainder==0,
        run_contract_sha256=digest,model_sha256=marker['model_sha256'],model_bytes=model.stat().st_size,
        assets_sha256=contract['assets_sha256'],tokenizer_sha256=contract['tokenizer_sha256'])
    inference=dict(format='pi05_demo_lora_v1',model_config=contract['model_config'],action_hz=50,
        native_action_dim=10,action_horizon=50,task_text_required=False,assets='assets.json',tokenizer='tokenizer.model',
        source=f'{prefix}/source',weights='model.safetensors',optimizer_included=False,base_source='gs://openpi-assets/checkpoints/pi05_base',base_sha256=contract['base_sha256'],
        lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],checkpoint_contains='all_trainable_parameters_only')
    files={f'{destination}/model.safetensors':model,f'{destination}/complete.json':checkpoint/'complete.json',
        f'{destination}/assets.json':assets,f'{destination}/tokenizer.model':tokenizer,
        f'{destination}/inference_config.json':canonical(inference),f'{destination}/upload.json':canonical(receipt),
        f'{prefix}/contract.json':run/'contract.json',f'{prefix}/dataset_manifest.json':manifest}
    card=f'''---
tags:
- robotics
- pi05
- visual-in-context-learning
- roboicl
---
# π0.5 + RGB Demo on RoboICL3000

Trained checkpoint available: [{label}]({destination}).

Optimizer updates: {step}; completed full Query-frame epochs: {epoch}. This checkpoint is {'an epoch boundary' if remainder==0 else 'the final partial epoch'}.

Inputs: independent top RGB Demo video sampled at 5 Hz, current Query top RGB, measured world-hand10 state. No task text or Demo actions/states are required. Outputs: 50 hand10 commands at 50 Hz.

Initialization: official OpenPI pi05_base. The project adds Demo modules while retaining pi05 discrete state tokens and time-conditioned adaRMSNorm. SigLIP and native transformer weights are frozen; attention and MLP projections use LoRA. New Demo modules and action/time projections remain trainable.

LoRA and trainable module weights, tokenizer, normalization and inference configuration are under the checkpoint directory. Frozen implementation and dependency versions are in [{prefix}/source]({prefix}/source). Dataset split and provenance are stored with each run. The matching official pi05_base is required for inference. Optimizer states are not uploaded.

Training uses RoboICL3000 with the A+B family split and independent Demo–Query pairs. A successful upload does not imply a measured task success rate; see run-specific evaluation results when available. Use remains subject to upstream OpenPI, Gemma and dataset terms.
'''
    files['README.md']=card.encode()
    for name,expected in contract['source_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=expected:raise ValueError(f'Frozen source changed: {name}')
        files[f'{prefix}/source/{name}']=p
    for name in ('README.md','vendor/openpi/LICENSE','vendor/openpi/NOTICE','configs/demo-top.json','configs/huggingface.json'):
        p=root/name
        if p.is_file():files[f'{prefix}/source/{name}']=p
    return files,receipt,destination

def remote_verify(api,repo,revision,destination,receipt):
    from huggingface_hub import get_hf_file_metadata,hf_hub_url
    metadata=get_hf_file_metadata(hf_hub_url(repo,f'{destination}/model.safetensors',revision=revision),token=True)
    if metadata.size!=receipt['model_bytes'] or metadata.etag!=receipt['model_sha256']:raise ValueError('Remote weights SHA256/size mismatch')
    import requests
    for name,key in (('assets.json','assets_sha256'),('tokenizer.model','tokenizer_sha256')):
        with requests.get(hf_hub_url(repo,f'{destination}/{name}',revision=revision),headers=api._build_hf_headers(),stream=True,timeout=(30,120)) as response:
            response.raise_for_status();h=hashlib.sha256()
            for block in response.iter_content(1024*1024):h.update(block)
        if h.hexdigest()!=receipt[key]:raise ValueError(f'Remote {name} checksum differs')

def publish_checkpoint(api,repo,root,run,step,*,verify=remote_verify):
    from huggingface_hub import CommitOperationAdd
    files,receipt,destination=checkpoint_payload(root,run,step)
    head=api.repo_info(repo,repo_type='model').sha
    marker=f'{destination}/upload.json'
    if api.file_exists(repo,marker,repo_type='model',revision=head):
        from huggingface_hub import hf_hub_download
        existing=hf_hub_download(repo_id=repo,filename=marker,repo_type='model',revision=head,token=True)
        if read(existing)!=receipt:raise ValueError('Different checkpoint already occupies this destination')
        verify(api,repo,head,destination,receipt)
        return dict(step=step,commit=head,destination=destination,verified=True,already_uploaded=True,receipt=receipt)
    if api.file_exists(repo,f'{destination}/model.safetensors',repo_type='model',revision=head):raise ValueError('Remote weights lack matching upload receipt; refusing overwrite')
    commit=api.create_commit(repo_id=repo,repo_type='model',parent_commit=head,
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=value) for name,value in files.items()],
        commit_message=f'{Path(run).name}: {"epoch" if receipt["epoch_boundary"] else "final"} checkpoint step {step}')
    verify(api,repo,commit.oid,destination,receipt)
    return dict(step=step,commit=commit.oid,destination=destination,verified=True,already_uploaded=False,receipt=receipt)

def prune_verified_checkpoints(run,contract,uploaded,keep_last):
    if keep_last<1:raise ValueError('Keep at least one full local checkpoint')
    run=Path(run).resolve()
    complete=sorted(int(p.name) for p in run.iterdir() if p.is_dir() and p.name.isdigit() and (p/'complete.json').is_file())
    protected=set(complete[-keep_last:]);removed=[]
    for step in complete:
        if step in protected or step not in uploaded:continue
        item=uploaded[step];folder=run/str(step);marker=read(folder/'complete.json')
        if folder.is_symlink() or folder.resolve().parent!=run:raise ValueError('Unsafe checkpoint cleanup target')
        if not item.get('verified') or marker['contract']!=contract or marker['model_sha256']!=item['receipt']['model_sha256']:
            raise ValueError('Refusing cleanup of unverified checkpoint')
        import shutil
        shutil.rmtree(folder);removed.append(step)
        print(json.dumps(dict(pruned_local_checkpoint=step,remote_commit=item['commit'],keep_last=keep_last)),flush=True)
    return removed

def run_publisher(api,repo,root,run,*,follow=False,parent_pid=None,poll=10):
    import fcntl
    run=Path(run);state=run/'upload';state.mkdir(exist_ok=True)
    with (state/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=read(run/'contract.json');steps=expected_steps(contract);uploaded={};failures={}
        keep_last=contract.get('config',{}).get('keep_last_full_checkpoints')
        digest=sha(run/'contract.json')
        if (state/'status.json').is_file():
            previous=read(state/'status.json')
            if previous.get('repo_id')!=repo:raise ValueError('Uploader repository changed')
            for item in previous.get('verified_checkpoints',[]):
                if item['step'] not in steps or not item.get('verified') or item.get('receipt',{}).get('run_contract_sha256')!=digest:
                    raise ValueError('Saved upload status does not match this run')
                remote_verify(api,repo,item['commit'],item['destination'],item['receipt'])
                uploaded[item['step']]=item

        while True:
            finished=read(state/'training-finished.json') if (state/'training-finished.json').exists() else None
            orphan=parent_pid is not None and os.getppid()!=parent_pid
            terminal=finished is not None or orphan or not follow
            for step in steps:
                if step in uploaded or not (run/str(step)/'complete.json').is_file():continue
                try:
                    # A terminal drain gets bounded retries; transient errors during training are retried by the next poll.
                    for attempt in range(5 if terminal else 1):
                        try:result=publish_checkpoint(api,repo,root,run,step);break
                        except ValueError:raise
                        except Exception:
                            if attempt+1==(5 if terminal else 1):raise
                            time.sleep(min(30,2**(attempt+1)))
                    uploaded[step]=result;failures.pop(step,None)
                    print(json.dumps(result),flush=True)
                except ValueError:
                    atomic_json(state/'status.json',dict(state='failed_integrity',step=step,verified_checkpoints=list(uploaded.values())))
                    raise
                except Exception as error:
                    failures[step]=dict(error_type=type(error).__name__,at=time.time())
            missing=[s for s in steps if s not in uploaded]
            success=terminal and not failures and (not missing or not (finished or {}).get('training_ok',False))
            status=dict(repo_id=repo,state=('complete' if not missing else 'partial') if terminal and success else ('failed' if terminal else ('retry_wait' if failures else 'watching')),
                verified_checkpoints=list(uploaded.values()),pending_steps=missing,errors=failures,training_finished=finished,orphaned=orphan)
            atomic_json(state/'status.json',status)
            if keep_last is not None:prune_verified_checkpoints(run,contract,uploaded,keep_last)
            if terminal:
                if failures or ((finished or {}).get('training_ok',False) and missing):raise RuntimeError('Checkpoint publication incomplete; inspect upload/status.json and retry publisher.py --run')
                return status
            time.sleep(poll)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--run',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--follow',action='store_true');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    from huggingface_hub import HfApi
    api=HfApi(token=True);print(json.dumps(check_access(api,a.repo)),flush=True)
    if a.check:return
    if not a.run or not a.run.is_absolute():p.error('--run must be absolute')
    run_publisher(api,a.repo,a.root,a.run,follow=a.follow,parent_pid=a.parent_pid)
if __name__=='__main__':main()
