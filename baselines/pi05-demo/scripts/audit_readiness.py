"""Read-only readiness audit; CPU only, no downloads or optimization."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
from data import read,sha,split_rows,validate_pairs
cfg=read(root/'configs/demo-top.json');raw=Path('/data/haoyuzhao/envs/video/zero-wam-ig10k/datasets/RoboICL/train3000');plan_path=Path('/data/haoyuzhao/exps/train/robokata-3000-finetune-20261113/cache-b1-top/plan.json')
report={'hostname':os.uname().nodename,'project':str(root),'config':cfg,'training_started_by_audit':False}
report['paths']={k:dict(path=cfg[k],exists=Path(cfg[k]).exists()) for k in ('data','assets','base_checkpoint','tokenizer')}
report['base_identity_exists']=(Path(cfg['base_checkpoint']).parent/'base_identity.json').is_file()
report['python']=sys.executable
for path in ('/data','/tmp'):
    usage=shutil.disk_usage(path);report.setdefault('disk',{})[path]=dict(free_bytes=usage.free,total_bytes=usage.total)
manifest=read(raw/'manifest.json');plan=read(plan_path);rows=split_rows(manifest,plan);validate_pairs(rows,plan['pairs'])
by_id={r['id']:r for r in rows}
report['raw_data']=dict(root=str(raw),episodes=len(rows),manifest_sha256=sha(raw/'manifest.json'),split_plan_sha256=sha(plan_path),
    action_hz=manifest['action_hz'],total_frames=sum(r['frames'] for r in rows),maximum_demo_frames_5hz=max(__import__('math').ceil(r['seconds']*5) for r in rows),
    recording_files_present=sum((raw/'records'/r['id']/'policy_recording.npz').is_file() for r in rows),
    supervision_files_present=sum((raw/'records'/r['id']/'supervision.npz').is_file() for r in rows),
    all_recording_hashes_rechecked=False,source_release_status=read(raw.parent/'release_status.json'))
report['sampling']={}
for split,pairs in plan['pairs'].items():
    ids={p['query'] for p in pairs};frames=sum(by_id[i]['frames'] for i in ids);updates=(frames//cfg['world_size'])//cfg['per_gpu_batch']
    report['sampling'][split]=dict(pairs=len(pairs),unique_queries=len(ids),query_frames=frames,updates_per_epoch=updates)
updates=report['sampling']['train']['updates_per_epoch'];report['schedule']=dict(updates_per_epoch=updates,steps=cfg['steps'],equivalent_epochs=cfg['steps']/updates,save_every_updates=updates)
report['top_rgb_estimated_bytes']=sum(r['frames'] for r in rows)*224*224*3
patch=subprocess.run([sys.executable,str(root/'vendor/openpi/examples/segmented_demo/prepare_transformers.py')],capture_output=True,text=True)
report['transformers_patch']=dict(passed=patch.returncode==0,details=json.loads(patch.stdout) if patch.returncode==0 else patch.stderr[-2000:])
import torch,transformers,numpy,jax,h5py,safetensors
report['versions']={k:getattr(v,'__version__','unknown') for k,v in dict(torch=torch,transformers=transformers,numpy=numpy,jax=jax,h5py=h5py,safetensors=safetensors).items()}
report['nccl_built']=torch.distributed.is_nccl_available()
try:
    from train import Loader
    spec=importlib.util.spec_from_file_location('audit_openpi_trainer',root/'vendor/openpi/scripts/train_pytorch.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report['trainer_import_passed']=True
except Exception as e:report['trainer_import_passed']=False;report['trainer_import_error']=f'{type(e).__name__}: {e}'
preflight=subprocess.run([sys.executable,str(root/'scripts/preflight.py'),'--config',str(root/'configs/demo-top.json')],capture_output=True,text=True)
report['preflight']=dict(passed=preflight.returncode==0,returncode=preflight.returncode,stderr=preflight.stderr[-2500:])
try:
    from huggingface_hub import HfApi
    hf=read(root/'configs/huggingface.json');api=HfApi(token=True);info=api.model_info(hf['repo_id'])
    report['huggingface']=dict(repo_id=info.id,private=info.private,files=api.list_repo_files(info.id),automatic_upload_enabled=hf['automatic_upload_enabled'])
except Exception as e:report['huggingface']=dict(error=f'{type(e).__name__}: {e}')
report['source_sha256']={p.relative_to(root).as_posix():sha(p) for p in [*root.glob('*.py'),*root.glob('configs/*.json'),root/'scripts/train.sh']}
report['assets_ready']=all(report['paths'][k]['exists'] for k in ('data','assets','base_checkpoint','tokenizer')) and report['preflight']['passed']
report['full_weight_gpu_ddp_verified']=False
report['ready_for_gpu_validation']=report['assets_ready']
report['ready_to_train']=False
report['blockers']=[]
if not report['paths']['data']['exists'] or not report['paths']['assets']['exists']:report['blockers'].append('Configured full dataset/assets absent')
if not report['paths']['base_checkpoint']['exists'] or not report['base_identity_exists']:report['blockers'].append('Configured official pi05_base weights/identity absent')
report['blockers'].append('Full-weight GPU forward/backward/optimizer/DDP capacity still unverified; no training launched')
print(json.dumps(report,indent=2,ensure_ascii=False))
