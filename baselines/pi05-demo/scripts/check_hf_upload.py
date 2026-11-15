"""Check real Hub write/read integrity with small JSON only; never upload fake weights."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from publisher import check_access,canonical,atomic_json
from data import read
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
import requests
cfg=read(root/'configs/huggingface.json');api=HfApi(token=True);access=check_access(api,cfg['repo_id'])
probe=canonical(dict(purpose='verify configured checkpoint upload access; no model weights',timestamp=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),training_started=False))
name='verification/upload-channel.json'
commit=api.create_commit(repo_id=cfg['repo_id'],repo_type='model',operations=[
    CommitOperationAdd(path_in_repo=name,path_or_fileobj=probe),
    CommitOperationAdd(path_in_repo='training/upload_policy.json',path_or_fileobj=canonical(cfg))],
    commit_message='Configure epoch checkpoint publication and verify upload channel')
response=requests.get(hf_hub_url(cfg['repo_id'],name,revision=commit.oid),headers=api._build_hf_headers(),timeout=(30,120));response.raise_for_status()
assert hashlib.sha256(response.content).digest()==hashlib.sha256(probe).digest()
report=dict(passed=True,**access,commit=commit.oid,remote_sha256_verified=True,trained_weights_uploaded=False,automatic_upload_enabled=True,large_checkpoint_upload_tested=False)
atomic_json(root/'verification/huggingface-upload-check.json',report);print(json.dumps(report,indent=2))
