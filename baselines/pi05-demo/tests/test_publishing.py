"""Publisher integrity, epoch selection, retries and final-drain tests; no Hub writes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from publisher import atomic_json,checkpoint_payload,expected_steps,publish_checkpoint,run_publisher
from data import sha,read
from publishing import EpochPublisher

class FakeApi:
    def __init__(self,root):self.root=Path(root);self.sha='initial';self.commits=0;self.fail_once=False
    def repo_info(self,*args,**kwargs):return SimpleNamespace(sha=self.sha)
    def file_exists(self,repo,name,**kwargs):return (self.root/name).is_file()
    def create_commit(self,**kwargs):
        if self.fail_once:self.fail_once=False;raise ConnectionError('injected transient failure')
        self.commits+=1;self.sha=f'commit-{self.commits}'
        for operation in kwargs['operations']:
            target=self.root/operation.path_in_repo;target.parent.mkdir(parents=True,exist_ok=True)
            value=operation.path_or_fileobj;target.write_bytes(value if isinstance(value,bytes) else Path(value).read_bytes())
        return SimpleNamespace(oid=self.sha)
    def download(self,repo_id,filename,**kwargs):return str(self.root/filename)

def verify(api,repo,revision,destination,receipt):
    if sha(api.root/destination/'model.safetensors')!=receipt['model_sha256']:raise ValueError('Remote weights corrupted')
    for name,key in (('assets.json','assets_sha256'),('tokenizer.model','tokenizer_sha256')):
        if sha(api.root/destination/name)!=receipt[key]:raise ValueError('Remote asset corrupted')

class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT);self.root=Path(self.temp.name);self.run=self.root/'runs/run';self.run.mkdir(parents=True)
        self.remote=FakeApi(self.root/'remote');self.data=self.root/'data';self.data.mkdir()
        atomic_json(self.data/'manifest.json',{'rows':[]});atomic_json(self.data/'assets.json',{'manifest_sha256':sha(self.data/'manifest.json')})
        (self.root/'tokenizer.model').write_bytes(b'test tokenizer');(self.root/'model.py').write_text('model source\n')
        self.contract=dict(model='demo_pi05_lora',updates_per_epoch=4,total_updates=10,model_config={},
            config=dict(data=str(self.data),assets=str(self.data/'assets.json'),tokenizer=str(self.root/'tokenizer.model'),lora_rank=232,lora_alpha=464,expected_trainable_parameters=4),base_sha256='test-base',
            assets_sha256=sha(self.data/'assets.json'),tokenizer_sha256=sha(self.root/'tokenizer.model'),source_sha256={'model.py':sha(self.root/'model.py')})
        atomic_json(self.run/'contract.json',self.contract)
        for step in (4,8,10):
            folder=self.run/str(step);folder.mkdir();
            import numpy as np
            from safetensors.numpy import save_file
            save_file({'test_weight':np.full(4,step,dtype=np.float32)},str(folder/'model.safetensors'),metadata=dict(format='pi05_demo_lora_v1',rank='232',alpha='464',base_sha256='test-base',trainable_parameters='4'))
            atomic_json(folder/'complete.json',dict(step=step,contract=self.contract,model_sha256=sha(folder/'model.safetensors')))
    def tearDown(self):self.temp.cleanup()
    def publish(self,step):return publish_checkpoint(self.remote,'owner/repo',self.root,self.run,step,verify=verify)
    def test_controller_starts_cpu_worker_and_marks_finish(self):
        process=SimpleNamespace(wait=lambda:0)
        with patch('publishing.subprocess.run') as check,patch('publishing.subprocess.Popen',return_value=process) as spawn:
            controller=EpochPublisher(self.root,self.run,'owner/repo').start()
            self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertIn('--follow',spawn.call_args.args[0]);check.assert_called_once()
            controller.finish(True)
        self.assertTrue(read(self.run/'upload/training-finished.json')['training_ok'])
    def test_upload_failure_surfaces_after_training(self):
        controller=EpochPublisher(self.root,self.run,'owner/repo');controller.process=SimpleNamespace(wait=lambda:1)
        with self.assertRaises(RuntimeError):controller.finish(True)
    def test_epoch_and_final_selection(self):
        self.assertEqual(expected_steps(self.contract),[4,8,10])
        files,receipt,destination=checkpoint_payload(self.root,self.run,4)
        self.assertTrue(receipt['epoch_boundary']);self.assertTrue(destination.endswith('epoch-001'))
        self.assertFalse(any(name.endswith(('optimizer.pt','metadata.pt')) for name in files))
        self.assertTrue(any(name.endswith('source/model.py') for name in files))
        _,receipt,destination=checkpoint_payload(self.root,self.run,10)
        self.assertFalse(receipt['epoch_boundary']);self.assertEqual(receipt['completed_epochs'],2);self.assertTrue(destination.endswith('final-step-10'))
    def test_idempotence_and_remote_corruption(self):
        with patch('huggingface_hub.hf_hub_download',self.remote.download):
            first=self.publish(4);again=self.publish(4);self.assertTrue(again['already_uploaded']);self.assertEqual(self.remote.commits,1)
            (self.remote.root/first['destination']/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):self.publish(4)
    def test_local_corruption_and_frozen_source(self):
        (self.run/'4/model.safetensors').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.publish(4)
        (self.root/'model.py').write_text('changed')
        with self.assertRaises(ValueError):self.publish(8)
    def test_partial_checkpoint_ignored_and_success_drain(self):
        (self.run/'upload').mkdir();atomic_json(self.run/'upload/training-finished.json',dict(training_ok=True))
        self.remote.fail_once=True
        original=publish_checkpoint
        with patch('publisher.publish_checkpoint',lambda api,repo,root,run,step:original(api,repo,root,run,step,verify=verify)),patch('publisher.time.sleep'):
            result=run_publisher(self.remote,'owner/repo',self.root,self.run)
        self.assertEqual(result['state'],'complete');self.assertEqual(len(result['verified_checkpoints']),3)
    def test_missing_epoch_is_failure_when_training_reports_success(self):
        (self.run/'4/complete.json').unlink();(self.run/'upload').mkdir();atomic_json(self.run/'upload/training-finished.json',dict(training_ok=True))
        original=publish_checkpoint
        with patch('publisher.publish_checkpoint',lambda api,repo,root,run,step:original(api,repo,root,run,step,verify=verify)):
            with self.assertRaises(RuntimeError):run_publisher(self.remote,'owner/repo',self.root,self.run)
        self.assertEqual(read(self.run/'upload/status.json')['state'],'failed')

if __name__=='__main__':unittest.main(verbosity=2)
