"""Rank-zero lifecycle of the independent CPU uploader."""
import os
from pathlib import Path
import subprocess
import sys
from publisher import atomic_json

class EpochPublisher:
    def __init__(self,root,run,repo):
        self.root,self.run,self.repo=Path(root),Path(run),repo
        self.state=self.run/'upload';self.process=self.log=self.lock=None
    def start(self):
        import fcntl
        self.state.mkdir(exist_ok=True)
        self.lock=(self.state/'controller.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (self.state/'publisher.lock').open('a') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='0',HF_HUB_CACHE=str(self.root/'tmp/hub'))
        command=[sys.executable,str(self.root/'publisher.py'),'--repo',self.repo]
        subprocess.run([*command,'--check'],env=env,check=True)
        (self.state/'training-finished.json').unlink(missing_ok=True)
        self.log=(self.state/'publisher.log').open('ab',buffering=0)
        self.process=subprocess.Popen([*command,'--root',str(self.root),'--run',str(self.run),'--follow','--parent-pid',str(os.getpid())],
            env=env,stdout=self.log,stderr=subprocess.STDOUT)
        return self
    def finish(self,training_ok):
        atomic_json(self.state/'training-finished.json',dict(training_ok=training_ok))
        try:
            rc=self.process.wait()
            if rc and training_ok:raise RuntimeError('Training finished but checkpoint upload failed; inspect upload/status.json')
        finally:
            if self.log:self.log.close()
            if self.lock:self.lock.close()
