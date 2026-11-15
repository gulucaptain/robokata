"""Create isolated Transformers overlay; reuse existing dependencies read-only."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--base-python',required=True);p.add_argument('--target',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];target=Path(a.target).resolve()
if target.exists():raise FileExistsError(target)
info=json.loads(subprocess.check_output([a.base_python,'-c',"import json,sysconfig,transformers;print(json.dumps(dict(site=sysconfig.get_paths()['purelib'],transformers=transformers.__path__[0],version=transformers.__version__)))"],text=True))
assert info['version']=='4.53.2'
subprocess.run([a.base_python,'-m','venv','--without-pip',str(target)],check=True)
python=str(target/'bin/python')
site=Path(subprocess.check_output([python,'-c',"import sysconfig;print(sysconfig.get_paths()['purelib'])"],text=True).strip())
shutil.copytree(info['transformers'],site/'transformers',ignore=shutil.ignore_patterns('__pycache__'))
(site/'read_only_dependencies.pth').write_text(info['site']+'\n')
subprocess.run([python,str(root/'vendor/openpi/examples/segmented_demo/prepare_transformers.py'),'--apply'],check=True)
(root/'environment.json').write_text(json.dumps(dict(base_python=a.base_python,dependency_site=info['site'],python=python,transformers='private copy with OpenPI replacement'),indent=2)+'\n')
