"""Explicit large-download/conversion command. Never invoked by training automatically."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
from data import write,sha
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
os.environ.setdefault('OPENPI_DATA_HOME',str(root/'download-cache'))
os.environ['PYTHONPATH']=os.pathsep.join([str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src'),os.environ.get('PYTHONPATH','')])
from openpi.shared.download import maybe_download
source='gs://openpi-assets/checkpoints/pi05_base'
checkpoint=maybe_download(source)
subprocess.run([sys.executable,str(root/'vendor/openpi/examples/convert_jax_model_to_pytorch.py'),
    '--checkpoint-dir',str(checkpoint),'--config-name','pi05_aloha','--output-path',str(out),'--precision','bfloat16'],check=True)
write(out/'base_identity.json',dict(source=source,model_sha256=sha(out/'model.safetensors'),conversion_config='pi05_aloha',
    converter_sha256=sha(root/'vendor/openpi/examples/convert_jax_model_to_pytorch.py')))
