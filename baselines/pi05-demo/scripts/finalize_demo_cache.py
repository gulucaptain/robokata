"""Commit an index only after every shard and every immutable file is complete."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
from data import read,write,sha
from demo_cache import identity,file_receipt,manifest_digest

def finalize(cfg,out,shards=4):
    out=Path(out);signature=identity(cfg);manifest=read(Path(cfg['data'])/'manifest.json')
    wanted={p['demo'] for pairs in manifest['pairs'].values() for p in pairs}
    by={r['id']:r for r in manifest['rows']};records={};total=0
    for shard in range(shards):
        summary=read(out/f'shard-{shard}-complete.json')
        if summary['identity']!=signature or summary['digest']!=manifest_digest(summary['records']):raise ValueError('Shard identity/digest mismatch')
        for key,digest in summary['records']:
            if key in records:raise ValueError('Duplicate shard ownership')
            _,marker=file_receipt(out,by[key],signature,verify_hash=False)
            if marker['sha256']!=digest:raise ValueError('Worker/file receipt mismatch')
            records[key]=digest;total+=marker['bytes']
    if set(records)!=wanted:raise ValueError('Incomplete or extraneous Demo cache')
    producer=out.parent
    import inspect,torch
    from transformers import SiglipVisionModel
    result=dict(identity=signature,records=records,bytes=total,complete=True,shards=shards,
        producer_source_sha256={name:sha(producer/name) for name in ('scripts/build_demo_cache.py','demo_cache.py','data.py')},
        torch_version=torch.__version__,siglip_implementation_sha256=sha(inspect.getfile(SiglipVisionModel)))
    write(out/'index.json',result)
    print(dict(complete=True,recordings=len(records),GiB=total/2**30),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    finalize(read(a.config),a.output)
