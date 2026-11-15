"""Verify prepared ICL assets and recompute train-only normalization; no training."""
from pathlib import Path
import json,sys
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'vendor/openpi/src'),str(root/'vendor/openpi/packages/openpi-client/src')]
from data import read,write,sha,Frames,validate_pairs
from inputs import observation_from_samples
cfg=read(root/'configs/demo-top.json');data=Path(cfg['data']);manifest=read(data/'manifest.json');assets=read(cfg['assets']);receipt=read(data/'preparation_receipt.json')
assert receipt['passed'] and not receipt['smoke_only'] and receipt['episodes']==3000
assert len(manifest['rows'])==3000 and not manifest['smoke_only']
assert receipt['manifest_sha256']==sha(data/'manifest.json')==assets['manifest_sha256']
assert receipt['assets_sha256']==sha(cfg['assets'])
validate_pairs(manifest['rows'],manifest['pairs'])
parts={'state':[],'action':[]}
for row in manifest['rows']:
    folder=data/'records'/row['id'];done=read(folder/'prepared.json');assert done==row
    assert sha(folder/'supervision.npz')==row['labels_sha256']
    rgb=np.load(folder/'rgb.npy',mmap_mode='r');st=(folder/'rgb.npy').stat()
    assert rgb.shape==(row['frames'],1,224,224,3) and rgb.dtype==np.uint8
    assert st.st_size==row['rgb_bytes'] and st.st_mtime_ns==row['rgb_mtime_ns'];del rgb
    with np.load(folder/'supervision.npz') as d:
        for key in parts:
            values=d[key];assert values.shape==(row['frames'],10) and np.isfinite(values).all()
            if row['split']=='train':parts[key].append(values)
for key,values in parts.items():
    x=np.concatenate(values);q=np.quantile(x,[.01,.99],axis=0)
    for name,actual in dict(q01=q[0],q99=q[1],mean=x.mean(0),std=x.std(0)).items():
        np.testing.assert_allclose(actual,assets['normalization'][key][name],atol=1e-8,rtol=1e-7)
train=Frames(cfg);validation=Frames(cfg,split='validation')
samples=[train[i] for i in range(cfg['per_gpu_batch'])];obs=observation_from_samples(samples)
assert tuple(obs.demo_rgb.shape)==(16,60,1,3,224,224) and tuple(obs.state.shape)==(16,32)
assert all(np.isfinite(s['actions']).all() and s['valid'].sum()>0 and s['mask'].sum()<200 for s in samples)
for dataset in (train,validation):
    sample=dataset[len(dataset)-1];assert sample['valid'].sum()==10
report=dict(passed=True,episodes=3000,total_frames=sum(r['frames'] for r in manifest['rows']),
    training_queries=len(train.rows),training_frames=len(train),validation_queries=len(validation.rows),validation_frames=len(validation),
    training_pairs=len(manifest['pairs']['train']),validation_pairs=len(manifest['pairs']['validation']),
    normalization_independently_recomputed=True,normalization_train_only=True,source_hashes_verified_during_preparation=True,
    rgb_hashes_recorded_during_preparation=True,all_labels_and_rgb_headers_verified=True,physical_batch_16_collation_passed=True,
    updates_per_epoch=len(train)//cfg['global_batch'],manifest_sha256=sha(data/'manifest.json'),assets_sha256=sha(cfg['assets']),
    gpu_used=False,training_started=False)
write(root/'verification/full-data-verified.json',report);print(json.dumps(report,indent=2))
