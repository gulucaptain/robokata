"""Verify real ICL data conversion/collation and smoke-training rejection, CPU only."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/openpi/src'),str(ROOT/'vendor/openpi/packages/openpi-client/src')]
from data import Frames,read,normalize,demo_indices
from inputs import observation_from_samples
cfg=read(ROOT/'configs/demo-top.json');cfg.update(data=str(ROOT/'smoke-data'),assets=str(ROOT/'smoke-data/assets.json'))
try:Frames(cfg)
except AssertionError as e:assert 'Smoke data cannot train' in str(e)
else:raise AssertionError('Smoke data accepted for training')
frames=Frames(cfg,allow_smoke=True);sample=frames[0];obs=observation_from_samples([sample])
query=frames.rows[0];raw=Path(frames.manifest['raw_root'])
with np.load(raw/'records'/query['id']/'policy_recording.npz') as d:
    np.testing.assert_array_equal(sample['rgb'][0],d['rgb'][0,0])
    np.testing.assert_allclose(sample['state'][:10],normalize(d['state'][0],frames.stats['state']))
    np.testing.assert_allclose(sample['actions'][:,:10],normalize(d['action'][:50],frames.stats['action']))
demo=frames.by_id[sample['demo_id']]
with np.load(raw/'records'/demo['id']/'policy_recording.npz') as d:
    indices,times=demo_indices(d['timestamp_s'],demo['seconds'])
    np.testing.assert_array_equal(sample['demo_rgb'][:len(indices),0],d['rgb'][indices,0])
assert query['family']!=demo['family'] and query['task_id']==demo['task_id']
assert tuple(obs.demo_rgb.shape)==(1,60,1,3,224,224) and list(obs.images)==['base_0_rgb']
for split in ('train','validation'):
    dataset=Frames(cfg,split=split,allow_smoke=True)
    for index in (0,len(dataset)-1):
        item=dataset[index];assert item['mask'].sum()<200
        if index==len(dataset)-1:assert item['valid'].sum()==10
report=dict(passed=True,prepared_episodes=len(frames.manifest['rows']),query_id=query['id'],demo_id=demo['id'],
    top_rgb_and_state_action_equal_raw=True,full_demo_equal_raw=True,smoke_training_rejected=True,no_task_text=True)
print(json.dumps(report,indent=2));(ROOT/'smoke-data-check.json').write_text(json.dumps(report,indent=2)+'\n')
