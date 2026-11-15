"""ICL3000 RGB Demo/Query contract with frozen A+B pairs and native pi05 state tokens."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

SCHEMA = 'pi05_icl3000_demo_top_hand10_v1'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()

def normalize(x, stats):
    lo, hi = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return ((np.asarray(x, np.float32)-lo)/(hi-lo+1e-6)*2-1).astype(np.float32)

def denormalize(x, stats):
    lo, hi = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return ((np.asarray(x, np.float32)+1)/2*(hi-lo+1e-6)+lo).astype(np.float32)

def action_window(action, frame, stats, horizon=50):
    ids = np.arange(frame, frame+horizon)
    out = np.pad(normalize(action[ids.clip(max=len(action)-1)], stats), ((0,0),(0,22)))
    valid = np.zeros((horizon,32), bool)
    valid[ids<len(action),:10] = True
    return out, valid

def split_rows(manifest, plan):
    rows = manifest['rows']; assert len({r['id'] for r in rows}) == len(rows)
    assert set(plan['records']) == {r['id'] for r in rows}
    held = set(plan['validation_families'])
    result = []
    for row in rows:
        r = dict(row); p = plan['records'][r['id']]
        assert r['frames'] == p['frames'] and r['task_id'] == p['task_id']
        assert r['recording_sha256'] == p['source_sha256']
        r.update(family=p['family'], split='validation' if p['family'] in held else 'train')
        result.append(r)
    # Preserve every raw episode; the frozen pairs select eligible Query/Demo roles.
    return result

def prepare_episode(arguments):
    raw,output,row,resume=arguments
    raw,output=Path(raw),Path(output);folder=output/'records'/row['id']
    if resume and (folder/'prepared.json').is_file():
        done=read(folder/'prepared.json');assert done['source_sha256']==row['recording_sha256']
        assert sha(folder/'supervision.npz')==done['labels_sha256'] and sha(folder/'rgb.npy')==done['rgb_sha256']
        with np.load(folder/'supervision.npz') as d:labels={key:d[key] for key in ('state','action')}
        return done,labels
    src=raw/'records'/row['id']/'policy_recording.npz'
    assert sha(src)==row['recording_sha256'],row['id']
    folder.mkdir(parents=True,exist_ok=resume)
    with np.load(src,allow_pickle=False) as d:
        labels={key:d[key] for key in ('state','action','timestamp_s')}
        assert np.allclose(labels['timestamp_s'],np.arange(row['frames'])/50,atol=1e-8,rtol=0)
        for key in ('state','action'):
            x=labels[key];assert x.shape==(row['frames'],10) and x.dtype==np.float32 and np.isfinite(x).all()
        np.savez(folder/'supervision.npz',**labels)
    with zipfile.ZipFile(src) as z,z.open('rgb.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        assert shape==(row['frames'],2,224,224,3) and dtype==np.uint8 and not fortran
        rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='w+',dtype=np.uint8,shape=(row['frames'],1,224,224,3))
        size=2*224*224*3
        for i in range(row['frames']):
            block=f.read(size);assert len(block)==size
            rgb[i,0]=np.frombuffer(block,np.uint8).reshape(2,224,224,3)[0]
        assert f.read(1)==b'';rgb.flush();del rgb
    st=(folder/'rgb.npy').stat()
    done=dict(id=row['id'],task_id=row['task_id'],family=row['family'],split=row['split'],frames=row['frames'],seconds=row['seconds'],
        source_sha256=row['recording_sha256'],labels_sha256=sha(folder/'supervision.npz'),rgb_bytes=st.st_size,rgb_mtime_ns=st.st_mtime_ns,rgb_sha256=sha(folder/'rgb.npy'))
    write(folder/'prepared.json',done)
    return done,{key:labels[key] for key in ('state','action')}

def prepare(raw,plan_path,output,*,limit=None,resume=False,workers=1):
    raw,output=Path(raw),Path(output);source=read(raw/'manifest.json');assert source['action_hz']==50
    rows=split_rows(source,read(plan_path));plan=read(plan_path);pairs=validate_pairs(rows,plan['pairs'])
    if limit is None:assert len(rows)==3000,'Full ICL3000 preparation requires exactly 3000 episodes'
    if output.exists() and not resume:raise FileExistsError(f'Use --resume for partial preparation: {output}')
    if limit is not None:
        if limit<4:raise ValueError('Smoke needs both endpoints of a train/validation pair')
        ids=set()
        for split in ('train','validation'):
            first=pairs[split][0];ids.update((first['demo'],first['query']))
        ids.update(r['id'] for r in rows[:max(0,limit-len(ids))]);rows=[r for r in rows if r['id'] in ids]
        pairs={k:[p for p in ps if p['demo'] in ids and p['query'] in ids] for k,ps in pairs.items()}
    assert workers>=1
    output.mkdir(parents=True,exist_ok=resume)
    signature=dict(source_manifest_sha256=sha(raw/'manifest.json'),split_plan_sha256=sha(plan_path),ids=[r['id'] for r in rows])
    if (output/'input_identity.json').exists():assert read(output/'input_identity.json')==signature
    write(output/'input_identity.json',signature)
    from concurrent.futures import ProcessPoolExecutor
    import contextlib
    parts={'state':[],'action':[]};prepared=[]
    jobs=[(str(raw),str(output),row,resume) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) if workers>1 else contextlib.nullcontext() as pool:
        results=pool.map(prepare_episode,jobs,chunksize=1) if pool else map(prepare_episode,jobs)
        for done,labels in results:
            if done['split']=='train':
                for key in parts:parts[key].append(labels[key])
            prepared.append(done)
            write(output/'progress.json',dict(prepared=len(prepared),total=len(rows),latest=done['id'],workers=workers))
            print(f"prepared {len(prepared)}/{len(rows)} {done['id']} {done['split']}",flush=True)
    stats={}
    for key,values in parts.items():
        x=np.concatenate(values);q=np.quantile(x,[.01,.99],axis=0)
        stats[key]=dict(q01=q[0].tolist(),q99=q[1].tolist(),mean=x.mean(0).tolist(),std=x.std(0).tolist())
    manifest=dict(schema=SCHEMA,smoke_only=limit is not None,rows=prepared,action_hz=50,action_dim=10,
        raw_root=str(raw.resolve()),source_manifest_sha256=sha(raw/'manifest.json'),split_plan_sha256=sha(plan_path),
        query_image_keys=['base_0_rgb'],demo_views=1,pairs=pairs,sampling='uniform_query_frames_epoch_seeded_demo',demo_sample_hz=5,
        maximum_demo_frames=max(int(np.ceil(r['seconds']*5)) for r in rows),
        split_policy='exact A+B frozen nonself independent-family pairs; no cross-split pairs')
    write(output/'manifest.json',manifest)
    write(output/'assets.json',dict(manifest_sha256=sha(output/'manifest.json'),normalization=stats,
        normalization_method='q01/q99, no clipping, training families only',action_layout='unchanged raw world-hand10'))
    write(output/'preparation_receipt.json',dict(passed=True,episodes=len(prepared),source_recordings_sha256_verified=True,
        rgb_files_sha256_recorded=True,manifest_sha256=sha(output/'manifest.json'),assets_sha256=sha(output/'assets.json'),smoke_only=limit is not None))

class Frames:
    def __init__(self,cfg,split='train',tokenizer=None,allow_smoke=False):
        self.root=Path(cfg['data']);self.manifest=read(self.root/'manifest.json')
        assert self.manifest['schema']==SCHEMA
        assert allow_smoke or not self.manifest['smoke_only'], 'Smoke data cannot train'
        all_rows=self.manifest['rows'];pairs=validate_pairs(all_rows,self.manifest['pairs'])[split]
        self.candidates={}
        for pair in pairs:self.candidates.setdefault(pair['query'],[]).append(pair['demo'])
        self.by_id={r['id']:r for r in all_rows}
        self.rows=[r for r in all_rows if r['id'] in self.candidates];assert self.rows
        self.seed=cfg['seed'];self.epoch=0;self.demo_frames=cfg['demo_frames']
        assert self.manifest['maximum_demo_frames']<=self.demo_frames,'Demo cap must cover full videos'
        self.ends=np.cumsum([r['frames'] for r in self.rows]);self.assets=read(cfg['assets'])
        assert self.assets['manifest_sha256']==sha(self.root/'manifest.json')
        self.stats=self.assets['normalization']
        if tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            tokenizer=PaligemmaTokenizer(200,tokenizer_path=Path(cfg['tokenizer']))
        self.tokenizer=tokenizer
        for r in all_rows:
            folder=self.root/'records'/r['id'];assert sha(folder/'supervision.npz')==r['labels_sha256']
            st=(folder/'rgb.npy').stat();assert st.st_size==r['rgb_bytes'] and st.st_mtime_ns==r['rgb_mtime_ns']
    def __len__(self):return int(self.ends[-1])
    @functools.lru_cache(maxsize=32)
    def labels(self,identity):
        with np.load(self.root/'records'/identity/'supervision.npz') as d:return {k:d[k] for k in ('state','action','timestamp_s')}
    @functools.lru_cache(maxsize=32)
    def rgb(self,identity):return np.load(self.root/'records'/identity/'rgb.npy',mmap_mode='r')
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        ri=int(np.searchsorted(self.ends,index,side='right'));r=self.rows[ri]
        frame=int(index-(self.ends[ri-1] if ri else 0));labels=self.labels(r['id'])
        state=np.pad(normalize(labels['state'][frame],self.stats['state']),(0,22))
        actions,valid=action_window(labels['action'],frame,self.stats['action'])
        tokens,mask=self.tokenizer.tokenize('',state)
        assert mask.sum()<200,'Discrete state exceeds tokenizer context'
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(index)]))
        demo_id=self.candidates[r['id']][int(rng.integers(len(self.candidates[r['id']])))]
        demo=self.by_id[demo_id];clock=self.labels(demo_id)['timestamp_s']
        indices,times=demo_indices(clock,demo['seconds'])
        payload=self.demo_payload(demo_id,indices,times)
        return dict(rgb=np.array(self.rgb(r['id'])[frame],copy=True),state=state,actions=actions,valid=valid,
            tokens=tokens,mask=mask,demo_id=demo_id,**payload)

    def demo_payload(self,demo_id,indices,times):
        demo_rgb=np.zeros((self.demo_frames,1,224,224,3),np.uint8)
        demo_valid=np.zeros((self.demo_frames,1),bool);demo_times=np.zeros(self.demo_frames,np.float32)
        demo_rgb[:len(indices)]=self.rgb(demo_id)[indices];demo_valid[:len(indices)]=True;demo_times[:len(indices)]=times
        return dict(demo_rgb=demo_rgb,demo_valid=demo_valid,demo_timestamps=demo_times)


def validate_pairs(rows,pairs):
    by_id={r['id']:r for r in rows}
    for split,items in pairs.items():
        assert split in ('train','validation') and items
        seen=set()
        for pair in items:
            d,q=by_id[pair['demo']],by_id[pair['query']]
            assert d['id']!=q['id'] and d['family']!=q['family'],'Demo must use independent physical start'
            assert d['split']==q['split']==split,'Cross split pair'
            assert d['task_id']==q['task_id']==pair['task_id'],'Different tasks'
            identity=(d['id'],q['id']);assert identity not in seen,'Duplicate pair';seen.add(identity)
    return pairs

def demo_indices(clock,duration):
    clock=np.asarray(clock,np.float64)
    assert clock.ndim==1 and len(clock) and clock[0]==0 and np.all(np.diff(clock)>0)
    assert np.isfinite(clock).all() and duration>clock[-1]
    targets=np.arange(int(np.ceil(duration*5)),dtype=np.float64)/5
    targets=targets[targets<duration-1e-8]
    indices=np.searchsorted(clock,targets+1e-8,side='right')-1
    assert np.all(indices>=0) and len(np.unique(indices))==len(indices)
    assert np.all(targets-clock[indices]>=-1e-8) and np.all(targets-clock[indices]<.020001)
    return indices,clock[indices].astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();prepare(a.raw,a.plan,a.output,limit=a.limit,resume=a.resume,workers=a.workers)
if __name__=='__main__':main()
