"""Development-only model selection followed by one locked test pass."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import joblib
from adapters import External

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ['PSYPATH_GSIT_OUTPUT']).resolve()
BASE.mkdir(parents=True,exist_ok=True)
CACHE=Path(os.environ['PSYPATH_CACHE']).resolve()
CONTRACT=Path(os.environ['PSYPATH_CONTRACT']).resolve()
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import run_published_models_on_meter_cache as m
m.configure_runtime(1)
TASKS=['daic_depression','edaic_depression','edaic_ptsd']
NAMES=['gsit']
SEEDS=[7,17,27,41,53]

def write(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    tmp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def threshold(y,p):
    grid=np.linspace(.1,.9,161)
    h=p[None,:]>=grid[:,None]; yy=y[None,:]
    tp=(h&(yy==1)).sum(1);tn=(~h&(yy==0)).sum(1)
    fp=(h&(yy==0)).sum(1);fn=(~h&(yy==1)).sum(1)
    f1=2*tp/np.maximum(2*tp+fp+fn,1);f0=2*tn/np.maximum(2*tn+fp+fn,1)
    w=(f1*(y==1).sum()+f0*(y==0).sum())/len(y)
    i=max(range(len(grid)),key=lambda i:(w[i],(f1[i]+f0[i])/2,-abs(grid[i]-.5),grid[i]))
    return float(grid[i])

def worker(mode,task,name,seed):
    out=BASE/'runs'/task/name/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    m.seed_all(seed)
    cache=joblib.load(BASE/task/'train_dev.joblib')
    records=cache['records']; net=External(name)
    device=torch.device('cpu')
    if mode=='train':
        start=time.time()
        loaders={s:m.make_loader(rr,16,train=s=='train',seed=seed,weighted_sampling=False) for s,rr in records.items()}
        counts=np.bincount([r['label'] for r in records['train']],minlength=2)
        weights=torch.tensor(len(records['train'])/(2*counts),dtype=torch.float32)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
        bestloss=float('inf');bestkey=None;stale=0;history=[];selected={}
        for epoch in range(1,101):
            net.train(); losses=[];aux=[]
            for batch in loaders['train']:
                opt.zero_grad(set_to_none=True)
                logits=net(*(batch[k] for k in ['text','audio','visual','length']))
                ce=torch.nn.functional.cross_entropy(logits,batch['label'],weight=weights)
                loss=ce+net.auxiliary
                assert torch.isfinite(loss), (task,name,seed,epoch,'loss')
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None), 'gradient'
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step();losses.append(float(ce.detach()));aux.append(float(net.auxiliary.detach()))
            ids,y,p=m.predict(net,loaders['dev'],device)
            assert np.isfinite(p).all()
            th=threshold(y,p);dev=m.metric_block(y,p,th)
            bce=float(torch.nn.functional.binary_cross_entropy(torch.tensor(p),torch.tensor(y,dtype=torch.float64)))
            row=dict(epoch=epoch,train_ce=float(np.mean(losses)),auxiliary=float(np.mean(aux)),dev_bce=bce,development=dev,probability_sd=float(np.std(p)))
            history.append(row);key=(dev['weighted_f1'],dev['auroc'],-bce);improved=[]
            if bce<bestloss-1e-5:bestloss=bce;stale=0;improved.append('bce')
            else:stale+=1
            if bestkey is None or key>bestkey:bestkey=key;improved.append('weighted_f1')
            for criterion in improved:
                path=out/f'{criterion}.pt';torch.save(net.state_dict(),path)
                selected[criterion]=dict(epoch=epoch,threshold=th,development=dev,checkpoint_sha256=sha(path),development_predictions=dict(pid=ids,label=y.tolist(),probability=p.tolist()))
            write(out/'progress.json',dict(task=task,model=name,seed=seed,history=history,elapsed_s=time.time()-start))
            if epoch==10:
                assert max(h['probability_sd'] for h in history)>1e-5, 'constant model; stop before test'
            if epoch>=20 and stale>=15:break
        write(out/'LOCK.json',dict(task=task,model=name,seed=seed,selected=selected,history=history,elapsed_s=time.time()-start,parameter_count=sum(p.numel() for p in net.parameters()),primary='weighted_f1',protocol_sha256=sha(BASE/'PROTOCOL.json')))
    else:
        assert (BASE/'ALL_LOCKED.json').exists()
        lock=json.loads((out/'LOCK.json').read_text())
        raw=m.load_records(CACHE,CONTRACT,task,allowed_splits=('test',),split_policy='official')['test']
        loader=m.make_loader([m.adapt_record(r,cache['adapters'],120) for r in raw],16,train=False,seed=seed)
        results={}
        for criterion,v in lock['selected'].items():
            path=out/f'{criterion}.pt';assert sha(path)==v['checkpoint_sha256']
            net.load_state_dict(torch.load(path,weights_only=True))
            ids,y,p=m.predict(net,loader,device)
            results[criterion]=dict(metrics=m.metric_block(y,p,v['threshold']),pid=ids,label=y.tolist(),probability=p.tolist(),threshold=v['threshold'])
        write(out/'TEST.json',dict(task=task,model=name,seed=seed,tests=results,lock_sha256=sha(out/'LOCK.json')))

def main():
    os.chdir(ROOT)
    protocol=json.loads((ROOT/'configs/gsit.json').read_text())
    if (BASE/'PROTOCOL.json').exists():assert json.loads((BASE/'PROTOCOL.json').read_text())==protocol
    else:write(BASE/'PROTOCOL.json',protocol)
    for task in TASKS:
        dest=BASE/task;dest.mkdir(exist_ok=True)
        if not (dest/'train_dev.joblib').exists():
            raw=m.load_records(CACHE,CONTRACT,task,allowed_splits=('train','dev'),split_policy='official')
            assert not set(r['pid'] for r in raw['train']) & set(r['pid'] for r in raw['dev'])
            adapters=m.fit_adapters(raw['train'],32)
            joblib.dump(dict(records={s:[m.adapt_record(r,adapters,120) for r in rr] for s,rr in raw.items()},adapters=adapters),dest/'train_dev.joblib')
    pairs=[(t,n,s) for s in SEEDS for t in TASKS for n in NAMES]
    def run(pair,mode='train'):
        task,name,seed=pair;out=BASE/'runs'/task/name/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/('LOCK.json' if mode=='train' else 'TEST.json')).exists():return
        with (out/f'{mode}.log').open('w') as log:
            p=subprocess.run([sys.executable,__file__,'--worker',mode,task,name,str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(str(out/f'{mode}.log'))
        print(task,name,seed,mode,'complete',flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(run,pairs))
    locks={str(p.relative_to(BASE)):sha(p) for p in (BASE/'runs').glob('*/*/*/LOCK.json')}
    assert len(locks)==15
    write(BASE/'ALL_LOCKED.json',locks)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda p:run(p,'test'),pairs))
    write(BASE/'COMPLETE.json',dict(test_runs=15,completed=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    if '--worker' in sys.argv:
        mode,task,name,seed=sys.argv[-4:];worker(mode,task,name,int(seed))
    else:main()
