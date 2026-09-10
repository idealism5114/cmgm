"""Formal full-information benchmark. Training is explicit via --stage tune/final/all."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import joblib
import numpy as np
import torch
from cmgm import config
from cmgm.models.formal_baselines_v2 import ORDER,CLASSICAL,NEURAL,KEYS,make_neural
from cmgm.scripts.formal_v2_protocol import (PROTOCOL,SEEDS,grid,classical_model,loader,atomic_json,sha,
    input_audit,arrays,metrics,distribution,neural_predictions,classical_predictions,neural_train,seed_all)
from cmgm.scripts.formal_v2_audit import ROOT,provenance,sanity


def save(r,out):
    from cmgm.scripts.formal_v2_report import report
    report(r,out)
    atomic_json(out/'results.json',r)


def job_id(stage,name,seed,candidate=None):
    return f'{stage}_{KEYS[name]}_seed{seed}'+('' if candidate is None else f'_candidate{candidate}')


def prepare(args,data):
    if config.SEQ_LEN!=20 or config.FEATURE_DIM!=21 or list(config.MULTI_HORIZONS)!=[1,5,10,20] or config.TARGET_TYPE!='return':
        raise ValueError('Formal D0B data configuration differs; STOP')
    if config.HUBER_DELTA!=.02:raise ValueError('Huber delta differs')
    prov=provenance();audit=input_audit(data)
    if not audit['PASS']:raise AssertionError('Input equality/target-order failure')
    root=Path(args.output_dir)
    if args.resume:
        options=sorted(root.glob('*/results.json'))
        out=options[-1].parent if args.resume=='latest' and options else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['input_audit']!=audit:raise ValueError('Protocol/data differs from saved experiment')
        if r['provenance']!=prov:raise ValueError('Official source/package versions changed; STOP')
        if r['cpu_jobs']!=args.cpu_jobs:raise ValueError('CPU resource policy differs from saved run')
    else:
        out=root/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        r=dict(status='PREFLIGHT',protocol=PROTOCOL,provenance=prov,input_audit=audit,sanity={},jobs={},selected={},formal={},
            preflight_complete=False,selection_frozen=False,cpu_jobs=args.cpu_jobs,
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            checkpoint_dir=str((Path(args.checkpoint_dir)/out.name).resolve()),
            historical_checkpoint=str(Path(args.d0b_checkpoint).resolve()),historical_checksum=sha(args.d0b_checkpoint))
    if sha(args.d0b_checkpoint)!=r['historical_checksum']:raise ValueError('Historical D0B checkpoint changed')
    source_files=[ROOT/'cmgm/models/formal_baselines_v2.py',*sorted((ROOT/'cmgm/scripts').glob('formal_v2_*.py')),
        Path(__file__),ROOT/'cmgm/config.py',ROOT/'cmgm/models/hetero_mixhop_model.py',ROOT/'cmgm/scripts/main_ablation.py',
        ROOT/'cmgm/data/data_loader.py',ROOT/'cmgm/data/feature_builder.py',ROOT/'cmgm/training/metric_standard.py']
    r['current_source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    for file,value in [('protocol.json',PROTOCOL),('baseline_provenance.json',prov),('input_equality_audit.json',audit)]:atomic_json(out/file,value)
    return r,out


def preflight(r,out,data,device,args):
    if r['preflight_complete']:return
    fixed=next(iter(loader(data['loaders']['val'].dataset,42)))[0][:2].to(device)
    for name in NEURAL:
        seed_all(42);m=make_neural(name,data,device);check=sanity(name,m,fixed)
        r['sanity'][name]=dict(initial=check,params=sum(p.numel() for p in m.parameters()),PASS=check['PASS'])
        del m
        if not check['PASS']:save(r,out);raise AssertionError(f'{name} initial sanity FAIL')
    for name in CLASSICAL:
        estimator=classical_model(name,grid(name)[0],42,args.cpu_jobs)
        r['sanity'][name]=dict(PASS=True,status='lossless input and estimator configuration checked; fitted prediction checks pending',
            estimator_class=f'{type(estimator).__module__}.{type(estimator).__name__}',input_scalars=int(fixed[0].numel()))
    # Explicit historical reference exception, before Stage A. Never a formal row.
    from cmgm.scripts.baseline_protocol import evaluate
    from cmgm.scripts.baseline_comparison import REFERENCE
    from cmgm.scripts.d0b_5d_error_regime_diagnostic import checkpoint_payload
    m=make_neural('D0B',data,device);cp=checkpoint_payload(args.d0b_checkpoint)
    m.load_state_dict(cp.get('model_state_dict',cp.get('state_dict',cp)));m.requires_grad_(False)
    ref=evaluate(m,{s:loader(l.dataset,42) for s,l in data['loaders'].items()},device,data['feature_names'][data['market_indices']['commodity'][0]:data['market_indices']['commodity'][1]])
    actual=ref['metrics']['test']['5'];passed=all(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9) for k,v in REFERENCE.items())
    r['historical_reference']=dict(actual=actual,PASS=bool(passed),best_epoch=cp.get('best_epoch'),formal_row=False)
    if not passed:save(r,out);raise AssertionError('Historical D0B reference mismatch; STOP')
    r['preflight_complete']=True;r['status']='PREFLIGHT_COMPLETE; no benchmark training executed'
    atomic_json(out/'sanity_checks.json',r['sanity']);save(r,out)


def fitted_sanity(estimator,x):
    sample=x[:4];p=classical_predictions(estimator,sample)
    order=np.arange(len(sample))[::-1]
    batch=float(np.max(np.abs(classical_predictions(estimator,sample[order])-p[order])))
    single=float(np.max(np.abs(classical_predictions(estimator,sample[:1])-p[:1])))
    bound=32*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(p))))
    return dict(OutputShape=list(p.shape),Finite=bool(np.isfinite(p).all()),BatchPerm=batch,SingleSample=single,
        FullInformation=True,CommodityOrdering='reshape96 horizon-major commodity-minor',roundoff_bound=bound,
        PASS=np.isfinite(p).all().item() and max(batch,single)<=bound)


def fit_job(r,out,data,device,args,name,hp,seed,stage,candidate=None):
    key=job_id(stage,name,seed,candidate);previous=r['jobs'].get(key)
    cpdir=Path(r['checkpoint_dir']);cpdir.mkdir(parents=True,exist_ok=True)
    suffix='.joblib' if name in CLASSICAL else '.pt';path=cpdir/(key+'_best'+suffix)
    if previous and previous['status']=='DONE':
        if sha(path)!=previous['sha256']:raise ValueError('Completed artifact changed')
        return previous
    if previous and not args.retry_invalid:raise RuntimeError(f'{key} was interrupted/invalid; inspect cause before --retry-invalid')
    if path.exists():
        # Fully fitted checkpoint may survive an interruption between file and manifest writes.
        obj=joblib.load(path) if name in CLASSICAL else torch.load(path,map_location='cpu',weights_only=False)
        if obj.get('training_complete'):
            md=obj['metadata']
            if md['model']!=name or md['hp']!=hp or md['seed']!=seed or md['stage']!=stage:raise ValueError('Recovery identity mismatch')
            entry=dict(status='DONE',path=str(path),sha256=sha(path),model=name,hp=hp,seed=seed,stage=stage,
                summary=obj['summary'],source_hashes=md['source_hashes'],sanity=obj.get('fitted_sanity',{}))
            # Neural checkpoint training completion precedes its mandatory best-checkpoint audit.
            if name in NEURAL:
                m=make_neural(name,data,device);m.load_state_dict(obj['state_dict'])
                check=sanity(name,m,next(iter(loader(data['loaders']['val'].dataset,seed)))[0][:2].to(device));del m
                if not check['PASS']:raise AssertionError('Recovered best checkpoint sanity failed')
                entry['sanity']=check
            r['jobs'][key]=entry;save(r,out);return entry
        path.rename(path.with_name(path.stem+'_invalid_'+datetime.now().strftime('%Y%m%d_%H%M%S')+suffix))
    metadata=dict(model=name,hp=hp,seed=seed,stage=stage,protocol=PROTOCOL,source_hashes=r['current_source_hashes'],git_sha=r['git_sha'])
    r['jobs'][key]=dict(status='RUNNING',model=name,hp=hp,seed=seed,stage=stage,path=str(path));r['status']=key;save(r,out)
    print(f'[V2] Starting {stage} {name}, seed={seed}, HP={hp}; artifact={path}',flush=True)
    seed_all(seed)
    if name in CLASSICAL:
        tx,ty=arrays(data['loaders']['train'].dataset,out/'arrays','train')
        vx,vy=arrays(data['loaders']['val'].dataset,out/'arrays','val')
        m=classical_model(name,hp,seed,args.cpu_jobs);start=time.perf_counter();m.fit(tx,ty);seconds=time.perf_counter()-start
        check=fitted_sanity(m,vx)
        if not check['PASS']:raise AssertionError('Classical fitted sanity failed')
        p=classical_predictions(m,vx);vm=metrics(p,vy.reshape(-1,4,24))
        size=int(m.coef_.size+m.intercept_.size) if name=='Ridge Regression' else (hp['n_estimators'] if name=='Random Forest' else 500*96)
        summary=dict(val5_MAE=vm['5']['MAE'],val_metrics=vm,prediction_audit=distribution(p,vy.reshape(-1,4,24)),
            train_seconds=seconds,best_epoch=None,complexity=size,complexity_unit={'Ridge Regression':'fitted coefficients including intercepts','Random Forest':'trees','XGBoost':'estimators × outputs'}[name])
        obj=dict(estimator=m,metadata=metadata,training_complete=True,summary=summary,fitted_sanity=check)
        temp=Path(str(path)+'.tmp');joblib.dump(obj,temp);os.replace(temp,path)
    else:
        m=make_neural(name,data,device)
        summary=neural_train(m,loader(data['loaders']['train'].dataset,seed),loader(data['loaders']['val'].dataset,seed),device,hp['lr'],path,metadata)
        check=sanity(name,m,next(iter(loader(data['loaders']['val'].dataset,seed)))[0][:2].to(device))
        if not check['PASS']:raise AssertionError('Neural best checkpoint sanity failed')
    del m
    entry=dict(status='DONE',path=str(path),sha256=sha(path),model=name,hp=hp,seed=seed,stage=stage,summary=summary,sanity=check,source_hashes=metadata['source_hashes'])
    r['jobs'][key]=entry;save(r,out)
    if device.type=='cuda':torch.cuda.empty_cache()
    return entry


def tune(r,out,data,device,args):
    if r['selection_frozen']:return
    for name in ORDER:
        trials=[fit_job(r,out,data,device,args,name,hp,42,'A',i) for i,hp in enumerate(grid(name))]
        best=min(trials,key=lambda t:t['summary']['val5_MAE'])
        r['selected'][name]=dict(hp=best['hp'],val5_MAE=best['summary']['val5_MAE'],artifact=best['path'],artifact_sha256=best['sha256'],
            criterion='VAL5 MAE only; ties choose first registered candidate')
        save(r,out)
    r['selection_frozen']=True
    atomic_json(out/'selected_hyperparameters.json',r['selected']);r['selection_checksum']=sha(out/'selected_hyperparameters.json')
    r['status']='STAGE_A_COMPLETE; all hyperparameters frozen; no formal TEST evaluated';save(r,out)


def final_job_keys():
    return [job_id('B',name,s) for name in ORDER for s in ((42,) if name=='Ridge Regression' else SEEDS)]


def evaluation_allowed(r):
    return r['selection_frozen'] and all(r['jobs'].get(k,{}).get('status')=='DONE' for k in final_job_keys())


def final(r,out,data,device,args):
    if not r['selection_frozen']:raise RuntimeError('All Stage A selections must be frozen first; run --stage tune')
    if sha(out/'selected_hyperparameters.json')!=r['selection_checksum']:raise ValueError('Frozen selections changed')
    for name in ORDER:
        if name=='Ridge Regression':
            chosen=r['selected'][name];key=job_id('B',name,42)
            if key not in r['jobs']:
                original=next(t for t in r['jobs'].values() if t['path']==chosen['artifact'])
                r['jobs'][key]=dict(original,stage='B',reuse='deterministic Ridge selected fit; no fake repeated seeds');save(r,out)
            continue
        for seed in SEEDS:fit_job(r,out,data,device,args,name,r['selected'][name]['hp'],seed,'B')
    if not evaluation_allowed(r):raise AssertionError('TEST barrier failed')
    # TEST is materialized only after every architecture/seed formal checkpoint is fixed.
    names=data['feature_names'];cs,ce=data['market_indices']['commodity']
    for key in final_job_keys():
        entry=r['jobs'][key]
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Formal artifact modified')
        if key in r['formal']:continue
        name=entry['model'];seed=entry['seed'];seed_all(seed);path=Path(entry['path'])
        if name in CLASSICAL:m=joblib.load(path)['estimator']
        else:
            m=make_neural(name,data,device);cp=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict'])
            branch=getattr(m,'switching_latent_transformer',None)
            if branch is not None:branch.set_epoch(cp['best_epoch'])
        result=dict(model=name,seed=seed,hp=entry['hp'],summary=entry['summary'],metrics={},prediction_audit={},per_commodity=[])
        for split in ('val','test'):
            ds=data['loaders'][split].dataset
            if name in CLASSICAL:
                x,y=arrays(ds,out/'arrays',split);p=classical_predictions(m,x);y=y.reshape(-1,4,24)
            else:p,y,_=neural_predictions(m,loader(ds,seed),device)
            result['metrics'][split]=metrics(p,y);result['prediction_audit'][split]=distribution(p,y)
            if split=='test':
                idx=config.MULTI_HORIZONS.index(5)
                from cmgm.training.metric_standard import population_metrics
                result['per_commodity']=[dict(commodity=str(names[cs+i]),**population_metrics(p[:,idx,i],y[:,idx,i])) for i in range(ce-cs)]
        r['formal'][key]=result;save(r,out);del m
        if device.type=='cuda':torch.cuda.empty_cache()
    r['status']='COMPLETE';save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('preflight','tune','final','all'),default='preflight')
    p.add_argument('--cpu-check',action='store_true',help='CPU preflight only, never formal neural training')
    p.add_argument('--cpu-jobs',type=int,default=min(8,len(os.sched_getaffinity(0))),help='RF/XGB worker limit, frozen per suite')
    p.add_argument('--resume',help='latest or existing V2 report directory')
    p.add_argument('--retry-invalid',action='store_true',help='After fixing an invalid/interrupted run, never for poor valid results')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/formal_baseline_benchmark_v2')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/formal_baselines_v2')
    p.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.cpu_jobs<1:raise ValueError('cpu-jobs must be positive')
    if args.cpu_check and args.stage!='preflight':raise ValueError('CPU checking cannot launch formal training')
    if not args.cpu_check and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
    device=torch.device('cpu' if args.cpu_check else 'cuda')
    from cmgm.scripts.main_ablation import build_data
    seed_all(42);data=build_data(args);r,out=prepare(args,data)
    try:
        preflight(r,out,data,device,args)
        if args.stage in ('tune','all'):tune(r,out,data,device,args)
        if args.stage in ('final','all'):final(r,out,data,device,args)
    except Exception as error:
        affected=[dict(model=j['model'],stage=j['stage'],seed=j['seed'],hp=j['hp']) for j in r['jobs'].values() if j['status']=='RUNNING']
        for job in r['jobs'].values():
            if job['status']=='RUNNING':job['status']='INVALID'
        r['status']='STOP: '+type(error).__name__
        r.setdefault('errors',[]).append(dict(message=str(error),stage=args.stage,affected_runs=affected,
            peak_cpu_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None))
        save(r,out);raise
    finally:
        if sha(args.d0b_checkpoint)!=r['historical_checksum']:raise AssertionError('Historical checkpoint changed')
    print(f'V2 {r["status"]}: {out}; STOP',flush=True)


if __name__=='__main__':main()
