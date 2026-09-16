"""Fixed single-run benchmark. Default prepares/reuses; --run fits missing models only."""
import argparse
from datetime import datetime
import fcntl
import gc
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
from cmgm.models.formal_baselines_v2 import ORDER,CLASSICAL,KEYS,make_neural
from cmgm.scripts.fixed_baseline_protocol import PROTOCOL,FIXED,RUN_ORDER,TRAINABLE_NEURAL,estimator,reuse_plan,heartbeat
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,input_audit,arrays,metrics,distribution,neural_predictions,classical_predictions
from cmgm.scripts.formal_v2_audit import ROOT,provenance,sanity
from cmgm.scripts.baseline_protocol import seed_all,loaders,train_one


def save(r,out):
    from cmgm.scripts.fixed_baseline_report import report
    atomic_json(out/'results.json',r);atomic_json(out/'partial_results.json',r)
    report(r,out)


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def evaluate(name,m,data,device):
    result=dict(metrics={},prediction_audit={},per_commodity=[])
    if name in CLASSICAL:
        sources={s:l.dataset for s,l in data['loaders'].items()}
        for split,ds in sources.items():
            predictions=[];targets=[]
            # Avoid another large resident design matrix during evaluation.
            for batch in loaders(data,full=True)[split]:
                x,y=batch[:2];predictions.append(classical_predictions(m,x.numpy().reshape(len(x),-1)));targets.append(y.numpy())
            p,y=np.concatenate(predictions),np.concatenate(targets)
            result['metrics'][split]=metrics(p,y);result['prediction_audit'][split]=distribution(p,y)
            result['prediction_audit'][split]['pred_P50']=float(np.median(p))
            if split=='test':test_p,test_y=p,y
    else:
        for split,source in loaders(data,full=True).items():
            p,y,_=neural_predictions(m,source,device)
            result['metrics'][split]=metrics(p,y);result['prediction_audit'][split]=distribution(p,y)
            result['prediction_audit'][split]['pred_P50']=float(np.median(p))
            if split=='test':test_p,test_y=p,y
    cs,ce=data['market_indices']['commodity'];idx=config.MULTI_HORIZONS.index(5)
    from cmgm.training.metric_standard import population_metrics
    result['per_commodity']=[dict(commodity=str(data['feature_names'][cs+i]),**population_metrics(test_p[:,idx,i],test_y[:,idx,i])) for i in range(ce-cs)]
    result['scope']='pooled all origins ×24 commodities; no metric clipping or inverse target scaling'
    return result


def setup(args,data):
    if (config.SEQ_LEN,config.FEATURE_DIM,tuple(config.MULTI_HORIZONS))!=(20,21,(1,5,10,20)):raise ValueError('Wrong fixed data config')
    if config.TARGET_TYPE!='return' or config.HUBER_DELTA!=.02:raise ValueError('Wrong target/loss config')
    prov=provenance();audit=input_audit(data)
    if not audit['PASS']:raise AssertionError('Input equality failed')
    root=Path(args.output_dir)
    if args.resume:
        options=sorted(root.glob('*/results.json'))
        out=options[-1].parent if args.resume=='latest' and options else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['fixed_configurations']!=FIXED or r['input_audit']!=audit:raise ValueError('Fixed protocol or data changed')
        if r['provenance']!=prov:raise ValueError('Package versions or official code changed')
        if r['cpu_jobs']!=args.cpu_jobs:raise ValueError('Saved parallelism differs')
    else:
        out=root/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        options=sorted(Path(args.old_dir).glob('*/results.json'))
        old_path=options[-1] if options else None;old=json.loads(old_path.read_text()) if old_path else {}
        plan=reuse_plan(old,audit,prov,ROOT)
        r=dict(status='PREPARING',protocol=PROTOCOL,fixed_configurations=FIXED,provenance=prov,input_audit=audit,
            plan=plan,models={},jobs={},sanity={},cpu_jobs=args.cpu_jobs,parallelism='RF n_jobs=cpu_jobs; XGBoost outer=1, inner=cpu_jobs (one level only)',
            checkpoint_dir=str((Path(args.checkpoint_dir)/out.name).resolve()),old_experiment=str(old_path.parent) if old_path else None,
            historical_checkpoint=str(args.d0b_checkpoint.resolve()),historical_sha256=sha(args.d0b_checkpoint),
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            old_status=old.get('status'),old_completed=sum(v['status']=='DONE' for v in old.get('jobs',{}).values()),
            computation=dict(old_grid_runs=35,old_multiseed_runs=24,old_xgb_output_regressors=672,
                new_grid_runs=0,new_repeat_seeds=0,new_D0B_training=0,new_xgb_output_regressors=96,
                new_fits_required=sum(plan[n]['action']!='Reuse' for n in RUN_ORDER)),
            initial_sanity_complete=False)
    if sha(args.d0b_checkpoint)!=r['historical_sha256']:raise ValueError('Historical D0B changed')
    files=[ROOT/'cmgm/models/formal_baselines_v2.py',Path(__file__),ROOT/'cmgm/scripts/fixed_baseline_protocol.py',ROOT/'cmgm/scripts/fixed_baseline_report.py',
        ROOT/'cmgm/scripts/baseline_protocol.py',ROOT/'cmgm/scripts/formal_v2_protocol.py',ROOT/'cmgm/scripts/formal_v2_audit.py']
    r['current_source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in files}
    for name,value in [('protocol.json',PROTOCOL),('fixed_configurations.json',FIXED),('baseline_provenance.json',prov),('input_equality_audit.json',audit)]:atomic_json(out/name,value)
    notes=(ROOT/'third_party/baselines/ADAPTATION_NOTES.md').read_text()
    notes=notes.replace('Huber/Adam/VAL5-selection protocol','Huber/Adam/multi-horizon-VAL-Huber selection protocol')
    (out/'ADAPTATION_NOTES.md').write_text(notes,encoding='utf-8')
    print('OLD PROTOCOL: 35 grid runs; 24 multi-seed runs; 672 XGBoost output regressors.')
    print('NEW PROTOCOL: one fixed fit/model, no grid/repeated seeds, frozen D0B.',r['computation'])
    for name,row in r['plan'].items():print(name,row['action'],row['reason'],flush=True)
    print('Completed:',list(r['models']),'; reusable:',[n for n,p in r['plan'].items() if p['action']=='Reuse'],
          '; pending fixed fits:',[n for n in RUN_ORDER if n not in r['models'] and r['plan'][n]['action']!='Reuse'],flush=True)
    save(r,out);return r,out


def prepare_checks(r,out,data,device):
    if r['initial_sanity_complete']:return
    x=next(iter(loaders(data,full=True)['val']))[0][:2].to(device)
    for name in TRAINABLE_NEURAL:
        seed_all(42);m=make_neural(name,data,device);check=sanity(name,m,x)
        r['sanity'][name]=dict(initial=check,params=sum(p.numel() for p in m.parameters()),PASS=check['PASS'])
        del m;release();save(r,out)
        if not check['PASS']:raise AssertionError(name+' initial sanity failed')
    r['initial_sanity_complete']=True;save(r,out)


def frozen_d0b(r,out,data,device):
    if 'D0B' in r['models']:return
    from cmgm.scripts.d0b_5d_error_regime_diagnostic import checkpoint_payload
    seed_all(42);m=make_neural('D0B',data,device);cp=checkpoint_payload(r['historical_checkpoint'])
    m.load_state_dict(cp.get('model_state_dict',cp.get('state_dict',cp)));m.requires_grad_(False)
    check=sanity('D0B',m,next(iter(loaders(data,full=True)['val']))[0][:2].to(device))
    if not check['PASS']:raise AssertionError('D0B sanity failed')
    result=evaluate('D0B',m,data,device);actual=result['metrics']['test']['5']
    expected=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)
    if not all(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9) for k,v in expected.items()):raise AssertionError('STOP: historical D0B evaluator mismatch')
    r['sanity']['D0B']=dict(PASS=True,best=check)
    r['plan']['D0B']['sanity_PASS']=True
    r['models']['D0B']=dict(result,status='COMPLETE',reused=True,path=r['historical_checkpoint'],sha256=r['historical_sha256'],
        summary=dict(best_epoch=cp.get('best_epoch'),train_seconds=None,device='historical; inference '+str(device),complexity=sum(p.numel() for p in m.parameters()),complexity_unit='parameters'),
        reason='Frozen historical formal checkpoint; reproduction PASS; historical seed metadata not assumed.')
    del m;release();save(r,out)


def process_model(name,r,out,data,device,args,allow_fit):
    if name=='D0B':raise ValueError('D0B may never enter fitting path')
    if name in r['models']:
        if sha(r['models'][name]['path'])!=r['models'][name]['sha256']:raise ValueError('Completed artifact changed')
        return
    row=r['plan'][name];is_classical=name in CLASSICAL
    suffix='.joblib' if is_classical else '.pt'
    path=Path(r['checkpoint_dir'])/(KEYS[name]+'_seed42_best'+suffix)
    recovering=False;reused=row['action']=='Reuse'
    if reused:path=Path(row['source']['path'])
    elif not allow_fit:return
    previous=r['jobs'].get(name,{})
    seed_all(42)
    if path.exists():
        obj=joblib.load(path) if is_classical else torch.load(path,map_location='cpu',weights_only=False)
        if not obj.get('training_complete'):
            if reused:raise ValueError('Reuse artifact incomplete')
            if not args.retry_invalid:raise RuntimeError('Incomplete artifact; inspect before explicit retry')
            path.rename(path.with_name(path.stem+'_invalid_'+datetime.now().strftime('%Y%m%d_%H%M%S')+suffix));del obj
        else:
            if reused:
                if sha(path)!=row['source']['sha256']:raise ValueError('Reuse artifact changed')
            elif obj['metadata']['fixed_configurations']!=FIXED or obj['metadata']['model']!=name or obj['metadata']['seed']!=42:
                raise ValueError('Recovery configuration mismatch')
            elif obj['metadata']['data']!=r['input_audit']['data']['split_fingerprint']:
                raise ValueError('Recovery data mismatch')
            if is_classical:m=obj['estimator'];summary=obj['summary']
            else:m=make_neural(name,data,device);m.load_state_dict(obj['model_state_dict']);summary=obj['training_summary']
            recovering=True
    if not recovering:
        if not allow_fit:raise AssertionError('A new fit is not authorized by this invocation')
        if previous.get('status') in ('RUNNING','INVALID') and not args.retry_invalid:
            raise RuntimeError('Inspect interrupted fit, then use --retry-invalid; never retry for poor results')
        path.parent.mkdir(parents=True,exist_ok=True)
        r['jobs'][name]=dict(status='RUNNING',path=str(path),start=datetime.now().isoformat());r['status']='FITTING '+name;save(r,out)
        md=dict(model=name,seed=42,protocol=PROTOCOL,fixed_configurations=FIXED,source_hashes=r['current_source_hashes'],git_sha=r['git_sha'],data=r['input_audit']['data']['split_fingerprint'])
        print('[Fixed benchmark] Fitting',name,'seed42; one configuration only',flush=True)
        if is_classical:
            tx,ty=arrays(data['loaders']['train'].dataset,out/'arrays','train')
            m=estimator(name,args.cpu_jobs);start=time.perf_counter()
            with heartbeat(name,out):m.fit(tx,ty)
            seconds=time.perf_counter()-start;del tx,ty;gc.collect()
            size=int(m.coef_.size+m.intercept_.size) if name=='Ridge Regression' else (300 if name=='Random Forest' else 200*96)
            summary=dict(train_seconds=seconds,best_epoch=None,device='CPU',complexity=size,complexity_unit='coefficients incl intercept' if name=='Ridge Regression' else 'trees',peak_cpu_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            temp=Path(str(path)+'.tmp');joblib.dump(dict(estimator=m,metadata=md,training_complete=True,summary=summary),temp);os.replace(temp,path)
        else:
            if device.type!='cuda':raise RuntimeError('Formal neural fitting requires GPU')
            m=make_neural(name,data,device);sources=loaders(data)
            torch.cuda.reset_peak_memory_stats(device)
            summary,history=train_one(m,sources['train'],sources['val'],device,path,md,
                on_epoch=lambda h:atomic_json(out/f'history_{KEYS[name]}.json',h))
            summary.update(device='GPU',peak_gpu_bytes=int(torch.cuda.max_memory_allocated(device)),complexity=sum(p.numel() for p in m.parameters()),complexity_unit='parameters')
            # Preserve completed checkpoint before performing any potentially failing reporting.
            cp=torch.load(path,map_location='cpu',weights_only=False);cp['training_summary']=summary
            temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path);del cp
    r['jobs'][name]=dict(status='FITTED',path=str(path),sha256=sha(path),reused=reused);save(r,out)
    if is_classical:
        from cmgm.scripts.formal_baseline_benchmark_v2 import fitted_sanity
        x=next(iter(loaders(data,full=True)['val']))[0][:4].numpy().reshape(4,-1);check=fitted_sanity(m,x)
    else:check=sanity(name,m,next(iter(loaders(data,full=True)['val']))[0][:2].to(device))
    r['sanity'].setdefault(name,{})['best']=check;r['sanity'][name]['PASS']=check['PASS'];save(r,out)
    if not check['PASS']:raise AssertionError(name+' fitted sanity failed; no valid result')
    result=evaluate(name,m,data,device)
    r['models'][name]=dict(result,status='COMPLETE',path=str(path),sha256=sha(path),reused=reused,summary=summary,
        source_experiment=r['old_experiment'] if reused else str(out),reason=row['reason'] if reused else 'One new seed42 fixed-config fit')
    r['jobs'][name]['status']='DONE';r['plan'][name]['sanity_PASS']=True
    print('[Fixed benchmark] COMPLETE',name,'TEST5:',result['metrics']['test']['5'],flush=True)
    del m
    if 'obj' in locals():del obj
    release();save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Fit only missing fixed configurations; no grid/seeds/D0B training')
    p.add_argument('--resume',help='latest or a single-run report directory')
    p.add_argument('--retry-invalid',action='store_true')
    p.add_argument('--cpu-check',action='store_true',help='Read-only preparation/evaluation, never new fitting')
    p.add_argument('--cpu-jobs',type=int,default=min(8,len(os.sched_getaffinity(0))))
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/formal_baseline_benchmark_v2_single_run')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/formal_baselines_single_run')
    p.add_argument('--old-dir',type=Path,default=ROOT/'experiments/formal_baseline_benchmark_v2')
    p.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.cpu_check and args.run:raise ValueError('CPU check cannot fit new models')
    if args.cpu_jobs<1:raise ValueError('Invalid worker count')
    if not args.cpu_check and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
    # One process for the entire fixed suite, even if a second timestamp is requested.
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    with (args.checkpoint_dir/'.active.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A fixed benchmark process is already running')
        if args.run and not args.resume and any(args.output_dir.glob('*/results.json')):
            raise RuntimeError('Existing fixed suite found; use --resume latest --run, not a repeated experiment')
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=setup(args,data)
        try:
            prepare_checks(r,out,data,device);frozen_d0b(r,out,data,device)
            for name in RUN_ORDER:process_model(name,r,out,data,device,args,allow_fit=args.run)
            r['status']='COMPLETE' if len(r['models'])==9 else 'PREPARED: reused results saved; fixed fits pending'
            save(r,out)
        except BaseException as error:
            for name,job in r['jobs'].items():
                if job['status'] in ('RUNNING','FITTED'):job['status']='INVALID'
            r['status']='STOP: '+type(error).__name__
            r.setdefault('errors',[]).append(dict(message=str(error),time=datetime.now().isoformat(),peak_CPU_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            save(r,out);raise
        finally:
            if sha(args.d0b_checkpoint)!=r['historical_sha256']:raise AssertionError('Historical D0B changed')
        print('Fixed single-run report:',out,'; STOP',flush=True)


if __name__=='__main__':main()
