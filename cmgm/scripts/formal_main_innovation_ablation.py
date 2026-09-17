"""Revised main ablations: audit/re-evaluate by default; --run fits only missing controls."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from cmgm import config
from cmgm.models.formal_d0b_main_ablation import NAMES,KEY,REUSE,NEW,DEFINITIONS,MainInnovationAblation
from cmgm.models.formal_d0b_ablation import FormalD0BAblation
from cmgm.scripts.formal_main_ablation_audit import init_audit,sanity
from cmgm.scripts.baseline_protocol import seed_all,loaders,data_audit
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,metrics
from cmgm.scripts.formal_ablation_study import PROTOCOL as OLD_PROTOCOL,free
from cmgm.training.train import train_epoch,validate_epoch
from cmgm.training.metric_standard import population_metrics
ROOT=Path(__file__).resolve().parents[2]
SUPPLEMENTARY=('w/o Switch KL','CommodityOnly','w/o Stock','w/o Bond')
PROTOCOL={**OLD_PROTOCOL,'name':'D0B FORMAL MAIN-INNOVATION ABLATION STUDY — REVISED',
          'configurations':list(NAMES),'formal_reference':'Full D0B: audited same-protocol FullD0B-Control seed42 checkpoint',
          'reuse_policy':'Exact data, source implementation, protocol, seed, definition and completed sanity; reevaluate; never retrain eligible artifacts',
          'KL':'Native beta_max=.0005 warmup20; NoTemporal disabled; uniform p/prior zero; SharedG retains native KL',
          'near_tie_relative_percent':.1}


def save(r,out):
    from cmgm.scripts.formal_main_ablation_report import report
    atomic_json(out/'results.json',r);atomic_json(out/'partial_results.json',r);report(r,out)


def audit_data(data):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS)!=(20,21,[1,5,10,20]):raise ValueError('Frozen data config changed')
    audit=data_audit(data)
    audit['mapping']=[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]
    if not audit['PASS']:raise ValueError('Data/order audit failed')
    return json.loads(json.dumps(audit))


def audit_reuse(name,old_name,source,audit):
    old=json.loads(source.read_text());entry=old.get('results',{}).get(old_name)
    row=dict(old_name=old_name,source_experiment=str(source.parent),old_checkpoint_found=bool(entry),definition_exact_match=True,reused=False)
    if not entry:
        row['reason']='No completed artifact';return row
    path=Path(entry['path']);row['path']=str(path)
    if not path.exists():row['reason']='Checkpoint missing';return row
    cp=torch.load(path,map_location='cpu',weights_only=False);md=cp.get('metadata',{})
    checks=dict(hash=sha(path)==entry['sha256'],complete=bool(cp.get('training_complete')),name=md.get('name')==old_name,
        seed42=md.get('seed')==42,protocol=md.get('protocol')==OLD_PROTOCOL,
        data=md.get('data')==audit,source_metadata=md.get('source_hashes')==old.get('source_hashes'),
        implementation=bool(md.get('source_hashes')) and all((ROOT/s).exists() and sha(ROOT/s)==v for s,v in md.get('source_hashes',{}).items()),
        sanity=entry.get('sanity',{}).get('PASS') is True)
    row.update(checks=checks,reused=all(checks.values()),sha256=sha(path),best_epoch=cp.get('best_epoch'))
    row['reason']='Exact mathematical definition and complete protocol match; reevaluate only' if row['reused'] else 'Reuse rejected: '+','.join(k for k,v in checks.items() if not v)
    return row


def setup(args,data):
    audit=audit_data(data)
    sources=['cmgm/models/formal_d0b_main_ablation.py','cmgm/scripts/formal_main_innovation_ablation.py',
             'cmgm/scripts/formal_main_ablation_audit.py','cmgm/scripts/formal_main_ablation_report.py',
             'cmgm/models/model.py','cmgm/training/metric_standard.py','cmgm/scripts/baseline_protocol.py']
    old_sources=json.loads(args.source_results.read_text())['source_hashes']
    hashes={s:sha(ROOT/s) for s in sorted(set(sources)|set(old_sources))}
    if args.resume:
        options=sorted(args.output_dir.glob('*/results.json'))
        if args.resume=='latest' and not options:raise ValueError('No prepared revised experiment')
        out=options[-1].parent if args.resume=='latest' else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['data']!=audit or r['source_hashes']!=hashes:raise ValueError('Resume data/protocol/implementation mismatch; inspect rather than silently rerun')
    else:
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        reuse={n:audit_reuse(n,old,args.source_results,audit) for n,old in REUSE.items()}
        for n in NEW:reuse[n]=dict(old_checkpoint_found=False,definition_exact_match=False,reused=False,reason='New single-factor mechanism, no old equivalent')
        supplement={n:audit_reuse(n,n,args.source_results,audit) for n in SUPPLEMENTARY}
        r=dict(protocol=PROTOCOL,data=audit,source_hashes=hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            reuse=reuse,supplementary_reuse=supplement,jobs={},results={},supplementary={},initialization={},sanity={},status='PREPARING',
            checkpoint_dir=str(args.checkpoint_dir/out.name))
    save(r,out);return r,out


@torch.no_grad()
def evaluate(model,data,device):
    result=dict(metrics={},per_commodity=[],mechanism={})
    name=getattr(model,'main_name',model.ablation_name)
    for split,loader in loaders(data,full=True).items():
        preds=[];targets=[];probs=[];longs=[];micros=[];disagreements=[];model.eval()
        for batch in loader:
            pred=model(batch[0].to(device));y=batch[1]
            if pred.shape!=y.shape or not torch.isfinite(pred).all():raise ValueError('Prediction shape/finite failure')
            preds.append(pred.cpu().numpy());targets.append(y.numpy());b=model.switching_latent_transformer
            if split=='test' and name in ('Full D0B','w/o Adaptive Regime Routing','w/o Switch KL'):
                probs.append(b.last_regime_probabilities.cpu())
            if split=='test' and name in ('Full D0B','w/o Regime-Specific Transitions'):
                c=b.last_latent_candidates
                disagreements.append(torch.stack([(c[:,:,i]-c[:,:,j]).abs().mean(-1) for i,j in ((0,1),(0,2),(1,2))],-1).cpu())
            if name in ('Full D0B','w/o Balanced Readout'):
                longs.append(b.last_h_long.norm(dim=-1).cpu());micros.append(b.last_h_micro.norm(dim=-1).cpu())
        p,y=np.concatenate(preds),np.concatenate(targets);result['metrics'][split]=metrics(p,y)
        if split=='test':
            idx=config.MULTI_HORIZONS.index(5);cs,ce=data['market_indices']['commodity']
            result['per_commodity']=[dict(commodity=str(data['feature_names'][cs+i]),**population_metrics(p[:,idx,i],y[:,idx,i])) for i in range(ce-cs)]
        if probs:
            prob=torch.cat(probs).double()
            result['mechanism']['regime']=dict(mean_p=prob.mean((0,1)).tolist(),entropy=float(-(prob*torch.log(prob+1e-8)).sum(-1).mean()),
                hard_occupancy=torch.bincount(prob.argmax(-1).flatten(),minlength=3).double().div(prob.shape[0]*prob.shape[1]).tolist(),
                occupancy_note='Argmax tie-breaking assigns uniform routing to index0; this is not learned collapse.')
        if disagreements:result['mechanism']['candidate_pairwise_mean_absolute_disagreement']=float(torch.cat(disagreements).double().mean())
        if longs:
            h=float(torch.cat(longs).double().mean());z=float(torch.cat(micros).double().mean())
            result['mechanism'][split]=dict(long_norm=h,micro_norm=z,micro_long_ratio=z/(h+1e-8))
    return result


def preflight(r,out,data,device):
    x,y=next(iter(loaders(data,full=True)['val']))[:2];x=x[:2].to(device);y=y[:2].to(device)
    for n in NAMES:
        if r['sanity'].get(n,{}).get('PASS'):continue
        model,initial=init_audit(n,data,device,x);r['initialization'][n]=initial
        if not initial['PASS']:save(r,out);raise AssertionError(n+' shared initialization failed')
        check=sanity(model,x,y);r['sanity'][n]=check
        print('[Revised sanity]',n,'params',initial['total_instantiated'],'shared diff',initial['shared_parameter_initial_max_abs_diff'],'PASS',check['PASS'],flush=True)
        del model;free();save(r,out)
        if not check['PASS']:raise AssertionError(n+' sanity failed')


def evaluate_reused(r,out,data,device):
    for n in NAMES:
        row=r['reuse'][n]
        if not row['reused']:continue
        if sha(row['path'])!=row['sha256']:raise ValueError('Reused checkpoint changed')
        if n in r['results']:continue
        cp=torch.load(row['path'],map_location='cpu',weights_only=False)
        seed_all(42);m=MainInnovationAblation(n,data).to(device);m.load_state_dict(cp['model_state_dict'],strict=True)
        x,y=next(iter(loaders(data,full=True)['val']))[:2]
        check=sanity(m,x[:2].to(device),y[:2].to(device))
        if not check['PASS']:raise AssertionError(n+' reused best sanity failed')
        result=evaluate(m,data,device)
        r['results'][n]=dict(result,path=row['path'],sha256=row['sha256'],reused=True,source_experiment=row['source_experiment'],runtime=cp['runtime'],sanity=check)
        print('[Reevaluated]',n,'TEST5 MAE',result['metrics']['test']['5']['MAE'],flush=True)
        del m,cp;free();save(r,out)
    for n,row in r['supplementary_reuse'].items():
        if not row['reused']:continue
        if sha(row['path'])!=row['sha256']:raise ValueError('Supplementary checkpoint changed')
        if n in r['supplementary']:continue
        cp=torch.load(row['path'],map_location='cpu',weights_only=False)
        seed_all(42);m=FormalD0BAblation(n,data).to(device);m.load_state_dict(cp['model_state_dict'],strict=True)
        from cmgm.scripts.formal_ablation_audit import sanity as old_sanity
        x,y=next(iter(loaders(data,full=True)['val']))[:2];check=old_sanity(m,x[:2].to(device),y[:2].to(device))
        if not check['PASS']:raise AssertionError(n+' supplementary sanity failed; never retrain here')
        r['supplementary'][n]=dict(evaluate(m,data,device),path=row['path'],sha256=row['sha256'],reused=True,sanity=check)
        print('[Supplementary reuse only]',n,flush=True)
        del m,cp;free();save(r,out)


def fit(n,m,data,device,path,metadata,out):
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    criterion=torch.nn.HuberLoss(delta=.02);sources=loaders(data);history=[];best=float('inf');stale=0;start=time.perf_counter()
    edge_index=torch.empty((2,0),dtype=torch.long,device=device);edge_weight=torch.empty(0,device=device)
    for epoch in range(1,201):
        tic=time.perf_counter();beta=m.switching_latent_transformer.set_epoch(epoch)
        train=train_epoch(m,sources['train'],edge_index,edge_weight,optimizer,criterion,device)
        val=validate_epoch(m,sources['val'],edge_index,edge_weight,criterion,device)
        if not np.isfinite(train) or not np.isfinite(val):raise FloatingPointError('Invalid nonfinite training run')
        from cmgm.scripts.baseline_protocol import validation
        _,secondary=validation(m,sources['val'],device)
        scheduler.step(val)
        row=dict(epoch=epoch,train_total_loss=train,val_objective=val,val5_MAE=secondary['MAE'],val5_MSE=secondary['MSE'],LR=optimizer.param_groups[0]['lr'],switch_beta=0 if m.disable_switch_kl else beta,seconds=time.perf_counter()-tic)
        history.append(row)
        if val<best:
            best=val;stale=0
            cp=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},best_epoch=epoch,best_val_objective=val,metadata=metadata,training_complete=False)
            temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
        else:stale+=1
        atomic_json(out/f'history_{KEY[n]}.json',history)
        print('[Formal ablation]',n,'epoch',epoch,'train',train,'val',val,'secondary VAL5',secondary['MAE'],flush=True)
        if stale>=10:break
    cp=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(cp['model_state_dict']);m.switching_latent_transformer.set_epoch(cp['best_epoch'])
    cp.update(training_complete=True,history=history,runtime=dict(best_epoch=cp['best_epoch'],train_seconds=time.perf_counter()-start,seconds_per_epoch=float(np.mean([h['seconds'] for h in history])),val5_at_best=history[cp['best_epoch']-1]['val5_MAE']))
    temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
    return cp


def run_one(n,r,out,data,device,args):
    if n in r['results']:
        if sha(r['results'][n]['path'])!=r['results'][n]['sha256']:raise ValueError('Completed artifact changed')
        return
    if r['reuse'][n]['reused']:raise AssertionError('Eligible checkpoint must be reevaluated, never retrained')
    seed_all(42);m=MainInnovationAblation(n,data).to(device)
    path=Path(r['checkpoint_dir'])/(KEY[n]+'_seed42.pt');path.parent.mkdir(parents=True,exist_ok=True)
    md=dict(name=n,definition=DEFINITIONS[n],seed=42,protocol=PROTOCOL,data=r['data'],source_hashes=r['source_hashes'])
    cp=None
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['metadata']!=md:raise ValueError('Checkpoint implementation/data/protocol mismatch')
        if not cp['training_complete']:raise ValueError('Interrupted run; STOP for manual implementation/resource review. No automatic retraining.')
        m.load_state_dict(cp['model_state_dict'],strict=True)
    if cp is None:
        if n in r['jobs']:raise ValueError('Previously started run cannot be silently restarted')
        r['jobs'][n]=dict(status='RUNNING',path=str(path));r['status']='TRAINING '+n;save(r,out)
        cp=fit(n,m,data,device,path,md,out)
    r['jobs'][n]=dict(status='FITTED',path=str(path));save(r,out)
    x,y=next(iter(loaders(data,full=True)['val']))[:2]
    check=sanity(m,x[:2].to(device),y[:2].to(device))
    if not check['PASS']:
        r['sanity'][n]['best']=check;save(r,out);raise AssertionError(n+' best-checkpoint sanity failed')
    result=evaluate(m,data,device)
    r['results'][n]=dict(result,path=str(path),sha256=sha(path),reused=False,runtime=cp['runtime'],sanity=check)
    r['jobs'][n]['status']='DONE';print('[Completed]',n,flush=True)
    del m,cp;free();save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--resume');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--source-results',type=Path,default=ROOT/'experiments/formal_ablation_study/20260915_153018/results.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/formal_main_innovation_ablation')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.cpu_check and args.run:raise ValueError('CPU formal training prohibited')
    if not args.cpu_check and not torch.cuda.is_available():raise ValueError('GPU unavailable; no CPU training fallback')
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    # Share the old ablation lock too, so these two D0B-family suites cannot overlap.
    old_lock=ROOT/'checkpoints/formal_ablation/.active.lock';old_lock.parent.mkdir(parents=True,exist_ok=True)
    with old_lock.open('a+') as old, (args.checkpoint_dir/'.active.lock').open('a+') as lock:
        for handle in (old,lock):
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('A D0B formal-ablation process is active; do not run concurrently')
        if not args.resume and any(args.output_dir.glob('*/results.json')):raise ValueError('Use --resume latest; never duplicate the suite')
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=setup(args,data)
        try:
            print('REUSE:',[n for n in NAMES if r['reuse'][n]['reused']],flush=True)
            print('NEED NEW TRAINING:',[n for n in NAMES if not r['reuse'][n]['reused'] and n not in r['results']],flush=True)
            preflight(r,out,data,device);evaluate_reused(r,out,data,device)
            if args.run:
                for n in (*NEW,*(n for n in NAMES if n not in NEW)):run_one(n,r,out,data,device,args)
            pending=[n for n in NAMES if n not in r['results']]
            r['status']='COMPLETE' if not pending else 'PREPARED: '+str(len(pending))+' unique runs pending'
            save(r,out)
        except BaseException as exc:
            r['status']='STOP '+type(exc).__name__;r.setdefault('errors',[]).append(str(exc));save(r,out);raise
        finally:
            for row in list(r['reuse'].values())+list(r['supplementary_reuse'].values()):
                if row['reused'] and sha(row['path'])!=row['sha256']:raise AssertionError('Read-only source checkpoint changed')
        print('Revised report:',out,'STOP',flush=True)


if __name__=='__main__':main()
