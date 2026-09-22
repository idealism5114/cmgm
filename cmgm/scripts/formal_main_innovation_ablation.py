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
from cmgm.models.formal_d0b_main_ablation import NAMES,KEY,REUSE,NEW,DEFINITIONS,FULL,CANDIDATE,GLOBAL,MainInnovationAblation
from cmgm.scripts.formal_main_ablation_audit import init_audit,sanity
from cmgm.scripts.baseline_protocol import seed_all,loaders,data_audit
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,metrics
from cmgm.scripts.formal_ablation_study import PROTOCOL as OLD_PROTOCOL,free
from cmgm.training.train import train_epoch,validate_epoch
from cmgm.training.metric_standard import population_metrics
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL={**OLD_PROTOCOL,'name':'D0B Candidate-Aware 2-Expert MoE FORMAL MAIN-INNOVATION ABLATION',
          'configurations':list(NAMES),'formal_reference':CANDIDATE,
          'reuse_policy':'Only audited Candidate Full / exact global mixture; old Adaptive-Gate internal ablations forbidden',
          'selection':'prediction-only four-horizon VAL Huber batch mean',
          'KL':'Native beta_max=.0005 warmup20; uniform p/prior mathematically zero; all other variants retain native KL',
          'moe_auxiliary_loss':None,'routing_warmup':None,'near_tie_relative_percent':.1}


def save(r,out):
    from cmgm.scripts.formal_main_ablation_report import report
    atomic_json(out/'results.json',r);atomic_json(out/'partial_results.json',r);report(r,out)


def audit_data(data):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS)!=(20,21,[1,5,10,20]):raise ValueError('Frozen data config changed')
    audit=data_audit(data)
    audit['mapping']=[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]
    if not audit['PASS']:raise ValueError('Data/order audit failed')
    return json.loads(json.dumps(audit))


def audit_reuse(name,source,path,audit):
    """Only the two completed native Candidate-family references are eligible."""
    row=dict(source_experiment=str(source.parent),source_report=str(source),path=str(path),
        old_checkpoint_found=path.exists(),definition_exact_match=False,reused=False)
    if not source.exists() or not path.exists():
        row['reason']='No completed native reference artifact';return row
    old=json.loads(source.read_text());cp=torch.load(path,map_location='cpu',weights_only=False);md=cp.get('metadata',{})
    # Imports are local: the historical runners import this module's audit_data.
    if name==FULL:
        from cmgm.scripts.d0b_candidate_2expert_moe import PROTOCOL as expected
    else:
        from cmgm.scripts.d0b_candidate_2expert_global_mixture import PROTOCOL as expected
    history=cp.get('history',{});vals=history.get('val_loss',[]);best_epoch=cp.get('best_epoch',0)
    completed=(bool(vals) and 0<best_epoch<=len(vals) and
        (len(vals)>=200 or len(vals)-best_epoch>=10) and
        np.isclose(cp.get('best_val_loss',float('inf')),min(vals),rtol=0,atol=1e-12))
    checks=dict(variant=md.get('variant')==REUSE[name],seed42=md.get('seed')==42,
        complete=old.get('status')=='COMPLETE' and old.get('trained') is True and completed and cp.get('training_complete',True) is True,
        protocol=md.get('config')==expected==old.get('config'),data=md.get('data')==audit==old.get('data'),
        source_metadata=md.get('source_hashes')==old.get('source_hashes'),
        history=all(v==old.get('history',{}).get(k) for k,v in history.items()) and
            set(old.get('history',{}))-set(history)<={'train_time'},
        best_epoch=best_epoch==old.get('best_checkpoint_metadata',{}).get('best_epoch'),
        initialization=old.get('initialization',{}).get('PASS') is True,
        sanity=old.get('sanity',{}).get('PASS') is True and old.get('best_sanity',{}).get('PASS') is True)
    if name!=FULL:checks['expert_initialization']=old.get('expert_initialization',{}).get('PASS') is True
    if sha(path)!=old.get('checkpoint_sha256'):raise ValueError('Reference checkpoint hash mismatch; STOP rather than retrain')
    # Record historical source drift explicitly. Current strict-load/native equivalence,
    # shared initialization and fresh predictions must pass before accepting reuse.
    drift=[s for s,h in old.get('source_hashes',{}).items() if not (ROOT/s).exists() or sha(ROOT/s)!=h]
    core=['cmgm/models/candidate_moe_fusion.py','cmgm/models/switching_latent_transformer.py']
    if name!=FULL:core.append('cmgm/models/global_mixture_fusion.py')
    checks['core_source_unchanged']=all(s in old.get('source_hashes',{}) and s not in drift for s in core)
    row.update(checks=checks,reused=all(checks.values()),definition_exact_match=checks['variant'] and checks['protocol'],
        sha256=sha(path),source_report_sha256=sha(source),best_epoch=best_epoch,
        historical_source_hashes=old.get('source_hashes',{}),source_drift=drift,
        expected_metrics=old.get('evaluation',{}).get('metrics',{}),
        training_complete=checks['complete'],completion_evidence='Legacy checkpoint has no completion flag: audited COMPLETE/trained report, identical checkpoint history, valid best epoch and early-stop/epoch-budget termination; report-only train_time is added after checkpoint serialization',
        report_only_history_keys=sorted(set(old.get('history',{}))-set(history)),
        runtime=dict(best_epoch=best_epoch,train_seconds=old.get('history',{}).get('train_time'),seconds_per_epoch=None))
    row['reason']='Eligible pending strict load and fresh reproduction; no retraining' if row['reused'] else 'REUSE REJECTED: '+','.join(k for k,v in checks.items() if not v)
    return row


def setup(args,data):
    if (config.TARGET_TYPE,config.LOSS_TYPE,config.HUBER_DELTA,config.LEARNING_RATE,config.WEIGHT_DECAY)!=('return','huber',.02,1e-4,1e-5):
        raise ValueError('Frozen loss/optimizer configuration changed')
    audit=audit_data(data)
    sources=['cmgm/models/formal_d0b_main_ablation.py','cmgm/scripts/formal_main_innovation_ablation.py',
        'cmgm/scripts/formal_main_ablation_audit.py','cmgm/scripts/formal_main_ablation_report.py',
        'cmgm/models/candidate_moe_fusion.py','cmgm/models/global_mixture_fusion.py','cmgm/models/hetero_mixhop_model.py',
        'cmgm/models/switching_latent_transformer.py','cmgm/models/model.py','cmgm/graph/adaptive_graph.py',
        'cmgm/training/train.py','cmgm/training/metric_standard.py','cmgm/scripts/baseline_protocol.py',
        'cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py','cmgm/config.py']
    hashes={s:sha(ROOT/s) for s in sources}
    if args.resume:
        options=[]
        for path in sorted(args.output_dir.glob('*/results.json')):
            if json.loads(path.read_text()).get('protocol')==PROTOCOL:options.append(path)
        if args.resume=='latest' and not options:raise ValueError('No prepared Candidate-MoE main ablation; historical Adaptive-Gate suite is ineligible')
        out=options[-1].parent if args.resume=='latest' else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['data']!=audit or r['source_hashes']!=hashes:raise ValueError('Resume data/protocol/implementation mismatch; STOP review')
        if r.get('errors'):raise ValueError('Invalid/interrupted experiment requires review, no silent rerun')
        if r['checkpoint_dir']!=str(args.checkpoint_dir/out.name):raise ValueError('Checkpoint destination changed')
        for name,source,path in ((FULL,args.full_results,args.full_checkpoint),('w/o Candidate-Aware Routing',args.global_results,args.global_checkpoint)):
            if Path(r['reuse'][name]['path']).resolve()!=path.resolve() or Path(r['reuse'][name]['source_report']).resolve()!=source.resolve():
                raise ValueError('Resume reference paths changed')
    else:
        for path in args.output_dir.glob('*/results.json'):
            if json.loads(path.read_text()).get('protocol')==PROTOCOL:raise ValueError('Use --resume latest; duplicate Candidate suite forbidden')
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        reuse={FULL:audit_reuse(FULL,args.full_results,args.full_checkpoint,audit),
            'w/o Candidate-Aware Routing':audit_reuse('w/o Candidate-Aware Routing',args.global_results,args.global_checkpoint,audit)}
        if not reuse[FULL]['reused']:raise ValueError('Full Candidate reference audit failed: '+str(reuse[FULL]))
        for n in NEW:reuse[n]=dict(old_checkpoint_found=False,definition_exact_match=False,reused=False,
            reason='Must estimate under final Candidate MoE; historical Adaptive-Gate results ineligible')
        r=dict(protocol=PROTOCOL,data=audit,source_hashes=hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            reuse=reuse,jobs={},results={},initialization={},sanity={},status='PREPARING',checkpoint_dir=str(args.checkpoint_dir/out.name))
    save(r,out);return r,out


@torch.no_grad()
def evaluate(model,data,device):
    result=dict(metrics={},per_commodity=[],mechanism={})
    name=model.main_name
    for split,loader in loaders(data,full=True).items():
        preds=[];targets=[];probs=[];longs=[];micros=[];disagreements=[];routing=[];model.eval()
        for batch in loader:
            pred=model(batch[0].to(device));y=batch[1]
            if pred.shape!=y.shape or not torch.isfinite(pred).all():raise ValueError('Prediction shape/finite failure')
            preds.append(pred.cpu().numpy());targets.append(y.numpy());b=model.switching_latent_transformer
            if hasattr(model,'candidate_moe_fusion'):routing.append(model.candidate_moe_fusion.last['pi'].cpu())
            if split=='test' and name in (FULL,'w/o Adaptive Regime Routing'):
                probs.append(b.last_regime_probabilities.cpu())
            if split=='test' and name in (FULL,'w/o Regime-Specific Transitions'):
                c=b.last_latent_candidates
                disagreements.append(torch.stack([(c[:,:,i]-c[:,:,j]).abs().mean(-1) for i,j in ((0,1),(0,2),(1,2))],-1).cpu())
            if name in (FULL,'w/o Balanced Readout'):
                longs.append(b.last_h_long.norm(dim=-1).cpu());micros.append(b.last_h_micro.norm(dim=-1).cpu())
        p,y=np.concatenate(preds),np.concatenate(targets);result['metrics'][split]=metrics(p,y)
        if split=='test':
            idx=config.MULTI_HORIZONS.index(5);cs,ce=data['market_indices']['commodity']
            result['per_commodity']=[dict(commodity=str(data['feature_names'][cs+i]),**population_metrics(p[:,idx,i],y[:,idx,i])) for i in range(ce-cs)]
        if routing:
            pi=torch.cat(routing).double()
            result['mechanism'].setdefault('router',{})[split]=dict(mean_pi=pi.mean(0).tolist(),std_pi=pi.std(0,unbiased=False).tolist(),
                P10=torch.quantile(pi,.1,dim=0).tolist(),P50=torch.quantile(pi,.5,dim=0).tolist(),P90=torch.quantile(pi,.9,dim=0).tolist(),
                entropy=float(-(pi*(pi+1e-8).log()).sum(-1).mean()))
        if hasattr(model,'global_mixture_fusion'):
            result['mechanism']['global_mixture']=dict(model.global_mixture_fusion.weight_diagnostics(),scope='one global pair across every sample; not sample-dependent')
        if hasattr(model,'simple_fusion'):
            result['mechanism']['simple_fusion']=dict(parameters=sum(p.numel() for p in model.simple_fusion.parameters()),
                experts_absent=not hasattr(model,'candidate_moe_fusion'),router_absent=not hasattr(model,'candidate_moe_fusion'),input='[s||t]',shared_head='unchanged')
        if probs:
            prob=torch.cat(probs).double()
            result['mechanism']['regime']=dict(mean_p=prob.mean((0,1)).tolist(),entropy=float(-(prob*torch.log(prob+1e-8)).sum(-1).mean()),
                hard_occupancy=torch.bincount(prob.argmax(-1).flatten(),minlength=3).double().div(prob.shape[0]*prob.shape[1]).tolist(),
                occupancy_note='This is deterministic tie breaking under uniform probabilities, not learned collapse.' if name=='w/o Adaptive Regime Routing' else 'Descriptive learned regime occupancy.')
        if disagreements:result['mechanism']['candidate_pairwise_mean_absolute_disagreement']=float(torch.cat(disagreements).double().mean())
        if longs:
            h=float(torch.cat(longs).double().mean());z=float(torch.cat(micros).double().mean())
            result['mechanism'][split]=dict(long_norm=h,micro_norm=z,micro_long_ratio=z/(h+1e-8))
    return result


def preflight(r,out,data,device):
    x,y=next(iter(loaders(data,full=True)['train']))[:2];x=x[:2].to(device);y=y[:2].to(device)
    for n in NAMES:
        if r['sanity'].get(n,{}).get('PASS') and r['sanity'][n].get('device')==str(device):continue
        model,initial=init_audit(n,data,device,x);r['initialization'][n]=initial
        if not initial['PASS']:save(r,out);raise AssertionError(n+' shared initialization failed')
        check=sanity(model,x,y);check.update(device=str(device),split='TRAIN');r['sanity'][n]=check
        print('[Revised sanity]',n,'params',initial['total_instantiated'],'shared diff',initial['shared_parameter_initial_max_abs_diff'],'PASS',check['PASS'],flush=True)
        del model;free();save(r,out)
        if not check['PASS']:raise AssertionError(n+' sanity failed')


def evaluate_reused(r,out,data,device):
    for n in (FULL,'w/o Candidate-Aware Routing'):
        row=r['reuse'][n]
        if not row['reused']:continue
        if sha(row['path'])!=row['sha256'] or sha(row['source_report'])!=row['source_report_sha256']:
            raise ValueError('Read-only reference artifact changed')
        if n in r['results']:continue
        cp=torch.load(row['path'],map_location='cpu',weights_only=False)
        seed_all(42);m=MainInnovationAblation(n,data).to(device);m.load_state_dict(cp['model_state_dict'],strict=True)
        x,y=next(iter(loaders(data,full=True)['train']))[:2]
        check=sanity(m,x[:2].to(device),y[:2].to(device))
        if not check['PASS']:raise AssertionError(n+' reused best sanity failed')
        result=evaluate(m,data,device);expected=row['expected_metrics']
        errors={s:{h:{k:abs(v-expected[s][h][k]) for k,v in values.items()} for h,values in hm.items()} for s,hm in result['metrics'].items()}
        ok=all(np.isclose(v,expected[s][h][k],rtol=1e-5,atol=1e-9) for s,hm in result['metrics'].items() for h,values in hm.items() for k,v in values.items())
        row['fresh_reproduction']=dict(PASS=bool(ok),absolute_errors=errors,rtol=1e-5,atol=1e-9)
        if not ok:save(r,out);raise ValueError(n+' fresh evaluation does not reproduce audited artifact; STOP')
        row['strict_load']=True;row['reuse_status']='PASS'
        r['results'][n]=dict(result,path=row['path'],sha256=row['sha256'],reused=True,source_experiment=row['source_experiment'],runtime=row['runtime'],sanity=check)
        print('[Fresh reference]',n,'TEST5 MAE',result['metrics']['test']['5']['MAE'],flush=True)
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
    if n==FULL:raise AssertionError('Full Candidate must never train')
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
    p.add_argument('--full-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_moe/20260921_151711/results.json')
    p.add_argument('--full-checkpoint',type=Path,default=ROOT/'checkpoints'/f'{CANDIDATE}_best.pt')
    p.add_argument('--global-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_global_mixture/20260921_214032/results.json')
    p.add_argument('--global-checkpoint',type=Path,default=ROOT/'checkpoints'/f'{GLOBAL}_best.pt')
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
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=setup(args,data)
        try:
            print('REUSE:',[n for n in NAMES if r['reuse'][n]['reused']],flush=True)
            print('NEED NEW TRAINING:',[n for n in NAMES if not r['reuse'][n]['reused'] and n not in r['results']],flush=True)
            evaluate_reused(r,out,data,device);preflight(r,out,data,device)
            if args.run:
                for n in (*NEW,*(n for n in NAMES if n not in NEW)):run_one(n,r,out,data,device,args)
            pending=[n for n in NAMES if n not in r['results']]
            r['status']='COMPLETE' if not pending else 'PREPARED: '+str(len(pending))+' unique runs pending'
            save(r,out)
        except BaseException as exc:
            r['status']='STOP '+type(exc).__name__;r.setdefault('errors',[]).append(str(exc));save(r,out);raise
        finally:
            for row in r['reuse'].values():
                if row['reused'] and (sha(row['path'])!=row['sha256'] or sha(row['source_report'])!=row['source_report_sha256']):raise AssertionError('Read-only source artifact changed')
        print('Revised report:',out,'STOP',flush=True)


if __name__=='__main__':main()
