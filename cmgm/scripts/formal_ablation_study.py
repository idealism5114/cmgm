"""D0B formal ablations: read-only preparation by default, --run fits 13 controls once."""
import argparse
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from cmgm import config
from cmgm.models.formal_d0b_ablation import NAMES,KEY,FormalD0BAblation
from cmgm.scripts.formal_ablation_audit import init_audit,sanity
from cmgm.scripts.baseline_protocol import seed_all,loaders,data_audit
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,metrics
from cmgm.training.train import train_epoch,validate_epoch
from cmgm.training.metric_standard import population_metrics
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=dict(name='D0B FORMAL ABLATION STUDY',configurations=list(NAMES),seed=42,seq_len=20,features=21,horizons=[1,5,10,20],
    optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,epochs=200,patience=10,scheduler='ReduceLROnPlateau factor=.5 patience5',
    prediction_loss='sum four Huber delta=.02',selection='multi-horizon VAL Huber batch mean',early_stopping='same VAL Huber',
    KL='beta_max=.0005; native epoch1=0, epoch20=.0005; except NoKL/NoTemporal contribution0, NoMarkov fixed-uniform p/prior KL0',
    formal_reference='FullD0B-Control (new seed42 training), never historical checkpoint',
    neutral='Zero: main_ablation.build_data uses per-node/per-feature TRAIN mean/std centering before Dataset. This is the TRAIN center, not a VAL/TEST-derived value.',
    single_run=True,train_shuffle=False,train_drop_last=True,CUDA_deterministic=True,CUDA_benchmark=False)


def save(r,out):
    from cmgm.scripts.formal_ablation_report import report
    atomic_json(out/'results.json',r);atomic_json(out/'partial_results.json',r);report(r,out)


def free():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(m,data,device):
    result=dict(metrics={},per_commodity=[],mechanism={})
    for split,loader in loaders(data,full=True).items():
        ps=[];ys=[];probs=[];longs=[];micros=[];m.eval()
        for batch in loader:
            p=m(batch[0].to(device));y=batch[1]
            if not torch.isfinite(p).all() or p.shape!=y.shape:raise ValueError('Invalid prediction')
            ps.append(p.cpu().numpy());ys.append(y.numpy())
            b=m.switching_latent_transformer
            if m.ablation_name in ('FullD0B-Control','w/o Switch KL') and split=='test':probs.append(b.last_regime_probabilities.cpu())
            if m.ablation_name in ('FullD0B-Control','w/o Balanced Readout'):
                longs.append(b.last_h_long.norm(dim=-1).cpu());micros.append(b.last_h_micro.norm(dim=-1).cpu())
        p,y=np.concatenate(ps),np.concatenate(ys);result['metrics'][split]=metrics(p,y)
        if split=='test':
            idx=config.MULTI_HORIZONS.index(5);cs,ce=data['market_indices']['commodity']
            result['per_commodity']=[dict(commodity=str(data['feature_names'][cs+i]),**population_metrics(p[:,idx,i],y[:,idx,i])) for i in range(ce-cs)]
        if probs:
            prob=torch.cat(probs).double();result['mechanism']['regime']=dict(mean_p=prob.mean((0,1)).tolist(),entropy=float(-(prob*torch.log(prob+1e-8)).sum(-1).mean()),hard_occupancy=torch.bincount(prob.argmax(-1).flatten(),minlength=3).double().div(prob.shape[0]*prob.shape[1]).tolist())
        if longs:
            h=float(torch.cat(longs).double().mean());z=float(torch.cat(micros).double().mean())
            result['mechanism'][split]=dict(long_norm=h,micro_norm=z,micro_long_ratio=z/(h+1e-8))
    return result


def setup(args,data):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS)!=(20,21,[1,5,10,20]):raise ValueError('Wrong data configuration')
    audit=data_audit(data)
    audit['mapping']=[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]
    if not audit['PASS']:raise ValueError('Target order/split audit failed')
    sources=['cmgm/models/formal_d0b_ablation.py','cmgm/models/hetero_mixhop_model.py','cmgm/models/switching_latent_transformer.py',
             'cmgm/scripts/formal_ablation_study.py','cmgm/scripts/formal_ablation_audit.py','cmgm/training/train.py','cmgm/scripts/main_ablation.py','cmgm/config.py']
    hashes={s:sha(ROOT/s) for s in sources}
    if args.resume:
        options=sorted(args.output_dir.glob('*/results.json'));out=options[-1].parent if args.resume=='latest' and options else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['data']!=audit or r['source_hashes']!=hashes:raise ValueError('Resume protocol/data/implementation mismatch')
    else:
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        # Past development variants do not establish these exact bypass/initialization semantics.
        candidates=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'checkpoints').rglob('*.pt'))
        r=dict(protocol=PROTOCOL,data=audit,source_hashes=hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            history_inventory=candidates,jobs={},results={},initialization={},sanity={},masking={},status='PREPARING',
            checkpoint_dir=str(args.checkpoint_dir/out.name),historical_path=str(args.d0b_checkpoint),historical_sha256=sha(args.d0b_checkpoint),
            reuse_policy='Only completed formal artifacts with exact source/data/protocol metadata may resume. Historical D0B is sanity only; legacy development variants lack exact formal implementation match.')
        r['existing_audit']={n:dict(existing_checkpoint=any(KEY[n] in Path(p).name for p in candidates),exact_protocol_match=False,reuse=False,need_training=True,
            reason='New Full reference required' if n==NAMES[0] else 'No previous artifact has this exact formal bypass implementation/protocol metadata') for n in NAMES}
    if sha(args.d0b_checkpoint)!=r['historical_sha256']:raise ValueError('Historical checkpoint changed')
    atomic_json(out/'protocol.json',PROTOCOL);save(r,out);return r,out


def preflight(r,out,data,device):
    x,y=next(iter(loaders(data,full=True)['val']))[:2];x=x[:2].to(device);y=y[:2].to(device)
    if 'historical' not in r:
        from cmgm.scripts.d0b_5d_error_regime_diagnostic import checkpoint_payload
        seed_all(42);m=FormalD0BAblation(NAMES[0],data).to(device);cp=checkpoint_payload(r['historical_path'])
        m.load_state_dict(cp.get('model_state_dict',cp.get('state_dict',cp)))
        result=evaluate(m,data,device);ref=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)
        if not all(np.isclose(result['metrics']['test']['5'][k],v,rtol=1e-5,atol=1e-9) for k,v in ref.items()):raise AssertionError('STOP: historical D0B mismatch')
        r['historical']=dict(result,role='evaluator sanity only, never formal delta reference',epoch=cp.get('best_epoch'));del m,cp;free();save(r,out)
    for n in NAMES:
        if r['sanity'].get(n,{}).get('PASS'):continue
        m,init=init_audit(n,data,device);r['initialization'][n]=init
        if not init['PASS']:save(r,out);raise AssertionError(n+' shared initialization mismatch')
        check=sanity(m,x,y);r['sanity'][n]=check
        if 'masking' in check:r['masking'][n]=check['masking']
        print('[Formal ablation sanity]',n,'params',init['total_instantiated'],'init diff',init['max_abs_diff'],'PASS',check['PASS'],flush=True)
        del m;free();save(r,out)
        if not check['PASS']:raise AssertionError(n+' sanity failed')


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
    seed_all(42);m=FormalD0BAblation(n,data).to(device);path=Path(r['checkpoint_dir'])/(KEY[n]+'_seed42.pt');path.parent.mkdir(parents=True,exist_ok=True)
    md=dict(name=n,seed=42,protocol=PROTOCOL,data=r['data'],source_hashes=r['source_hashes']);cp=None
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['metadata']!=md:raise ValueError('Checkpoint exact-match failure')
        if cp['training_complete']:m.load_state_dict(cp['model_state_dict'])
        else:
            if not args.retry_invalid:raise ValueError('Interrupted fit: inspect before --retry-invalid; no poor-performance retries')
            path.rename(path.with_name(path.stem+'_invalid_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pt'));cp=None
    if cp is None:
        if r['jobs'].get(n,{}).get('status') in ('RUNNING','INVALID') and not args.retry_invalid:raise ValueError('Interrupted fit requires explicit review/retry')
        r['jobs'][n]=dict(status='RUNNING',path=str(path));r['status']='TRAINING '+n;save(r,out)
        cp=fit(n,m,data,device,path,md,out)
    r['jobs'][n]=dict(status='FITTED',path=str(path));save(r,out)
    x,y=next(iter(loaders(data,full=True)['val']))[:2]
    check=sanity(m,x[:2].to(device),y[:2].to(device))
    if not check['PASS']:r['sanity'][n]['best']=check;save(r,out);raise AssertionError(n+' trained sanity failed')
    m.switching_latent_transformer.set_epoch(cp['best_epoch']);result=evaluate(m,data,device)
    r['results'][n]=dict(result,path=str(path),sha256=sha(path),runtime=cp['runtime'],sanity=check)
    r['jobs'][n]['status']='DONE';r['existing_audit'][n].update(need_training=False,exact_protocol_match=True)
    del m,cp;free();save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume');p.add_argument('--retry-invalid',action='store_true');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/formal_ablation_study');p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/formal_ablation')
    p.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt');p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.cpu_check and args.run:raise ValueError('No CPU formal training')
    if not args.cpu_check and not torch.cuda.is_available():raise ValueError('GPU unavailable; no training fallback')
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    with (args.checkpoint_dir/'.active.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A formal ablation process is already active')
        if args.run and not args.resume and any(args.output_dir.glob('*/results.json')):raise ValueError('Use --resume latest --run; no duplicate control suite')
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=setup(args,data)
        try:
            preflight(r,out,data,device)
            if args.run:
                for n in NAMES:run_one(n,r,out,data,device,args)
            r['status']='COMPLETE' if len(r['results'])==13 else 'PREPARED: 13 unique fixed runs pending';save(r,out)
        except BaseException as e:
            r['status']='STOP '+type(e).__name__;r.setdefault('errors',[]).append(str(e));save(r,out);raise
        finally:
            if sha(args.d0b_checkpoint)!=r['historical_sha256']:raise AssertionError('Historical checkpoint changed')
        print('Formal ablation report:',out,'STOP',flush=True)


if __name__=='__main__':main()
