"""One fixed D0B MoE experiment. Default: verify only; --run: one GPU fit."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from cmgm import config
from cmgm.models.moe_fusion import VARIANT,RoutingAccumulator
from cmgm.scripts.d0b_moe_audit import initialization,sanity,make_model
from cmgm.scripts.baseline_protocol import seed_all,loaders
from cmgm.scripts.formal_main_innovation_ablation import audit_data
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,metrics
from cmgm.training.train import train
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=dict(variant=VARIANT,seed=42,seq_len=20,features=21,horizons=[1,5,10,20],
    optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,max_epochs=200,patience=10,
    scheduler='ReduceLROnPlateau factor=.5 patience5',train_shuffle=False,train_drop_last=True,
    selection='multi-horizon validation Huber batch mean',prediction_loss='sum four Huber delta=.02',
    switch_kl='native beta_max=.0005 warmup20 unchanged',balance_coefficient=1e-4,balance_epsilon=1e-8,
    expert_order=['Temporal','Spatial','Interaction'],expert_dropout=.1,
    warmup='gamma=min(1,epoch/10), epoch starts at1; same gamma during validation',
    inference='Restore saved best epoch buffer; retain its gamma even for best epoch<10',
    initialization='standard PyTorch Linear; native shared modules constructed first',
    weight_training='All active shared/expert parameters train; only architecture definitions are frozen',
    single_run=True)
CONTROLS={'TemporalOnly':'w/o Spatial Branch','Fixed Equal Fusion':'w/o Adaptive Fusion','Adaptive Gated Fusion / Full D0B':'Full D0B'}


def save(r,out):
    from cmgm.scripts.d0b_moe_report import report
    atomic_json(out/'results.json',r);report(r,out)


def controls(source,data):
    old=json.loads(source.read_text());audit=audit_data(data)
    if old['data']!=audit:raise ValueError('Reference data fingerprints/order differ')
    result={}
    for label,name in CONTROLS.items():
        e=old['results'][name]
        if not e['sanity']['PASS'] or sha(e['path'])!=e['sha256']:raise ValueError('Invalid formal reference '+name)
        cp=torch.load(e['path'],map_location='cpu',weights_only=False);md=cp['metadata'];p=md['protocol']
        if not cp.get('training_complete') or md['seed']!=42 or md['data']!=audit:raise ValueError('Incomplete/different reference run')
        for k,v in dict(optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,epochs=200,patience=10,
                       seq_len=20,features=21,horizons=[1,5,10,20],train_shuffle=False,train_drop_last=True,
                       scheduler='ReduceLROnPlateau factor=.5 patience5',early_stopping='same VAL Huber',
                       prediction_loss='sum four Huber delta=.02',selection='multi-horizon VAL Huber batch mean').items():
            if p.get(k)!=v:raise ValueError('Reference protocol mismatch: '+k)
        result[label]=dict(metrics=e['metrics'],checkpoint=e['path'],checkpoint_sha256=e['sha256'],
            source_report=str(source),source_report_sha256=sha(source),source_variant=name,best_epoch=cp['best_epoch'],
            role='Existing formal evaluated control, reused read-only; never retrained',checkpoint_metadata=md)
    return result


def prepare(args,data,device):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS,config.HUBER_DELTA,config.LOSS_TYPE)!=(20,21,[1,5,10,20],.02,'huber'):raise ValueError('Frozen configuration mismatch')
    files=['cmgm/models/moe_fusion.py','cmgm/models/hetero_mixhop_model.py','cmgm/models/switching_latent_transformer.py',
           'cmgm/training/train.py','cmgm/scripts/d0b_moe_fusion.py','cmgm/scripts/d0b_moe_audit.py','cmgm/scripts/d0b_moe_report.py',
           'cmgm/scripts/main_ablation.py','cmgm/scripts/baseline_protocol.py','cmgm/training/metric_standard.py','cmgm/config.py']
    sources={f:sha(ROOT/f) for f in files};audit=audit_data(data)
    if args.resume:
        found=sorted(args.output_dir.glob('*/results.json'))
        if args.resume=='latest' and not found:raise ValueError('No prepared MoE run')
        out=found[-1].parent if args.resume=='latest' else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['config']!=PROTOCOL or r['data']!=audit or r['source_hashes']!=sources:raise ValueError('Resume source/protocol/data mismatch')
        if Path(r['checkpoint']).resolve()!=args.checkpoint.resolve():raise ValueError('Resume checkpoint destination mismatch')
    else:
        if any(args.output_dir.glob('*/results.json')):raise ValueError('Use --resume latest; one configuration, no duplicate experiment')
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        r=dict(config=PROTOCOL,data=audit,source_hashes=sources,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               controls=controls(args.reference_results,data),status='PREPARING',checkpoint=str(args.checkpoint),initialization={},sanity={},trained=False)
        save(r,out)
    # Always check the device used for this launch before allowing a formal fit.
    if not r['sanity'].get('PASS') or r['sanity'].get('device')!=str(device):
        x,y=next(iter(loaders(data,full=True)['val']))[:2]
        m,a=initialization(data,device);r['initialization']=a
        if not a['PASS']:save(r,out);raise AssertionError('Shared initialization failed')
        c=sanity(m,x[:2].to(device),y[:2].to(device));c['device']=str(device);r['sanity']=c
        save(r,out)
        if not c['PASS']:raise AssertionError('MoE structural sanity failed')
        del m
    for c in r['controls'].values():
        if sha(c['checkpoint'])!=c['checkpoint_sha256'] or sha(c['source_report'])!=c['source_report_sha256']:raise ValueError('Reference artifact changed')
    r['status']='COMPLETE' if r.get('evaluation') else 'PREPARED: one MoE GPU run pending';save(r,out)
    return r,out


@torch.no_grad()
def evaluate(model,data,device):
    model.eval();result={};diagnostics={}
    for split,loader in loaders(data,full=True).items():
        ps=[];ys=[];pis=[];effective=[];experts=[];fused=[];stats=RoutingAccumulator()
        for batch in loader:
            p=model(batch[0].to(device));y=batch[1]
            if p.shape!=y.shape or not torch.isfinite(p).all():raise ValueError('Invalid evaluated prediction')
            ps.append(p.cpu().numpy());ys.append(y.numpy());f=model.moe_fusion;stats.update(f)
            pis.append(f.last_pi.double().cpu());effective.append(f.last_effective_pi.double().cpu())
            if split=='test':experts.append(f.last_experts.double().cpu());fused.append(f.last_fused.double().cpu())
        result[split]=metrics(np.concatenate(ps),np.concatenate(ys))
        pi=torch.cat(pis);eff=torch.cat(effective);mean=pi.mean(0)
        d=dict(stats.summary(model.moe_fusion),mean_pi=mean.tolist(),mean_effective_pi=eff.mean(0).tolist(),
               hard_occupancy=torch.bincount(pi.argmax(-1),minlength=3).double().div(len(pi)).tolist(),
               pi_quantiles={str(q):torch.quantile(pi,q,dim=0).tolist() for q in (.1,.5,.9,.99)},
               pi_min=pi.min(0).values.tolist(),pi_max=pi.max(0).values.tolist(),
               effective_expert_count=float(np.exp(stats.summary(model.moe_fusion)['routing_entropy'])),
               uniform_entropy=float(np.log(3)),best_epoch=int(model.moe_fusion.epoch))
        if split=='test':
            e=torch.cat(experts);h=torch.cat(fused)
            d.update(expert_mean_L2=e.norm(dim=-1).mean(0).tolist(),h_moe_mean_L2=float(h.norm(dim=-1).mean()),
                disagreement={label:float((e[:,i]-e[:,j]).abs().mean()) for label,i,j in [('T_S',0,1),('T_ST',0,2),('S_ST',1,2)]})
        diagnostics[split]=d
    return dict(metrics=result,diagnostics=diagnostics)


def execute(r,out,data,device):
    if r.get('evaluation'):return
    path=Path(r['checkpoint']);path.parent.mkdir(parents=True,exist_ok=True)
    md=dict(variant=VARIANT,seed=42,config=PROTOCOL,data=r['data'],source_hashes=r['source_hashes'],git_sha=r['git_sha'])
    seed_all(42);m=make_model(data).to(device)
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp.get('metadata')!=md:raise ValueError('Existing checkpoint belongs to another run/source; do not overwrite')
        history=cp['history'];m.load_state_dict(cp['model_state_dict'],strict=True)
    else:
        if r.get('started'):raise ValueError('Interrupted fit requires manual review; never silently repeat')
        r['started']=True;r['status']='TRAINING';save(r,out)
        sources=loaders(data)
        def history_callback(history):
            atomic_json(out/'training_history.json',history)
            atomic_json(out/'routing_history.json',history.get('moe_routing_history',[]))
        history=train(m,sources['train'],sources['val'],torch.empty((2,0),dtype=torch.long,device=device),torch.empty(0,device=device),device,
            num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,checkpoint_path=str(path),checkpoint_metadata=md,
            epoch_history_callback=history_callback)
        cp=torch.load(path,map_location='cpu',weights_only=False)
    m.load_state_dict(cp['model_state_dict'],strict=True)
    if int(m.moe_fusion.epoch)!=cp['best_epoch']:raise ValueError('Best checkpoint routing epoch mismatch')
    m.switching_latent_transformer.set_epoch(cp['best_epoch'])
    r['trained']=True;r['history']=history;r['checkpoint_sha256']=sha(path)
    r['best_checkpoint_metadata']=dict(best_epoch=cp['best_epoch'],best_val_loss=cp['best_val_loss'],
        moe_epoch=int(m.moe_fusion.epoch),moe_gamma=m.moe_fusion.gamma,parameter_count=sum(p.numel() for p in m.parameters()),metadata=md)
    r['status']='EVALUATING';save(r,out)
    x,y=next(iter(loaders(data,full=True)['val']))[:2]
    # autograd.grad sanity needs fresh .grad=None after training; no weights change.
    m.zero_grad(set_to_none=True)
    r['best_sanity']=sanity(m,x[:2].to(device),y[:2].to(device),require_initial_activity=False)
    if not r['best_sanity']['PASS']:save(r,out);raise AssertionError('Best-checkpoint sanity failed; do not interpret performance')
    r['evaluation']=evaluate(m,data,device);r['status']='COMPLETE';save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/d0b_moe_fusion')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints'/f'{VARIANT}_best.pt')
    p.add_argument('--reference-results',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation/20260916_154242/results.json')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.cpu_check and args.run:raise ValueError('No CPU formal training')
    if not args.cpu_check and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU training fallback')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda')
        r,out=prepare(args,data,device)
        try:
            if args.run:execute(r,out,data,device)
        except BaseException as exc:
            r['status']='STOP '+type(exc).__name__;r.setdefault('errors',[]).append(str(exc));save(r,out);raise
        finally:
            for c in r['controls'].values():
                if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise AssertionError('Control checkpoint modified')
        print('MoE report:',out,'STOP',flush=True)


if __name__=='__main__':main()
