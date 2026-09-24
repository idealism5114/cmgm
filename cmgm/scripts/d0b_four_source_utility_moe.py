"""Four-source predictive utility MoE. Synthetic default; explicit --run only."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from cmgm.models.four_source_utility_moe import VARIANT, LABELS, utility_targets
from cmgm.scripts import d0b_candidate_moe_bottleneck16 as shared
from cmgm.scripts.d0b_candidate_gated_interaction import reference_audit, REFERENCE, REFERENCE_CP
from cmgm.scripts.d0b_four_source_audit import make_model, initialization, sanity, counts
from cmgm.scripts.baseline_protocol import prediction_loss
from cmgm.scripts.d0b_candidate_moe_diagnostics import distribution
from cmgm.scripts.formal_v2_protocol import atomic_json, sha, metrics
from cmgm.training.train import train
ROOT=Path(__file__).resolve().parents[2]
OUT_ROOT=ROOT/'experiments/d0b_four_source_utility_moe'
PROTOCOL={k:v for k,v in shared.PROTOCOL.items() if k not in ('variant','expert_hidden_dim','future_controls','future_control_policy')}
PROTOCOL.update(variant=VARIANT,expert_order=list(LABELS),expert_input_dims=[64,64,64,192],
    expert_structure='Linear(input,64),ReLU,Dropout(.3),Linear(64,96)',
    router='3 independent LN64; Linear192,64; ReLU; Linear64,4; zero final init',
    utility='TRAIN-only router-only .1*stopgrad(mean error)*KL(q||pi_aux)',
    tau=1.,lambda_route=.1,epsilon=1e-8,moe_auxiliary_loss='TRAIN-only .1*detached scale*KL(q||pi_aux)',
    expert_dropout=.3,
    initialization='Native backbone first; independent deepcopies of head; Joint first layer new',
    readout='live balanced long/micro + spatial; no state_readout or branch projections; prediction-level mixture')

def source_record():
    result=shared.source_record()
    for file in (Path(__file__),ROOT/'cmgm/scripts/d0b_four_source_audit.py',ROOT/'cmgm/scripts/d0b_candidate_gated_interaction.py'):
        result['hashes'][str(file.relative_to(ROOT))]=sha(file)
    result['git_status']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    return result

def probabilities(pi):
    return dict(by_expert={name:distribution(pi[:,k]) for k,name in enumerate(LABELS)},
        mean=pi.mean(0).tolist(),std=pi.std(0).tolist(),entropy=float(-(pi*np.log(pi+1e-8)).sum(-1).mean()))

@torch.no_grad()
def evaluate_frozen(model,data,seed,device,out):
    model.eval();arrays={};result={};mean_pi=None
    for split,loader in shared.data_loaders(data,seed,full=True).items():
        rows=[];experts=[];pis=[];ys=[];losses=[]
        for x,y in loader:
            pred=model(x.to(device));f=model.four_source_moe
            assert pred.shape==y.shape and torch.isfinite(pred).all()
            rows.append(pred.cpu().numpy());experts.append(f.predictions.cpu().numpy());pis.append(f.pi.cpu().numpy());ys.append(y.numpy())
            losses.append(float(prediction_loss(pred,y.to(device))))
        p,e,pi,y=map(np.concatenate,(rows,experts,pis,ys))
        if split=='train':mean_pi=pi.astype(np.float64).mean(0) # No labels used to set weights.
        if mean_pi is None:raise ValueError('Full TRAIN must precede VAL')
        fixed=(e*mean_pi[None,:,None,None]).sum(1)
        _,_,errors=utility_targets(torch.from_numpy(e),torch.from_numpy(y))
        result[split]=dict(metrics=metrics(p,y),prediction_only_huber_batch_mean=float(np.mean(losses)),
            experts={name:metrics(e[:,k],y) for k,name in enumerate(LABELS)},routing=probabilities(pi),
            expert_error={name:distribution(errors[:,k].numpy()) for k,name in enumerate(LABELS)},
            disagreement={LABELS[i]+' vs '+LABELS[j]:float(np.abs(e[:,i]-e[:,j]).mean()) for i in range(4) for j in range(i+1,4)},
            fixed_mean_metrics=metrics(fixed,y),fixed_minus_dynamic_MAE={h:metrics(fixed,y)[h]['MAE']-metrics(p,y)[h]['MAE'] for h in ('1','5','10','20')},
            sample_prediction_change=distribution(np.abs(fixed-p).mean((1,2))))
        arrays.update({split+'_prediction':p,split+'_experts':e,split+'_pi':pi,split+'_target':y,split+'_fixed_mean_prediction':fixed,
                       split+'_sample_prediction_change':np.abs(fixed-p).mean((1,2))})
    np.savez_compressed(out/'train_val_predictions.npz',**arrays)
    return dict(formal=result,train_mean_pi=mean_pi.tolist(),fixed_mean_policy='Frozen full TRAIN eval probabilities only, no labels; no optimization')

def report(r,out):
    lines=['# Four-source prediction experts + utility routing','',r['status'],'',
      'Overall architecture exploration: expert sources, prediction-level fusion and utility objective change together; no single-factor attribution.',
      'Sources: live LN(W_H H_last), LN(W_Z Z_last), h_spatial, concat of all three. Native K=3 unchanged.',
      'Y=sum_k pi_k Y_k; four independent Linear(input,64)-ReLU-Dropout(.3)-Linear(64,96) experts.',
      'ell_bk=sum_h mean_c Huber(.02); q=softmax(-detach(ell)/(mean_k detach(ell)+1e-8));',
      'L=L_pred+native KL+.1*detach(mean ell)*mean KL(q||Router(detached sources)).',
      'VAL selection/scheduler/early stopping: prediction-only four-horizon batch-mean Huber. No VAL utility optimization.',
      f"Reference: {r.get('reference',{}).get('status','PENDING')}; {r.get('reference',{}).get('reason','')}",
      f"Parameters: {r.get('initialization',{}).get('parameters',{})}",
      f"Best epoch: {r.get('best_epoch','N/A')}; train seconds: {r.get('training_seconds','N/A')}",'',
      '| Split | Predictor | Horizon | MAE | MSE | RMSE | Hit% |','|---|---|---:|---:|---:|---:|---:|']
    for split,row in r.get('evaluation',{}).get('formal',{}).items():
        for label,ms in [('Mixture',row['metrics']),*row['experts'].items(),('Frozen TRAIN-mean',row['fixed_mean_metrics'])]:
            for h,m in ms.items():lines.append(f"| {split} | {label} | {h} | {m['MAE']:.10g} | {m['MSE']:.10g} | {m['RMSE']:.10g} | {100*m['Hit']:.6f} |")
        lines+=['',f"{split} prediction-only Huber batch mean: {row['prediction_only_huber_batch_mean']:.12g}"]
    if r.get('reference',{}).get('status')=='PASS' and r.get('evaluation'):
        lines+=['','Audited Candidate64 comparison (100*(new-old)/old; negative MAE change is lower error):','',
                '| Split | Horizon | Candidate64 MAE | New MAE | Change % |','|---|---:|---:|---:|---:|']
        for split in ('train','val'):
            for h in ('1','5','10','20'):
                old=r['reference']['metrics'][split][h]['MAE'];new=r['evaluation']['formal'][split]['metrics'][h]['MAE']
                change=100*(new-old)/old if old else None
                lines.append(f'| {split} | {h} | {old:.10g} | {new:.10g} | {change} |')
        lines+=['',f"Candidate64 best epoch={r['reference']['best_epoch']}; params={r['reference']['parameters']}; selection VAL Huber={r['reference']['selection_val_huber']:.12g}"]
    lines+=['','Distributions, expert errors, prediction disagreement, frozen TRAIN-mean deltas and per-sample changes: diagnostics.json and train_val_predictions.npz.',
      'Utility q is a relative-error soft target, not optimal mixing weights or an oracle. Labels never enter normal forward.',
      'TRAIN/VAL only; single run, no TEST or ablations. No conclusions from routing std/occupancy alone; no claim of mechanism recovery.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(out/'results.json',r)

def preflight(seed,out):
    shared.check_config();data=dict(n_nodes=284,market_indices=dict(stock=(0,248),bond=(248,260),commodity=(260,284)))
    model,init=initialization(data,seed);g=torch.Generator().manual_seed(seed+1)
    checks=sanity(model,torch.randn(2,20,284,21,generator=g),torch.randn(2,4,24,generator=g)*.02)
    out.mkdir(parents=True,exist_ok=False)
    r=dict(status='SYNTHETIC PREFLIGHT PASS — NO TRAINING',config={**PROTOCOL,'seed':seed},initialization=init,sanity=checks,
           source=source_record(),reference=dict(status='PENDING',reason='Synthetic only; no real data/reference checkpoint accessed'))
    for key,filename in [('config','config'),('initialization','initialization_audit'),('sanity','structural_sanity'),('source','source_hashes')]:atomic_json(out/(filename+'.json'),r[key])
    report(r,out);print(json.dumps(dict(output=str(out),parameters=init['parameters'],shared_max_abs_diff=init['shared_max_abs_diff'],PASS=checks['PASS']),indent=2))

def finish(model,data,r,out,device):
    before=sha(r['checkpoint'])
    if before!=r['checkpoint_sha256']:raise ValueError('Checkpoint drift')
    r['evaluation']=evaluate_frozen(model,data,r['config']['seed'],device,out)
    if sha(r['checkpoint'])!=before:raise ValueError('Frozen checkpoint modified')
    if abs(r['evaluation']['formal']['val']['prediction_only_huber_batch_mean']-r['best_val_loss'])>1e-8:raise ValueError('Restored selection objective mismatch')
    r['status']='COMPLETE — TRAIN/VAL ONLY'
    atomic_json(out/'diagnostics.json',r['evaluation']);atomic_json(out/'train_val_metrics.json',{s:v['metrics'] for s,v in r['evaluation']['formal'].items()})
    report(r,out)

def execute(args):
    shared.check_config();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    data,audit=shared.train_val_data(args.data_audit)
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory; do not overwrite any experiment')
    out.mkdir(parents=True)
    model,init=initialization(data,args.seed);model.to(device)
    x,y=next(iter(shared.data_loaders(data,args.seed,full=True)['val']))
    checks=sanity(model,x[:2].to(device),y[:2].to(device))
    ref=reference_audit(audit,args.seed,data,args.reference_results,args.reference_checkpoint)
    r=dict(status='DATA PREFLIGHT PASS',config={**PROTOCOL,'seed':args.seed},data=audit,
           source=source_record(),initialization=init,sanity=checks,reference=ref)
    for filename,value in [('config',r['config']),('data_audit',audit),('source_hashes',r['source']),
                           ('initialization_audit',init),('structural_sanity',checks),('reference_provenance',ref)]:
        atomic_json(out/(filename+'.json'),value)
    report(r,out)
    if not args.run:
        print(f'DATA PREFLIGHT PASS, no training: {out}');return
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    receipt=OUT_ROOT/f'seed{args.seed}_formal_run.json'
    with (OUT_ROOT/f'seed{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists():raise RuntimeError(f'Formal run already reserved: {receipt}; review, never auto retry')
        cp=ROOT/'checkpoints/four_source_utility_moe'/f'seed{args.seed}'/out.name/(VARIANT+'_best.pt')
        cp.parent.mkdir(parents=True,exist_ok=True)
        if cp.exists():raise FileExistsError(cp)
        atomic_json(receipt,dict(status='STARTED',output=str(out),checkpoint=str(cp)))
        r.update(status='TRAINING',checkpoint=str(cp));report(r,out)
        del model
        model=make_model(data,args.seed).to(device)  # Reset RNG after audits, not from fitted weights.
        loaders=shared.data_loaders(data,args.seed)
        def callback(history):
            atomic_json(out/'training_history.json',history)
            atomic_json(out/'routing_history.json',history['four_source_history'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        try:
            history=train(model,loaders['train'],loaders['val'],torch.empty((2,0),dtype=torch.long,device=device),
                          torch.empty(0,device=device),device,num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
                          checkpoint_path=str(cp),checkpoint_metadata=dict(variant=VARIANT,seed=args.seed,
                          config=r['config'],data=audit,source=r['source']),epoch_history_callback=callback)
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-start
            saved=torch.load(cp,map_location=device,weights_only=False)
            model.load_state_dict(saved['model_state_dict'],strict=True)
            model.switching_latent_transformer.set_epoch(saved['best_epoch'])
            saved.update(training_complete=True,training_seconds=seconds,history=history)
            temp=cp.with_suffix('.tmp');torch.save(saved,temp);temp.replace(cp)
            r.update(status='TRAINING COMPLETE; CHECKPOINT FROZEN',training_complete=True,training_seconds=seconds,
                     checkpoint_sha256=sha(cp),best_epoch=saved['best_epoch'],best_val_loss=saved['best_val_loss'])
            callback(history);report(r,out)
            atomic_json(out/'best_checkpoint_metadata.json',{k:r[k] for k in ('checkpoint','checkpoint_sha256','best_epoch','best_val_loss','training_seconds','training_complete')})
            atomic_json(receipt,dict(status='TRAINING COMPLETE',output=str(out),checkpoint=str(cp),sha256=sha(cp)))
            finish(model,data,r,out,device)
        except BaseException as exc:
            r.update(status='EVALUATION INTERRUPTED' if r.get('training_complete') else 'INTERRUPTED — REVIEW REQUIRED',error=f'{type(exc).__name__}: {exc}')
            report(r,out);raise


def evaluate_completed(args):
    out=args.evaluate_completed.resolve();r=json.loads((out/'results.json').read_text())
    if r['config']!={**PROTOCOL,'seed':r['config']['seed']} or not r.get('training_complete'):
        raise ValueError('Expected a completed run of this protocol')
    if r['checkpoint_sha256']!=sha(r['checkpoint']) or r['source']['hashes']!=source_record()['hashes']:
        raise ValueError('Checkpoint/source drift')
    data,audit=shared.train_val_data(args.data_audit)
    if audit!=r['data']:raise ValueError('Data drift')
    device=torch.device(args.device);model=make_model(data,r['config']['seed']).to(device)
    cp=torch.load(r['checkpoint'],map_location=device,weights_only=False)
    if cp['metadata']['config']!=r['config'] or not cp.get('training_complete'):
        raise ValueError('Checkpoint metadata mismatch')
    model.load_state_dict(cp['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(cp['best_epoch'])
    finish(model,data,r,out,device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--run',action='store_true')
    group.add_argument('--data-preflight',action='store_true')
    group.add_argument('--evaluate-completed',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path)
    p.add_argument('--data-audit',type=Path,default=shared.DEFAULT_AUDIT)
    p.add_argument('--reference-results',type=Path,default=REFERENCE)
    p.add_argument('--reference-checkpoint',type=Path,default=REFERENCE_CP)
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.output is None:
        mode='run' if args.run else 'data_preflight' if args.data_preflight else 'synthetic_preflight'
        args.output=OUT_ROOT/f'{mode}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.evaluate_completed:evaluate_completed(args)
    elif args.run or args.data_preflight:execute(args)
    else:preflight(args.seed,args.output)


if __name__=='__main__':main()
