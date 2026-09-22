"""One fixed utility-supervised prediction-level MoE experiment. Default: verify only; --run: one GPU fit."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import torch
from cmgm import config
from cmgm.models.utility_routed_moe import VARIANT
from cmgm.scripts.d0b_utility_moe_audit import initialization,sanity,make_model
from cmgm.scripts.d0b_utility_moe_diagnostics import formal_evaluation
from cmgm.scripts.baseline_protocol import seed_all,loaders
from cmgm.scripts.formal_main_innovation_ablation import audit_data
from cmgm.scripts.formal_v2_protocol import atomic_json,sha
from cmgm.training.train import train
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=dict(variant=VARIANT,seed=42,seq_len=20,features=21,horizons=[1,5,10,20],
    optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,max_epochs=200,patience=10,
    scheduler='ReduceLROnPlateau factor=.5 patience5',train_shuffle=False,train_drop_last=True,
    selection='prediction-only multi-horizon validation Huber batch mean',prediction_loss='sum four Huber delta=.02',
    switch_kl='native beta_max=.0005 warmup20 unchanged',
    utility_loss='TRAIN only: stopgrad(mean((loss_T+loss_ST)/2))*mean KL(detached soft utility q || pi); no coefficient',
    utility_error='per-sample Huber(.02), mean24 commodities, sum all4 horizons',epsilon=1e-8,
    utility_gradient='Re-evaluate deterministic router on detached candidates and detached context; only Router parameters receive auxiliary gradients',
    routing_warmup=None,load_balancing=None,expert_order=['Temporal','Interaction'],expert_dropout=.1,
    router='LayerNorm64 each candidate; [u_T,u_ST,absdiff,product,detached p,gap,entropy]; Linear26364/ReLU/Linear642/softmax',
    router_final_initialization='weight=0 bias=0',heads='Temporal=self.head; Interaction=deepcopy(self.head), independent',
    fusion='prediction-level dense sample mixture B,4,24',
    weight_training='All active shared/expert/head/router parameters train normally under primary loss',single_run=True,
    fixed_control='Frozen best checkpoint full TRAIN-mean pi only; no fitting/tuning',
    oracle='EX-POST TEST-label selector using four-horizon per-sample Huber; not a deployable model',
    rank_thirds='Stable pi_ST sort, ties by origin order; no artificial routing-variance success cutoff')


def save(r,out):
    from cmgm.scripts.d0b_utility_moe_report import report
    atomic_json(out/'results.json',r);report(r,out)


def controls(source,three_source,candidate_source,data):
    from cmgm.scripts.d0b_candidate_2expert_moe import controls as previous_controls,PROTOCOL as CANDIDATE_PROTOCOL
    result=previous_controls(source,three_source,data)
    del result['Fixed Equal Fusion']
    old=json.loads(candidate_source.read_text());audit=audit_data(data)
    if old['status']!='COMPLETE' or not old['trained'] or old['data']!=audit or old['config']!=CANDIDATE_PROTOCOL:
        raise ValueError('Candidate reference incomplete or protocol/data mismatch')
    if not old['sanity']['PASS'] or not old['best_sanity']['PASS'] or sha(old['checkpoint'])!=old['checkpoint_sha256']:
        raise ValueError('Invalid candidate reference')
    cp=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)
    if cp['metadata']['source_hashes']!=old['source_hashes'] or cp['metadata']['config']!=CANDIDATE_PROTOCOL or cp['metadata']['data']!=audit:
        raise ValueError('Candidate checkpoint provenance mismatch')
    result['Candidate-Aware 2-Expert Representation MoE']=dict(metrics=old['evaluation']['metrics'],checkpoint=old['checkpoint'],
        checkpoint_sha256=old['checkpoint_sha256'],source_report=str(candidate_source),source_report_sha256=sha(candidate_source),
        best_epoch=cp['best_epoch'],checkpoint_metadata=cp['metadata'],routing=old['evaluation']['routing'],
        utility=old['evaluation']['utility'],role='Completed audited candidate control; read-only, never retrained')
    return result


def prepare(args,data,device):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS,config.HUBER_DELTA,config.LOSS_TYPE)!=(20,21,[1,5,10,20],.02,'huber'):raise ValueError('Frozen configuration mismatch')
    files=['cmgm/models/utility_routed_moe.py','cmgm/models/hetero_mixhop_model.py','cmgm/models/switching_latent_transformer.py',
           'cmgm/training/train.py','cmgm/scripts/d0b_utility_routed_moe.py','cmgm/scripts/d0b_utility_moe_audit.py',
           'cmgm/scripts/d0b_utility_moe_report.py','cmgm/scripts/d0b_utility_moe_diagnostics.py',
           'cmgm/scripts/d0b_candidate_moe_diagnostics.py','cmgm/scripts/formal_v2_protocol.py',
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
               controls=controls(args.reference_results,args.three_reference_results,args.candidate_reference_results,data),status='PREPARING',checkpoint=str(args.checkpoint),initialization={},sanity={},trained=False)
        save(r,out)
    # Always check the device used for this launch before allowing a formal fit.
    if not r['sanity'].get('PASS') or r['sanity'].get('device')!=str(device):
        # Utility-target/gradient audits must use TRAIN labels, even without a step.
        x,y=next(iter(loaders(data,full=True)['train']))[:2]
        m,a=initialization(data,device);r['initialization']=a
        if not a['PASS']:save(r,out);raise AssertionError('Shared initialization failed')
        c=sanity(m,x[:2].to(device),y[:2].to(device));c['device']=str(device);c['split']='TRAIN';r['sanity']=c
        save(r,out)
        if not c['PASS']:raise AssertionError('MoE structural sanity failed')
        del m
    for c in r['controls'].values():
        if sha(c['checkpoint'])!=c['checkpoint_sha256'] or sha(c['source_report'])!=c['source_report_sha256']:raise ValueError('Reference artifact changed')
    r['status']='COMPLETE' if r.get('evaluation') else 'PREPARED: one utility MoE GPU run pending';save(r,out)
    return r,out


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
            rows=history.get('utility_routing_history',[])
            atomic_json(out/'routing_history.json',rows)
            atomic_json(out/'utility_training_history.json',[dict(epoch=row['epoch'],**row['train']) for row in rows])
        history=train(m,sources['train'],sources['val'],torch.empty((2,0),dtype=torch.long,device=device),torch.empty(0,device=device),device,
            num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,checkpoint_path=str(path),checkpoint_metadata=md,
            epoch_history_callback=history_callback)
        cp=torch.load(path,map_location='cpu',weights_only=False)
    m.load_state_dict(cp['model_state_dict'],strict=True)
    m.switching_latent_transformer.set_epoch(cp['best_epoch'])
    r['trained']=True;r['history']=history;r['checkpoint_sha256']=sha(path)
    r['best_checkpoint_metadata']=dict(best_epoch=cp['best_epoch'],best_val_loss=cp['best_val_loss'],
        parameter_count=sum(p.numel() for p in m.parameters()),metadata=md)
    r['status']='EVALUATING';save(r,out)
    x,y=next(iter(loaders(data,full=True)['train']))[:2]
    # autograd.grad sanity needs fresh .grad=None after training; no weights change.
    m.zero_grad(set_to_none=True)
    r['best_sanity']=sanity(m,x[:2].to(device),y[:2].to(device),initial=False)
    r['best_sanity']['split']='TRAIN'
    if not r['best_sanity']['PASS']:save(r,out);raise AssertionError('Best-checkpoint sanity failed; do not interpret performance')
    frozen={k:v.detach().clone() for k,v in m.state_dict().items()}
    evaluation=formal_evaluation(m,data,device,out)
    if sha(path)!=r['checkpoint_sha256'] or any(not torch.equal(v,m.state_dict()[k]) for k,v in frozen.items()):
        r['status']='INVALID: frozen checkpoint changed during evaluation';save(r,out)
        raise AssertionError('Post-fit diagnostic must not modify model')
    r['evaluation']=evaluation;r['status']='COMPLETE';save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/d0b_utility_routed_moe')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints'/f'{VARIANT}_best.pt')
    p.add_argument('--reference-results',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation/20260916_154242/results.json')
    p.add_argument('--three-reference-results',type=Path,default=ROOT/'experiments/d0b_moe_fusion/20260921_141206/results.json')
    p.add_argument('--candidate-reference-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_moe/20260921_151711/results.json')
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
            if not r['status'].startswith('INVALID'):
                r['status']=('INVALID IMPLEMENTATION FAILURE: ' if isinstance(exc,FloatingPointError) else 'STOP ')+type(exc).__name__
            r.setdefault('errors',[]).append(str(exc));save(r,out);raise
        finally:
            for c in r['controls'].values():
                if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise AssertionError('Control checkpoint modified')
        print('Utility MoE report:',out,'STOP',flush=True)


if __name__=='__main__':main()
