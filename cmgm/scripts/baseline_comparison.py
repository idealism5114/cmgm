"""Fixed six-baseline suite plus read-only D0B; never train or alter D0B."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from cmgm import config
from cmgm.models.comparison_baselines import ORDER,TRAINABLE,make_model
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.baseline_protocol import seed_all,loaders,parameter_counts,sanity,evaluate,train_one,data_audit
from cmgm.scripts.d0b_5d_error_regime_diagnostic import ROOT,checkpoint_payload,sha256
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results

BASE='switching_latent_balanced_readout'
KEYS=dict(Linear='linear',GRU='gru',VanillaTransformer='transformer',iTransformer='itransformer',MTGNN='mtgnn')
REFERENCE=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)
PROTOCOL=dict(seed=42,seq_len=20,batch_size=64,epochs=200,patience=10,lr=1e-4,weight_decay=1e-5,
    loss='sum four Huber(delta=.02), no baseline auxiliary loss',optimizer='Adam, one parameter group',
    scheduler='ReduceLROnPlateau(mode=min,factor=.5,patience=5)',selection='mean of batch-mean multi-horizon VAL Huber, matching current D0B',
    train_shuffle=False,train_drop_last=True,full_evaluation_drop_last=False,adapter_std='population, correction=0',
    flatten_order='token-major, then feature-major: channel=21*token+feature',determinism='Python/NumPy/PyTorch/CUDA/loader seeds42, cudnn deterministic=True, benchmark=False')


def flush(r,out):
    from cmgm.scripts.baseline_report import write_report
    write_report(r,out)
    save_results(r,out/'results.json');save_results(dict(data=r.get('data_audit'),models=r['sanity']),out/'sanity_checks.json')


def completed_training(path,name,audit):
    """Recover a finished training after an evaluation/report interruption, never retrain it."""
    if path is None or not path.exists():return None
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if not cp.get('training_complete'):return None
    metadata=cp['metadata']
    if metadata['model']!=name or metadata['protocol']!=PROTOCOL or metadata['data_fingerprint']!=audit['split_fingerprint']:
        raise ValueError('Completed training metadata differs; STOP')
    return cp


def run_suite(args,device,data):
    import json
    if (config.SEQ_LEN,config.FEATURE_DIM,tuple(config.MULTI_HORIZONS),config.HUBER_DELTA,config.LEARNING_RATE,config.WEIGHT_DECAY)!=(20,21,(1,5,10,20),.02,1e-4,1e-5):raise ValueError('Current config differs from fixed formal protocol; STOP')
    if config.TARGET_TYPE!='return' or config.LOSS_TYPE!='huber':raise ValueError('Expected original return/Huber pipeline')
    if not args.sanity_only and device.type!='cuda':raise RuntimeError('Formal baseline suite requires GPU; no silent CPU fallback')
    outroot=Path(args.output_dir);cpdir=Path(args.checkpoint_dir);basepath=Path(args.d0b_checkpoint)
    if args.resume:
        if args.resume=='latest':
            candidates=sorted(p.parent for p in outroot.glob('*/results.json') if not json.loads(p.read_text()).get('sanity_only'))
            if not candidates:raise FileNotFoundError('No suite to resume')
            out=candidates[-1]
        else:out=Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL:raise ValueError('Saved protocol differs; cannot resume')
        if r.get('sanity_only'):raise ValueError('Sanity-only reports are not training runs; start a fresh formal suite')
    else:
        if not args.sanity_only and any((cpdir/f'{key}_best.pt').exists() for key in KEYS.values()):raise FileExistsError('Existing baseline checkpoint(s); use explicit resume of the original suite, not a repeat run')
        out=outroot/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        r=dict(status='PREFLIGHT',sanity_only=args.sanity_only,protocol=PROTOCOL,models={},sanity={},model_status={},
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            training_executed=False,baseline_checkpoint=str(basepath.resolve()),checkpoint_dir=str(cpdir.resolve()))
    if r['checkpoint_dir']!=str(cpdir.resolve()):raise ValueError('Resume checkpoint directory differs')
    before=sha256(basepath)
    if r.get('baseline_checksum') and r['baseline_checksum']!=before:raise ValueError('D0B checkpoint changed; STOP')
    audit=data_audit(data)
    if r.get('data_audit') and r['data_audit']!=audit:raise ValueError('Data/order differs from saved suite; STOP')
    r['data_audit']=audit;r['baseline_checksum']=before
    if not audit['PASS']:r['status']='INVALID DATA';flush(r,out);raise ValueError('Target/order sanity failed')
    source_names=['cmgm/config.py','cmgm/models/comparison_baselines.py','cmgm/scripts/baseline_protocol.py','cmgm/scripts/baseline_comparison.py','cmgm/scripts/baseline_report.py',
        'cmgm/models/hetero_mixhop_model.py','cmgm/graph/adaptive_graph.py','cmgm/models/model.py','cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py','cmgm/training/metric_standard.py']
    source_hashes={name:sha256(ROOT/name) for name in source_names};r['current_source_hashes']=source_hashes
    full=loaders(data,full=True);names=data['feature_names'][data['market_indices']['commodity'][0]:data['market_indices']['commodity'][1]]
    # Use VAL only for sanity; each trained baseline's formal TEST evaluation occurs once after selection.
    fixed=next(iter(full['val']))[0][:4].to(device);r['fixed_sanity_input_shape']=list(fixed.shape)
    seed_all()
    if 'D0B' not in r['models']:
        b=HeteroMixHopCMGM(data['n_nodes'],24,n_stock=data['market_indices']['stock'][1],n_bond=data['market_indices']['bond'][1]-data['market_indices']['bond'][0],variant=BASE).to(device).eval()
        counts=parameter_counts(b);payload=checkpoint_payload(basepath)
        b.load_state_dict(payload.get('model_state_dict',payload.get('state_dict',payload)),strict=True)
        b.requires_grad_(False)
        reference=evaluate(b,full,device,names);actual=reference['metrics']['test']['5']
        ok=all(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9) for k,v in REFERENCE.items())
        r['reference_check']=dict(actual=actual,expected=REFERENCE,PASS=bool(ok))
        if not ok:r['status']='STOP: D0B reference mismatch';flush(r,out);raise ValueError(r['status'])
        r['models']['D0B']=dict(**reference,parameters=counts,training=dict(best_epoch=payload.get('best_epoch'),train_seconds=None,seconds_per_epoch_mean=None),
            input_view='native full 284-node D0B (Ours)',frozen_in_suite=True,effective_trainable_parameters_in_suite=0,
            metadata=payload.get('metadata',{}),source_hashes=source_hashes)
        del b;flush(r,out)
    for name in ORDER:
        if r['model_status'].get(name)=='COMPLETE':
            entry=r['models'][name]
            if name in KEYS and sha256(cpdir/f'{KEYS[name]}_best.pt')!=entry['checkpoint_sha256']:raise ValueError('Completed checkpoint changed; no reevaluation or retraining')
            continue
        previous=r['model_status'].get(name)
        path=cpdir/f'{KEYS[name]}_best.pt' if name in KEYS else None
        recovered=completed_training(path,name,audit) if previous else None
        if previous and recovered is None and not (previous=='INVALID' and args.retry_invalid):raise ValueError(f'{name} status={previous}; no automatic rerun')
        if recovered is not None and previous=='INVALID' and not args.retry_invalid:raise ValueError('Invalid evaluation requires --retry-invalid after fixing its cause; finished training will be reused')
        if path and path.exists() and recovered is None:
            if previous!='INVALID' or not args.retry_invalid:raise FileExistsError(path)
            path.rename(path.with_name(path.stem+'_invalid_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pt'))
        seed_all();m=make_model(name,data['market_indices']).to(device)
        counts=parameter_counts(m);check=sanity(m,fixed);r['sanity'][name]=dict(initial=check,PASS=check['PASS'])
        entry=dict(parameters=counts,input_view=('neutral 28-node ×21-feature graph' if name=='MTGNN' else 'neutral 28×21 token-major sequence (588 channels)'),source_hashes=source_hashes if recovered is None else recovered['metadata']['source_hashes'])
        if name=='ZeroReturn':entry['input_view']='zero prediction, shape only'
        r['models'][name]=entry
        if not check['PASS']:
            r['model_status'][name]='INVALID';r['status']=f'STOP: {name} sanity failed';flush(r,out);raise AssertionError(r['status'])
        if args.sanity_only:
            if name=='ZeroReturn':entry.update(evaluate(m,full,device,names),training=dict(best_epoch=None,train_seconds=0.,seconds_per_epoch_mean=0.))
            r['model_status'][name]='SANITY_ONLY';flush(r,out);continue
        try:
            if name=='ZeroReturn':entry['training']=dict(best_epoch=None,train_seconds=0.,seconds_per_epoch_mean=0.)
            else:
                if recovered is not None:
                    m.load_state_dict(recovered['model_state_dict'],strict=True)
                    summary=recovered['training_summary']
                    print(f'[{name}] Reusing completed training; no optimizer is created.',flush=True)
                else:
                    cpdir.mkdir(parents=True,exist_ok=True);r['model_status'][name]='RUNNING';r['status']=f'TRAINING {name}';r['training_executed']=True;flush(r,out)
                    # Each model gets a fresh identical seeded chronological loader.
                    train_loaders=loaders(data)
                    metadata=dict(model=name,protocol=PROTOCOL,git_sha=r['git_sha'],source_hashes=source_hashes,data_fingerprint=audit['split_fingerprint'],parameters=counts)
                    summary,history=train_one(m,train_loaders['train'],train_loaders['val'],device,path,metadata,
                        on_epoch=lambda history,key=KEYS[name]:save_results(history,out/f'history_{key}.json'))
                entry['training']=summary;entry['checkpoint_sha256']=sha256(path)
                r['model_status'][name]='TRAINED';flush(r,out)
                best_check=sanity(m,fixed);r['sanity'][name]['best']=best_check;r['sanity'][name]['PASS']=best_check['PASS']
                if not best_check['PASS']:raise AssertionError('Formal best checkpoint sanity failure')
            # Sole formal TEST call for this baseline, only after formal selection and sanity.
            entry.update(evaluate(m,full,device,names));r['model_status'][name]='COMPLETE';flush(r,out)
        except Exception as error:
            r['model_status'][name]='INVALID';r['status']=f'STOP: {name} invalid run';r.setdefault('errors',[]).append(dict(model=name,type=type(error).__name__,message=str(error)))
            flush(r,out)
            if isinstance(error,torch.cuda.OutOfMemoryError):print('OOM: batch size remains 64. STOP for implementation/memory review; no automatic smaller batch.',flush=True)
            raise
        finally:
            del m
            if device.type=='cuda':torch.cuda.empty_cache()
    assert sha256(basepath)==before
    r['D0B_checkpoint_unchanged']=True;r['status']='SANITY ONLY — five trainings left to user' if args.sanity_only else 'COMPLETE — single-run controlled comparison'
    flush(r,out)
    if not args.sanity_only:
        from cmgm.scripts.d0b_risk_probe_analysis import table
        print('============================================================\nBASELINE COMPARISON — STANDARDIZED TEST 5D\n============================================================')
        print(table(['Model','Params','MAE','MSE','RMSE','Hit%','ΔMAE vs D0B','Relative ΔMAE %','Train Seconds'],
            [[row[k] for k in ('Model','Params','TEST5_MAE','TEST5_MSE','TEST5_RMSE','TEST5_Hit','DeltaMAE_vs_D0B','RelativeDeltaMAE_percent','TrainSeconds')] for row in r['tables']['overall']]))
        print('============================================================\nMULTI-HORIZON MAE\n============================================================')
        print(table(['Model','1d','5d','10d','20d'],[[name,*[entry['metrics']['test'][str(h)]['MAE'] for h in config.MULTI_HORIZONS]] for name,entry in r['models'].items()]))
    print(f'Baseline comparison report: {out.resolve()}; STOP',flush=True)
    return r


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from cmgm.scripts.main_ablation import build_data
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('experiments/baseline_comparison'))
    p.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints/baselines'))
    p.add_argument('--d0b-checkpoint',type=Path,default=Path('checkpoints')/f'{BASE}_best.pt')
    p.add_argument('--sanity-only',action='store_true');p.add_argument('--no-cuda',action='store_true')
    p.add_argument('--resume',help='Existing suite directory or latest; never repeats a completed model')
    p.add_argument('--retry-invalid',action='store_true',help='Explicit retry of a recorded INVALID code/numerical run only, never a completed poor result')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if args.no_cuda and not args.sanity_only:raise ValueError('--no-cuda is for sanity-only; formal run requires GPU')
    if not args.no_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    seed_all();data=build_data(args);run_suite(args,torch.device('cpu' if args.no_cuda else 'cuda'),data)


if __name__=='__main__':main()
