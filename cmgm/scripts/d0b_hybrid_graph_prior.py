"""One controlled D0B HybridGraphPriorHeads run; no ratio/selection search."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import train
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0e_diagnostics import diagnostic_context,model_arguments
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results
from cmgm.scripts.d0b_5d_error_regime_diagnostic import ROOT,checkpoint_payload,sha256
from cmgm.scripts.d0b_hybrid_graph_diagnostics import (
    BASE,VARIANT,DISPLAY,initialization_check,fixed_sanity,attention_diagnostics,
    gradient_diagnostics,graph_diagnostics,
)

REFERENCE=dict(MAE=.0219947839,MSE=.000912875426,RMSE=.0302138284,Hit=.468401487)


def validate_protocol(args):
    expected=dict(seq_len=20,seed=42,batch_size=64,epochs=200,patience=10)
    for key,value in expected.items():
        if getattr(args,key)!=value:raise ValueError(f'Controlled Hybrid experiment requires {key}={value}')
    assert config.SEQ_LEN==20 and config.MULTI_HORIZONS==[1,5,10,20]
    assert config.TARGET_TYPE=='return' and config.LOSS_TYPE=='huber' and config.HUBER_DELTA==.02
    assert (config.LEARNING_RATE,config.WEIGHT_DECAY)==(1e-4,1e-5)


@torch.no_grad()
def collect_arrays(model,loaders,device):
    model.eval();out={};idx=config.MULTI_HORIZONS.index(5)
    for split,source in loaders.items():
        loader=DataLoader(source.dataset,batch_size=64,shuffle=False,drop_last=False)
        p=[];y=[]
        for batch in loader:
            p.append(model(batch[0].to(device))[:,idx].cpu().numpy());y.append(batch[1][:,idx].numpy())
        out[split]=dict(prediction=np.concatenate(p),target=np.concatenate(y))
    return out


def verify_reference(arrays):
    actual=population_metrics(arrays['test']['prediction'],arrays['test']['target'])
    checks={k:bool(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9)) for k,v in REFERENCE.items()}
    return dict(actual=actual,reference=REFERENCE,checks=checks,PASS=all(checks.values()))


def full_sanity(model,x):
    r=fixed_sanity(model,x)
    if r['PASS']:return r
    # Only roundoff in asset relabeling may be audited in higher precision.
    if max(r['causality'].values())>=1e-6 or r['batch_permutation']>=1e-6 or r['single_sample']>=1e-6:
        return r
    with diagnostic_context(model):
        copy=HeteroMixHopCMGM(variant=model.variant,**model_arguments(model)).to(device=x.device,dtype=torch.float64)
        copy.load_state_dict(model.state_dict());high=fixed_sanity(copy,x.double())
    r['float64_relabeling_audit']=high
    r['PASS']=high['PASS'] and max(high['within_market_relabeling'].values())<1e-10
    r['note']='Raw FP32 values retained; only within-market summation roundoff audited on a separate FP64 model. Batch/prefix thresholds never relaxed.'
    return r


def run_hybrid(args,device,data):
    from cmgm.scripts.d0b_hybrid_graph_report import build_comparison,write_report
    validate_protocol(args)
    if not args.no_cuda and device.type!='cuda':
        raise RuntimeError('GPU requested but CUDA is unavailable. No CPU training fallback.')
    out=Path(args.hybrid_report_dir)/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    path=Path(args.checkpoint_dir)/f'{VARIANT}_best.pt'
    if path.exists() and not getattr(args,'sanity_only',False):
        raise FileExistsError(f'{path} already exists; use a distinct checkpoint directory to preserve the completed run')
    set_seed(args.seed)
    cs,ce=data['market_indices']['commodity']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],variant=VARIANT,
        n_stock=data['market_indices']['stock'][1],n_bond=data['market_indices']['bond'][1]-data['market_indices']['bond'][0],
        feat_dim=config.FEATURE_DIM).to(device)
    r=dict(display=DISPLAY,variant=VARIANT,seq_len=20,seed=42,status='PREFLIGHT',primary_case=None,
        diagnostic_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_path=str(path.resolve()),fixed_TEST_shape=None,
        training_protocol='Unchanged Adam/lr/weight_decay/Huber+switch KL; scheduler/early stopping/checkpoint use original multi-horizon validation loss. VAL5 is secondary only.',
        no_other_variants_trained=True,training_executed=False)
    r['implementation_sha256']={name:sha256(ROOT/name) for name in (
        'cmgm/config.py','cmgm/models/hetero_mixhop_model.py','cmgm/graph/adaptive_graph.py',
        'cmgm/training/train.py','cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py',
        'cmgm/data/feature_builder.py','cmgm/scripts/d0b_hybrid_graph_prior.py',
        'cmgm/scripts/d0b_hybrid_graph_diagnostics.py','cmgm/scripts/d0b_hybrid_graph_report.py')}
    with diagnostic_context(model):
        train_fixed=next(iter(data['loaders']['train']));train_fixed=tuple(t[:4].to(device) for t in train_fixed[:2])
        test_x=next(iter(data['loaders']['test']))[0][:4].to(device)
        r['fixed_TEST_shape']=list(test_x.shape)
        shared=initialization_check(model,args.seed)
        if shared['D0B_params']!=520549 or shared['Hybrid_params']!=520549:
            raise AssertionError('STOP: formal 20-step D0B parameter count must be 520549 for both models')
        r['sanity']=dict(shared_init=shared,initial=full_sanity(model,test_x))
        r['attention']={'initial':attention_diagnostics(model,test_x)}
        r['graph']={'initial':graph_diagnostics(model)}
        r['gradients']={'initial':gradient_diagnostics(model,train_fixed)}
        baseline=HeteroMixHopCMGM(variant=BASE,**model_arguments(model)).to(device).eval()
        basepath=Path(args.d0b_checkpoint)
        payload=checkpoint_payload(basepath)
        state=payload.get('model_state_dict',payload.get('state_dict',payload))
        baseline.load_state_dict(state,strict=True)
        baseline.switching_latent_transformer.set_epoch(payload.get('best_epoch') or 1)
        metadata=payload.get('metadata',{})
        recorded=metadata.get('variant',payload.get('variant'))
        if recorded is not None and recorded!=BASE:raise ValueError('D0B checkpoint variant mismatch')
        r['baseline_checkpoint']=dict(path=str(basepath.resolve()),sha256=sha256(basepath),best_epoch=payload.get('best_epoch'),
            seed=metadata.get('seed'),training_SHA=metadata.get('git_sha'))
        base_arrays=collect_arrays(baseline,data['loaders'],device)
        r['baseline_reference']=verify_reference(base_arrays)
        r['attention']['D0B_best']=attention_diagnostics(baseline,test_x)
        r['graph']['D0B_best']=graph_diagnostics(baseline)
        r['gradients']['D0B_best']=gradient_diagnostics(baseline,train_fixed)
        r['sanity']['D0B_legacy_all_heads']=r['attention']['D0B_best']['PASS']
        r['sanity']['initial_prior_partition']=r['attention']['initial']['PASS']
        checks=[shared['PASS'],r['sanity']['initial']['PASS'],r['baseline_reference']['PASS'],r['sanity']['D0B_legacy_all_heads'],r['sanity']['initial_prior_partition']]
        r['sanity']['PASS']=all(checks)
        del baseline
    save_results(r,out/'results.json');save_results(r['sanity'],out/'sanity_checks.json')
    if not r['sanity']['PASS']:
        r['status']='STOPPED: core sanity/reference failed; DO NOT INTERPRET PERFORMANCE'
        save_results(r,out/'results.json');write_report(r,out)
        raise AssertionError(r['status'])
    if getattr(args,'sanity_only',False):
        r['status']='SANITY ONLY — no formal training and no performance classification'
        save_results(r,out/'results.json');write_report(r,out)
        return r
    path.parent.mkdir(parents=True,exist_ok=True)
    def diagnostic(active,stage):
        # Fixed TEST inputs describe attention only; no TEST loss/metrics enter training.
        result=dict(attention=attention_diagnostics(active,test_x),graph=graph_diagnostics(active),
                    gradients=gradient_diagnostics(active,train_fixed))
        key='best' if stage.startswith('best(') else stage
        r['attention'][key]=result['attention'];r['graph'][key]=result['graph'];r['gradients'][key]=result['gradients']
        return result
    start=time.perf_counter()
    history=train(model,data['loaders']['train'],data['loaders']['val'],
        torch.empty((2,0),dtype=torch.long,device=device),torch.empty(0,device=device),device,
        num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),
        checkpoint_metadata=dict(variant=VARIANT,display_name=DISPLAY,seed=args.seed,seq_len=20,
            parameter_count=520549,git_sha=r['diagnostic_git_sha'],graph_prior_heads=4,
            formal_selection='original multi-horizon validation objective'),epoch_diagnostic=diagnostic)
    elapsed=time.perf_counter()-start
    # Explicitly reload the formal checkpoint selected by multi-horizon validation.
    formal=checkpoint_payload(path);model.load_state_dict(formal['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(formal['best_epoch'])
    r.update(training_executed=True,history=history,train_time_seconds=elapsed,best_epoch=formal['best_epoch'],
        best_formal_val_objective=formal['best_val_loss'],checkpoint_sha256=sha256(path))
    r['sanity']['best']=full_sanity(model,test_x)
    r['sanity']['best_prior_partition']=attention_diagnostics(model,test_x)['PASS']
    r['sanity']['PASS']=r['sanity']['PASS'] and r['sanity']['best']['PASS'] and r['sanity']['best_prior_partition']
    arrays=collect_arrays(model,data['loaders'],device)
    assert sha256(Path(args.d0b_checkpoint))==r['baseline_checkpoint']['sha256']
    names=data['feature_names'][cs:ce]
    r.update(build_comparison(base_arrays,arrays,names,r['attention']['best'],r['sanity']['PASS'],out))
    r['status']='COMPLETE' if r['sanity']['PASS'] else 'STOPPED: core sanity failed; DO NOT INTERPRET PERFORMANCE'
    for filename,key in [('training_history','history'),('attention_head_diagnostics','attention'),('gradient_diagnostics','gradients'),('sanity_checks','sanity')]:
        save_results(r[key],out/(filename+'.json'))
    save_results(r,out/'results.json');write_report(r,out)
    print(f'[{DISPLAY}] report={out.resolve()}; STOP',flush=True)
    metrics=r['metrics']['Hybrid']['test'];zero=population_metrics(np.zeros_like(arrays['test']['target']),arrays['test']['target'])
    legacy={**metrics,'Hit_Ratio':metrics['Hit']}
    return dict(variant=DISPLAY,params=520549,time=elapsed,MAE=metrics['MAE'],MSE=metrics['MSE'],RMSE=metrics['RMSE'],
        Hit_Ratio=metrics['Hit'],vs_zero_pct=100*(zero['MAE']-metrics['MAE'])/zero['MAE'],mn=legacy,mo=legacy,report=str(out))


def main():
    from cmgm.scripts.main_ablation import build_data
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints'))
    parser.add_argument('--d0b-checkpoint',type=Path,default=Path('checkpoints')/f'{BASE}_best.pt')
    parser.add_argument('--hybrid-report-dir',type=Path,default=Path('experiments')/'d0b_hybrid_graph_prior')
    parser.add_argument('--sanity-only',action='store_true')
    parser.add_argument('--no-cuda',action='store_true',help='Explicit CPU override; GPU is otherwise required')
    parser.set_defaults(seq_len=20,batch_size=64,seed=42,epochs=200,patience=10)
    args=parser.parse_args();validate_protocol(args)
    if not args.no_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; GPU training will not fall back to CPU')
    device=torch.device('cpu' if args.no_cuda else 'cuda')
    run_hybrid(args,device,build_data(args))


if __name__=='__main__':main()
