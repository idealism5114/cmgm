"""One formal QKNorm run with unchanged D0B training/selection protocol."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import train
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0e_diagnostics import diagnostic_context,model_arguments
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results
from cmgm.scripts.d0b_5d_error_regime_diagnostic import ROOT,checkpoint_payload,sha256
from cmgm.scripts.d0b_hybrid_graph_prior import collect_arrays,full_sanity,validate_protocol as _validate_protocol
from cmgm.scripts.d0b_qknorm_diagnostics import (
    VARIANT,BASE,DISPLAY,initialization_check,geometry_diagnostics,gradient_diagnostics,graph_diagnostics,
)

REFERENCE=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)


def validate_protocol(args):
    try:
        _validate_protocol(args)
    except ValueError as error:
        raise ValueError(str(error).replace('Hybrid', 'QKNorm')) from error


def verify_reference(arrays):
    actual=population_metrics(arrays['test']['prediction'],arrays['test']['target'])
    checks={k:bool(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9)) for k,v in REFERENCE.items()}
    return dict(actual=actual,reference=REFERENCE,checks=checks,PASS=all(checks.values()),
        RMSE_squared_minus_MSE_abs=abs(actual['RMSE']**2-actual['MSE']))


def assert_backbone(model):
    b=model.switching_latent_transformer
    assert model.variant==VARIANT
    assert model.use_gcn and model.use_edge_attn and model.use_gate and model.use_lstm and not model.use_mixhop
    assert b.balanced_readout and not b.use_latent_memory and not b.use_dynamic_slope and not b.use_balanced_transition_input
    assert not b.use_regime_relative_memory
    assert not b.regime_filter.learnable_sticky_alpha and b.regime_filter.sticky_alpha==.5
    for layer in (model.attn_mixhop1,model.attn_mixhop2):
        assert layer.qk_norm and layer.qk_norm_eps==1e-6 and layer.graph_prior_heads is None
        assert layer.n_heads==8 and not layer.hard_mask and layer.cross_mask is None
        assert layer.K==2 and layer.beta==.05 and layer.prior_scale==.5


def export(r,out):
    for filename,key in [('training_history','history'),('qk_norm_diagnostics','qk_norm'),
        ('attention_diagnostics','attention'),('gradient_diagnostics','gradients'),('graph_diagnostics','graph'),('sanity_checks','sanity')]:
        save_results(r.get(key,{}),out/(filename+'.json'))
    save_results(r,out/'results.json')


def run_qknorm(args,device,data):
    from cmgm.scripts.d0b_qknorm_report import build_comparison,write_report
    validate_protocol(args)
    if not args.no_cuda and device.type!='cuda':raise RuntimeError('QKNorm requires GPU unless CPU is explicitly requested; no silent fallback')
    path=Path(args.checkpoint_dir)/f'{VARIANT}_best.pt'
    if path.exists() and not getattr(args,'sanity_only',False):raise FileExistsError(f'{path} already exists; no automatic rerun or overwrite')
    out=Path(args.qknorm_report_dir)/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    set_seed(args.seed)
    cs,ce=data['market_indices']['commodity']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],variant=VARIANT,
        n_stock=data['market_indices']['stock'][1],n_bond=data['market_indices']['bond'][1]-data['market_indices']['bond'][0],feat_dim=config.FEATURE_DIM).to(device)
    assert_backbone(model)
    r=dict(display=DISPLAY,variant=VARIANT,seq_len=20,seed=42,status='PREFLIGHT',training_executed=False,primary_case=None,
        checkpoint_path=str(path.resolve()),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qk_norm={},attention={},gradients={},graph={},sanity={'geometry':{}},
        protocol='Only Q/K normalization changed; all 8 graph priors, V, D0B temporal/fusion/head and training protocol unchanged. Formal selection uses original multi-horizon validation objective; pooled VAL5 is secondary only.')
    files=['cmgm/config.py','cmgm/models/hetero_mixhop_model.py','cmgm/graph/adaptive_graph.py','cmgm/training/train.py',
        'cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py',
        'cmgm/scripts/d0b_qknorm_graph_attention.py','cmgm/scripts/d0b_qknorm_diagnostics.py','cmgm/scripts/d0b_qknorm_report.py',
        'cmgm/scripts/d0b_hybrid_graph_prior.py','cmgm/scripts/d0b_hybrid_graph_diagnostics.py','cmgm/scripts/d0e_diagnostics.py',
        'cmgm/training/metric_standard.py']
    r['implementation_sha256']={name:sha256(ROOT/name) for name in files}
    def capture(active,stage):
        # These four TRAIN samples match the preceding Hybrid gradient definition.
        pack=geometry_diagnostics(active,test_x)
        r['qk_norm'][stage]=pack['qk_norm'];r['attention'][stage]=pack['attention']
        r['graph'][stage]=graph_diagnostics(active);r['gradients'][stage]=gradient_diagnostics(active,train_fixed)
        r['sanity']['geometry'][stage]=pack['PASS']
        return pack['PASS']
    with diagnostic_context(model):
        batch=next(iter(data['loaders']['train']));train_fixed=tuple(t[:4].to(device) for t in batch[:2])
        test_x=next(iter(data['loaders']['test']))[0][:4].to(device)
        r['fixed_TRAIN_shape']=list(train_fixed[0].shape);r['fixed_TEST_shape']=list(test_x.shape)
        r['split_origins']={s:len(l.dataset) for s,l in data['loaders'].items()}
        r['sanity']['shared_init']=initialization_check(model,args.seed)
        assert r['sanity']['shared_init']['D0B_params']==r['sanity']['shared_init']['QKNorm_params']==520549
        print(f'[{DISPLAY} shared init] {r["sanity"]["shared_init"]}',flush=True)
        r['sanity']['initial']=full_sanity(model,test_x)
        capture(model,'initial')
        baseline=HeteroMixHopCMGM(variant=BASE,**model_arguments(model)).to(device).eval()
        cp=Path(args.d0b_checkpoint);payload=checkpoint_payload(cp);metadata=payload.get('metadata',{})
        recorded=metadata.get('variant',payload.get('variant'))
        if recorded is not None and recorded!=BASE:raise ValueError('Formal D0B checkpoint variant mismatch')
        baseline.load_state_dict(payload.get('model_state_dict',payload.get('state_dict',payload)),strict=True)
        baseline.switching_latent_transformer.set_epoch(payload.get('best_epoch') or 1)
        r['baseline_checkpoint']=dict(path=str(cp.resolve()),sha256=sha256(cp),best_epoch=payload.get('best_epoch'),
            seed=metadata.get('seed'),training_SHA=metadata.get('git_sha'))
        base_arrays=collect_arrays(baseline,data['loaders'],device)
        r['baseline_reference']=verify_reference(base_arrays)
        r['baseline_metrics']={s:population_metrics(a['prediction'],a['target']) for s,a in base_arrays.items()}
        print(f'[{DISPLAY} native D0B pooled reference] {r["baseline_reference"]}',flush=True)
        capture(baseline,'D0B_best')
        r['sanity']['legacy_qk_norm_false']=not baseline.attn_mixhop1.qk_norm and not baseline.attn_mixhop2.qk_norm and r['baseline_reference']['PASS']
        del baseline
    r['sanity']['PASS']=r['sanity']['shared_init']['PASS'] and r['sanity']['initial']['PASS'] and r['sanity']['legacy_qk_norm_false'] and all(r['sanity']['geometry'].values())
    if not r['sanity']['PASS']:
        r['status']='STOPPED: core sanity/reference failed; DO NOT INTERPRET PERFORMANCE'
        export(r,out);write_report(r,out);raise AssertionError(r['status'])
    if getattr(args,'sanity_only',False):
        r['status']='SANITY ONLY — formal training not run; no performance case assigned'
        export(r,out);write_report(r,out);print(f'[{DISPLAY}] sanity PASS: {out.resolve()}',flush=True);return r
    export(r,out)
    path.parent.mkdir(parents=True,exist_ok=True)
    def diagnostic(active,stage):
        key='best' if stage.startswith('best(') else stage
        passed=capture(active,key);export(r,out)
        if not passed:
            r['status']='STOPPED: Q/K or prior sanity failed during training; no performance interpretation'
            export(r,out);write_report(r,out);raise AssertionError(r['status'])
        return dict(geometry_PASS=passed,gradients=r['gradients'][key])
    r['training_executed']=True;r['status']='TRAINING'
    start=time.perf_counter()
    history=train(model,data['loaders']['train'],data['loaders']['val'],torch.empty((2,0),dtype=torch.long,device=device),
        torch.empty(0,device=device),device,num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),
        checkpoint_metadata=dict(variant=VARIANT,display_name=DISPLAY,seed=args.seed,seq_len=20,git_sha=r['git_sha'],
            parameter_count=520549,qk_norm=True,qk_norm_eps=1e-6,graph_prior_heads='all 8',
            formal_selection='original multi-horizon validation objective'),epoch_diagnostic=diagnostic)
    r['train_time_seconds']=time.perf_counter()-start;r['history']=history
    formal=checkpoint_payload(path);model.load_state_dict(formal['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(formal['best_epoch'])
    r.update(best_epoch=formal['best_epoch'],best_formal_val_objective=formal['best_val_loss'],checkpoint_sha256=sha256(path))
    r['sanity']['best']=full_sanity(model,test_x)
    r['sanity']['PASS']=r['sanity']['PASS'] and r['sanity']['best']['PASS'] and all(r['sanity']['geometry'].values())
    if not r['sanity']['PASS']:
        r['status']='STOPPED: best sanity failed; DO NOT INTERPRET PERFORMANCE'
        export(r,out);write_report(r,out);raise AssertionError(r['status'])
    arrays=collect_arrays(model,data['loaders'],device)
    assert sha256(Path(args.d0b_checkpoint))==r['baseline_checkpoint']['sha256']
    r.update(build_comparison(base_arrays,arrays,data['feature_names'][cs:ce],r,out))
    r['status']='COMPLETE';export(r,out);write_report(r,out)
    print(f'[{DISPLAY}] report={out.resolve()}; STOP',flush=True)
    m=r['metrics']['QKNorm']['test'];zero=population_metrics(np.zeros_like(arrays['test']['target']),arrays['test']['target'])
    legacy={**m,'Hit_Ratio':m['Hit']}
    return dict(variant=DISPLAY,params=520549,time=r['train_time_seconds'],MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],
        Hit_Ratio=m['Hit'],vs_zero_pct=100*(zero['MAE']-m['MAE'])/zero['MAE'],mn=legacy,mo=legacy,report=str(out))


def main():
    from cmgm.scripts.main_ablation import build_data
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints'))
    parser.add_argument('--d0b-checkpoint',type=Path,default=Path('checkpoints')/f'{BASE}_best.pt')
    parser.add_argument('--qknorm-report-dir',type=Path,default=Path('experiments')/'d0b_qknorm_graph_attention')
    parser.add_argument('--sanity-only',action='store_true')
    parser.add_argument('--no-cuda',action='store_true')
    parser.set_defaults(seq_len=20,batch_size=64,seed=42,epochs=200,patience=10)
    args=parser.parse_args();validate_protocol(args)
    if not args.no_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    device=torch.device('cpu' if args.no_cuda else 'cuda')
    run_qknorm(args,device,build_data(args))


if __name__=='__main__':main()
