"""One global complementary fusion experiment; original D0B protocol and controlled inference."""
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
from cmgm.scripts.d0b_hybrid_graph_prior import validate_protocol as _validate_protocol
from cmgm.scripts.d0b_fusion_diagnostics import (
    VARIANT,BASE,DISPLAY,initialization_check,gradient_diagnostics,structural_sanity,assert_backbone,collect,ablation_analysis,
)

REFERENCE=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)


def validate_protocol(args):
    try:
        _validate_protocol(args)
    except ValueError as error:
        raise ValueError(str(error).replace('Hybrid', 'ResidualComplementaryFusion')) from error


def verify_reference(arrays):
    actual=population_metrics(arrays['test']['prediction'],arrays['test']['target'])
    checks={k:bool(np.isclose(actual[k],v,rtol=1e-5,atol=1e-9)) for k,v in REFERENCE.items()}
    return dict(actual=actual,reference=REFERENCE,checks=checks,PASS=all(checks.values()),
        RMSE_squared_minus_MSE_abs=abs(actual['RMSE']**2-actual['MSE']))


def export(r,out):
    for filename,key in [('training_history','history'),('fusion_residual_diagnostics','residual'),
        ('ablation_metrics','ablations'),('gradient_diagnostics','gradients'),('sanity_checks','sanity')]:
        save_results(r.get(key,{}),out/(filename+'.json'))
    save_results(r,out/'results.json')


def run_fusion(args,device,data):
    from cmgm.scripts.d0b_fusion_report import build_comparison,write_report
    validate_protocol(args)
    if not args.no_cuda and device.type!='cuda':raise RuntimeError('ResidualComplementaryFusion requires GPU unless CPU is explicitly requested; no silent fallback')
    path=Path(args.checkpoint_dir)/f'{VARIANT}_best.pt'
    if path.exists() and not getattr(args,'sanity_only',False):raise FileExistsError(f'{path} already exists; no automatic rerun or overwrite')
    out=Path(args.fusion_report_dir)/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    set_seed(args.seed)
    cs,ce=data['market_indices']['commodity']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],variant=VARIANT,
        n_stock=data['market_indices']['stock'][1],n_bond=data['market_indices']['bond'][1]-data['market_indices']['bond'][0],feat_dim=config.FEATURE_DIM).to(device)
    assert_backbone(model)
    r=dict(display=DISPLAY,variant=VARIANT,seq_len=20,seed=42,status='PREFLIGHT',training_executed=False,primary_case=None,
        checkpoint_path=str(path.resolve()),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        residual={},gradients={},sanity={},
        protocol='Only one shared zero-initialized global fusion residual added; original raw Q/K, all 8 priors and complete D0B global path unchanged. Formal selection uses original multi-horizon validation objective; pooled VAL5 is secondary only.')
    files=['cmgm/config.py','cmgm/models/hetero_mixhop_model.py','cmgm/graph/adaptive_graph.py','cmgm/training/train.py',
        'cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py',
        'cmgm/scripts/d0b_residual_complementary_fusion.py','cmgm/scripts/d0b_fusion_diagnostics.py','cmgm/scripts/d0b_fusion_report.py',
        'cmgm/scripts/d0b_hybrid_graph_prior.py','cmgm/scripts/d0b_hybrid_graph_diagnostics.py','cmgm/scripts/d0e_diagnostics.py',
        'cmgm/training/metric_standard.py']
    r['implementation_sha256']={name:sha256(ROOT/name) for name in files}
    def capture(active,stage):
        g=gradient_diagnostics(active,train_fixed)
        r['gradients'][stage]=g
        r['residual'][stage]=g
        return g
    with diagnostic_context(model):
        batch=next(iter(data['loaders']['train']));train_fixed=tuple(t[:4].to(device) for t in batch[:2])
        test_x=next(iter(data['loaders']['test']))[0][:4].to(device)
        r['fixed_TRAIN_shape']=list(train_fixed[0].shape);r['fixed_TEST_shape']=list(test_x.shape)
        r['split_origins']={s:len(l.dataset) for s,l in data['loaders'].items()}
        r['sanity']['shared_init']=initialization_check(model,test_x,args.seed)
        assert r['sanity']['shared_init']['D0B_params']==520549 and r['sanity']['shared_init']['New_params']==526725
        print(f'[{DISPLAY} shared init] {r["sanity"]["shared_init"]}',flush=True)
        r['sanity']['initial']=structural_sanity(model,test_x)
        first=capture(model,'initial')
        r['sanity']['initial_gradient']=dict(final_gradient=first['norms']['residual_final'],first_gradient=first['norms']['residual_first'],
            PASS=first['norms']['residual_final']>0 and first['norms']['residual_first']==0)
        baseline=HeteroMixHopCMGM(variant=BASE,**model_arguments(model)).to(device).eval()
        cp=Path(args.d0b_checkpoint);payload=checkpoint_payload(cp);metadata=payload.get('metadata',{})
        recorded=metadata.get('variant',payload.get('variant'))
        if recorded is not None and recorded!=BASE:raise ValueError('Formal D0B checkpoint variant mismatch')
        baseline.load_state_dict(payload.get('model_state_dict',payload.get('state_dict',payload)),strict=True)
        baseline.switching_latent_transformer.set_epoch(payload.get('best_epoch') or 1)
        r['baseline_checkpoint']=dict(path=str(cp.resolve()),sha256=sha256(cp),best_epoch=payload.get('best_epoch'),
            seed=metadata.get('seed'),training_SHA=metadata.get('git_sha'))
        base_arrays,base_stats=collect(baseline,data['loaders'],device)
        r['residual']['D0B_full_loader']=base_stats
        r['baseline_reference']=verify_reference(base_arrays)
        r['baseline_metrics']={s:population_metrics(a['prediction'],a['target']) for s,a in base_arrays.items()}
        print(f'[{DISPLAY} native D0B pooled reference] {r["baseline_reference"]}',flush=True)
        r['sanity']['legacy_D0B_reference']=not baseline.attn_mixhop1.qk_norm and not baseline.attn_mixhop2.qk_norm and r['baseline_reference']['PASS']
        del baseline
    r['sanity']['PASS']=r['sanity']['shared_init']['PASS'] and r['sanity']['initial']['PASS'] and r['sanity']['legacy_D0B_reference'] and r['sanity']['initial_gradient']['PASS']
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
        result=capture(active,key);export(r,out)
        return result
    r['training_executed']=True;r['status']='TRAINING'
    start=time.perf_counter()
    history=train(model,data['loaders']['train'],data['loaders']['val'],torch.empty((2,0),dtype=torch.long,device=device),
        torch.empty(0,device=device),device,num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),
        checkpoint_metadata=dict(variant=VARIANT,display_name=DISPLAY,seed=args.seed,seq_len=20,git_sha=r['git_sha'],
            parameter_count=526725,added_parameters=6176,residual='shared global 128-32-64; final bias=False and zero initialized',
            qk_norm=False,graph_prior_heads='all 8',
            formal_selection='original multi-horizon validation objective'),epoch_diagnostic=diagnostic)
    r['train_time_seconds']=time.perf_counter()-start;r['history']=history
    formal=checkpoint_payload(path);model.load_state_dict(formal['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(formal['best_epoch'])
    r.update(best_epoch=formal['best_epoch'],best_formal_val_objective=formal['best_val_loss'],checkpoint_sha256=sha256(path))
    r['sanity']['best']=structural_sanity(model,test_x)
    r['sanity']['PASS']=r['sanity']['PASS'] and r['sanity']['best']['PASS'] and r['sanity']['initial_gradient']['PASS']
    if not r['sanity']['PASS']:
        r['status']='STOPPED: best sanity failed; DO NOT INTERPRET PERFORMANCE'
        export(r,out);write_report(r,out);raise AssertionError(r['status'])
    arrays,stats=collect(model,data['loaders'],device)
    r['residual']['New_full_loader']=stats
    r['ablations']=ablation_analysis(base_arrays,arrays)
    assert sha256(Path(args.d0b_checkpoint))==r['baseline_checkpoint']['sha256']
    r.update(build_comparison(base_arrays,arrays,data['feature_names'][cs:ce],r,out))
    r['status']='COMPLETE';export(r,out);write_report(r,out)
    print(f'[{DISPLAY}] report={out.resolve()}; STOP',flush=True)
    m=r['metrics']['ResidualComplementaryFusion']['test']
    legacy={**m,'Hit_Ratio':m['Hit']}
    return dict(variant=DISPLAY,params=526725,time=r['train_time_seconds'],MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],
        Hit_Ratio=m['Hit'],vs_zero_pct=float('nan'),mn=legacy,mo=legacy,report=str(out))


def main():
    from cmgm.scripts.main_ablation import build_data
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints'))
    parser.add_argument('--d0b-checkpoint',type=Path,default=Path('checkpoints')/f'{BASE}_best.pt')
    parser.add_argument('--fusion-report-dir',type=Path,default=Path('experiments')/'d0b_residual_complementary_fusion')
    parser.add_argument('--sanity-only',action='store_true')
    parser.add_argument('--no-cuda',action='store_true')
    parser.set_defaults(seq_len=20,batch_size=64,seed=42,epochs=200,patience=10)
    args=parser.parse_args();validate_protocol(args)
    if not args.no_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    device=torch.device('cpu' if args.no_cuda else 'cuda')
    run_fusion(args,device,build_data(args))


if __name__=='__main__':main()
