"""D0B-TargetScaleHuber: one frozen TRAIN-scale objective and checkpoint diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.target_scale_huber import (
    VARIANT, OBJECTIVE, HORIZONS, TargetScaleHuber, estimate_training_scales,
    horizon_terms, restore_scale_metadata,
)
from cmgm.training.train import _prediction_loss, make_loss
from cmgm.scripts.d0b_5d_only_diagnostics import (
    BASE_VARIANT, TRACE_KEYS, assert_backbone, full_reference, collect_model,
)
from cmgm.scripts.d0b_grouped_diagnostics import extra_fixed_diagnostics
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0b_regime_routing_diagnostics import ROOT, difference, sha256
from cmgm.scripts.d0e_diagnostics import diagnostic_context, model_arguments, functional_values
from cmgm.scripts.d0b_huber_horizon_scale_diagnostic import (
    residual_statistics, network_gradients, split_statistics, markdown_table,
)
from cmgm.training.metric_standard import STANDARD

DISPLAY = 'D0B-TargetScaleHuber'
CONTROLS = {'uniform': {'mode':'uniform'}, 'zero-micro': {'zero_component':'Z'}, 'zero-long': {'zero_component':'H'}}


def initialization_check(model, batch, seed=42):
    assert model.variant == VARIANT
    assert_backbone(model)
    x,y = [v.to(next(model.parameters()).device) for v in batch[:2]]
    with diagnostic_context(model),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(x.device).eval()
        old,new = dict(baseline.named_parameters()),dict(model.named_parameters())
        assert old.keys()==new.keys() and all(old[k].shape==new[k].shape for k in old)
        diffs = {k:float((old[k]-new[k]).abs().max()) for k in old}
        before,after = full_reference(baseline,x),full_reference(model,x)
        forward = {k:difference(after[k],before[k]) for k in (*TRACE_KEYS,'evidence')}
        raw,weighted = horizon_terms(after['prediction'],y,model.target_scale_huber,config.MULTI_HORIZONS)
        base_losses = {str(h):make_loss()(before['prediction'][:,config.MULTI_HORIZONS.index(h)],y[:,config.MULTI_HORIZONS.index(h)]).item() for h in HORIZONS}
        counts = [sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        result = {'seed':seed,'fixed_TRAIN_batch_shape':list(x.shape), 'D0B_params':counts[0],
                  'TargetScaleHuber_params':counts[1],'difference':counts[1]-counts[0],
                  'max_abs_diff':max(diffs.values()),'mismatch_count':sum(v!=0 for v in diffs.values()),
                  'forward_differences':forward,'D0B_raw_huber':base_losses,'D0B_sum':sum(base_losses.values()),
                  'raw_huber_delta_h':{k:v.item() for k,v in raw.items()},
                  'weighted_huber':{k:v.item() for k,v in weighted.items()},
                  'weighted_sum':_prediction_loss(model,after['prediction'],y,make_loss()).item(),
                  'five_day_loss_abs_diff':abs(base_losses['5']-weighted['5'].item()),
                  'calibration':model.target_scale_huber.metadata(),
                  'counterfactual_initial_batch':calibration_statistics(before['prediction'].cpu().numpy(),y.cpu().numpy(),model.target_scale_huber),
                  'native_initial_batch':calibration_statistics(before['prediction'].cpu().numpy(),y.cpu().numpy(),None)}
        result['PASS'] = result['difference']==0 and result['mismatch_count']==0 and result['five_day_loss_abs_diff']==0 and all(v['max']<=2e-6 for v in forward.values())
        if not result['PASS']: raise AssertionError('D0B shared initialization / forward / anchor equality failed')
        print('[D0B-TargetScaleHuber shared init] '+json.dumps(result),flush=True)
        return result


def calibration_statistics(p,y,calibration):
    rows={}
    for h in HORIZONS:
        idx=config.MULTI_HORIZONS.index(h)
        ci=calibration.horizons.index(h) if calibration is not None else idx
        delta=calibration.deltas[ci] if calibration is not None else .02
        weight=calibration.weights[ci] if calibration is not None else 1.
        row=residual_statistics(p[:,idx],y[:,idx],delta)
        e=p[:,idx].astype(np.float64)-y[:,idx].astype(np.float64)
        a=np.abs(e)
        ga=weight*np.minimum(a,delta)
        sat=a>=delta
        gradient={'mean_abs':float(ga.mean()),'median_abs':float(np.median(ga)),
                  'P90_abs':float(np.percentile(ga,90)),'P95_abs':float(np.percentile(ga,95)),
                  'RMS':float(np.sqrt(np.mean(ga**2))),'max_abs':float(ga.max()),
                  'saturation_fraction':float(sat.mean()),
                  'saturated_mean_abs':float(ga[sat].mean()) if sat.any() else None,
                  'cap':weight*delta,'cap_error':abs(weight*delta-.02)}
        assert gradient['max_abs']<=.02+1e-12 and gradient['cap_error']<=1e-12
        if sat.any(): assert abs(gradient['saturated_mean_abs']-.02)<=1e-12
        row.update(delta=delta,weight=weight,weighted_huber_loss=weight*row['huber_loss'],weighted_gradient=gradient)
        rows[str(h)]=row
    total=sum(v['weighted_huber_loss'] for v in rows.values())
    return {'horizons':rows,'prediction_loss_population_mean':total,
            'quadratic_fraction_range':float(np.ptp([v['quadratic_fraction'] for v in rows.values()])),
            'weighted_shares':{h:v['weighted_huber_loss']/total if total else None for h,v in rows.items()},
            'definition':'full sample × commodity; analytical gradients are unreduced, mean-loss derivative additionally divides by population size'}


def gradient_probe(model,batch):
    with diagnostic_context(model):
        callback=None
        if model.variant==VARIANT:
            callback=lambda p,y,h: horizon_terms(p,y,model.target_scale_huber,config.MULTI_HORIZONS)[1][str(h)]
        return network_gradients(model,batch,next(model.parameters()).device,horizon_loss=callback)


def precision_checked_sanity(model,x):
    """Retain the original float32 threshold; audit permutation roundoff on a float64 copy."""
    from cmgm.scripts.d0e_diagnostics import fixed_sanity
    with diagnostic_context(model):
        original=fixed_sanity(model,x,label=DISPLAY,raise_on_failure=False)
        if original['PASS']:
            return original
        # A future-prefix or batch failure remains a hard failure.
        if max([*original['causality'].values(),original['batch_permutation_max'],original['single_sample_max']])>3e-6:
            raise AssertionError(f'Prefix/batch sanity failed: {original}')
        copied=HeteroMixHopCMGM(variant=model.variant,**model_arguments(model)).to(device=x.device,dtype=torch.float64).eval()
        copied.load_state_dict(model.state_dict(),strict=True)
        high=fixed_sanity(copied,x.double(),label=DISPLAY+' float64 numeric audit')
        values=[*high['causality'].values(),high['batch_permutation_max'],high['single_sample_max'],
                *[v for row in high['within_market'].values() for v in row.values()]]
        if max(values)>1e-10:
            raise AssertionError(f'Permutation discrepancy persists in float64: {high}')
        return {'PASS':True,'float32_original':original,'float64_copy':high,
                'float64_max_diff':max(values),'float64_audit_tolerance':1e-10,
                'interpretation':'Original float32 permutation threshold exceeded; discrepancy vanishes in float64 copy. Training model/precision untouched; original float32 result retained.'}


def performance_comparison(models):
    rows={}
    for s in ('TRAIN','VAL','TEST'):
        rows[s]={}
        for h in map(str,HORIZONS):
            before=models['D0B']['splits'][s]['native_metrics'][h]
            after=models['TargetScaleHuber']['splits'][s]['native_metrics'][h]
            rows[s][h]={'D0B':before,'TargetScaleHuber':after,
                        **{f'delta_{k}':after[k]-before[k] for k in ('MAE','MSE')},
                        **{f'relative_{k}':(after[k]-before[k])/before[k] if before[k] else None for k in ('MAE','MSE')}}
    return rows


def case_assessment(comparison,mechanically_aligned=True,healthy=True):
    # Tolerance is numerical, not a statistical/materiality cutoff. No tuning.
    tie=1e-6
    mae=[comparison[s]['5']['relative_MAE'] for s in ('VAL','TEST')]
    mse=[comparison[s]['5']['relative_MSE'] for s in ('VAL','TEST')]
    result={'case':None,'candidate_final_model':False,'MAE_relative_VAL_TEST':mae,'MSE_relative_VAL_TEST':mse,
            'numerical_tie_tolerance':tie,'materiality_note':'Single-seed directional classification; inspect raw magnitude, tails and health. No statistical significance is inferred.'}
    if not all(v is not None and np.isfinite(v) for v in mae+mse) or not healthy:
        result['reason']='Sanity or regime-health failure; do not force a performance Case or replace D0B.'
    elif all(v < -tie for v in mae):
        if any(v > tie for v in mse):
            result.update(case='Case E',reason='MAE improves with an MSE tail-error tradeoff; magnitude must be reviewed; do not auto-accept.')
        elif mechanically_aligned:
            result.update(case='Case A',candidate_final_model=True,
                          reason='Both 5d MAEs improve, MSE does not worsen beyond numerical tolerance, working regions align and regime sanity passes. This supports calibration in this run; inspect effect sizes before calling it material.')
        else:
            result['reason']='Forecast errors improve but working-region condition does not hold; Case A mechanism is not established.'
    elif all(v < -tie for v in mse):
        result.update(case='Case F',reason='MSE improves without consistent MAE improvement; primarily a tail-error effect, no automatic replacement.')
    elif all(v > tie for v in mae):
        result.update(case='Case C',reason='Both 5d MAEs worsen; supports useful implicit horizon weighting of fixed-delta Huber.')
    elif all(abs(v)<=tie for v in mae+mse) and mechanically_aligned:
        result.update(case='Case B',reason='Calibration aligns working regions but forecasts remain numerically tied; no demonstrated forecasting bottleneck.')
    else:
        result.update(case='Case D',reason='VAL/TEST or error metrics give inconsistent evidence; keep D0B.')
    return result


def checkpoint_report(model,payload,data,d0b_path,output,checkpoint_path,seed=42):
    output.mkdir(parents=True,exist_ok=False)
    paths={'D0B':Path(d0b_path),'TargetScaleHuber':Path(checkpoint_path)}
    hashes={k:sha256(v) for k,v in paths.items()}
    snapshots={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.target_scale_huber=restore_scale_metadata(payload['metadata'])
    market=data['market_indices'];names=data['feature_names'][market['commodity'][0]:market['commodity'][1]]
    models={}
    with diagnostic_context(model):
        fixed=next(iter(data['loaders']['test']))
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(next(model.parameters()).device)
        saved=_load_checkpoint(baseline,paths['D0B'],next(model.parameters()).device)
        for name,active,record in [('D0B',baseline,saved),('TargetScaleHuber',model,payload)]:
            active.switching_latent_transformer.set_epoch(record.get('best_epoch',1))
            collected=collect_model(active,record,data,output/name,fixed,seed,label=DISPLAY,
                                    extra_fixed=extra_fixed_diagnostics,controls=CONTROLS,gradient_probe=gradient_probe,sanity_probe=precision_checked_sanity)
            collected['train_time_seconds']=record.get('metadata',{}).get('training_elapsed_seconds',record.get('history',{}).get('training_elapsed_seconds'))
            for s,split in collected['splits'].items():
                a=np.load(output/name/f'{s.lower()}_predictions.npz')
                calibration=model.target_scale_huber if name=='TargetScaleHuber' else None
                split['fixed_delta_descriptive_losses']=split.pop('prediction_losses')
                split['calibration']=calibration_statistics(a['native'],a['target'],calibration)
                if name=='D0B':
                    split['counterfactual_calibration']=calibration_statistics(a['native'],a['target'],model.target_scale_huber)
                error=split_statistics(a['native'],a['target'],names,SimpleNamespace(variant=BASE_VARIANT))
                split['commodity_error']={k:error[k] for k in ('commodity_5d','commodity_heterogeneity','highest_MSE_commodities','lowest_MSE_commodities')}
            models[name]=collected
    assert all(torch.equal(v,snapshots[k]) for k,v in model.state_dict().items())
    assert all(sha256(paths[k])==digest for k,digest in hashes.items())
    comparison=performance_comparison(models)
    aligned=all(models['TargetScaleHuber']['splits'][s]['calibration']['quadratic_fraction_range'] < models['D0B']['splits'][s]['calibration']['quadratic_fraction_range'] for s in ('TRAIN','VAL','TEST'))
    # Joint concentration indicators, not argmax occupancy alone.
    collapse={}
    for s,sp in models['TargetScaleHuber']['splits'].items():
        p=sp['normal_regime']
        collapse[s]={'warning':max(p['occupancy'])>=.99 and p['mean_max']>=.99 and p['entropy']<=.1,
                     'definition':'joint extreme-concentration warning only: occupancy>=.99 AND mean max>=.99 AND entropy<=.1; not a forecast acceptance cutoff'}
    sane=all(m['sanity']['PASS'] for m in models.values()) and not any(v['warning'] for v in collapse.values())
    consistency=max(abs(v['RMSE']**2-v['MSE']) for m in models.values() for sp in m['splits'].values() for v in sp['native_metrics'].values())
    report={'display_name':DISPLAY,'variant':VARIANT,'objective':OBJECTIVE,'metric_standard':STANDARD,
            'metadata':payload['metadata'],'checkpoint_paths':{k:str(v.resolve()) for k,v in paths.items()},
            'checkpoint_sha256':hashes,'fixed_TEST_shape':list(fixed[0].shape),'models':models,
            'comparison':comparison,'working_regions_more_aligned':aligned,'regime_over_specialization':collapse,
            'assessment':case_assessment(comparison,aligned,sane),
            'integrity':{'parameters_unchanged_by_diagnostics':True,'checkpoint_files_unchanged':True,'max_RMSE_squared_minus_MSE':consistency}}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    from cmgm.scripts.d0b_target_scale_huber_report import write_report
    write_report(report,output/'REPORT.md')
    print('[D0B-TargetScaleHuber REPORT] '+str(output/'REPORT.md'),flush=True)
    return report


def validate_protocol(args):
    expected={'seed':42,'batch_size':64,'seq_len':20,'epochs':200,'patience':10}
    if any(getattr(args,k)!=v for k,v in expected.items()):
        raise ValueError(f'This controlled experiment requires {expected}; no tuning runs')
    if (config.LEARNING_RATE!=1e-4 or config.WEIGHT_DECAY!=1e-5 or config.HUBER_DELTA!=.02
            or config.LOSS_TYPE!='huber' or config.MULTI_HORIZONS!=list(HORIZONS)):
        raise ValueError('Original D0B configuration changed; stop')


@torch.no_grad()
def reference_counterfactual(reference,train_loader,calibration,device):
    """Before training: identical trained D0B residuals, only threshold changes.

    Calibration is already frozen from targets. Residuals never fit any constant.
    """
    reference.eval()
    full=DataLoader(train_loader.dataset,batch_size=train_loader.batch_size,shuffle=False,drop_last=False)
    predictions,targets=[],[]
    for batch in full:
        predictions.append(reference(batch[0].to(device)).cpu().numpy())
        targets.append(batch[1].numpy())
    p,y=np.concatenate(predictions),np.concatenate(targets)
    native=calibration_statistics(p,y,None)
    counterfactual=calibration_statistics(p,y,calibration)
    result={'source':'existing trained D0B; complete TRAIN residuals; computed before new training',
            'samples':len(p),'native_quadratic_range':native['quadratic_fraction_range'],
            'counterfactual_quadratic_range':counterfactual['quadratic_fraction_range'],
            'horizons':{h:{'native_Q':native['horizons'][h]['quadratic_fraction'],
                           'counterfactual_Q':counterfactual['horizons'][h]['quadratic_fraction']}
                        for h in map(str,HORIZONS)}}
    print('[D0B-TargetScaleHuber pre-training D0B counterfactual] '+json.dumps(result),flush=True)
    return result


def prepare(args,data,device):
    validate_protocol(args)
    calibration,scale_sanity=estimate_training_scales(data['loaders']['train'].dataset)
    print('[D0B-TargetScaleHuber TRAIN scale] '+json.dumps(scale_sanity),flush=True)
    set_seed(args.seed)
    m=data['market_indices']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=m['stock'][1]-m['stock'][0],
                          n_bond=m['bond'][1]-m['bond'][0],feat_dim=config.FEATURE_DIM,variant=VARIANT)
    model.target_scale_huber=calibration
    # Diagnostic loader iteration and model construction must not advance training RNG.
    with torch.random.fork_rng(devices=[]):
        fixed=next(iter(data['loaders']['train']))
    initial=initialization_check(model,fixed,args.seed)
    if initial['D0B_params']!=520549: raise ValueError('Current architecture parameter baseline changed; stop for review')
    model.to(device)
    branch=model.switching_latent_transformer
    initial_logits=branch.regime_filter.transition_logits.detach().cpu().clone()
    branch._initial_transition_logits=initial_logits
    metadata={**calibration.metadata(),'variant':VARIANT,'display_name':DISPLAY,'seed':args.seed,
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'parameter_count':initial['TargetScaleHuber_params'],'initialization':initial,'scale_sanity':scale_sanity,
              'initial_transition_logits':initial_logits.tolist(),'metric_standard':STANDARD,
              'protocol':{'seed':42,'batch_size':64,'epochs':200,'patience':10,'seq_len':20,
                          'optimizer':'Adam','lr':1e-4,'weight_decay':1e-5,'scheduler':'ReduceLROnPlateau',
                          'validation_policy':'existing D0B prediction-only; mean of batch means; no switch KL in selection',
                          'switch_beta_max':5e-4,'warmup_epochs':20},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'cmgm/training/target_scale_huber.py',ROOT/'cmgm/training/train.py',ROOT/'cmgm/models/hetero_mixhop_model.py']}}
    return model,metadata


def run_target_scale_huber(args,device,data):
    from cmgm.training.train import train
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant
    validate_protocol(args)
    if not args.d0b_checkpoint.is_file(): raise FileNotFoundError(args.d0b_checkpoint)
    path=_checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    if path.exists(): raise FileExistsError(f'{path} exists; this entry will not overwrite/retrain the one-run experiment. Use checkpoint-only diagnostics.')
    model,metadata=prepare(args,data,device)
    # Verify the requested reference before spending the one authorized training run.
    # Context restores initialization RNG; reference weights never enter the new model.
    with diagnostic_context(model):
        reference=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(device)
        reference_payload=_load_checkpoint(reference,args.d0b_checkpoint,device)
        metadata['D0B_reference']={'path':str(args.d0b_checkpoint.resolve()),
                                   'sha256':sha256(args.d0b_checkpoint),
                                   'best_epoch':reference_payload.get('best_epoch'),
                                   'metadata':reference_payload.get('metadata',{})}
        metadata['pre_training_D0B_counterfactual']=reference_counterfactual(
            reference,data['loaders']['train'],model.target_scale_huber,device)
        del reference,reference_payload
    output=args.target_scale_report_dir/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    (output/'preflight.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    # One training call only. The new model is randomly initialized, never initialized from D0B checkpoint.
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata)
    payload=_load_checkpoint(model,path,device)
    report=checkpoint_report(model,payload,data,args.d0b_checkpoint,output/'checkpoint_diagnostics',path,args.seed)
    return ablation_result(report,payload,output/'checkpoint_diagnostics',data)


def ablation_result(report,payload,output,data):
    """Existing main_ablation / ExperimentLogger contract, using saved full predictions."""
    from cmgm.training.evaluate import compute_metrics,inverse_transform_predictions
    values=np.load(output/'TargetScaleHuber'/'test_predictions.npz')
    idx=config.MULTI_HORIZONS.index(5)
    prediction,target=values['native'][:,idx],values['target'][:,idx]
    normalized=compute_metrics(prediction,target)
    p_original,y_original=inverse_transform_predictions(prediction,target,data['norm_stats'],
                                                        data['raw_prices_test'],data['market_indices'],target_type=config.TARGET_TYPE)
    original=compute_metrics(p_original,y_original)
    return {'variant':DISPLAY,'params':report['models']['TargetScaleHuber']['params'],
            'time':payload['history']['training_elapsed_seconds'],
            **{k:normalized[k] for k in ('MAE','MSE','RMSE','Hit_Ratio')},
            'vs_zero_pct':(normalized['MAE']/float(np.abs(target).mean())-1)*100,
            'mn':normalized,'mo':original,
            'report_path':str(output/'REPORT.md'),'case':report['assessment']['case']}


def write_preflight_report(metadata, output):
    initial=metadata['initialization']
    lines=['# D0B-TargetScaleHuber — preflight only', '',
           '仅代码与真实数据验证；未执行训练实验。VAL/TEST performance、best epoch 与 Case A–F 均待正式训练后报告。', '',
           'TRAIN target scale source = population std, float64, ddof=0；全 1396 窗口×24 commodities，含尾 batch。', '',
           markdown_table(['Horizon','TRAIN std','delta_h','weight_h','gradient cap'],
                          [[h,metadata[f'target_scale_{h}'],metadata[f'delta_{h}'],metadata[f'weight_{h}'],metadata['caps'][str(h)]] for h in HORIZONS]),
           markdown_table(['Check','Result'],[['D0B params',initial['D0B_params']],['TargetScaleHuber params',initial['TargetScaleHuber_params']],
                                              ['Parameter difference',initial['difference']],['Parameter max diff',initial['max_abs_diff']],
                                              ['5d loss abs diff',initial['five_day_loss_abs_diff']],['Fixed TRAIN shape',initial['fixed_TRAIN_batch_shape']],
                                              ['Shared init PASS',initial['PASS']],['Scale sanity PASS',metadata['scale_sanity']['PASS']],
                                              ['Causality/batch/market PASS',metadata['fixed_TEST_sanity']['PASS']]]),
           markdown_table(['Trace','Mean diff','Max diff'],[[k,*v.values()] for k,v in initial['forward_differences'].items()]),
           '训练前 raw/weighted Huber decomposition 与 counterfactual coverage、完整 sanity 数值见 preflight.json。', '',
           '唯一正式训练命令（在 Commedities 目录）：', '', '```bash',
           'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.main_ablation --variants D0B-TargetScaleHuber --seed 42 --batch-size 64 --seq-len 20 --epochs 200 --patience 10',
           '```', '', 'CPU 追加 --no-cuda。训练后自动生成 REPORT.md；不自动执行其他实验。', '']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description='TargetScaleHuber preflight or checkpoint-only diagnostics; training uses main_ablation')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--no-cuda',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    args.seed=42;args.batch_size=64;args.seq_len=20;args.epochs=200;args.patience=10
    device=torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    if args.output.exists(): raise FileExistsError(args.output)
    from cmgm.scripts.main_ablation import build_data
    data=build_data(args)
    if args.preflight:
        model,metadata=prepare(args,data,device)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'preflight_initialization.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        with diagnostic_context(model):
            fixed=next(iter(data['loaders']['test']))
            metadata['fixed_TEST_sanity']=precision_checked_sanity(model,fixed[0].to(device))
        (args.output/'preflight.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
        write_preflight_report(metadata,args.output)
        print('[D0B-TargetScaleHuber preflight only] PASS; no training',flush=True)
    else:
        m=data['market_indices']
        model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=m['stock'][1],n_bond=m['bond'][1]-m['bond'][0],feat_dim=21,variant=VARIANT).to(device)
        payload=_load_checkpoint(model,args.checkpoint,device)
        checkpoint_report(model,payload,data,args.d0b_checkpoint,args.output,args.checkpoint)


if __name__=='__main__':
    main()
