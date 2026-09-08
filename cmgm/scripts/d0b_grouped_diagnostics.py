"""One fixed 4/3*(5d+10d+20d) objective and three-checkpoint diagnostics."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import (
    GROUPED_VARIANT as VARIANT, GROUPED_MULTIPLIER, GROUPED_HORIZONS,
    GROUPED_OBJECTIVE, _prediction_loss, _horizon_loss_values, make_loss,
)
from cmgm.scripts.d0b_5d_only_diagnostics import (
    VARIANT as FIVE_VARIANT, BASE_VARIANT, TRACE_KEYS, assert_backbone,
    collect_model, full_reference,
)
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0b_regime_routing_diagnostics import ROOT, array, difference, sha256
from cmgm.scripts.d0e_diagnostics import diagnostic_context, model_arguments

DISPLAY = 'D0B-MidLongGroupedObjective'
PREFIX = 'D0B grouped'


def initialization_check(model, batch, seed=42):
    assert model.variant == VARIANT
    assert_backbone(model)
    device = next(model.parameters()).device
    x, y = [v.to(device) for v in batch[:2]]
    with diagnostic_context(model), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT, **model_arguments(model)).to(device).eval()
        left, right = dict(baseline.named_parameters()), dict(model.named_parameters())
        assert left.keys() == right.keys()
        assert all(left[k].shape == right[k].shape for k in left)
        differences = {k:(left[k]-right[k]).abs().max().item() for k in left}
        before, after = full_reference(baseline,x), full_reference(model,x)
        traces = {k:difference(after[k],before[k]) for k in TRACE_KEYS}
        counts = [sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        losses = _horizon_loss_values(after['prediction'],y,make_loss())
        raw = sum(losses[str(h)] for h in GROUPED_HORIZONS)
        scaled = _prediction_loss(model,after['prediction'],y,make_loss()).item()
        multi = sum(losses.values())
        result = {'seed':seed,'fixed_TRAIN_batch_shape':list(x.shape),
                  'D0B_params':counts[0],'Grouped_params':counts[1],'difference':counts[1]-counts[0],
                  'max_abs_diff':max(differences.values()),'mismatch_count':sum(v!=0 for v in differences.values()),
                  'forward_differences':traces,
                  'loss_scale':{**{f'L{h}':v for h,v in losses.items()},'sum_multi':multi,
                                'group_raw':raw,'group_scaled':scaled,'ratio':scaled/multi if multi>0 else None,
                                'multiplier':GROUPED_MULTIPLIER,'description':'scale-matched grouped-objective diagnostic'}}
        result['PASS'] = result['difference']==0 and result['mismatch_count']==0 and all(v['max']<=2e-6 for v in traces.values())
        print(f'[{PREFIX} shared init] {json.dumps(result)}',flush=True)
        if not result['PASS']:
            raise AssertionError('Grouped model differs from D0B at initialization')
        return result


def extra_fixed_diagnostics(model, native):
    """Real native forward caches; QK/base statistics use causal entries only."""
    b = model.switching_latent_transformer
    memory = b.long_memory
    bias = memory.last_base_relative_bias
    mask = torch.ones(bias.shape[-2:],dtype=torch.bool,device=bias.device).tril()
    base_mean = bias[...,mask].abs().mean().item()
    qk_means = [layer.attention.last_qk_logits[...,mask].abs().mean().item() for layer in memory.layers]
    qk_mean = float(np.mean(qk_means))
    long_norm = b.long_memory_readout.weight.norm().item()
    micro_norm = b.micro_state_readout.weight.norm().item()
    W = b.state_readout.weight
    half = W.shape[1]//2
    return {'base_rpe':{'norm':memory.base_rpe.norm().item(),'mean_abs_QK':qk_mean,
                        'mean_abs_base_bias':base_mean,'base_QK_ratio':base_mean/(qk_mean+1e-8),
                        'layer_mean_abs_QK':qk_means,
                        'definition':'QK/sqrt(head_dim), before bias/masking; lower triangle including diagonal; equal layer mean'},
            'readout_weights':{'W_long_norm':long_norm,'W_micro_norm':micro_norm,
                               'W_micro_W_long':micro_norm/(long_norm+1e-8),
                               'definition':'Frobenius norms of long_memory_readout.weight W_H and micro_state_readout.weight W_Z',
                               'state_readout_long_block_norm':W[:,:half].norm().item(),
                               'state_readout_micro_block_norm':W[:,half:].norm().item()},
            'representation_norms':{k:native[k].norm(dim=-1).mean().item()
                                    for k in ('H','Z','h_temporal','h_spatial')}}


def checked_load(model,path,device,objective=None):
    payload = _load_checkpoint(model,Path(path),device)
    metadata = payload.get('metadata',{})
    if objective is not None and metadata.get('objective') != objective:
        raise ValueError(f'{path}: expected objective={objective}; identical state shapes cannot identify objective')
    if metadata.get('variant',model.variant) != model.variant:
        raise ValueError(f'{path}: checkpoint variant does not match {model.variant}')
    return payload


def performance_comparison(models):
    result = {}
    for split in ('TRAIN','VAL','TEST'):
        rows = {}
        for horizon in map(str,config.MULTI_HORIZONS):
            values = {k:m['splits'][split]['native_metrics'][horizon] for k,m in models.items()}
            delta = {}
            for baseline in ('D0B','5dOnly'):
                old,new = values[baseline]['MAE'],values['Grouped']['MAE']
                delta[baseline] = {'delta_MAE':new-old,'relative_change':(new-old)/old if old>0 else None}
            rows[horizon] = {'metrics':values,'grouped_vs':delta}
        result[split] = rows
    return result


def case_assessment(comparison, sane=True):
    """Conservative reporting rules; no thresholds influence training/selection."""
    effect, numerical_tie = .001, 1e-6
    versus_base = [comparison[s]['5']['grouped_vs']['D0B']['relative_change'] for s in ('VAL','TEST')]
    versus_only = [comparison[s]['5']['grouped_vs']['5dOnly']['relative_change'] for s in ('VAL','TEST')]
    healthy = all(comparison[s][str(h)]['metrics']['Grouped'][metric] <=
                  (1+effect)*comparison[s][str(h)]['metrics']['D0B'][metric]
                  for s in ('VAL','TEST') for h in (10,20) for metric in ('MAE','RMSE'))
    result = {'case':None,'candidate_objective_justified':False,
              'auxiliary_10_20_healthy':healthy,'relative_effect_convention':effect,
              'numerical_tie_relative_tolerance':numerical_tie,
              'health_definition':'10d/20d VAL and TEST MAE and RMSE do not worsen by >0.1% vs D0B',
              'note':'Single seed and approximate scale matching; descriptive evidence, not a statistical significance test.'}
    if not sane or any(v is None or not np.isfinite(v) for v in versus_base+versus_only):
        result['reason']='Invalid baseline or sanity failure; no mechanism conclusion.'
    elif all(abs(v)<=numerical_tie for v in versus_base):
        result.update(case='Case E',reason='5d metrics essentially tied; little measured net effect of 1d supervision.')
    elif all(v < -numerical_tie for v in versus_base):
        if any(v > -effect for v in versus_base):
            result.update(case='Case B',reason='Both splits improve, but at least one improves by <0.1%; weak negative-transfer evidence.')
        elif all(v<=-effect for v in versus_only) and healthy:
            result.update(case='Case A',candidate_objective_justified=True,
                          reason='Grouped materially beats D0B and 5d-only on both splits while 10d/20d remain healthy; supports beneficial 10/20 regularization and harmful 1d supervision.')
        else:
            result['reason']='Both splits beat D0B, but strongest Case A conditions fail (5d-only comparison or 10/20 health); supplied A–E cases do not fully cover this outcome. Do not force Case A.'
    elif all(v > numerical_tie for v in versus_base):
        result.update(case='Case C',reason='Removing 1d worsens 5d VAL and TEST; auxiliary 1d supervision is beneficial here.')
    else:
        result.update(case='Case D',reason='No consistent two-split improvement; removing 1d does not provide robust generalization benefit.')
    return result


def checkpoint_report(model,payload,data,d0b_path,five_path,output,seed=42,initialization=None,checkpoint_path=None):
    assert model.variant == VARIANT
    output.mkdir(parents=True,exist_ok=False)
    paths = {'D0B':Path(d0b_path),'5dOnly':Path(five_path)}
    if checkpoint_path is not None:
        paths['Grouped'] = Path(checkpoint_path)
    hashes = {k:sha256(p) for k,p in paths.items()}
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    with diagnostic_context(model):
        fixed = next(iter(data['loaders']['test']))
        device = next(model.parameters()).device
        references = {}
        for label,variant,obj in (('D0B',BASE_VARIANT,None),('5dOnly',FIVE_VARIANT,'4x_5d_only')):
            baseline = HeteroMixHopCMGM(variant=variant,**model_arguments(model)).to(device)
            references[label] = (baseline,checked_load(baseline,paths[label],device,obj))
        references['Grouped'] = (model,payload)
        models = {}
        for label,(active,saved) in references.items():
            active.switching_latent_transformer.set_epoch(saved.get('best_epoch',1))
            models[label] = collect_model(active,saved,data,output/label,fixed,seed,
                                          label=PREFIX,extra_fixed=extra_fixed_diagnostics)
            elapsed = saved.get('metadata',{}).get('training_elapsed_seconds',saved.get('history',{}).get('training_elapsed_seconds'))
            models[label]['train_time_seconds'] = elapsed
            models[label]['train_time_note'] = ('Training loop elapsed time; excludes post-training report' if elapsed is not None
                                                else 'Not recorded in historical checkpoint; not inferred from wall time including diagnostics')
            for split in models[label]['splits'].values():
                losses = split['prediction_losses']
                raw = sum(losses['per_horizon'][str(h)] for h in GROUPED_HORIZONS)
                losses.update(group_raw=raw,group_scaled=GROUPED_MULTIPLIER*raw)
        drift = {}
        for label in ('D0B','5dOnly'):
            with np.load(output/label/'fixed_native.npz') as before, np.load(output/'Grouped/fixed_native.npz') as after:
                drift[f'Grouped-vs-{label}'] = {k:difference(torch.from_numpy(after[k]),torch.from_numpy(before[k])) for k in TRACE_KEYS}
        prediction_impact = {}
        for split in ('TRAIN','VAL','TEST'):
            prediction_impact[split] = {}
            for label in ('D0B','5dOnly'):
                filename=f'{split.lower()}_predictions.npz'
                with np.load(output/label/filename) as before, np.load(output/'Grouped'/filename) as after:
                    np.testing.assert_array_equal(before['target'],after['target'])
                    prediction_impact[split][label] = {str(h):float(np.abs(after['native'][:,i]-before['native'][:,i]).mean())
                                                       for i,h in enumerate(config.MULTI_HORIZONS)}
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(sha256(paths[k])==digest for k,digest in hashes.items())
    comparison = performance_comparison(models)
    report = {'objective':GROUPED_OBJECTIVE,'description':'scale-matched grouped-objective diagnostic',
              'seed':seed,'fixed_TEST_batch_shape':list(fixed[0].shape),'initialization':initialization,
              'checkpoint_paths':{k:str(p.resolve()) for k,p in paths.items()},'checkpoint_sha256':hashes,
              'metadata':payload.get('metadata',{}),'models':models,'performance_comparison':comparison,
              'representation_drift':drift,'prediction_impact':prediction_impact,
              'assessment':case_assessment(comparison,all(m['sanity']['PASS'] for m in models.values())),
              'integrity':{'parameters_unchanged':True,'checkpoint_files_unchanged':True}}
    (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    from cmgm.scripts.d0b_grouped_report import write_report
    write_report(report,output/'REPORT.md')
    print(f'[{PREFIX} REPORT] {output / "REPORT.md"}',flush=True)
    return report


def run_grouped(args,device,data):
    from cmgm.training.train import train
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant, evaluate_primary_horizon, print_diagnostics
    from cmgm.training.evaluate import compute_metrics
    for path in (args.d0b_checkpoint,args.five_day_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f'Required existing comparison checkpoint missing: {path}')
    if args.epochs < 1:
        raise ValueError('Grouped training requires at least one epoch')
    # Check reference identity before committing time to training; no forward.
    for ref_path,ref_variant,ref_objective in (
        (args.d0b_checkpoint,BASE_VARIANT,None),
        (args.five_day_checkpoint,FIVE_VARIANT,'4x_5d_only'),
    ):
        ref_meta = torch.load(ref_path,map_location='cpu',weights_only=True).get('metadata',{})
        if ref_meta.get('variant',ref_variant) != ref_variant:
            raise ValueError(f'{ref_path}: unexpected reference variant')
        if ref_objective is not None and ref_meta.get('objective') != ref_objective:
            raise ValueError('5d-only reference must have objective=4x_5d_only metadata')
    set_seed(args.seed)
    market=data['market_indices']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],
                          n_stock=market['stock'][1]-market['stock'][0],
                          n_bond=market['bond'][1]-market['bond'][0],feat_dim=config.FEATURE_DIM,variant=VARIANT)
    with torch.random.fork_rng(devices=[]):
        fixed=next(iter(data['loaders']['train']))
    initial=initialization_check(model,fixed,args.seed)
    model=model.to(device)
    model._experiment_seed=args.seed
    branch=model.switching_latent_transformer
    branch._initial_transition_logits=branch.regime_filter.transition_logits.detach().cpu().clone()
    sources=[Path(__file__),ROOT/'cmgm/training/train.py',ROOT/'cmgm/models/hetero_mixhop_model.py',
             ROOT/'cmgm/models/switching_latent_transformer.py',ROOT/'cmgm/scripts/main_ablation.py']
    metadata={'variant':VARIANT,'display_name':DISPLAY,'objective':GROUPED_OBJECTIVE,'seed':args.seed,
              'seq_len':args.seq_len,'initialization_check':initial,
              'initial_transition_logits':array(branch._initial_transition_logits).tolist(),
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
              'protocol':{'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
                          'lr':config.LEARNING_RATE,'weight_decay':config.WEIGHT_DECAY,'optimizer':'Adam',
                          'scheduler':'ReduceLROnPlateau','Huber_delta':config.HUBER_DELTA,
                          'horizons':config.MULTI_HORIZONS,'selection_objective':GROUPED_OBJECTIVE}}
    path=_checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    if path.resolve() in (args.d0b_checkpoint.resolve(),args.five_day_checkpoint.resolve()):
        raise ValueError('Grouped checkpoint must not overwrite either reference')
    start=time.time()
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata)
    payload=checked_load(model,path,device,GROUPED_OBJECTIVE)
    normalized,original,target=evaluate_primary_horizon(model,data['loaders']['test'],data,device)
    legacy=print_diagnostics(model,VARIANT,data['loaders'],device)
    output=args.grouped_report_dir/time.strftime('%Y%m%d_%H%M%S')
    report=checkpoint_report(model,payload,data,args.d0b_checkpoint,args.five_day_checkpoint,output,args.seed,initial,path)
    zero=compute_metrics(np.zeros_like(target),target)
    return {'variant':DISPLAY,'params':initial['Grouped_params'],'time':time.time()-start,
            'MAE':normalized['MAE'],'MSE':normalized['MSE'],'RMSE':normalized['RMSE'],'Hit_Ratio':normalized['Hit_Ratio'],
            'vs_zero_pct':(normalized['MAE']/zero['MAE']-1)*100,'mn':normalized,'mo':original,
            'diagnostics':legacy,'report_path':str(output/'REPORT.md'),'case':report['assessment']['case']}


def main():
    parser=argparse.ArgumentParser(description='Three-checkpoint grouped-objective diagnostics; no training')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--five-day-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_5d_only_best.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=config.RANDOM_SEED)
    parser.add_argument('--batch-size',type=int,default=config.BATCH_SIZE)
    parser.add_argument('--seq-len',type=int,default=config.SEQ_LEN)
    parser.add_argument('--no-cuda',action='store_true')
    args=parser.parse_args()
    set_seed(args.seed)
    device=torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    from cmgm.scripts.main_ablation import build_data
    data=build_data(SimpleNamespace(batch_size=args.batch_size,seq_len=args.seq_len,seed=args.seed))
    market=data['market_indices']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=market['stock'][1]-market['stock'][0],
                          n_bond=market['bond'][1]-market['bond'][0],variant=VARIANT).to(device)
    payload=checked_load(model,args.checkpoint,device,GROUPED_OBJECTIVE)
    checkpoint_report(model,payload,data,args.d0b_checkpoint,args.five_day_checkpoint,args.output,args.seed,
                      payload.get('metadata',{}).get('initialization_check'),args.checkpoint)


if __name__=='__main__':
    main()
