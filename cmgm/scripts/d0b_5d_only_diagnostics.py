"""Scale-matched 5d-only objective: training integration and checkpoint report.

This module creates only the registered D0B objective variant. Its model is
identical to D0B; uniform/forced controls are checkpoint diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import (
    FIVE_DAY_ONLY_VARIANT as VARIANT, FIVE_DAY_OBJECTIVE_MULTIPLIER,
    GROUPED_VARIANT, NO_SWITCH_KL_VARIANT,
    _prediction_loss, _horizon_loss_values, make_loss,
)
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0b_regime_routing_diagnostics import (
    ROOT, array, difference, evaluate_split, impacts, normal_reference,
    run_intervention, sha256,
)
# Reuse diagnostic math and contexts, not D0E model construction/training.
from cmgm.scripts.d0e_diagnostics import (
    BASE_VARIANT, CONTROLS, diagnostic_context, fixed_sanity, flattened_gradients,
    functional_values, gradient_groups, micro_diagnostics, model_arguments,
    transition_diagnostics,
)

DISPLAY = 'D0B-5dOnlyObjective'
PREFIX = 'D0B-5dOnly'
TRACE_KEYS = ('E', 'H', 'prior', 'p', 'candidates', 'Z', 'h_long', 'h_micro',
              'h_temporal', 'h_spatial', 'gate', 'prediction')


def assert_backbone(model):
    b = model.switching_latent_transformer
    assert model.variant in (BASE_VARIANT, VARIANT, GROUPED_VARIANT, NO_SWITCH_KL_VARIANT)
    assert b.balanced_readout and b.K == 3
    assert not any((b.use_dynamic_slope, b.use_balanced_transition_input,
                    b.use_latent_memory, b.use_regime_relative_memory,
                    b.regime_filter.learnable_sticky_alpha))
    assert b.regime_filter.sticky_alpha_value() == .5


def full_reference(model, x):
    """Capture fusion inputs/gate from the actual forward, without editing it."""
    captured = {}
    def capture_gate(module, inputs, output):
        spatial, temporal = inputs[0].chunk(2, dim=-1)
        captured.update(h_spatial=spatial.detach().clone(),
                        gate=output.sigmoid().detach().clone())
    handle = model.gate_fc.register_forward_hook(capture_gate)
    try:
        native, _ = normal_reference(model, x)
    finally:
        handle.remove()
    native.update(captured, E=model.switching_latent_transformer.last_market_tokens.clone())
    return native


def initialization_check(model, fixed_batch, seed=42):
    """Independent same-seed D0B; eval forward and fixed TRAIN loss scale."""
    assert_backbone(model)
    assert model.variant == VARIANT
    x, target = fixed_batch[:2]
    device = next(model.parameters()).device
    x, target = x.to(device), target.to(device)
    with diagnostic_context(model), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT, **model_arguments(model)).to(device).eval()
        old, new = dict(baseline.named_parameters()), dict(model.named_parameters())
        assert old.keys() == new.keys()
        assert all(old[k].shape == new[k].shape for k in old)
        diffs = {k: (old[k] - new[k]).abs().max().item() for k in old}
        before, after = full_reference(baseline, x), full_reference(model, x)
        trace_diffs = {k: difference(after[k], before[k]) for k in TRACE_KEYS}
        counts = [sum(p.numel() for p in m.parameters()) for m in (baseline, model)]
        losses = _horizon_loss_values(after['prediction'], target, make_loss())
        multi = sum(losses.values())
        scaled = _prediction_loss(model, after['prediction'], target, make_loss()).item()
        result = {'fixed_TRAIN_batch_shape': list(x.shape), 'seed': seed,
                  'D0B_params': counts[0], '5dOnly_params': counts[1], 'difference': counts[1]-counts[0],
                  'max_abs_diff': max(diffs.values()), 'mismatch_count': sum(v != 0 for v in diffs.values()),
                  'forward_differences': trace_diffs,
                  'loss_scale': {**{f'L{h}': v for h,v in losses.items()},
                                 'sum_multi': multi, '4x_L5': scaled,
                                 'ratio': scaled/multi if multi > 0 else None,
                                 'multiplier': FIVE_DAY_OBJECTIVE_MULTIPLIER,
                                 'description': 'scale-matched 5d-only diagnostic objective'}}
        result['PASS'] = (result['difference'] == 0 and result['mismatch_count'] == 0
                          and all(v['max'] <= 2e-6 for v in trace_diffs.values()))
        print(f'[{PREFIX} shared init] {json.dumps(result)}', flush=True)
        if not result['PASS']:
            raise AssertionError('5d-only differs from D0B before objective computation')
        return result


def horizon_gradients(model, batch, label=PREFIX):
    """Unscaled single-horizon Huber gradients; no .grad writes or optimizer."""
    b = model.switching_latent_transformer
    groups = gradient_groups(model)
    groups.update({'Market Encoder': list(b.market_encoder.parameters()),
                   'LongMemory': list(b.long_memory.layers.parameters()),
                   'Base RPE': [b.long_memory.base_rpe],
                   'state readout': list(b.state_readout.parameters())})
    x, y = [v.to(next(model.parameters()).device) for v in batch[:2]]
    vectors, norms, losses = {}, {}, {}
    with diagnostic_context(model), torch.enable_grad():
        for i,h in enumerate(config.MULTI_HORIZONS):
            loss = make_loss()(model(x)[:, i, :], y[:, i, :])
            flat = flattened_gradients(loss, groups)
            vectors[str(h)] = flat
            norms[str(h)] = {name: g.norm().item() for name,g in flat.items()}
            losses[str(h)] = loss.item()
    cosines = {name: {f'5d-vs-{h}d': F.cosine_similarity(vectors['5'][name], vectors[str(h)][name], dim=0, eps=1e-12).item()
                      for h in (1,10,20)}
               for name in ('regime evidence','transition logits','generators','balanced readouts')}
    result = {'norms': norms, 'cosines': cosines, 'raw_horizon_losses': losses,
              'five_day_multiplier': 1., 'LongMemory_definition': 'transformer layers; Base RPE listed separately',
              'method': 'eval single-horizon raw Huber; autograd.grad; no optimizer step'}
    print(f'[{label} {model.variant} horizon gradients] {json.dumps(result)}', flush=True)
    return result


def prediction_losses_from_arrays(prediction, target, batch_size):
    """Same mean-of-batch-means as validate_epoch; also retain sample means."""
    criterion = make_loss()
    p, y = torch.from_numpy(prediction), torch.from_numpy(target)
    rows = [_horizon_loss_values(p[i:i+batch_size],y[i:i+batch_size],criterion)
            for i in range(0,len(p),batch_size)]
    means = {h: float(np.mean([row[h] for row in rows])) for h in rows[0]}
    sample = _horizon_loss_values(p,y,criterion)
    return {'raw_L5': means['5'], 'scaled_5d_loss': 4. * means['5'],
            'sum_multi': sum(means.values()), 'per_horizon': means,
            'sample_mean_per_horizon': sample,
            'aggregation': 'mean of batch means, matching existing validate_epoch; TRAIN evaluation includes tail'}


def collect_model(model, payload, data, output, fixed, seed, label=PREFIX, extra_fixed=None):
    assert_backbone(model)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with diagnostic_context(model):
        with torch.no_grad():
            native = full_reference(model, fixed[0].to(device))
            extra = extra_fixed(model, native) if extra_fixed is not None else {}
            fixed_controls = {}
            for name, spec in CONTROLS.items():
                altered = run_intervention(model, native['h_spatial'], native, spec, None, None)
                torch.testing.assert_close(altered['p'],native['p'],rtol=0,atol=0)
                fixed_controls[name] = impacts(altered,native)
            micro = micro_diagnostics(model,native)
            gate = native['gate']
            gate_stats = {'mean': gate.mean().item(), 'std': gate.std().item(),
                          'min': gate.min().item(), 'max': gate.max().item(),
                          'definition': 'gate multiplies projected temporal; (1-gate) projected spatial'}
            np.savez_compressed(output/'fixed_native.npz', **{k:array(v) for k,v in native.items()},
                                X=array(fixed[0]), target=array(fixed[1]))
        splits = {}
        for split,loader in data['loaders'].items():
            name = split.upper()
            full = DataLoader(loader.dataset,batch_size=loader.batch_size,shuffle=False,drop_last=False)
            # TRAIN native metrics/regime/candidates are available; controls are
            # needed on VAL/TEST only. An empty spec avoids TRAIN routing-only mode.
            specs = {} if name == 'TRAIN' else CONTROLS
            print(f'[{label} {model.variant} {name}]', flush=True)
            result = evaluate_split(model,full,name,specs,device,seed,output)
            with np.load(output/f'{name.lower()}_predictions.npz') as values:
                result['prediction_losses'] = prediction_losses_from_arrays(values['native'], values['target'], full.batch_size)
            splits[name] = result
        return {'params': sum(p.numel() for p in model.parameters()),
                'best_epoch': payload.get('best_epoch'), 'metadata': payload.get('metadata',{}),
                'history': payload.get('history',{}), 'splits': splits,
                'transition': transition_diagnostics(model,payload.get('metadata',{}).get('initial_transition_logits')),
                'fixed': {'micro': micro, 'gate': gate_stats, 'controls': fixed_controls, **extra},
                'sanity': fixed_sanity(model,fixed[0].to(device),label=label),
                'gradients': horizon_gradients(model,fixed,label=label)}


def performance_comparison(models):
    result = {}
    for split in ('TRAIN','VAL','TEST'):
        rows = {}
        for h in map(str,config.MULTI_HORIZONS):
            baseline = models['D0B']['splits'][split]['native_metrics'][h]
            changed = models['5dOnly']['splits'][split]['native_metrics'][h]
            delta = changed['MAE']-baseline['MAE']
            rows[h] = {'D0B': baseline, '5dOnly': changed, 'delta_MAE': delta,
                       'relative_change': delta/baseline['MAE'] if baseline['MAE'] > 0 else None}
        result[split] = rows
    return result


def case_assessment(comparison, sane=True):
    """One descriptive case; ~0.1% is reporting convention, never tuning."""
    val, test = [comparison[s]['5']['relative_change'] for s in ('VAL','TEST')]
    if val is None or test is None or not sane:
        return {'case': 'Case C', 'next_5_10_20_eligible': False,
                'reason': 'No robust evidence: invalid baseline or mechanism sanity failure.'}
    if val < 0 and test < 0:
        case = 'Case A' if max(val,test) <= -.001 else 'Case B'
        reason = ('Both splits improve by at least 0.1%; supports negative transfer in this controlled run.'
                  if case == 'Case A' else 'Both splits improve, but at least one improves by <0.1%; not a major demonstrated bottleneck.')
    elif val > 0 and test > 0:
        case,reason = 'Case D','Auxiliary horizon supervision benefits 5d generalization in this comparison.'
    elif val < 0 and test > 0:
        case,reason = 'Case E','Validation specialization fails to transfer to TEST; retain D0B.'
    else:
        case,reason = 'Case C','No robust two-split generalization evidence; TEST-only improvement does not establish negative transfer.'
    return {'case':case,'reason':reason,'next_5_10_20_eligible':case=='Case A',
            'materiality_convention_relative_MAE':.001,
            'note':'Single seed, approximate loss-scale matching and changed selection objective; not statistical proof. No follow-up run is launched.'}


def checkpoint_report(model,payload,data,d0b_checkpoint,output,seed=42,initialization=None,checkpoint_path=None):
    assert model.variant == VARIANT
    output.mkdir(parents=True,exist_ok=False)
    hashes = {'D0B':sha256(d0b_checkpoint)}
    if checkpoint_path is not None:
        hashes['5dOnly'] = sha256(checkpoint_path)
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    with diagnostic_context(model):
        fixed = next(iter(data['loaders']['test']))
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(next(model.parameters()).device)
        base_payload = _load_checkpoint(baseline,Path(d0b_checkpoint),next(model.parameters()).device)
        for active,saved in ((baseline,base_payload),(model,payload)):
            active.switching_latent_transformer.set_epoch(saved.get('best_epoch',1))
        results = {label:collect_model(active,saved,data,output/label,fixed,seed)
                   for label,active,saved in (('D0B',baseline,base_payload),('5dOnly',model,payload))}
        with np.load(output/'D0B/fixed_native.npz') as left, np.load(output/'5dOnly/fixed_native.npz') as right:
            drift = {k:difference(torch.from_numpy(right[k]),torch.from_numpy(left[k])) for k in TRACE_KEYS}
        specialization = {}
        for split in ('TRAIN','VAL','TEST'):
            file = f'{split.lower()}_predictions.npz'
            with np.load(output/'D0B'/file) as left, np.load(output/'5dOnly'/file) as right:
                np.testing.assert_array_equal(left['target'],right['target'])
                specialization[split] = {str(h):float(np.abs(right['native'][:,i]-left['native'][:,i]).mean())
                                         for i,h in enumerate(config.MULTI_HORIZONS)}
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert sha256(d0b_checkpoint) == hashes['D0B']
    if checkpoint_path is not None:
        assert sha256(checkpoint_path) == hashes['5dOnly']
    comparison = performance_comparison(results)
    report = {'objective':'scale-matched 5d-only diagnostic objective',
              'checkpoint_paths':{'D0B':str(Path(d0b_checkpoint).resolve()),
                                  '5dOnly':str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None},
              'checkpoint_sha256':hashes,'metadata':payload.get('metadata',{}),
              'seed':seed,'fixed_TEST_batch_shape':list(fixed[0].shape),
              'initialization':initialization,'models':results,'performance_comparison':comparison,
              'representation_drift':drift,'prediction_specialization':specialization,
              'case_assessment':case_assessment(comparison,all(m['sanity']['PASS'] for m in results.values())),
              'integrity':{'diagnostic_parameters_unchanged':True,'checkpoint_files_unchanged':True}}
    (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    write_report(report,output/'REPORT.md')
    print(f'[{PREFIX} REPORT] {output / "REPORT.md"}',flush=True)
    return report


def run_five_day_only(args,device,data):
    """Exactly one run of the existing train(); same optimizer/protocol."""
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant, evaluate_primary_horizon, print_diagnostics
    from cmgm.training.train import train
    from cmgm.training.evaluate import compute_metrics
    if not args.d0b_checkpoint.is_file():
        raise FileNotFoundError(f'D0B reference checkpoint missing: {args.d0b_checkpoint}')
    if args.epochs < 1:
        raise ValueError('Training requires at least one epoch')
    set_seed(args.seed)
    market = data['market_indices']
    model = HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],
                            n_stock=market['stock'][1]-market['stock'][0],
                            n_bond=market['bond'][1]-market['bond'][0],
                            feat_dim=config.FEATURE_DIM,variant=VARIANT)
    with torch.random.fork_rng(devices=[]):
        fixed = next(iter(data['loaders']['train']))
    initial = initialization_check(model,fixed,args.seed)
    model = model.to(device)
    model._experiment_seed = args.seed
    branch = model.switching_latent_transformer
    branch._initial_transition_logits = branch.regime_filter.transition_logits.detach().cpu().clone()
    source_files = [Path(__file__),ROOT/'cmgm/training/train.py',ROOT/'cmgm/models/hetero_mixhop_model.py',
                    ROOT/'cmgm/models/switching_latent_transformer.py',ROOT/'cmgm/scripts/main_ablation.py']
    metadata = {'variant':VARIANT,'display_name':DISPLAY,'objective':'4x_5d_only',
                'seed':args.seed,'seq_len':args.seq_len,'initialization_check':initial,
                'initial_transition_logits':array(branch._initial_transition_logits).tolist(),
                'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source_files},
                'protocol':{'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
                            'lr':config.LEARNING_RATE,'weight_decay':config.WEIGHT_DECAY,
                            'optimizer':'Adam','scheduler':'ReduceLROnPlateau','Huber_delta':config.HUBER_DELTA,
                            'horizons':config.MULTI_HORIZONS,'selection_objective':'4x_5d_only'}}
    path = _checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    start = time.time()
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata)
    payload = _load_checkpoint(model,path,device)
    normalized,original,target = evaluate_primary_horizon(model,data['loaders']['test'],data,device)
    legacy = print_diagnostics(model,VARIANT,data['loaders'],device)
    output = args.five_day_report_dir / time.strftime('%Y%m%d_%H%M%S')
    report = checkpoint_report(model,payload,data,args.d0b_checkpoint,output,args.seed,initial,path)
    zero = compute_metrics(np.zeros_like(target),target)
    return {'variant':DISPLAY,'params':initial['5dOnly_params'],'time':time.time()-start,
            'MAE':normalized['MAE'],'RMSE':normalized['RMSE'],'Hit_Ratio':normalized['Hit_Ratio'],
            'vs_zero_pct':(normalized['MAE']/zero['MAE']-1)*100,'mn':normalized,'mo':original,
            'diagnostics':legacy,'report_path':str(output/'REPORT.md'),
            'case':report['case_assessment']['case']}


def main():
    parser = argparse.ArgumentParser(description='D0B 5d-only checkpoint comparison; no training')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=config.BATCH_SIZE)
    parser.add_argument('--seq-len',type=int,default=config.SEQ_LEN)
    parser.add_argument('--seed',type=int,default=config.RANDOM_SEED)
    parser.add_argument('--no-cuda',action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=args.batch_size,seq_len=args.seq_len,seed=args.seed))
    market = data['market_indices']
    model = HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=market['stock'][1]-market['stock'][0],
                            n_bond=market['bond'][1]-market['bond'][0],variant=VARIANT).to(device)
    payload = _load_checkpoint(model,args.checkpoint,device)
    if payload.get('metadata',{}).get('objective') != '4x_5d_only':
        raise ValueError('Checkpoint metadata must identify objective=4x_5d_only; D0B has identical state keys')
    checkpoint_report(model,payload,data,args.d0b_checkpoint,args.output,args.seed,
                      payload['metadata'].get('initialization_check'),args.checkpoint)


def write_report(report,path):
    # Defined below independently of the training/forward paths.
    from cmgm.scripts.d0b_5d_only_report import write_report as render
    render(report,path)


if __name__ == '__main__':
    main()
