"""One D0B horizon-state-readout experiment; preflight and immutable checkpoint probes."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import HORIZON_READOUT_VARIANT as VARIANT, _prediction_loss, make_loss
from cmgm.training.metric_standard import STANDARD, population_metrics
from cmgm.scripts.d0b_5d_only_diagnostics import (
    BASE_VARIANT, TRACE_KEYS, assert_backbone, full_reference, prediction_losses_from_arrays,
)
from cmgm.scripts.d0b_regime_routing_diagnostics import (
    ROOT, array, difference, complete_readout, horizon_metrics, probability_stats,
    specialization, sha256,
)
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0e_diagnostics import (
    diagnostic_context, model_arguments, fixed_sanity, micro_diagnostics, transition_diagnostics,
)
from cmgm.scripts.d0b_grouped_diagnostics import extra_fixed_diagnostics

DISPLAY = 'D0B-HorizonSpecificStateReadout'
HORIZONS = (1, 5, 10, 20)
NEW = 'HorizonSpecific'


def is_horizon(model):
    return model.switching_latent_transformer.horizon_specific_state_readout


def readout_modules(model):
    b = model.switching_latent_transformer
    return {str(h): (b.state_readout if h == 5 or not is_horizon(model)
                     else b.horizon_state_readouts[str(h)]) for h in HORIZONS}


def initialization_check(model, batch, seed=42):
    assert model.variant == VARIANT
    assert_backbone(model)
    x, y = [v.to(next(model.parameters()).device) for v in batch[:2]]
    with diagnostic_context(model), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline = HeteroMixHopCMGM(variant=BASE_VARIANT, **model_arguments(model)).to(x.device).eval()
        old, new = dict(baseline.named_parameters()), dict(model.named_parameters())
        shared = {k: difference(new[k], p)['max'] for k, p in old.items()}
        extra = set(new) - set(old)
        expected = {f'switching_latent_transformer.horizon_state_readouts.{h}.{k}'
                    for h in (1, 10, 20) for k in ('weight', 'bias')}
        assert extra == expected
        count = [sum(p.numel() for p in m.parameters()) for m in (baseline, model)]
        before, after = full_reference(baseline, x), full_reference(model, x)
        traces = {k: difference(after[k], before[k]) for k in (*TRACE_KEYS, 'evidence')
                  if k not in ('gate', 'h_temporal')}
        for key in ('gate', 'h_temporal'):
            for i, h in enumerate(HORIZONS):
                traces[f'{key}_{h}'] = difference(after[key][:, i], before[key])
        clone = {h: {k: difference(getattr(m, k), getattr(model.switching_latent_transformer.state_readout, k))['max']
                     for k in ('weight', 'bias')} for h, m in readout_modules(model).items()}
        losses = {}
        for name, active, trace in [('D0B', baseline, before), (NEW, model, after)]:
            branch = active.switching_latent_transformer
            raw = [make_loss()(trace['prediction'][:, i], y[:, i]).item() for i in range(4)]
            # Probe fully active KL as well as the epoch-1 zero-beta objective.
            kl = branch.regime_filter._last_switch_loss.item()
            pred_loss = _prediction_loss(active, trace['prediction'], y, make_loss()).item()
            saved_epoch = branch.regime_filter.current_epoch
            branch.set_epoch(20)
            weighted_kl = branch.switch_loss().item()
            branch.set_epoch(saved_epoch)
            losses[name] = {'raw_horizons': raw, 'prediction': pred_loss, 'raw_KL': kl,
                            'switch_epoch1': 0., 'switch_epoch20': weighted_kl,
                            'total_epoch1': pred_loss, 'total_epoch20': pred_loss + weighted_kl}
        loss_diff = {k: abs(losses[NEW][k] - losses['D0B'][k]) for k in losses['D0B'] if k != 'raw_horizons'}
        result = {'seed': seed, 'fixed_TRAIN_shape': list(x.shape), 'prediction_shape': list(after['prediction'].shape),
                  'D0B_params': count[0], 'new_params': count[1], 'difference': count[1]-count[0],
                  'expected_difference': 24768, 'shared_parameter_count': len(shared),
                  'shared_parameter_numel': count[0], 'max_abs_diff': max(shared.values()),
                  'mismatch_count': sum(v != 0 for v in shared.values()), 'clone_differences': clone,
                  'forward_differences': traces, 'losses': losses, 'loss_differences': loss_diff}
        result['PASS'] = (result['difference'] == 24768 and result['mismatch_count'] == 0
                          and all(v == 0 for r in clone.values() for v in r.values())
                          and max(v['max'] for v in traces.values()) <= 2e-6
                          and max(loss_diff.values()) <= 2e-8)
        if not result['PASS']:
            raise AssertionError(result)
        print(f'[{DISPLAY} shared init] ' + json.dumps(result), flush=True)
        return result


def precision_sanity(model, x):
    with diagnostic_context(model):
        low = fixed_sanity(model, x, label=DISPLAY, raise_on_failure=False)
        if low['PASS']:
            return low
        if max([*low['causality'].values(), low['batch_permutation_max'], low['single_sample_max']]) > 3e-6:
            raise AssertionError(low)
        # Retain the FP32 result; audit roundoff on a separate FP64 model only.
        high_model = HeteroMixHopCMGM(variant=model.variant, **model_arguments(model)).to(device=x.device, dtype=torch.float64).eval()
        high_model.load_state_dict(model.state_dict(), strict=True)
        high = fixed_sanity(high_model, x.double(), label=DISPLAY+' FP64 audit')
        vals = [*high['causality'].values(), high['batch_permutation_max'], high['single_sample_max'],
                *[v for row in high['within_market'].values() for v in row.values()]]
        if max(vals) > 1e-10:
            raise AssertionError(high)
        return {'PASS': True, 'float32': low, 'float64_copy': high, 'audit_tolerance': 1e-10,
                'note': 'FP32 market-permutation roundoff audited on FP64 copy; original model/RNG untouched.'}


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    denom = a.norm() * b.norm()
    return float(torch.dot(a, b) / denom) if denom > 1e-30 else None


def gradient_probe(model, batch):
    """Raw per-horizon Huber, real shared u graph, no .grad mutation or step."""
    b = model.switching_latent_transformer
    x, y = [v.to(next(model.parameters()).device) for v in batch[:2]]
    groups = {
        'Market Encoder': list(b.market_encoder.parameters()),
        'LongMemory Transformer': list(b.long_memory.parameters()),
        'Base RPE': [b.long_memory.base_rpe],
        'regime evidence': list(b.regime_filter.regime_evidence.parameters()),
        'transition logits': [b.regime_filter.transition_logits],
        **{f'G{k}': list(g.parameters()) for k, g in enumerate(b.latent_transition.generators)},
        'generators combined': list(b.latent_transition.parameters()),
        'long_memory_readout': list(b.long_memory_readout.parameters()) + list(b.long_memory_norm.parameters()),
        'micro_state_readout': list(b.micro_state_readout.parameters()) + list(b.micro_state_norm.parameters()),
        'shared gate_fc': list(model.gate_fc.parameters()),
        'shared lstm_proj': list(model.lstm_proj.parameters()),
        'shared gcn_proj': list(model.gcn_proj.parameters()),
        'shared head body': list(model.head[:3].parameters()),
        'final output layer': list(model.head[3].parameters()),
        **{f'readout_{h}': list(m.parameters()) for h, m in readout_modules(model).items()},
    }
    groups['Balanced H/Z projections'] = groups['long_memory_readout'] + groups['micro_state_readout']
    parameters = list(model.parameters())
    slots = {id(p): i for i, p in enumerate(parameters)}
    captured = []
    vectors, u_vectors, own_weights, norms, loss_values = {}, {}, {}, {}, {}
    with diagnostic_context(model):
        handle = b.state_readout.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        try:
            pred = model(x)
        finally:
            handle.remove()
        assert len(captured) == 1
        u = captured[0]
        for i, h in enumerate(HORIZONS):
            loss = make_loss()(pred[:, i], y[:, i])
            grad = torch.autograd.grad(loss, parameters + [u], allow_unused=True, retain_graph=i < 3)
            vectors[str(h)] = {
                name: torch.cat([(torch.zeros_like(p) if grad[slots[id(p)]] is None else grad[slots[id(p)]]).detach().reshape(-1)
                                 for p in params]).cpu() for name, params in groups.items()}
            u_vectors[str(h)] = grad[-1].detach().cpu()
            own = readout_modules(model)[str(h)].weight
            own_weights[str(h)] = grad[slots[id(own)]].detach().cpu()
            norms[str(h)] = {name: float(v.norm()) for name, v in vectors[str(h)].items()}
            loss_values[str(h)] = float(loss.detach())
    cosines = {name: {str(h): cosine(vectors['5'][name], vectors[str(h)][name]) for h in (1, 10, 20)} for name in groups}
    isolation = {h: {other: norms[h][f'readout_{other}'] for other in map(str, HORIZONS)} for h in map(str, HORIZONS)}
    if is_horizon(model):
        assert all(v == 0 for h, row in isolation.items() for other, v in row.items() if h != other)
    return {'method': 'eval; raw single-horizon Huber delta=.02; autograd.grad; no optimizer step',
            'norms': norms, 'losses': loss_values, 'cosines_vs_5d': cosines,
            'readout_loss_gradient_matrix': isolation, 'readout_isolation_PASS': True if is_horizon(model) else None,
            'own_readout_weight_cosines_vs_5d': {str(h): cosine(own_weights['5'], own_weights[str(h)]) for h in (1, 10, 20)},
            'u_gradient': {'norms': {h: float(v.norm()) for h,v in u_vectors.items()},
                           'cosines_vs_5d': {str(h): cosine(u_vectors['5'], u_vectors[str(h)]) for h in (1, 10, 20)}},
            'interpretation': 'Negative cosine is descriptive local geometry, not proof of negative transfer.'}


def readout_statistics(model, initial=None):
    modules = readout_modules(model)
    anchor = modules['5'].weight.detach()
    rows = {}
    for h, module in modules.items():
        w, bias = module.weight.detach(), module.bias.detach()
        rows[h] = {'weight_norm': float(w.norm()), 'bias_norm': float(bias.norm()),
                   'distance_from_5d': float((w-anchor).norm()),
                   'relative_distance_from_5d': float((w-anchor).norm() / (anchor.norm()+1e-8)),
                   'weight_distance_from_init': None if initial is None else float((w-w.new_tensor(initial['weight'])).norm()),
                   'bias_distance_from_init': None if initial is None else float((bias-bias.new_tensor(initial['bias'])).norm())}
    return {'horizons': rows, 'pairwise_weight_cosine': {f'{i}-{j}': cosine(modules[str(i)].weight, modules[str(j)].weight)
                                                     for i,j in itertools.combinations(HORIZONS,2)},
            'init_source': 'recorded exact seed-42 initialization' if initial is not None else 'unavailable'}


def temporal_statistics(temporal, gate):
    temporal, gate = temporal.astype(np.float64), gate.astype(np.float64)
    def pair(a,b):
        diff = a-b
        denom = np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
        return {'mean_abs':float(np.abs(diff).mean()), 'L1':float(np.abs(diff).sum(-1).mean()),
                'L2':float(np.linalg.norm(diff,axis=-1).mean()), 'cosine':float(((a*b).sum(-1)/denom).mean())}
    five = HORIZONS.index(5)
    return {'horizons':{str(h):{'mean_norm':float(np.linalg.norm(temporal[:,i],axis=-1).mean()),
                              'vs_5d':pair(temporal[:,i],temporal[:,five]),
                              'gate':{'mean':float(gate[:,i].mean()),'std':float(gate[:,i].std()),
                                      'min':float(gate[:,i].min()),'max':float(gate[:,i].max())}}
                        for i,h in enumerate(HORIZONS)},
            'pairwise':{f'{a}-{b}':pair(temporal[:,HORIZONS.index(a)],temporal[:,HORIZONS.index(b)])
                        for a,b in itertools.combinations(HORIZONS,2)}}


@torch.no_grad()
def collect_split(model, loader, split, output, names):
    """Full population including tail; H/p/Z once per native sample batch."""
    full = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, drop_last=False)
    values = {k:[] for k in ('target','native','zero-micro','zero-long','shared-readout')}
    cache = {k:[] for k in ('p','prior','candidates','temporal','gate')}
    micro, rpe, counts = [], [], []
    for batch in full:
        x, target = batch[:2]
        trace = full_reference(model, x.to(next(model.parameters()).device))
        counts.append(len(x))
        values['target'].append(array(target)); values['native'].append(array(trace['prediction']))
        for k in ('p','prior','candidates'):
            cache[k].append(array(trace[k]))
        for k, source in [('temporal','h_temporal'),('gate','gate')]:
            v = trace[source]
            cache[k].append(array(v if v.dim()==3 else v[:,None,:].expand(-1,4,-1)))
        micro.append(micro_diagnostics(model, trace))
        rpe.append(extra_fixed_diagnostics(model, trace)['base_rpe'])
        for name, component in [('zero-micro','Z'),('zero-long','H')]:
            changed = complete_readout(model,trace['h_spatial'],trace,component)
            values[name].append(array(changed['prediction']))
        if is_horizon(model):
            b = model.switching_latent_transformer
            temporal = b.readout_by_horizon(trace['H'][:,-1],trace['Z'][:,-1],shared_readout=True)
            shared = model._market_token_predict_by_horizon(trace['h_spatial'],temporal)
        else:
            shared = trace['prediction']
        torch.testing.assert_close(shared[:,HORIZONS.index(5)],trace['prediction'][:,HORIZONS.index(5)],atol=0,rtol=0)
        values['shared-readout'].append(array(shared))
        print(f'[{DISPLAY} {model.variant} {split}] {sum(counts)}/{len(full.dataset)}',flush=True)
    arrays = {k:np.concatenate(v) for k,v in values.items()}
    c = {k:np.concatenate(v) for k,v in cache.items()}
    np.savez_compressed(output/f'{split.lower()}_predictions.npz',**arrays)
    # Keep per-sample horizon representations available for audit.
    np.savez_compressed(output/f'{split.lower()}_readouts.npz',temporal=c['temporal'],gate=c['gate'])
    impacts = {mode:{str(h):{'mean':float(np.abs(arrays['native'][:,i]-arrays[mode][:,i]).mean()),
                               'max':float(np.abs(arrays['native'][:,i]-arrays[mode][:,i]).max())}
                     for i,h in enumerate(HORIZONS)} for mode in ('zero-micro','zero-long','shared-readout')}
    commodity = [{ 'name':str(name), **population_metrics(arrays['native'][:,HORIZONS.index(5),i],arrays['target'][:,HORIZONS.index(5),i])}
                 for i,name in enumerate(names)]
    order = sorted(commodity,key=lambda row:row['MSE'])
    mse_values = np.array([v['MSE'] for v in commodity])
    return {'samples':sum(counts),'native_metrics':horizon_metrics(arrays['native'],arrays['target']),
            'control_metrics':{k:horizon_metrics(arrays[k],arrays['target']) for k in ('zero-micro','zero-long','shared-readout')},
            'prediction_losses':prediction_losses_from_arrays(arrays['native'],arrays['target'],full.batch_size),
            'normal_regime':probability_stats(c['p'],prior=c['prior']),
            'candidate_specialization':specialization(c['candidates'],c['p']),
            'micro':{k:float(np.average([r[k] for r in micro],weights=counts)) for k,v in micro[0].items() if isinstance(v,(int,float))},
            'base_rpe':{k:float(np.average([r[k] for r in rpe],weights=counts)) for k,v in rpe[0].items() if isinstance(v,(int,float))},
            'temporal_specialization':temporal_statistics(c['temporal'],c['gate']),
            'impacts':impacts,'micro_long_ratio':{str(h):impacts['zero-micro'][str(h)]['mean']/(impacts['zero-long'][str(h)]['mean']+1e-8) for h in HORIZONS},
            'commodity_5d':commodity,'highest_MSE_commodities':order[-5:][::-1], 'lowest_MSE_commodities':order[:5],
            'commodity_MSE_CV':float(mse_values.std()/(mse_values.mean()+1e-12))}


def validate_protocol(args):
    expected = {'seed':42,'batch_size':64,'seq_len':20,'epochs':200,'patience':10}
    if any(getattr(args,k)!=v for k,v in expected.items()):
        raise ValueError(f'One controlled run requires {expected}')
    if (config.MULTI_HORIZONS!=list(HORIZONS) or config.TARGET_HORIZON!=5 or
        config.LEARNING_RATE!=1e-4 or config.WEIGHT_DECAY!=1e-5 or
        config.LOSS_TYPE!='huber' or config.HUBER_DELTA!=.02):
        raise ValueError('Original D0B training protocol must be retained')


def make_model(data, variant=VARIANT):
    m=data['market_indices']
    return HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=m['stock'][1]-m['stock'][0],
                           n_bond=m['bond'][1]-m['bond'][0],feat_dim=config.FEATURE_DIM,variant=variant)


def prepare(args, data, device):
    validate_protocol(args)
    set_seed(args.seed)
    model=make_model(data)
    with torch.random.fork_rng(devices=[]):
        fixed=next(iter(data['loaders']['train']))
    initial=initialization_check(model,fixed,args.seed)
    if initial['D0B_params']!=520549 or initial['new_params']!=545317:
        raise ValueError('Architecture baseline parameter count changed')
    b=model.switching_latent_transformer
    metadata={'variant':VARIANT,'display_name':DISPLAY,'seed':args.seed,
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'parameter_count':initial['new_params'],'initialization':initial,
              'initial_state_readout':{k:v.detach().cpu().tolist() for k,v in b.state_readout.state_dict().items()},
              'initial_transition_logits':b.regime_filter.transition_logits.detach().cpu().tolist(),
              'horizon_specific_state_readout':True,'forecast_horizons':list(HORIZONS),
              '5d_uses_original_state_readout':True,'extra_readout_parameter_count':24768,
              'loss_type':'huber','delta':.02,'switch_beta_max':5e-4,'warmup_epochs':20,
              'objective':'sum_1d_5d_10d_20d','metric_standard':STANDARD,
              'protocol':{'seed':42,'batch_size':64,'seq_len':20,'epochs':200,'patience':10,
                          'optimizer':'Adam','lr':1e-4,'weight_decay':1e-5,'scheduler':'ReduceLROnPlateau',
                          'validation_selection':'prediction-only four-horizon sum; existing mean of batch means'},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
                 ROOT/'cmgm/models/switching_latent_transformer.py',ROOT/'cmgm/models/hetero_mixhop_model.py',
                 ROOT/'cmgm/training/train.py',ROOT/'cmgm/training/metric_standard.py']}}
    return model.to(device),metadata


def performance_comparison(models):
    return {s:{str(h):{'D0B':models['D0B']['splits'][s]['native_metrics'][str(h)],
                       NEW:models[NEW]['splits'][s]['native_metrics'][str(h)],
                       **{f'{kind}_{metric}':((models[NEW]['splits'][s]['native_metrics'][str(h)][metric]-models['D0B']['splits'][s]['native_metrics'][str(h)][metric]) /
                          (models['D0B']['splits'][s]['native_metrics'][str(h)][metric] if kind=='relative' else 1))
                          for metric in ('MAE','MSE') for kind in ('delta','relative')}}
                  for h in HORIZONS} for s in ('TRAIN','VAL','TEST')}


def case_assessment(comparison, specialization_evidence, healthy=True):
    """Fixed descriptive conventions, not tuned thresholds or significance tests."""
    mae=[comparison[s]['5']['relative_MAE'] for s in ('VAL','TEST')]
    mse=[comparison[s]['5']['relative_MSE'] for s in ('VAL','TEST')]
    negligible=.001  # ~0.1% relative; retain raw deltas for human interpretation.
    material_mse=.001
    if all(abs(v)<negligible for v in mae+mse):
        case,reason='Case E','MAE and MSE effectively tied; no material readout bottleneck demonstrated.'
    elif all(v<0 for v in mae) and any(v>material_mse for v in mse):
        case,reason='Case F','MAE improves with a material MSE tail-error tradeoff; retain D0B.'
    elif all(v<0 for v in mae) and healthy:
        if specialization_evidence:
            case,reason='Case A','Both 5d MAEs improve without material MSE harm, with weight and functional specialization evidence.'
        else:
            case,reason='Case B','Performance gain without strong specialization evidence.'
    elif all(v>0 for v in mae) and specialization_evidence:
        case,reason='Case C','Specialization accompanies worse generalization; shared decoder regularization was beneficial.'
    else:
        case,reason='Case D','No robust generalization evidence across VAL/TEST or mechanism health; retain D0B.'
    return {'case':case,'reason':reason,'candidate_for_human_review':case=='Case A',
            'MAE_relative_VAL_TEST':mae,'MSE_relative_VAL_TEST':mse,
            'negligible_relative_convention':negligible,'material_MSE_relative_convention':material_mse,
            'healthy':healthy,'specialization_evidence':specialization_evidence,
            'note':'Single-seed descriptive result; no statistical significance claim, automatic replacement, or next experiment.'}


def checkpoint_report(model, payload, data, d0b_path, output, checkpoint_path, seed=42):
    output.mkdir(parents=True,exist_ok=False)
    paths={'D0B':Path(d0b_path),NEW:Path(checkpoint_path)}
    hashes={k:sha256(p) for k,p in paths.items()}
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    models={}
    names=data['feature_names'][slice(*data['market_indices']['commodity'])]
    with diagnostic_context(model):
        fixed=next(iter(data['loaders']['test']))
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(next(model.parameters()).device)
        base_payload=_load_checkpoint(baseline,paths['D0B'],next(model.parameters()).device)
        for label,active,record in [('D0B',baseline,base_payload),(NEW,model,payload)]:
            assert_backbone(active)
            active.switching_latent_transformer.set_epoch(record.get('best_epoch',1))
            directory=output/label;directory.mkdir()
            with diagnostic_context(active):
                sanity=precision_sanity(active,fixed[0].to(next(active.parameters()).device))
                with torch.no_grad():
                    trace=full_reference(active,fixed[0].to(next(active.parameters()).device))
                    np.savez_compressed(directory/'fixed_native.npz',**{k:array(v) for k,v in trace.items()},X=array(fixed[0]),target=array(fixed[1]))
                splits={s.upper():collect_split(active,loader,s.upper(),directory,names) for s,loader in data['loaders'].items()}
                metadata=record.get('metadata',{})
                # At construction both had the exact same original readout.
                init=payload.get('metadata',{}).get('initial_state_readout')
                models[label]={'params':sum(p.numel() for p in active.parameters()),'best_epoch':record.get('best_epoch'),
                               'train_time':metadata.get('training_elapsed_seconds',record.get('history',{}).get('training_elapsed_seconds')),
                               'history':record.get('history',{}),'metadata':metadata,'sanity':sanity,'splits':splits,
                               'transition':transition_diagnostics(active,metadata.get('initial_transition_logits')),
                               'readouts':readout_statistics(active,init),'gradients':gradient_probe(active,fixed)}
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(sha256(paths[k])==v for k,v in hashes.items())
    comparison=performance_comparison(models)
    maximum_weight_distance=max(v['relative_distance_from_5d'] for v in models[NEW]['readouts']['horizons'].values())
    functional={s:max(models[NEW]['splits'][s]['impacts']['shared-readout'][str(h)]['mean'] /
                      (models[NEW]['splits'][s]['native_metrics'][str(h)]['MAE']+1e-12) for h in (1,10,20)) for s in ('VAL','TEST')}
    # Fixed descriptive conventions; show their raw evidence in report.
    evidence=maximum_weight_distance>.001 and all(v>.001 for v in functional.values())
    health={s: not (max(r['normal_regime']['occupancy'])>=.99 and r['normal_regime']['mean_max']>=.99
                   and r['normal_regime']['entropy']<=.1) for s,r in models[NEW]['splits'].items()}
    report={'display_name':DISPLAY,'variant':VARIANT,'metric_standard':STANDARD,'metadata':payload.get('metadata',{}),
            'checkpoint_paths':{k:str(p.resolve()) for k,p in paths.items()},'checkpoint_sha256':hashes,
            'fixed_TEST_shape':list(fixed[0].shape),'models':models,'comparison':comparison,
            'specialization_evidence':{'max_weight_relative_distance':maximum_weight_distance,
                'max_counterfactual_impact_over_native_MAE':functional,'descriptive_threshold':.001},
            'regime_health_no_extreme_concentration':health,
            'assessment':case_assessment(comparison,evidence,all(health.values())),
            'integrity':{'parameters_unchanged':True,'checkpoint_files_unchanged':True,
                         'max_RMSE_squared_minus_MSE':max(abs(v['RMSE']**2-v['MSE']) for m in models.values() for s in m['splits'].values() for v in s['native_metrics'].values())}}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    from cmgm.scripts.d0b_horizon_readout_report import write_report
    write_report(report,output/'REPORT.md')
    print(f'[{DISPLAY} REPORT] {output / "REPORT.md"}',flush=True)
    return report


def run_horizon_readout(args,device,data):
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant
    from cmgm.training.train import train
    validate_protocol(args)
    if not args.d0b_checkpoint.is_file():
        raise FileNotFoundError(args.d0b_checkpoint)
    path=_checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    if path.exists():
        raise FileExistsError(f'{path} exists; use checkpoint-only diagnostics; no overwrite or repeated training')
    model,metadata=prepare(args,data,device)
    with diagnostic_context(model):
        reference=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(device)
        _load_checkpoint(reference,args.d0b_checkpoint,device)
        metadata['D0B_checkpoint_sha256']=sha256(args.d0b_checkpoint)
        del reference
        fixed=next(iter(data['loaders']['test']))
        metadata['fixed_TEST_sanity']=precision_sanity(model,fixed[0].to(device))
    output=args.horizon_readout_report_dir/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    (output/'preflight.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    # Exactly one new training call; D0B checkpoint is reference only.
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata)
    payload=_load_checkpoint(model,path,device)
    report=checkpoint_report(model,payload,data,args.d0b_checkpoint,output/'checkpoint_diagnostics',path,args.seed)
    from cmgm.training.evaluate import compute_metrics,inverse_transform_predictions
    values=np.load(output/'checkpoint_diagnostics'/NEW/'test_predictions.npz')
    idx=HORIZONS.index(5);p,y=values['native'][:,idx],values['target'][:,idx]
    mn=compute_metrics(p,y)
    po,yo=inverse_transform_predictions(p,y,data['norm_stats'],data['raw_prices_test'],data['market_indices'],target_type=config.TARGET_TYPE)
    return {'variant':DISPLAY,'params':report['models'][NEW]['params'],'time':payload['history']['training_elapsed_seconds'],
            **{k:mn[k] for k in ('MAE','MSE','RMSE','Hit_Ratio')},'vs_zero_pct':(mn['MAE']/float(np.abs(y).mean())-1)*100,
            'mn':mn,'mo':compute_metrics(po,yo),'report_path':str(output/'checkpoint_diagnostics'/'REPORT.md')}


def main():
    parser=argparse.ArgumentParser(description='Horizon state readout preflight or checkpoint-only diagnostics; training via main_ablation')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--no-cuda',action='store_true');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    args.seed=42;args.batch_size=64;args.seq_len=20;args.epochs=200;args.patience=10
    if args.output.exists():raise FileExistsError(args.output)
    device=torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    from cmgm.scripts.main_ablation import build_data
    data=build_data(args)
    if args.preflight:
        model,metadata=prepare(args,data,device)
        with diagnostic_context(model):
            fixed=next(iter(data['loaders']['test']))
            metadata['fixed_TEST_shape']=list(fixed[0].shape)
            metadata['fixed_TEST_sanity']=precision_sanity(model,fixed[0].to(device))
            metadata['initial_gradients']=gradient_probe(model,fixed)
        args.output.mkdir(parents=True)
        (args.output/'preflight.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        from cmgm.scripts.d0b_horizon_readout_report import write_preflight
        write_preflight(metadata,args.output/'REPORT.md')
        print(f'[{DISPLAY} preflight] PASS; no training; {args.output}',flush=True)
    else:
        model=make_model(data).to(device)
        payload=_load_checkpoint(model,args.checkpoint,device)
        checkpoint_report(model,payload,data,args.d0b_checkpoint,args.output,args.checkpoint)


if __name__=='__main__':
    main()
