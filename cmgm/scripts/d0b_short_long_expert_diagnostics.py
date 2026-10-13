"""Frozen short/long expert diagnostics; no training and no TEST entry.

Default: synthetic preflight. --run restores the indexed seed42 best checkpoints,
reproduces TRAIN/VAL first, then measures the three existing candidates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import torch

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_dynamic_moe import A_DYNAMIC, B_DYNAMIC, ShortLongDynamicMoE
from cmgm.scripts.d0b_short_long_dynamic_moe import (
    ROOT, EXPERIMENT_ROOT, ALIASES, SOURCE_FILES, PROTOCOL, check_protocol,
    restricted_train_val, evaluate, route_statistics, preserved_rng, model_kwargs,
    sha256, write_json, full_loader,
    data_audit,
)
from cmgm.training.metric_standard import population_metrics

HORIZONS = (1, 5, 10, 20)
GROUPS = ((0, 1), (2, 3))
GROUP_NAMES = ('short', 'long')
CANDIDATES = ('shared', 'state', 'interaction')
TIE_ATOL, TIE_RTOL = 1e-12, 1e-9
CORRELATION_STD_EPS = 1e-12
PREDICTION_ATOL = 1e-6
# Installed PyTorch 2.7 float32 assert_close defaults, not a performance cutoff.
REPRODUCTION_ATOL, REPRODUCTION_RTOL = 1e-5, 1.3e-6
METRIC_ATOL = 1e-8
DIAGNOSED = ('b-dynamic', 'a-dynamic')


def huber(error, delta):
    error = np.abs(np.asarray(error, dtype=np.float64))
    return np.where(error <= delta, .5 * error ** 2, delta * (error - .5 * delta))


def group_losses(prediction, target, delta):
    """(N,4,C) -> (N,2); mean over both horizons and commodities, not batches."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1] != 4 or not p.size:
        raise ValueError('Expected matching nonempty (origins,4,commodities)')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite predictions or labels')
    return np.stack([huber(p[:, ids] - y[:, ids], delta).mean(axis=(1, 2)) for ids in GROUPS], axis=1)


def expert_losses(prediction, target, delta):
    """(N,3,4,C) -> (N,2,3); labels enter offline statistics only."""
    p = np.asarray(prediction)
    if p.shape != (len(target), 3, *target.shape[1:]):
        raise ValueError('Expert predictions must be (N,3,4,C)')
    return np.stack([group_losses(p[:, k], target, delta) for k in range(3)], axis=-1)


def best_candidates(losses):
    losses = np.asarray(losses, dtype=np.float64)
    minimum = losses.min(axis=-1, keepdims=True)
    tied = np.abs(losses - minimum) <= TIE_ATOL + TIE_RTOL * np.abs(minimum)
    # Lowest index among candidates within the published tolerance.
    return tied.argmax(axis=-1), tied, np.sort(losses, axis=-1)[..., 1] - minimum[..., 0]


def choose_predictions(experts, choices):
    """One candidate per sample/group, shared by its two horizon output rows."""
    n = len(experts)
    choices = np.asarray(choices)
    if choices.shape == (2,):
        choices = np.broadcast_to(choices, (n, 2))
    if choices.shape != (n, 2) or not np.isin(choices, (0, 1, 2)).all():
        raise ValueError('Choices must be (2,) or (N,2), with ids 0/1/2')
    result = np.empty_like(experts[:, 0])
    for g, ids in enumerate(GROUPS):
        selected = experts[np.arange(n), choices[:, g]]
        result[:, ids] = selected[:, ids]
    return result


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()),
                quantiles={str(q): float(np.quantile(v, q)) for q in (.1, .25, .5, .75, .9, .99)})


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() <= CORRELATION_STD_EPS or b.std() <= CORRELATION_STD_EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction, target, delta):
    return dict(group_huber=group_losses(prediction, target, delta).mean(0).tolist(),
                horizons={str(h): population_metrics(prediction[:, i], target[:, i])
                          for i, h in enumerate(HORIZONS)})


def analyze_split(alias, split, arrays, train_choice, val_choice, delta):
    p, y, pi = arrays['expert_predictions'], arrays['target'], arrays['pi'].astype(np.float64)
    losses = expert_losses(p, y, delta)
    best, tied, margin = best_candidates(losses)
    hard = pi.argmax(-1)
    n = len(y)
    if pi.shape != (n, 2, 3) or len(arrays['origin_dates']) != n:
        raise ValueError('Origins/labels/routes are not aligned')
    competence, pairwise, alignment, per_origin = [], [], [], []
    group_summary = {}
    for g, ids in enumerate(GROUPS):
        tie_count = tied[:, g].sum(-1)
        non_tie = tie_count == 1
        group_summary[GROUP_NAMES[g]] = dict(
            tie_origins=int((~non_tie).sum()), tie_fraction=float((~non_tie).mean()),
            winner_fractions_including_ties=tied[:, g].mean(0).tolist(),
            winner_fractions_fractional_ties=(tied[:, g] / tie_count[:, None]).mean(0).tolist(),
            deterministic_winner_fractions=[float((best[:, g] == k).mean()) for k in range(3)],
            first_second_loss_gap=distribution(margin[:, g]))
        for k, expert in enumerate(CANDIDATES):
            for i in ids:
                row = population_metrics(p[:, k, i], y[:, i])
                competence.append(dict(model=alias, split=split, group=GROUP_NAMES[g], expert=expert,
                    expert_id=k, horizon=HORIZONS[i], origins=n, group_huber=float(losses[:, g, k].mean()),
                    **row, Hit_percent=100 * row['Hit'],
                    winner_fraction_including_ties=float(tied[:, g, k].mean()),
                    winner_fraction_fractional_ties=float((tied[:, g, k] / tie_count).mean()),
                    deterministic_winner_fraction=float((best[:, g] == k).mean()),
                    group_tie_fraction=float((~non_tie).mean()),
                    first_second_gap_median=float(np.median(margin[:, g])),
                    first_second_gap_P90=float(np.quantile(margin[:, g], .9))))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            tol = TIE_ATOL + TIE_RTOL * np.maximum(np.abs(losses[:, g, a]), np.abs(losses[:, g, b]))
            gap = losses[:, g, a] - losses[:, g, b]
            for i in ids:
                per_date_diff = np.abs(p[:, a, i].astype(np.float64) - p[:, b, i]).mean(-1)
                largest = np.argsort(-per_date_diff, kind='stable')[:5]
                e_a, e_b = p[:, a, i].astype(np.float64) - y[:, i], p[:, b, i].astype(np.float64) - y[:, i]
                pairwise.append(dict(model=alias, split=split, group=GROUP_NAMES[g], horizon=HORIZONS[i],
                    expert_a=CANDIDATES[a], expert_b=CANDIDATES[b],
                    mean_abs_prediction_difference=float(per_date_diff.mean()),
                    signed_error_Pearson=correlation(e_a, e_b),
                    origin_MAE_Pearson=correlation(np.abs(e_a).mean(-1), np.abs(e_b).mean(-1)),
                    expert_a_group_win_fraction=float((gap < -tol).mean()),
                    expert_b_group_win_fraction=float((gap > tol).mean()),
                    pair_group_tie_fraction=float((np.abs(gap) <= tol).mean()),
                    top_disagreement_cases=json.dumps([dict(origin=int(j), date=str(arrays['origin_dates'][j]),
                        mean_abs_prediction_difference=float(per_date_diff[j]),
                        expert_a_group_loss=float(losses[j, g, a]), expert_b_group_loss=float(losses[j, g, b]),
                        group_huber_gap_a_minus_b=float(gap[j]),
                        horizon_MAE_gap_a_minus_b=float(np.abs(e_a[j]).mean() - np.abs(e_b[j]).mean()))
                        for j in largest], ensure_ascii=False)))
        hard_loss = losses[np.arange(n), g, hard[:, g]]
        oracle_loss = losses[np.arange(n), g, best[:, g]]
        alignment.append(dict(model=alias, split=split, group=GROUP_NAMES[g], origins=n,
            non_tie_origins=int(non_tie.sum()), tie_origins=int((~non_tie).sum()),
            hard_router_oracle_agreement_non_ties=float((hard[non_tie, g] == best[non_tie, g]).mean()) if non_tie.any() else None,
            mean_hard_choice_group_huber=float(hard_loss.mean()),
            mean_oracle_group_huber=float(oracle_loss.mean()),
            mean_hard_choice_minus_oracle_group_huber=float((hard_loss-oracle_loss).mean()),
            mean_hard_choice_minus_true_min_group_huber=float((hard_loss-losses[:, g].min(-1)).mean()),
            semantics='argmax-weight hard choice only, not the normal soft-mixture loss'))
        for j in range(n):
            per_origin.append(dict(model=alias, split=split, origin=j, date=str(arrays['origin_dates'][j]),
                group=GROUP_NAMES[g], pi_shared=float(pi[j, g, 0]), pi_state=float(pi[j, g, 1]),
                pi_interaction=float(pi[j, g, 2]), loss_shared=float(losses[j, g, 0]),
                loss_state=float(losses[j, g, 1]), loss_interaction=float(losses[j, g, 2]),
                oracle_expert=int(best[j, g]), tied_candidates=json.dumps(np.flatnonzero(tied[j, g]).tolist()),
                oracle_tied=bool(tie_count[j] > 1), router_argmax=int(hard[j, g]),
                first_second_loss_gap=float(margin[j, g])))
    strategies = {'normal_dynamic': arrays['prediction'],
                  'train_best_fixed': choose_predictions(p, train_choice),
                  'val_best_fixed_EX_POST': choose_predictions(p, val_choice),
                  'oracle_group_EX_POST': choose_predictions(p, best),
                  'train_mean_route': arrays['train_mean_prediction'],
                  'paired_route_shuffle': arrays['shuffle_prediction'],
                  'shared_only': p[:, 0]}
    scores = {name: metrics(pred, y, delta) for name, pred in strategies.items()}
    strategy_rows = []
    for name, score in scores.items():
        for g, ids in enumerate(GROUPS):
            for i in ids:
                h = str(HORIZONS[i]); metric = score['horizons'][h]
                normal = scores['normal_dynamic']
                strategy_rows.append(dict(model=alias, split=split, strategy=name, group=GROUP_NAMES[g],
                    horizon=HORIZONS[i], group_huber=score['group_huber'][g], **metric,
                    Hit_percent=100 * metric['Hit'],
                    delta_group_huber_vs_normal=score['group_huber'][g]-normal['group_huber'][g],
                    delta_MAE_vs_normal=metric['MAE']-normal['horizons'][h]['MAE'],
                    delta_RMSE_vs_normal=metric['RMSE']-normal['horizons'][h]['RMSE'],
                    delta_Hit_percentage_points_vs_normal=100*(metric['Hit']-normal['horizons'][h]['Hit']),
                    uses_VAL_labels_for_selection=name == 'val_best_fixed_EX_POST' or (name == 'oracle_group_EX_POST' and split == 'val'),
                    selection_note={'train_best_fixed':'Fixed by TRAIN mean group Huber; in-sample when shown on TRAIN',
                        'val_best_fixed_EX_POST':'Uses VAL labels, diagnostic only',
                        'oracle_group_EX_POST':'Uses current split labels; three single-candidate group selector; not deployable'}.get(name,'Frozen inference; no label-conditioned weights')))
    potential = {}
    for g, group in enumerate(GROUP_NAMES):
        oracle = scores['oracle_group_EX_POST']['group_huber'][g]
        potential[group] = {}
        for fixed in ('train_best_fixed', 'val_best_fixed_EX_POST'):
            fixed_loss = scores[fixed]['group_huber'][g]
            potential[group][fixed] = dict(absolute_huber_reduction=fixed_loss-oracle,
                relative_huber_reduction_percent=100*(fixed_loss-oracle)/fixed_loss if fixed_loss else None)
    summary = dict(scores=scores, potential=potential, group_summary=group_summary,
                   routing=route_statistics(pi), train_choice=train_choice.tolist(), val_choice=val_choice.tolist())
    derived = dict(expert_losses=losses, oracle_indices=best, tied_candidates=tied,
                   router_argmax=hard, first_second_gap=margin)
    return summary, dict(competence=competence, pairwise=pairwise, alignment=alignment,
                        per_origin=per_origin, strategies=strategy_rows), derived


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def capture_normal(model, loader, device):
    """Capture actual fusion inputs from this forward, never detached log caches."""
    batches = []
    def capture(module, args, kwargs):
        batches.append((tuple(x.to('cpu', copy=True) for x in args),
                        {k: v.to('cpu', copy=True) if torch.is_tensor(v) else v for k, v in kwargs.items()}))
    hook = model.short_long_dynamic_moe.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        metric, arrays = evaluate(model, loader, device)
    finally:
        hook.remove()
    if sum(len(args[0]) for args, _ in batches) != len(loader.dataset):
        raise ValueError('Fusion inputs and origins are not one-to-one')
    return metric, arrays, batches


@torch.inference_mode()
def predict_frozen_inputs(model, batches, weights, device):
    """Run the actual fusion with an override and the original head.

    Backbone outputs are invariant in eval and are reused to avoid re-encoding
    every candidate. Tensors come from the live fusion pre-hook of this run.
    """
    n = sum(len(args[0]) for args, _ in batches)
    weights = np.asarray(weights)
    if weights.shape not in ((2, 3), (n, 2, 3)):
        raise ValueError('Override weights do not match the frozen origins')
    predictions, offset = [], 0
    f = model.short_long_dynamic_moe
    for args, kwargs in batches:
        args = tuple(x.to(device) for x in args)
        kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        batch_weights = weights[offset:offset+len(args[0])] if weights.ndim == 3 else weights
        with f.intervention('override', batch_weights):
            states = f(*args, **kwargs)
            prediction = f.predict(model.head, states, model.n_commodities)
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite forced expert prediction')
        predictions.append(prediction.cpu().numpy()); offset += len(args[0])
    return np.concatenate(predictions)


def one_hot(candidate):
    weights = np.zeros((2, 3), dtype=np.float32)
    weights[:, candidate] = 1.
    return weights


@torch.inference_mode()
def verify_group_injection(model, loader, batches, arrays, device):
    """Real first-batch check of full forward vs reused fusion and group isolation."""
    first = [batches[0]]
    n = len(first[0][0][0])
    x = next(iter(full_loader(loader)))[0].to(device)
    f = model.short_long_dynamic_moe
    errors = []
    for k in range(3):
        both = predict_frozen_inputs(model, first, one_hot(k), device)
        with f.intervention('override', one_hot(k)):
            full = model(x).cpu().numpy()
        errors.append(float(np.abs(both.astype(np.float64)-full).max()))
        for g, ids in enumerate(GROUPS):
            weights = arrays['pi'][:n].copy()
            weights[:, g] = one_hot(k)[g]
            only = predict_frozen_inputs(model, first, weights, device)
            if (not np.allclose(only[:, ids], both[:, ids], atol=PREDICTION_ATOL, rtol=0)
                    or not np.allclose(only[:, GROUPS[1-g]], arrays['prediction'][:n, GROUPS[1-g]],
                                       atol=PREDICTION_ATOL, rtol=0)):
                raise ValueError('One-group intervention changed another group or mismatched its candidate')
    if max(errors) > PREDICTION_ATOL:
        raise ValueError('Full one-hot forward differs from actual reused fusion/head')
    if f.residual:
        args, kwargs = first[0]
        args = tuple(x.to(device) for x in args)
        kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        with f.intervention('override', one_hot(0)):
            states = f(*args, **kwargs)
        if not torch.equal(states, kwargs['z_base'][:, None].expand(-1, 4, -1)):
            raise ValueError('B shared candidate is not exactly z_base')
    return dict(PASS=True,full_vs_reused_fusion_max_abs_diff=max(errors),
                both_vs_one_group_checked=True,other_group_unchanged=True,
                B_shared_exact_z_base=True if f.residual else 'NOT_APPLICABLE')


def discover_index(explicit=None):
    if explicit is not None:
        return explicit, json.loads(explicit.read_text())
    indexes = []
    for path in sorted(EXPERIMENT_ROOT.glob('*/file_index.json')):
        index = json.loads(path.read_text())
        if set(index) == set(ALIASES) and all(isinstance(v, dict) and 'checkpoint_sha256' in v for v in index.values()):
            indexes.append((path, index))
    if not indexes:
        raise FileNotFoundError('No audited six-run file_index.json found; use --file-index')
    signatures = {tuple((a, v['checkpoint_sha256'], v['result']) for a, v in sorted(index.items())) for _, index in indexes}
    if len(signatures) != 1:
        raise ValueError('Conflicting indexed checkpoint sets; specify --file-index')
    return indexes[0]


def source_audit(cfg):
    """Accept the documented diagnostics-only dispatch repair, not model drift."""
    differences = {}
    for name, original in cfg['source_sha256'].items():
        current = sha256(ROOT/name)
        if current == original:
            continue
        if name == 'cmgm/scripts/d0b_short_long_dynamic_moe.py':
            note = 'Orchestration/calendar/override checks repaired; full data fingerprint and normal predictions must reproduce'
        elif name == 'cmgm/models/hetero_mixhop_model.py':
            text = (ROOT/name).read_text()
            start = text.index('        if self.variant in DYNAMIC_STUDY_VARIANTS:', text.index('    def get_gate_stats('))
            end = text.index('        if self.variant == "switching_latent_balanced_mean_fusion":', start)
            if hashlib.sha256((text[:start]+text[end:]).encode()).hexdigest() != original:
                raise ValueError('Model change extends beyond audited get_gate_stats repair')
            note = 'Only diagnostic get_gate_stats dispatch changed; removing it reproduces original source hash'
        else:
            raise ValueError(f'Unaudited model/data/training source drift: {name}')
        differences[name] = dict(training_sha256=original, current_sha256=current, explanation=note)
    return differences


def load_run(alias, item, device):
    result_path = Path(item['result']); directory = result_path.parent
    cfg = json.loads(Path(item['config']).read_text())
    result = json.loads(result_path.read_text())
    history = json.loads(Path(item['training_history']).read_text())
    status = json.loads((directory/'RUN_STATUS.json').read_text())
    if (result['status'] != 'COMPLETE' or status['status'] != 'COMPLETE'
            or cfg['variant'] != ALIASES[alias] or result['variant'] != cfg['variant']
            or cfg['seed'] != result['seed'] or cfg['seed'] != 42
            or result['test_evaluated'] or cfg['protocol'] != PROTOCOL):
        raise ValueError(f'Invalid indexed identity/completion/protocol: {alias}')
    ck = Path(item['checkpoint']); digest = sha256(ck)
    if digest != item['checkpoint_sha256'] or digest != result['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {alias}')
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    if (json.loads(json.dumps(payload['metadata'])) != cfg
            or payload['best_epoch'] != result['best_epoch'] or payload['best_epoch'] != history['best_epoch']
            or payload['best_epoch'] != int(np.argmin(history['val_loss']))+1
            or payload['history']['val_loss'] != history['val_loss']
            or payload['best_val_loss'] != min(history['val_loss'])
            or payload['best_val_loss'] != result['best_val_prediction_huber_batch_mean']):
        raise ValueError(f'Checkpoint/history metadata mismatch: {alias}')
    drift = source_audit(cfg)
    with preserved_rng():
        model = HeteroMixHopCMGM(variant=cfg['variant'], **cfg['model_kwargs'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.requires_grad_(False).to(device).eval()
    model.switching_latent_transformer.set_epoch(payload['best_epoch'])
    provenance = json.loads(Path(item['postprocess_provenance']).read_text()) if item.get('postprocess_provenance') else None
    if provenance is not None:
        original = Path(provenance['original_run'])
        if (sha256(original/'config.json') != sha256(Path(item['config']))
                or sha256(original/'training_history.json') != sha256(Path(item['training_history']))
                or provenance['checkpoint_sha256'] != digest):
            raise ValueError('Original/restored provenance conflict')
    return model, cfg, result, dict(checkpoint=str(ck), checkpoint_sha256=digest,
        best_epoch=payload['best_epoch'], strict_load=True, switch_beta=model.switching_latent_transformer.regime_filter.current_beta,
        training_git_commit=cfg['git_commit'], training_git_status=cfg['git_status'],
        training_environment=cfg['environment'], training_config=str(Path(item['config'])),
        checkpoint_selection=cfg['protocol']['checkpoint_selection'],
        training_source_sha256=cfg['source_sha256'], permitted_evaluation_source_differences=drift,
        result=str(result_path), postprocess_provenance=provenance)


def compare_saved(arrays, saved_path, current_metrics, saved_metrics):
    with np.load(saved_path, allow_pickle=False) as saved:
        difference = float(np.max(np.abs(arrays['prediction'].astype(np.float64)-saved['prediction'])))
        for name in ('prediction', 'target', 'pi', 'origin_dates', 'commodity_order'):
            if name not in arrays:
                continue
            if name not in saved or arrays[name].shape != saved[name].shape:
                raise ValueError(f'Normal reproduction shape mismatch: {name}')
            if name == 'prediction':
                matches = np.allclose(arrays[name], saved[name], atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL)
            elif name == 'pi':
                matches = np.allclose(arrays[name], saved[name], atol=PREDICTION_ATOL, rtol=0)
            else:
                matches = np.array_equal(arrays[name], saved[name])
            if not matches:
                raise ValueError(f'Normal reproduction mismatch: {name}; prediction max_abs_diff={difference:.9g}; STOP before expert analysis')
    for h in saved_metrics:
        for k in ('MAE', 'MSE', 'RMSE', 'Hit'):
            if (current_metrics[h][k] != saved_metrics[h][k] if k == 'Hit' else
                    not np.isclose(current_metrics[h][k], saved_metrics[h][k], atol=METRIC_ATOL, rtol=0)):
                raise ValueError(f'Normal metric reproduction mismatch: {h}/{k}')
    return difference


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


DATA_SOURCE_FILES = ('cmgm/config.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py',
                     'cmgm/scripts/main_ablation.py','cmgm/scripts/d0b_short_long_dynamic_moe.py')


def save_validated_data(path, data, fingerprint, reference):
    """Keep exact TRAIN/VAL arrays after audit to avoid repeating slow features."""
    meta=dict(data_audit=fingerprint,calendar_audit=data['calendar_audit'],
        reference_config_sha256=sha256(reference),source_sha256={p:sha256(ROOT/p) for p in DATA_SOURCE_FILES})
    arrays={'metadata':np.asarray(json.dumps(meta,ensure_ascii=False))}
    for split in ('train','val'):
        ds=data['loaders'][split].dataset
        for key in ('prices','feature_matrix','raw_prices'):
            arrays[f'{split}_{key}']=getattr(ds,key)
        arrays[f'{split}_origin_dates']=data['origin_dates'][split]
    np.savez_compressed(path,**arrays)


def load_validated_data(path, expected, reference):
    from cmgm.data.data_loader import MarketSequenceDataset
    from torch.utils.data import DataLoader
    with np.load(path,allow_pickle=False) as z:
        allowed={'metadata'}|{f'{s}_{k}' for s in ('train','val') for k in ('prices','feature_matrix','raw_prices','origin_dates')}
        if set(z.files)!=allowed:raise ValueError('Cache must contain TRAIN/VAL only')
        meta=json.loads(str(z['metadata']))
        if (meta['data_audit']!=expected or meta['reference_config_sha256']!=sha256(reference)
                or any(sha256(ROOT/p)!=v for p,v in meta['source_sha256'].items())):
            raise ValueError('Validated data cache provenance mismatch')
        market={k:tuple(v) for k,v in expected['market_indices'].items()}
        data=dict(loaders={},origin_dates={},market_indices=market,n_nodes=len(expected['node_order']),
                  n_commodities=len(expected['commodity_order']),feature_names=expected['node_order'],
                  calendar_audit=meta['calendar_audit'])
        for split in ('train','val'):
            ds=MarketSequenceDataset(z[f'{split}_prices'],market,seq_len=20,
                feature_matrix=z[f'{split}_feature_matrix'],raw_prices=z[f'{split}_raw_prices'],
                target_type='return',horizons=list(HORIZONS))
            data['loaders'][split]=DataLoader(ds,batch_size=64,shuffle=False,drop_last=split=='train')
            data['origin_dates'][split]=z[f'{split}_origin_dates']
    actual=data_audit(data)
    if actual!=expected:raise ValueError('Cache numeric fingerprint mismatch')
    return data,actual


def report_text(summary, references):
    lines = ['# 短/长任务 MoE 专家互补性与路由有效性诊断', '',
        '本报告只分析现有 seed42 TRAIN/VAL 冻结模型。没有训练、参数更新或 TEST 评估。', '',
        '结论先行：下面分别量化固定三候选集合的逐样本互补潜力，以及当前 router 对该潜力的利用；不以路由波动或专家名称宣称有效。', '']
    for alias in DIAGNOSED:
        s = summary[alias]['val']
        gain = [s['potential'][g]['val_best_fixed_EX_POST']['relative_huber_reduction_percent'] for g in GROUP_NAMES]
        normal = s['scores']['normal_dynamic']['horizons']['5']['MAE']
        mean_delta = s['scores']['train_mean_route']['horizons']['5']['MAE']-normal
        shuffle_delta = s['scores']['paired_route_shuffle']['horizons']['5']['MAE']-normal
        lines.append(f'- {alias}：理想逐样本选择相对 VAL 最佳固定专家的组内 Huber 降幅 short={gain[0]:.4f}%、long={gain[1]:.4f}%；正常 VAL MAE@5d={normal:.9f}。TRAIN 均值路由替换 ΔMAE@5d={mean_delta:+.12g}，整对路由打乱 ΔMAE@5d={shuffle_delta:+.12g}。')
    lines += ['', '## 策略与专家能力', '',
        '| 模型 | VAL 策略 | short Huber | long Huber | VAL MAE@5d | VAL RMSE@5d | VAL Hit@5d % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for alias in DIAGNOSED:
        for name, score in summary[alias]['val']['scores'].items():
            m=score['horizons']['5']
            lines.append(f"| {alias} | {name} | {score['group_huber'][0]:.9g} | {score['group_huber'][1]:.9g} | {m['MAE']:.9f} | {m['RMSE']:.9f} | {100*m['Hit']:.4f} |")
    lines += ['', 'TRAIN/VAL 各组×候选的完整能力矩阵、胜出比例和并列比例见 expert_competence.csv；所有周期和策略指标见 strategy_comparison.csv。', '',
        '## 胜出与路由一致性', '', '| 模型 | VAL 组 | TRAIN 固定选择 | VAL 事后固定选择 | 确定性胜出比例 [共享,状态,交互] | 并列比例 |', '|---|---|---|---|---|---:|']
    for alias in DIAGNOSED:
        s=summary[alias]['val']
        for g,name in enumerate(GROUP_NAMES):
            q=s['group_summary'][name]
            lines.append(f"| {alias} | {name} | {CANDIDATES[s['train_choice'][g]]} | {CANDIDATES[s['val_choice'][g]]} | {q['deterministic_winner_fractions']} | {q['tie_fraction']:.6f} |")
    lines += ['', 'routing_alignment.csv 报告非并列样本上的 argmax router 一致率和硬选择损失差；这些不是软融合模型的损失。expert_pairwise.csv 报告预测分歧、误差相关、成对胜负和最高分歧日期；近零方差相关性留空（JSON 为 null）。', '',
        '## 已有对照', '', '| 参考 | 原 VAL MAE@5d | 原四周期选择 Huber | Best epoch |', '|---|---:|---:|---:|']
    for alias, r in references.items():
        lines.append(f"| {alias} | {r['metrics']['val']['5']['MAE']:.9f} | {r['best_val_prediction_huber_batch_mean']:.9f} | {r['best_epoch']} |")
    lines += ['', '## 统计与解释边界', '',
        '- 组内 Huber 是每个样本的组内两个周期×全部商品的均值，然后样本等权平均，delta 来自原配置。这与历史选模的四周期损失求和及不等长 batch-mean 聚合不同，不把两者数值直接比较。',
        f'- float64 离线统计；并列规则 abs_tol={TIE_ATOL}、rel_tol={TIE_RTOL}，在最小损失容差内取最小候选编号。报告原始第一/第二损失差，不把微小差别强行称为分工。',
        '- TRAIN 最佳固定专家只使用 TRAIN 标签，在 VAL 不再选择；其 TRAIN 表现是选取后的 in-sample 结果。VAL 最佳固定专家使用 VAL 标签，仅作事后参照。',
        '- Oracle 使用对应区间真实标签，不能部署或列为正式模型性能；它仅是三个单专家中逐样本按组择优，不是全部软融合方法的理论性能上界。按组 Huber 择优不保证 5d MAE 最低。',
        '- B 的共享候选是共同训练后的 z_base，不等于独立训练的 D0B；各完整候选始终包含一次 z_base，没有额外归一化。',
        '- 交换或打乱路由是冻结模型干预，不是重训消融或经济因果识别。不同输入上的预测能力差异仍需可预测性验证，不能保证改 router 会成功。',
        '- 这是单种子探索性 TRAIN/VAL 诊断，历史 VAL 参与过模型研究。未做显著性声明，也未开展下一轮模型或训练。', '',
        '## 可复算产物', '',
        '每个模型、每个 split 的 *_predictions.npz 含 origin_dates、commodity_order、normal prediction、target、pi、三候选预测 (N,3,4,C)、TRAIN均值/打乱路由预测、逐样本专家损失、oracle选择、并列标记、argmax路由和损失差。audit.json 与 provenance.json 保存数据审计、源代码、原/恢复产物关系和前后不可变性校验。']
    return '\n'.join(lines)+'\n'


def run(args):
    check_protocol()
    index_path, index = discover_index(args.file_index)
    out = args.output or EXPERIMENT_ROOT/f'expert_diagnostics_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    audit = dict(status='RUNNING', test_evaluated=False, formal_training=False, file_index=str(index_path), models={})
    write_json(out/'audit.json', audit)
    print(f'DIAGNOSTIC_DIRECTORY {out.resolve()}', flush=True)
    started = time.perf_counter()
    try:
        torch.set_num_threads(args.threads)
        protected = {Path(v[k]): sha256(Path(v[k])) for v in index.values()
                     for k in ('result','config','training_history','checkpoint','TRAIN_predictions','VAL_predictions')}
        models, configs, references = {}, {}, {}
        for alias, item in index.items():
            m,c,r,p = load_run(alias,item,args.device)
            configs[alias],references[alias],audit['models'][alias] = c,r,p
            if alias in DIAGNOSED:
                models[alias]=m
        ref = configs['b-dynamic']
        if any(c['data_audit'] != ref['data_audit'] or c['protocol'] != ref['protocol'] for c in configs.values()):
            raise ValueError('Indexed runs have different data or protocol')
        match = re.fullmatch(r'sum_h Huber\(delta=([0-9.]+)\)', ref['protocol']['prediction_loss'])
        if match is None:
            raise ValueError('Original Huber delta missing from protocol')
        delta = float(match.group(1))
        reference = Path(ref['reference_config'])
        if sha256(reference) != ref['reference_config_sha256']:
            raise ValueError('Reference config hash mismatch')
        if args.data_cache:
            print('Loading fingerprint-validated TRAIN/VAL cache',flush=True)
            data,fingerprint=load_validated_data(args.data_cache,ref['data_audit'],reference)
        else:
            print('Reconstructing original TRAIN/VAL only; rolling feature statistics may take several minutes.', flush=True)
            data, fingerprint = restricted_train_val(reference)
        if fingerprint != ref['data_audit'] or json.loads(json.dumps(model_kwargs(data))) != ref['model_kwargs']:
            raise ValueError('Original dataset/model fingerprint changed')
        if not args.data_cache:
            save_validated_data(out/'validated_train_val_cache.npz',data,fingerprint,reference)
        audit.update(data_audit=fingerprint, calendar_audit=data['calendar_audit'], delta=delta,
                     origin_ranges={s:dict(first=str(d[0]),last=str(d[-1]),count=len(d)) for s,d in data['origin_dates'].items()})
        state_before = {a: state_digest(m) for a,m in models.items()}
        normal, frozen = {}, {}
        # Reproduce BOTH models on BOTH splits before interpreting any candidate.
        for alias in DIAGNOSED:
            model=models[alias]
            env=configs[alias]['environment']
            torch.backends.cudnn.deterministic=env['cudnn_deterministic']
            torch.backends.cudnn.benchmark=env['cudnn_benchmark']
            normal[alias],frozen[alias]={},{}
            for split in ('train','val'):
                score,arrays,batches=capture_normal(model,data['loaders'][split],args.device)
                arrays.update(origin_dates=data['origin_dates'][split],commodity_order=np.asarray(fingerprint['commodity_order'],dtype=str))
                try:
                    difference=compare_saved(arrays,Path(index[alias][split.upper()+'_predictions']),score,references[alias]['metrics'][split])
                except ValueError:
                    np.savez_compressed(out/f'reproduction_failure_{alias}_{split}.npz',**arrays)
                    raise
                audit['models'][alias].setdefault('normal_reproduction',{})[split]=dict(PASS=True,max_abs_prediction_difference=difference)
                normal[alias][split],frozen[alias][split]=arrays,batches
                print(f'NORMAL PASS {alias} {split}: {len(arrays["target"])} origins; max prediction diff={difference:.3g}',flush=True)
        audit['status']='NORMAL_REPRODUCED';write_json(out/'audit.json',audit)
        if args.compare_cpu:
            if not args.device.startswith('cuda'):
                raise ValueError('--compare-cpu requires a CUDA main evaluation')
            cpu_model,_,_,_=load_run('a-dynamic',index['a-dynamic'],'cpu')
            _,cpu_arrays,_=capture_normal(cpu_model,data['loaders']['val'],'cpu')
            gpu=normal['a-dynamic']['val']['prediction']
            diff=np.abs(cpu_arrays['prediction'].astype(np.float64)-gpu)
            worst=np.unravel_index(diff.argmax(),diff.shape)
            audit['cpu_vs_cuda_investigation']=dict(model='a-dynamic',split='val',
                max_abs_prediction_difference=float(diff.max()),mean_abs_prediction_difference=float(diff.mean()),
                elements_above_1e6_atol=int((diff>PREDICTION_ATOL).sum()),
                worst_origin=int(worst[0]),worst_date=str(data['origin_dates']['val'][worst[0]]),
                worst_horizon=HORIZONS[worst[1]],worst_commodity=fingerprint['commodity_order'][worst[2]],
                CPU_prediction=float(cpu_arrays['prediction'][worst]),CUDA_prediction=float(gpu[worst]),
                semantics='Numerical investigation only; CUDA reproduces original checkpoint outputs; CPU result does not replace formal predictions')
            np.savez_compressed(out/'numerical_investigation_a_dynamic_VAL.npz',
                cpu_prediction=cpu_arrays['prediction'],cuda_prediction=gpu,
                origin_dates=data['origin_dates']['val'],commodity_order=np.asarray(fingerprint['commodity_order'],dtype=str))
            print('CPU/CUDA investigation: '+json.dumps(audit['cpu_vs_cuda_investigation']),flush=True)
            del cpu_model
        # Reference metrics reuse their completed artifacts, never reinterpret a
        # name or substitute a historical TEST number. Recompute their NPZs.
        for alias in ALIASES:
            if alias in DIAGNOSED:
                continue
            for split in ('train','val'):
                with np.load(index[alias][split.upper()+'_predictions'],allow_pickle=False) as z:
                    if (not np.array_equal(z['target'],normal['b-dynamic'][split]['target'])
                            or not np.array_equal(z['origin_dates'],data['origin_dates'][split])
                            or z['commodity_order'].tolist()!=fingerprint['commodity_order']):
                        raise ValueError('Reference targets/origins/commodity order differ')
                    for i,h in enumerate(HORIZONS):
                        if population_metrics(z['prediction'][:,i],z['target'][:,i])!=references[alias]['metrics'][split][str(h)]:
                            raise ValueError('Reference NPZ metrics do not reproduce result.json')
            audit['models'][alias]['reference_saved_metrics_recomputed']=True
        summary={};tables={name:[] for name in ('competence','pairwise','alignment','per_origin','strategies')}
        for alias in DIAGNOSED:
            model=models[alias];collected={};permutations={}
            train_mean=normal[alias]['train']['pi'].mean(0,dtype=np.float64)
            for split in ('train','val'):
                arrays=normal[alias][split];batches=frozen[alias][split]
                audit['models'][alias].setdefault('forced_injection',{})[split]=verify_group_injection(
                    model,data['loaders'][split],batches,arrays,args.device)
                p=np.stack([predict_frozen_inputs(model,batches,one_hot(k),args.device) for k in range(3)],axis=1)
                arrays['expert_predictions']=p
                arrays['train_mean_prediction']=predict_frozen_inputs(model,batches,train_mean,args.device)
                permutation=np.random.default_rng(args.permutation_seed).permutation(len(p))
                arrays['shuffle_prediction']=predict_frozen_inputs(model,batches,arrays['pi'][permutation],args.device)
                permutations[split]=permutation.tolist()
                collected[split]=arrays
                print(f'FORCED + INTERVENTIONS {alias} {split} complete',flush=True)
            train_choice=best_candidates(expert_losses(collected['train']['expert_predictions'],collected['train']['target'],delta).mean(0))[0]
            val_choice=best_candidates(expert_losses(collected['val']['expert_predictions'],collected['val']['target'],delta).mean(0))[0]
            summary[alias]={}
            for split,arrays in collected.items():
                stats,rows,derived=analyze_split(alias,split,arrays,train_choice,val_choice,delta)
                summary[alias][split]=stats
                for key,values in rows.items():tables[key].extend(values)
                arrays.update(derived,train_mean_pi=train_mean,route_permutation=np.asarray(permutations[split]),
                              train_fixed_choice=train_choice,val_fixed_choice=val_choice)
                np.savez_compressed(out/f'{alias}_{split}_predictions.npz',**arrays)
            audit['models'][alias]['state_before']=state_before[alias]
            audit['models'][alias]['state_after']=state_digest(model)
            if audit['models'][alias]['state_before'] != audit['models'][alias]['state_after']:
                raise ValueError('Model state changed during frozen diagnosis')
            audit['models'][alias]['train_mean_pi']=train_mean.tolist()
            audit['models'][alias]['permutation_seed']=args.permutation_seed
        if any(sha256(path)!=digest for path,digest in protected.items()):
            raise ValueError('Original checkpoint/experiment artifact changed')
        outputs={'competence':'expert_competence.csv','pairwise':'expert_pairwise.csv',
                 'alignment':'routing_alignment.csv','per_origin':'per_origin_diagnostics.csv','strategies':'strategy_comparison.csv'}
        for name,filename in outputs.items():write_csv(out/filename,tables[name])
        write_json(out/'analysis_summary.json',summary)
        write_json(out/'reference_results.json',references)
        provenance=dict(file_index=str(index_path.resolve()),file_index_sha256=sha256(index_path),
            indexed_files=index,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            current_source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,'cmgm/scripts/d0b_short_long_expert_diagnostics.py')},
            environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=args.device,
                threads=args.threads,autocast=False,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            historical_environment_missing=['Original records do not document TF32 flags or thread count.'],
            target_scale=ref['target_scale'],features='Original 21 features and TRAIN-fitted statistics; exact original dataset fingerprints required',
            timing_seconds=time.perf_counter()-started,huberdelta=delta,tie_atol=TIE_ATOL,tie_rtol=TIE_RTOL,
            correlation_std_epsilon=CORRELATION_STD_EPS,prediction_atol=PREDICTION_ATOL,hit_storage='fraction; CSV Hit_percent and report display percent',
            reproduction_atol=REPRODUCTION_ATOL,reproduction_rtol=REPRODUCTION_RTOL,metric_reproduction_atol=METRIC_ATOL,
            reproduction_tolerance_basis='Installed PyTorch float32 default assert_close tolerances; population metrics checked at1e-8 and Hit exactly. Initial1e-6-only attempts retained as FAILED_STOP.',
            data_cache=str(args.data_cache.resolve()) if args.data_cache else str((out/'validated_train_val_cache.npz').resolve()),
            backbone_reuse='Actual fusion pre-hook tensors from normal forward; no detached model logging cache used. Same eval fusion and head rerun for each weight override.',
            test_evaluated=False,formal_training=False,original_artifacts_unchanged=True)
        write_json(out/'provenance.json',provenance)
        (out/'diagnostic_summary.md').write_text(report_text(summary,references))
        audit.update(status='COMPLETE',original_artifacts_unchanged=True,model_parameters_unchanged=True,
                     elapsed_seconds=time.perf_counter()-started)
        write_json(out/'audit.json',audit)
        print(f'COMPLETE {out.resolve()}',flush=True)
    except BaseException as exc:
        audit.update(status='FAILED_STOP',error=repr(exc));write_json(out/'audit.json',audit)
        raise


def preflight():
    torch.manual_seed(42)
    for variant in (A_DYNAMIC,B_DYNAMIC):
        f=ShortLongDynamicMoE(variant).eval()
        s,t,h,z,base=torch.randn(2,64),torch.randn(2,64),torch.randn(2,20,128),torch.randn(2,20,64),torch.randn(2,64)
        with torch.no_grad():
            if f.residual:
                for expert in (f.short_state,f.short_interaction,f.long_state,f.long_interaction):
                    expert.mlp[-1].weight.normal_(0,.01)
            states,parts=f(s,t,h,z,z_base=base,return_components=True)
            for k in range(3):
                with f.intervention('override',one_hot(k)):
                    chosen=f(s,t,h,z,z_base=base)
                expected_short=parts['experts'][:,k]
                expected_long=parts['experts'][:,(0,3,4)[k]]
                if not f.residual:
                    expected_short=f.fusion_norm(expected_short);expected_long=f.fusion_norm(expected_long)
                torch.testing.assert_close(chosen[:,0],expected_short)
                torch.testing.assert_close(chosen[:,2],expected_long)
            if f.residual:
                with f.intervention('override',one_hot(0)):
                    torch.testing.assert_close(f(s,t,h,z,z_base=base),base[:,None].expand(-1,4,-1),atol=0,rtol=0)
    p=np.zeros((1,3,4,1));p[0,0,0]=10;p[0,1,1]=2;p[0,2,1]=4
    losses=expert_losses(p,np.zeros((1,4,1)),.02)
    assert best_candidates(losses)[0][0,0]==1  # two-horizon choice, not 5d alone
    assert best_candidates(np.zeros((2,2,3)))[0].tolist()==[[0,0],[0,0]]
    print('SYNTHETIC PREFLIGHT PASS; formal evaluation NOT_RUN; TEST NOT_READ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Frozen TRAIN/VAL diagnostics; never trains')
    parser.add_argument('--file-index',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--permutation-seed',type=int,default=42)
    parser.add_argument('--compare-cpu',action='store_true',help='After CUDA reproduction, investigate A VAL CPU/CUDA differences without changing formal predictions')
    parser.add_argument('--data-cache',type=Path,help='Exact TRAIN/VAL arrays previously constructed and fully fingerprint audited; no new preprocessing')
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    if args.run:run(args)
    else:preflight()


if __name__=='__main__':
    main()
