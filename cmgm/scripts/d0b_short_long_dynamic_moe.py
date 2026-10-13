"""Short/long dynamic routing study: synthetic preflight unless --run is explicit.

Six fixed configurations, TRAIN/VAL only. No TEST evaluation entry exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_dynamic_moe import (
    A_STATIC, A_DYNAMIC, B_STATIC, B_DYNAMIC, DYNAMIC_STUDY_VARIANTS, EXPERT_NAMES,
)
from cmgm.models.task_conditioned_moe import BASE_VARIANT, GLOBAL_VARIANT as THREE_GLOBAL
from cmgm.scripts.d0b_task_conditioned_moe import (
    PROTOCOL, check_protocol, data_audit, full_loader, model_kwargs, sha256, write_json,
)
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / 'experiments/d0b_short_long_dynamic_moe'
CHECKPOINT_ROOT = ROOT / 'checkpoints/d0b_short_long_dynamic_moe'
DEFAULT_REFERENCE = ROOT / ('experiments/d0b_short_long_mask_moe/'
    'switching_latent_balanced_readout_seed42_20261012_173338_417491/config.json')
VARIANTS = (BASE_VARIANT, THREE_GLOBAL, *DYNAMIC_STUDY_VARIANTS)
ALIASES = dict(zip(('d0b', 'three-global', 'a-static', 'a-dynamic', 'b-static', 'b-dynamic'), VARIANTS))
SOURCE_FILES = (
    'cmgm/config.py', 'cmgm/data/data_loader.py', 'cmgm/data/feature_builder.py', 'cmgm/models/model.py',
    'cmgm/models/hetero_mixhop_model.py', 'cmgm/models/switching_latent_transformer.py',
    'cmgm/models/short_long_mask_moe.py', 'cmgm/models/short_long_dynamic_moe.py',
    'cmgm/models/task_conditioned_moe.py', 'cmgm/models/horizon_tensor_fusion.py',
    'cmgm/scripts/main_ablation.py', 'cmgm/scripts/d0b_task_conditioned_moe.py',
    'cmgm/scripts/d0b_short_long_dynamic_moe.py', 'cmgm/training/train.py',
    'cmgm/training/metric_standard.py',
)
VAL_RULE = dict(primary='VAL MAE@5d; decimal returns', checkpoint_selection=PROTOCOL['checkpoint_selection'],
    comparisons=[['a-dynamic', 'a-static'], ['b-dynamic', 'b-static'], ['b-dynamic', 'd0b'],
                 ['b-dynamic', 'three-global']],
    decision='Report strict lower/higher and absolute/relative differences; no TEST trigger.',
    caveat='A vs B changes experts, shared path and normalization; not a single-factor routing contrast.')


@contextmanager
def preserved_rng():
    py, npstate = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(npstate)


def seed_all(seed):
    random.seed(seed)
    set_seed(seed)


def copy_keys(source, target, keys):
    src, dst = source.state_dict(), target.state_dict()
    for k in keys:
        if k not in dst or src[k].shape != dst[k].shape:
            raise ValueError(f'Shared initialization key missing or changed: {k}')
        dst[k] = src[k].clone()
    target.load_state_dict(dst, strict=True)
    differences = {k: float((src[k]-target.state_dict()[k]).abs().max()) for k in keys
                   if not torch.equal(src[k], target.state_dict()[k])}
    return dict(copied_keys=list(keys), tensor_count=len(keys),
                parameter_elements=sum(src[k].numel() for k in keys),
                shared_max_abs_diff=max(differences.values(), default=0.),
                mismatch_count=len(differences), mismatches=differences, PASS=not differences)


def matched_models(kwargs, seed):
    """One fresh D0B state explicitly copied; pair experts copied from static.

    Audit constructors preserve caller RNG. Actual training resets all seeds once
    after construction, consistently for all six configurations.
    """
    with preserved_rng():
        seed_all(seed)
        base = HeteroMixHopCMGM(variant=BASE_VARIANT, **kwargs)
        models, audit = {BASE_VARIANT: base}, {}
        base_keys = list(base.state_dict())
        audit[BASE_VARIANT] = copy_keys(base, base, base_keys)
        for variant in VARIANTS[1:]:
            seed_all(seed)
            model = HeteroMixHopCMGM(variant=variant, **kwargs)
            keys = [k for k in base_keys if not k.startswith('gate_fc.') or hasattr(model, 'gate_fc')]
            audit[variant] = copy_keys(base, model, keys)
            models[variant] = model
        pairs = {}
        for static, dynamic in ((A_STATIC, A_DYNAMIC), (B_STATIC, B_DYNAMIC)):
            keys = [k for k in models[static].state_dict()
                    if not k.startswith('short_long_dynamic_moe.router.')]
            pairs[dynamic] = copy_keys(models[static], models[dynamic], keys)
    return models, dict(shared=audit, pairs=pairs, PASS=all(a['PASS'] for a in [*audit.values(), *pairs.values()]))


def parameters(model):
    count = lambda m: sum(p.numel() for p in m.parameters())
    result = dict(total=count(model), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if hasattr(model, 'short_long_dynamic_moe'):
        f = model.short_long_dynamic_moe
        result['experts'] = {name: count(getattr(f, name)) for name in EXPERT_NAMES if hasattr(f, name)}
        result['router'] = count(f.router)
        result['fusion_norm'] = count(f.fusion_norm) if hasattr(f, 'fusion_norm') else 0
        result['shared_expert_note'] = 'B shared expert reuses D0B gated state; no additional parameters' if f.residual else 'A existing shared expert'
    return result


def restricted_train_val(reference_path):
    """Read only TRAIN/VAL price rows, using original audited calendar and assets.

    Full date columns provide a superset calendar; audited row counts freeze splits.
    TEST price/feature/target rows are excluded at CSV parsing, not sliced later.
    Fingerprint mismatch aborts before optimizer creation; never silently realign.
    """
    import pandas as pd
    from cmgm.data.data_loader import (load_stock_prices, load_bond_prices, load_commodity_prices,
                                      normalize_data, compute_returns)
    from cmgm.scripts.main_ablation import build_data
    reference = json.loads(reference_path.read_text())
    if reference['protocol'] != PROTOCOL:
        raise ValueError('Reference protocol changed; STOP before training')
    expected = reference['data_audit']
    files = (config.STOCK_FILE, config.BOND_FILE, config.COMMODITY_FILE)
    date_rows = [pd.to_datetime(pd.read_csv(p, usecols=['date'], encoding='utf-8-sig')['date']) for p in files]
    common = pd.DatetimeIndex(date_rows[0].unique())
    for dates in date_rows[1:]:
        common = common.intersection(pd.DatetimeIndex(dates.unique()))
    common = common.sort_values()
    # Raw date columns are a SUPERSET of the loader calendars: pivot_table
    # drops dates for which every close is missing. Never split the raw calendar.
    # Freeze row counts from the audited windows, then find the matching effective
    # calendar without ever reading beyond its VAL end. Starting with N raw dates
    # is a lower bound; adding only the deficit cannot overshoot N effective dates.
    def raw_rows(split):
        audit = expected['splits'][split]
        if (audit['seq_len'] != config.SEQ_LEN
                or audit['horizons'] != list(config.MULTI_HORIZONS)
                or audit['target_type'] != config.TARGET_TYPE):
            raise ValueError('Audited window/target protocol changed; STOP')
        return int(audit['origins']) + int(audit['seq_len']) + max(audit['horizons']) - 1

    train_end = raw_rows('train')
    required_rows = train_end + raw_rows('val')
    if len(common) < required_rows:
        raise ValueError('Raw calendar is shorter than audited TRAIN/VAL; STOP')
    raw_count = required_rows
    attempts = []
    while True:
        cutoff = common[raw_count - 1]
        frames = []
        for path, dates, load in zip(files, date_rows, (load_stock_prices, load_bond_prices, load_commodity_prices)):
            # Header is row zero. Exclude future prices before CSV conversion.
            excluded = set((np.flatnonzero((dates > cutoff).to_numpy()) + 1).tolist())
            frames.append(load(path, read_csv_kwargs={'skiprows': lambda row, excluded=excluded: row in excluded}))
        effective = frames[0].index
        for frame in frames[1:]:
            effective = effective.intersection(frame.index)
        effective = effective.sort_values()
        if not effective.is_unique or not effective.isin(common[:raw_count]).all():
            raise ValueError('Unexpected effective calendar; STOP')
        deficit = required_rows - len(effective)
        attempts.append(dict(price_cutoff=str(cutoff), raw_common_rows=raw_count,
                             effective_common_rows=len(effective), deficit=deficit))
        print(f'[TRAIN/VAL calendar] cutoff={cutoff.date()} effective_rows={len(effective)} '
              f'required_rows={required_rows} deficit={deficit}', flush=True)
        if deficit == 0:
            retained = effective
            break
        if deficit < 0 or raw_count + deficit > len(common):
            raise ValueError('Cannot reproduce audited TRAIN/VAL calendar; STOP')
        raw_count += deficit
    market = {k: tuple(v) for k, v in expected['market_indices'].items()}
    selected = []
    for frame, group in zip(frames, ('stock', 'bond', 'commodity')):
        start, stop = market[group]
        names = expected['node_order'][start:stop]
        selected.append(frame.loc[retained, names])
    raw = pd.concat(selected, axis=1)
    if raw.isna().any().any():
        raise ValueError('Restricted TRAIN/VAL prices contain missing values; STOP')
    train_raw, val_raw = raw.values[:train_end], raw.values[train_end:]
    _, _, _, norm = normalize_data(train_raw, val_raw, np.empty((0, raw.shape[1]), dtype=raw.values.dtype))
    prepared = dict(raw_prices_train=train_raw, raw_prices_val=val_raw, norm_stats=norm,
                    market_indices=market, n_nodes=raw.shape[1], n_commodities=len(expected['commodity_order']),
                    train_returns=compute_returns(train_raw), feature_names=list(raw.columns))
    data = build_data(SimpleNamespace(batch_size=64, seq_len=20), prepared_train_val=prepared)
    actual = data_audit(data)
    if actual != expected:
        mismatched = [k for k in actual if actual[k] != expected.get(k)]
        raise ValueError(f'TRAIN/VAL fingerprint/order mismatch {mismatched}; STOP')
    data['origin_dates'] = {}
    for split, calendar in (('train', retained[:train_end]), ('val', retained[train_end:])):
        n = len(data['loaders'][split].dataset)
        data['origin_dates'][split] = calendar[19:19+n].to_numpy(dtype='datetime64[ns]')
    data['calendar_audit'] = dict(
        strategy='audited row counts + bounded effective-calendar reconstruction',
        raw_dates_removed=common[:raw_count].difference(retained).astype(str).tolist(),
        train_rows=train_end, val_rows=len(retained)-train_end, attempts=attempts,
        train_end_date=str(retained[train_end-1]), val_end_date=str(retained[-1]),
        test_prices_parsed=False, fingerprint_reproduced=True)
    assert set(data['loaders']) == {'train', 'val'}
    return data, actual


def route_statistics(pi):
    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 3 or pi.shape[1:] != (2, 3):
        raise ValueError('Expected (origins,2,3) routes')
    entropy = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum(-1)
    mean = pi.mean(0)
    return dict(group_order=['short', 'long'], candidate_order=['shared', 'state', 'interaction'],
        mean=mean.tolist(), std=pi.std(0).tolist(), min=pi.min(0).tolist(), max=pi.max(0).tolist(),
        quantiles={str(q): np.quantile(pi, q, axis=0).tolist() for q in (.1,.25,.5,.75,.9,.99)},
        mean_sample_entropy_nats=entropy(pi).mean(0).tolist(),
        entropy_of_mean_weights_nats=entropy(mean).tolist(),
        shared_weight_distribution={'mean':mean[:,0].tolist(), 'std':pi[:,:,0].std(0).tolist(),
                                    'quantiles':{str(q):np.quantile(pi[:,:,0],q,axis=0).tolist() for q in (.1,.5,.9)}})


@torch.inference_mode()
def evaluate(model, loader, device, *, intervention=None, override=None):
    model.eval()
    p, y, routes, residuals, relative, weighted, expert_norms = [], [], [], [], [], [], []
    base_norms, weighted_norms = [], []
    offset = 0
    f = getattr(model, 'short_long_dynamic_moe', None)
    if override is not None and np.ndim(override) == 3 and len(override) != len(loader.dataset):
        raise ValueError('Per-sample override must cover exactly all loader origins')
    for batch in full_loader(loader):
        x, target = batch[:2]
        weights = override
        if override is not None and np.ndim(override) == 3:
            weights = override[offset:offset+len(x)]
        context = f.intervention(intervention, weights) if intervention is not None else nullcontext()
        with context:
            pred = model(x.to(device))
        if pred.shape != target.shape or not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError('Invalid complete-population prediction/target')
        p.append(pred.cpu().numpy()); y.append(target.numpy()); offset += len(x)
        if f is not None:
            routes.append(f.last['pi'].cpu().numpy())
            expert_norms.append(f.last['experts'].norm(dim=-1).cpu().numpy())
            if f.residual:
                dn = f.last['deltas'].norm(dim=-1)
                bn = f.last['z_base'].norm(dim=-1).clamp_min(1e-12)
                residuals.append(dn.cpu().numpy())
                base_norms.append(f.last['z_base'].norm(dim=-1).cpu().numpy())
                weighted_norms.append(f.last['weighted_delta'].norm(dim=-1).cpu().numpy())
                relative.append((dn / bn[:,None]).cpu().numpy())
                weighted.append((f.last['weighted_delta'].norm(dim=-1) / bn[:,None]).cpu().numpy())
    predictions, targets = np.concatenate(p), np.concatenate(y)
    if len(predictions) != len(loader.dataset):
        raise ValueError('Full evaluation omitted or duplicated origins')
    metrics = {str(h): population_metrics(predictions[:,i], targets[:,i]) for i,h in enumerate((1,5,10,20))}
    arrays = dict(prediction=predictions, target=targets)
    if routes:
        arrays['pi'] = np.concatenate(routes)
        arrays['expert_norms'] = np.concatenate(expert_norms)
    if residuals:
        arrays.update(base_norms=np.concatenate(base_norms), weighted_residual_norms=np.concatenate(weighted_norms),
                      residual_norms=np.concatenate(residuals), residual_to_base=np.concatenate(relative),
                      weighted_residual_to_base=np.concatenate(weighted))
    return metrics, arrays


def frozen_diagnostics(model, loader, device, train_pi, val_pi, original, permutation_seed):
    # Float32 summation over ~1400 origins can move a valid probability mean
    # outside the 1e-6 simplex tolerance. Accumulate in float64; do not loosen
    # the override validator or silently normalize invalid source probabilities.
    train_pi = np.asarray(train_pi, dtype=np.float64)
    if (train_pi.ndim != 3 or train_pi.shape[1:] != (2, 3) or not len(train_pi)
            or not np.isfinite(train_pi).all() or (train_pi < 0).any()
            or not np.allclose(train_pi.sum(-1), 1., atol=1e-6, rtol=0)):
        raise ValueError('TRAIN routes must be finite normalized (N,2,3) probabilities')
    train_mean = train_pi.mean(axis=0, dtype=np.float64)
    val_pi = np.asarray(val_pi)
    if (val_pi.shape != (len(loader.dataset), 2, 3) or not np.isfinite(val_pi).all()
            or (val_pi < 0).any()
            or not np.allclose(val_pi.sum(-1), 1., atol=1e-6, rtol=0)):
        raise ValueError('VAL routes must match all loader origins and be normalized (N,2,3)')
    permutation = np.random.default_rng(permutation_seed).permutation(len(val_pi))
    specs = [('train_mean', 'override', train_mean),
             ('val_pair_permutation', 'override', val_pi[permutation]),
             ('shared_only', 'shared_only', None)]
    specs += [(f'mask_{g}_{k}', f'mask_{g}_{k}', None) for g in ('short','long') for k in (1,2)]
    result = dict(semantics='Frozen VAL inference intervention; not retrained ablation or economic causality',
                  permutation_seed=permutation_seed, permutation=permutation.tolist(),
                  train_mean=train_mean.tolist(), normal=original, interventions={})
    for label, operation, weights in specs:
        metrics, _ = evaluate(model, loader, device, intervention=operation, override=weights)
        result['interventions'][label] = dict(metrics=metrics, delta={
            h:{k:metrics[h][k]-original[h][k] for k in ('MAE','MSE','RMSE','Hit')} for h in original})
    return result


def preflight(seed=42):
    check_protocol()
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2, commodity_indices=(6,30))
    with preserved_rng():
        models, audit = matched_models(kwargs, seed)
        seed_all(seed)
        x = torch.randn(2,20,30,21)
        for model in models.values():
            model.eval()
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2,4,24) and torch.isfinite(y).all()
        equivalence = {}
        for variant in (B_STATIC,B_DYNAMIC):
            with torch.no_grad():
                candidate, original = models[variant](x), models[BASE_VARIANT](x)
                torch.testing.assert_close(candidate, original, atol=3e-7, rtol=1e-6)
                equivalence[variant] = float((candidate-original).abs().max())
        full_kwargs = dict(num_nodes=284, n_commodities=24, n_stock=248, n_bond=12, commodity_indices=(260,284))
        full_models, full_audit = matched_models(full_kwargs, seed)
        report = dict(status='PREFLIGHT_PASS', formal_training='NOT_RUN', real_data='NOT_READ', test_evaluated=False,
                      initialization=audit, full_initialization=full_audit,
                      zero_residual_prediction_max_abs_diff=equivalence,
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                      source_sha256={p:sha256(ROOT/p) for p in SOURCE_FILES},
                      software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                      parameters={v:parameters(m) for v,m in full_models.items()})
    print(json.dumps({k:v for k,v in report.items() if k not in ('initialization','full_initialization','source_sha256','git_status')}, indent=2))
    return report


def run(args):
    check_protocol()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    data, fingerprint = restricted_train_val(args.reference_config)
    models, init = matched_models(model_kwargs(data), args.seed)
    model = models[args.variant].to(args.device)
    del models
    name = f'{args.variant}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}'
    out, ck = EXPERIMENT_ROOT/name, CHECKPOINT_ROOT/name/'best.pt'
    out.mkdir(parents=True,exist_ok=False);ck.parent.mkdir(parents=True,exist_ok=False)
    metadata = dict(variant=args.variant, seed=args.seed, model_kwargs=model_kwargs(data), protocol=PROTOCOL,
        val_rule=VAL_RULE, data_audit=fingerprint, calendar_audit=data.get('calendar_audit'),
        origin_date_ranges={split:dict(first=str(dates[0]),last=str(dates[-1]),count=len(dates))
                            for split,dates in data['origin_dates'].items()}, reference_config=str(args.reference_config.resolve()),
        reference_config_sha256=sha256(args.reference_config),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        source_sha256={p:sha256(ROOT/p) for p in SOURCE_FILES}, parameters=parameters(model),
        initialization_audit=init, target_scale='decimal return', hit_storage='fraction',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         device=args.device,cuda=torch.version.cuda,
                         cudnn_deterministic=torch.backends.cudnn.deterministic,
                         cudnn_benchmark=torch.backends.cudnn.benchmark),
        rng_policy='seed all after fresh explicit shared-state copy; audit RNG isolated',
        router_initialization='dynamic final weight Normal(0,1e-3), bias 0; static logits zero',
        test_evaluated=False)
    write_json(out/'config.json',metadata)
    write_json(out/'data_audit.json',fingerprint)
    write_json(out/'initialization_audit.json',init)
    write_json(out/'RUN_STATUS.json',dict(status='RUNNING',test_evaluated=False))
    seed_all(args.seed)
    started = time.perf_counter()
    try:
        history = train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),
            torch.empty(0),torch.device(args.device),num_epochs=200,lr=1e-4,weight_decay=1e-5,
            patience=10,checkpoint_path=str(ck),checkpoint_metadata=metadata,
            epoch_history_callback=lambda h: write_json(out/'training_history.json',h))
        seconds = time.perf_counter()-started
        write_json(out/'training_history.json',history)
        payload = torch.load(ck,map_location=args.device,weights_only=False)
        if payload['metadata']['variant'] != args.variant or payload['best_epoch'] != history['best_epoch']:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(payload['model_state_dict'],strict=True)
        model.switching_latent_transformer.set_epoch(payload['best_epoch'])
        write_json(out/'best_checkpoint_metadata.json',dict(variant=args.variant,seed=args.seed,
            path=str(ck.resolve()),sha256=sha256(ck),best_epoch=payload['best_epoch'],
            best_val_prediction_huber_batch_mean=payload['best_val_loss']))
        complete_evaluation(model, data, fingerprint, out, ck, payload, seconds,
                            device=args.device, variant=args.variant, seed=args.seed,
                            permutation_seed=args.permutation_seed)
    except BaseException as exc:
        write_json(out/'RUN_STATUS.json',dict(status='INTERRUPTED_OR_FAILED',error=repr(exc),test_evaluated=False))
        raise


def complete_evaluation(model, data, fingerprint, out, ck, payload, seconds,
                        *, device, variant, seed, permutation_seed, reference_arrays=None):
    """Frozen TRAIN/VAL postprocessing only; never constructs an optimizer."""
    checkpoint_digest = sha256(ck)
    model.eval()
    all_metrics, summaries, split_arrays = {}, {}, {}
    for split in ('train','val'):
        metrics, arrays = evaluate(model,data['loaders'][split],device)
        arrays['origin_dates'] = data['origin_dates'][split]
        arrays['commodity_order'] = np.asarray(fingerprint['commodity_order'],dtype=str)
        if reference_arrays is not None:
            with np.load(reference_arrays/f'{split}_predictions_routes.npz', allow_pickle=False) as prior:
                for key in ('prediction', 'target', 'pi', 'origin_dates', 'commodity_order'):
                    if key not in arrays:
                        continue
                    if key not in prior or arrays[key].shape != prior[key].shape:
                        raise ValueError(f'Frozen reproduction shape mismatch: {split}/{key}')
                    same = (np.allclose(arrays[key], prior[key], atol=1e-6, rtol=0)
                            if key in ('prediction', 'pi') else np.array_equal(arrays[key], prior[key]))
                    if not same:
                        raise ValueError(f'Frozen reproduction mismatch: {split}/{key}; STOP')
        np.savez_compressed(out/f'{split}_predictions_routes.npz',**arrays)
        all_metrics[split], split_arrays[split] = metrics, arrays
        summaries[split] = dict(routing=route_statistics(arrays['pi']) if 'pi' in arrays else None,
            norms={k:arrays[k].mean(0).tolist() for k in ('expert_norms','base_norms','residual_norms','weighted_residual_norms','residual_to_base','weighted_residual_to_base') if k in arrays},
            norm_aggregation='mean over sample-wise L2 norms / ratios; eps=1e-12')
    diagnostics = {'status':'NOT_APPLICABLE'}
    if variant in DYNAMIC_STUDY_VARIANTS:
        diagnostics = frozen_diagnostics(model,data['loaders']['val'],device,
            split_arrays['train']['pi'],split_arrays['val']['pi'],all_metrics['val'],permutation_seed)
    if sha256(ck) != checkpoint_digest:
        raise ValueError('Checkpoint changed during frozen evaluation')
    result = dict(status='COMPLETE',variant=variant,seed=seed,checkpoint=str(ck.resolve()),
        checkpoint_sha256=checkpoint_digest,best_epoch=payload['best_epoch'],
        best_val_prediction_huber_batch_mean=payload['best_val_loss'],train_seconds=seconds,
        parameters=parameters(model),metrics=all_metrics,diagnostics=summaries,test_evaluated=False)
    write_json(out/'metrics.json',all_metrics)
    write_json(out/'routing_diagnostics.json',summaries)
    write_json(out/'val_frozen_interventions.json',diagnostics)
    write_json(out/'result.json',result)
    write_json(out/'RUN_STATUS.json',dict(status='COMPLETE',test_evaluated=False))
    print(f'COMPLETE: {out}\nCheckpoint: {ck}')


ROUTE_OVERRIDE_ERROR = "ValueError('Override must be normalized (2,3) or (B,2,3)')"


def finish_runs(paths, device, permutation_seed=42):
    """Complete failed diagnostics from frozen best checkpoints; NOT resume training.

    Original artifacts stay read-only. New outputs live in a postprocess_* child.
    Training provenance is retained separately from the new evaluation source.
    """
    check_protocol()
    data_cache = {}
    for directory in paths:
        directory = Path(directory).resolve()
        read = lambda name: json.loads((directory/name).read_text())
        status, cfg, history, best = [read(p) for p in (
            'RUN_STATUS.json', 'config.json', 'training_history.json', 'best_checkpoint_metadata.json')]
        ck = Path(best['path'])
        if sha256(ck) != best['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {ck}')
        if status['status'] == 'COMPLETE':
            print(f'SKIP already complete: {directory}', flush=True)
            continue
        if status.get('error') != ROUTE_OVERRIDE_ERROR:
            raise ValueError(f'Not the known post-training diagnostic failure: {directory}')
        completed = []
        for result_path in directory.glob('postprocess_*/result.json'):
            result = json.loads(result_path.read_text())
            done = json.loads((result_path.parent/'RUN_STATUS.json').read_text())
            if (result['status'] == done['status'] == 'COMPLETE'
                    and result['checkpoint_sha256'] == best['sha256']):
                completed.append(str(result_path))
        if completed:
            print(f'SKIP already postprocessed: {completed}', flush=True)
            continue
        if (cfg['variant'] not in DYNAMIC_STUDY_VARIANTS or cfg['protocol'] != PROTOCOL
                or not cfg['initialization_audit']['PASS'] or cfg['test_evaluated']):
            raise ValueError('Invalid original study identity/protocol')
        # The fix changes only this orchestration script; never waive model,
        # feature, dataset or training implementation provenance checks.
        changed = [p for p, digest in cfg['source_sha256'].items()
                   if p != 'cmgm/scripts/d0b_short_long_dynamic_moe.py' and sha256(ROOT/p) != digest]
        if changed:
            raise ValueError(f'Non-runner source changed: {changed}; STOP')
        payload = torch.load(ck, map_location='cpu', weights_only=False)
        losses = np.asarray(history['val_loss'], dtype=np.float64)
        epoch = history['best_epoch']
        if (not len(losses) or not np.isfinite(losses).all() or epoch != int(losses.argmin())+1
                or json.loads(json.dumps(payload['metadata'])) != cfg or payload['best_epoch'] != epoch
                or best['best_epoch'] != epoch or best['variant'] != cfg['variant']
                or best['seed'] != cfg['seed']
                or payload['history']['val_loss'] != history['val_loss']
                or not np.isclose(payload['best_val_loss'], losses.min(), atol=1e-12, rtol=0)
                or best['best_val_prediction_huber_batch_mean'] != payload['best_val_loss']
                or not np.isfinite(history.get('train_time', np.nan))):
            raise ValueError('Completed training/checkpoint/history identity mismatch')
        reference = Path(cfg['reference_config'])
        if sha256(reference) != cfg['reference_config_sha256']:
            raise ValueError('Reference config changed')
        key = (str(reference.resolve()), cfg['reference_config_sha256'])
        if key not in data_cache:
            data_cache[key] = restricted_train_val(reference)
        data, fingerprint = data_cache[key]
        # JSON turns tuple node bounds into lists.
        if (fingerprint != cfg['data_audit']
                or json.loads(json.dumps(model_kwargs(data))) != cfg['model_kwargs']):
            raise ValueError('Original TRAIN/VAL or model config mismatch')
        with preserved_rng():
            model = HeteroMixHopCMGM(variant=cfg['variant'], **cfg['model_kwargs'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        # evaluate() uses inference_mode throughout; retain requires_grad flags
        # so the reported trainable architecture size matches the trained model.
        model.to(device).eval()
        model.switching_latent_transformer.set_epoch(epoch)
        out = directory/f'postprocess_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}'
        out.mkdir(exist_ok=False)
        for name in ('config.json', 'data_audit.json', 'initialization_audit.json',
                     'training_history.json', 'best_checkpoint_metadata.json'):
            shutil.copyfile(directory/name, out/name)
        provenance = dict(original_run=str(directory), checkpoint_sha256=best['sha256'],
            mode='frozen postprocessing only; no training or optimizer', test_evaluated=False,
            permutation_seed=permutation_seed, device=device,
            source_sha256={p: sha256(ROOT/p) for p in SOURCE_FILES},
            training_time_source='original training_history.train_time; excludes this postprocessing',
            training_provenance='config.json preserved verbatim; evaluation source recorded here')
        write_json(out/'postprocess_provenance.json', provenance)
        write_json(out/'RUN_STATUS.json', dict(status='POSTPROCESSING', test_evaluated=False))
        try:
            complete_evaluation(model, data, fingerprint, out, ck, payload, history['train_time'],
                device=device, variant=cfg['variant'], seed=cfg['seed'],
                permutation_seed=permutation_seed, reference_arrays=directory)
            if sha256(ck) != best['sha256']:
                raise ValueError('Checkpoint changed during frozen evaluation')
        except BaseException as exc:
            write_json(out/'RUN_STATUS.json', dict(status='POSTPROCESSING_FAILED', error=repr(exc),
                                                  test_evaluated=False))
            raise


def compare(paths, output):
    if len(paths)!=6:
        raise ValueError('Exactly six seed42 result.json paths required')
    records, configs, index = {}, {}, {}
    for path in paths:
        r=json.loads(path.read_text());v=r['variant']
        if v not in VARIANTS or v in records or r['seed']!=42 or r['status']!='COMPLETE' or r['test_evaluated']:
            raise ValueError(f'Invalid result {path}')
        if sha256(Path(r['checkpoint']))!=r['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        c=json.loads((path.parent/'config.json').read_text())
        h=json.loads((path.parent/'training_history.json').read_text())
        if (c['variant']!=v or c['seed']!=42 or not c['initialization_audit']['PASS']
                or h['best_epoch']!=r['best_epoch']
                or not np.isclose(min(h['val_loss']),r['best_val_prediction_huber_batch_mean'],rtol=0,atol=1e-12)):
            raise ValueError('Result/config/history/init mismatch')
        if not all(np.isfinite(r['metrics']['val'][str(horizon)][metric])
                   for horizon in (1,5,10,20) for metric in ('MAE','MSE','RMSE','Hit')):
            raise ValueError('Nonfinite validation metrics')
        records[v]=r;configs[v]=c;index[v]=str(path.resolve())
    reference=configs[BASE_VARIANT]
    for c in configs.values():
        for key in ('protocol','data_audit','source_sha256','seed','rng_policy'):
            if c[key]!=reference[key]:raise ValueError(f'Incomparable {key}')
    mae=lambda v:records[v]['metrics']['val']['5']['MAE']
    pairs={f'{a}_vs_{b}':dict(delta=mae(ALIASES[a])-mae(ALIASES[b]),
               relative_improvement_percent=100*(mae(ALIASES[b])-mae(ALIASES[a]))/mae(ALIASES[b]))
           for a,b in VAL_RULE['comparisons']}
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for v in VARIANTS:
        r=records[v];row=dict(variant=v,best_epoch=r['best_epoch'],parameters=r['parameters']['total'],
            train_seconds=r['train_seconds'],val_huber=r['best_val_prediction_huber_batch_mean'])
        for h in (1,5,10,20):
            for metric in ('MAE','RMSE','Hit'):
                row[f'VAL_{metric}@{h}d']=r['metrics']['val'][str(h)][metric]*(100 if metric=='Hit' else 1)
        rows.append(row)
    with (output/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(output/'comparison.json',dict(status='COMPLETE',test_evaluated=False,val_rule=VAL_RULE,
        pairs=pairs,rows=rows,file_index=index,hit_display='percent',limitations='Single seed VAL; no causal architecture attribution or TEST claim'))
    report=['# Seed42 TRAIN/VAL dynamic task routing comparison', '',
            'Primary: VAL MAE@5d. Checkpoints selected by prediction-only four-horizon VAL Huber.', '',
            '| Variant | VAL MAE@1d | VAL MAE@5d | VAL MAE@10d | VAL MAE@20d | VAL Huber |',
            '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append('| '+row['variant']+' | '+' | '.join(
            f"{row[f'VAL_MAE@{h}d']:.9g}" for h in (1,5,10,20))+f" | {row['val_huber']:.9g} |")
    report += ['', 'Paired differences (candidate minus reference; negative is lower error):', '']
    for name,stats in pairs.items():
        report.append(f"- {name}: delta={stats['delta']:+.9g}, relative improvement={stats['relative_improvement_percent']:+.5f}%")
    report += ['', 'These are single-seed VAL comparisons. Routing variation alone is not evidence of predictive value; '
               'inspect each frozen VAL intervention alongside its static control. A/B differ in experts, normalization '
               'and shared path. No TEST evaluation or automatic follow-up.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--finish-run',type=Path,nargs='+',help='Frozen postprocessing of explicit failed run directories; never train')
    parser.add_argument('--finish-failed',action='store_true',help='Find only the known routing-mean diagnostic failures and finish them without training')
    parser.add_argument('--variant',choices=(*ALIASES,*VARIANTS),default='b-dynamic')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--permutation-seed',type=int,default=42)
    parser.add_argument('--reference-config',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--compare-results',type=Path,nargs=6)
    parser.add_argument('--comparison-output',type=Path,default=EXPERIMENT_ROOT/'seed42_comparison')
    parser.add_argument('--preflight-output',type=Path)
    args=parser.parse_args();args.variant=ALIASES.get(args.variant,args.variant)
    if sum(bool(x) for x in (args.run, args.finish_run, args.finish_failed, args.compare_results)) > 1:
        parser.error('Training, postprocessing and comparison are separate operations')
    if args.finish_run or args.finish_failed:
        paths = args.finish_run
        if args.finish_failed:
            paths = [p.parent for p in sorted(EXPERIMENT_ROOT.glob('*/RUN_STATUS.json'))
                     if json.loads(p.read_text()).get('error') == ROUTE_OVERRIDE_ERROR]
        if not paths:
            print('No matching post-training diagnostic failures found.')
        finish_runs(paths, args.device, args.permutation_seed)
        return
    if args.compare_results:
        if args.run:parser.error('Comparison and training are separate operations')
        compare(args.compare_results,args.comparison_output)
    elif args.run:run(args)
    else:
        report=preflight(args.seed)
        if args.preflight_output:
            if args.preflight_output.exists():raise FileExistsError(args.preflight_output)
            args.preflight_output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.preflight_output,report)


if __name__=='__main__':
    main()
