"""One Full bottleneck16 experiment. Default: synthetic preflight; never evaluates TEST.

No alternate trainer, no sweep, no automatic ablations, no reference retraining.
The model factory is the integration point for future shared-G0 / post-LN-null
controls; those controls are deliberately not exposed or executed by this CLI.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.candidate_moe_fusion import VARIANT as ORIGINAL, BOTTLENECK16_VARIANT as VARIANT
from cmgm.scripts.baseline_protocol import seed_all, data_audit
from cmgm.scripts.d0b_moe_audit import make_model as native_model
from cmgm.scripts.d0b_candidate_moe_audit import sanity
from cmgm.scripts.d0b_candidate_moe_diagnostics import routing, distribution
from cmgm.scripts.formal_v2_protocol import atomic_json, sha, metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / 'experiments/d0b_candidate_moe_bottleneck16'
DEFAULT_AUDIT = ROOT / 'experiments/baseline_comparison_deep_v3/20260922_143029/data_audit.json'
TRAIN_SCHEMA_CACHE = ROOT / 'experiments/formal_baseline_benchmark_v2/20260910_131615/arrays/train_x.npy'
PROTOCOL = dict(
    variant=VARIANT, expert_hidden_dim=16, seq_len=20, feature_dim=21,
    horizons=[1, 5, 10, 20], optimizer='Adam', lr=1e-4, weight_decay=1e-5,
    batch_size=64, max_epochs=200, patience=10,
    scheduler='ReduceLROnPlateau', scheduler_factor=.5, scheduler_patience=5,
    train_shuffle=False, train_drop_last=True, full_eval_drop_last=False,
    prediction_loss='sum four equally weighted Huber(delta=.02)',
    selection='prediction-only four-horizon VAL Huber; arithmetic mean of batch means',
    scheduler_metric='same prediction-only VAL objective', early_stopping_metric='same prediction-only VAL objective',
    val5='diagnostic only', switch_beta_max=5e-4, switch_warmup_epochs=20,
    switch_schedule='native: beta_max * clamp((epoch-1)/19, 0, 1)',
    moe_auxiliary_loss=None, routing_warmup=None, gradient_clipping=None,
    expert_dropout=.1, head_dropout=.3, router='unchanged candidate-aware 256->64->2; final layer zero',
    target_type='return', test_policy='No TEST dataset, features, labels, predictions or metrics in this entry point',
    future_controls=['shared-G0 with native KL', 'post-normalization micro readout null; recurrence and KL retained'],
    future_control_policy='Not run here; use bottleneck16 Full and identical seed/window/definition as reference',
)


def check_config():
    expected = dict(SEQ_LEN=20, FEATURE_DIM=21, MULTI_HORIZONS=[1, 5, 10, 20],
                    TARGET_TYPE='return', LOSS_TYPE='huber', HUBER_DELTA=.02)
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f'Frozen configuration changed: {key}')


def make_model(data, seed=42, variant=VARIANT):
    """Construct from seed only, never from a fitted checkpoint."""
    seed_all(seed)
    return native_model(data, variant)


def counts(model):
    f = model.candidate_moe_fusion
    count = lambda module: sum(p.numel() for p in module.parameters())
    return dict(total=count(model), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                temporal_expert=count(f.temporal_expert), interaction_expert=count(f.interaction_expert),
                experts=count(f.temporal_expert)+count(f.interaction_expert), router=count(f.router))


def initialization_audit(data, seed):
    original = make_model(data, seed, ORIGINAL)
    compressed = make_model(data, seed)
    a, b = dict(original.named_parameters()), dict(compressed.named_parameters())
    replaced = ('candidate_moe_fusion.temporal_expert.', 'candidate_moe_fusion.interaction_expert.')
    shared = sorted(n for n in a if not n.startswith(replaced))
    differences = {n: float((a[n]-b[n]).detach().abs().max()) for n in shared}
    mismatch = [n for n, value in differences.items() if value != 0]
    old, new = counts(original), counts(compressed)
    result = dict(seed=seed, n_nodes=data['n_nodes'], original=old, bottleneck16=new,
                  parameter_delta=new['total']-old['total'], shared_parameter_count=sum(a[n].numel() for n in shared),
                  shared_max_abs_diff=max(differences.values()), mismatch_count=len(mismatch),
                  mismatched_parameters=mismatch, shared_parameter_differences=differences,
                  replaced_parameters=[n for n in b if n.startswith(replaced)],
                  initialization='Original complete Candidate construction, then replace only two experts; no trained checkpoint')
    result['PASS'] = (not mismatch and a.keys() == b.keys() and old['experts'] == 20736
                      and new['temporal_expert'] == 2128 and new['interaction_expert'] == 3152
                      and result['parameter_delta'] == -15456 and old['router'] == new['router'])
    if not result['PASS']:
        raise ValueError(f'Initialization/parameter audit failed: {result}')
    return compressed, result


def source_record():
    files = sorted((ROOT / 'cmgm/models').glob('*.py')) + [
        Path(__file__), ROOT/'cmgm/scripts/baseline_protocol.py', ROOT/'cmgm/scripts/main_ablation.py',
        ROOT/'cmgm/scripts/d0b_candidate_moe_audit.py', ROOT/'cmgm/scripts/d0b_moe_audit.py',
        ROOT/'cmgm/scripts/d0b_candidate_moe_diagnostics.py', ROOT/'cmgm/scripts/formal_v2_protocol.py',
        ROOT/'cmgm/data/data_loader.py', ROOT/'cmgm/data/feature_builder.py', ROOT/'cmgm/config.py',
        ROOT/'cmgm/training/train.py', ROOT/'cmgm/training/metric_standard.py']
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
                versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))


def synthetic_preflight(seed, out):
    """No file containing real features or targets is opened here."""
    check_config()
    data = dict(n_nodes=284, market_indices=dict(stock=(0, 248), bond=(248, 260), commodity=(260, 284)))
    model, audit = initialization_audit(data, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(2, 20, 284, 21, generator=g)
    y = torch.randn(2, 4, 24, generator=g) * .02
    structural = sanity(model, x, y)
    if not structural['PASS']:
        raise ValueError(f'Structural preflight failed: {structural}')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'initialization_audit.json', audit)
    atomic_json(out/'structural_sanity.json', structural)
    atomic_json(out/'config.json', {**PROTOCOL, 'seed': seed, 'synthetic_only': True})
    atomic_json(out/'source_hashes.json', source_record())
    atomic_json(out/'results.json', dict(status='SYNTHETIC PREFLIGHT PASS; NO TRAINING',
                                        data_access='synthetic only; no TRAIN/VAL/TEST read', parameters=audit))
    print(json.dumps(dict(status='PREFLIGHT PASS', output=str(out), parameters=audit['bottleneck16'],
                          parameter_delta=audit['parameter_delta'], shared_max_abs_diff=audit['shared_max_abs_diff']), indent=2))


def restore_train_schema(aligned, mi, expected):
    """Recover historical node identities from TRAIN-only observations.

    Cache matching is only a schema proposal. The unchanged, exact SHA256 audit
    of BOTH full TRAIN/VAL feature and raw-price arrays below is authoritative.
    """
    mapping = expected['mapping']
    n = max(v.get('full_node', v.get('node_index')) for v in mapping)+1
    if aligned.shape[1] == n:
        return aligned, mi, dict(required=False)
    path = TRAIN_SCHEMA_CACHE
    if not path.exists():
        raise ValueError(f'Frozen TRAIN schema required: missing {path}; no TEST fallback')
    cached = np.load(path, mmap_mode='r', allow_pickle=False)
    origins = expected['split_fingerprint']['train']['origins']
    if cached.shape != (origins, 20*n*21) or cached.dtype != np.float32:
        raise ValueError('Historical TRAIN schema cache shape/dtype mismatch')
    raw = np.ascontiguousarray(aligned.to_numpy(dtype=np.float32))
    end = expected['split_fingerprint']['train']['timeline']
    normalized = (raw-raw[:end].mean(0, keepdims=True))/np.maximum(raw[:end].std(0, keepdims=True), config.FEAT_ZSCORE_EPS)
    anchors = sorted(set((0, origins//2, origins-1)))
    current = np.concatenate([normalized[i:i+20] for i in anchors])
    historical = np.concatenate([cached[i].reshape(20, n, 21)[:, :, 0] for i in anchors])
    indices = []
    for j in range(n):
        error = np.max(np.abs(current-historical[:, j:j+1]), axis=0)
        candidates = np.flatnonzero(error < 1e-5)
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous/missing TRAIN schema match for historical node {j}; STOP')
        indices.append(int(candidates[0]))
    if len(set(indices)) != n or indices != sorted(indices):
        raise ValueError('TRAIN schema does not preserve unique original node order')
    groups = [sum(mi[k][0] <= i < mi[k][1] for i in indices) for k in ('stock', 'bond', 'commodity')]
    if groups[2] != 24 or sum(groups[:2]) != mapping[0].get('full_node', mapping[0].get('node_index')):
        raise ValueError('Recovered market/commodity boundaries disagree with audit')
    restored = aligned.iloc[:, indices]
    new_mi = dict(stock=(0, groups[0]), bond=(groups[0], sum(groups[:2])), commodity=(sum(groups[:2]), n))
    record = dict(required=True, source=str(path), anchors=anchors, source_shape=list(cached.shape),
                  matched_TRAIN_price_channels_sha256=__import__('hashlib').sha256(historical.tobytes()).hexdigest(),
                  recovered_node_names=list(restored.columns),
                  verification='Schema proposal only; exact full TRAIN/VAL SHA256 equality is mandatory')
    return restored, new_mi, record


def train_val_data(reference_audit):
    """Reuse original preprocessing, but skip TEST value rows at the CSV parser.

    Only date metadata spans the full source, to retain the original chronological
    boundaries. Dropping future rows can expose historical bfill/column-selection
    dependencies: exact prior TRAIN/VAL fingerprints MUST match or we stop.
    """
    from cmgm.data.data_loader import (load_stock_prices, load_bond_prices, load_commodity_prices,
                                      align_markets, normalize_data, compute_returns)
    from cmgm.scripts.main_ablation import build_data
    expected = json.loads(Path(reference_audit).read_text())
    if expected.get('PASS') is not True:
        raise ValueError('A passing historical data audit is required before reading prices')
    paths = [config.STOCK_FILE, config.BOND_FILE, config.COMMODITY_FILE]
    dates = [pd.to_datetime(pd.read_csv(p, encoding='utf-8-sig', usecols=['date'])['date']) for p in paths]
    common = pd.DatetimeIndex(dates[0].unique())
    for d in dates[1:]:
        common = common.intersection(pd.DatetimeIndex(d.unique()))
    common = common.sort_values()
    # Use the audited split sizes, NOT ratios over raw CSV date rows: pivot_table
    # removes dates with no valid quotes. Raw date metadata is only an upper bound.
    train_end = expected['split_fingerprint']['train']['timeline']
    val_end = train_end + expected['split_fingerprint']['val']['timeline']
    if train_end < 104 or val_end-train_end < 40:
        raise ValueError('Insufficient original TRAIN/VAL history')
    calendar_position = val_end-1
    while True:
        if calendar_position >= len(common):
            raise ValueError('Cannot recover audited TRAIN/VAL calendar from source')
        cutoff = common[calendar_position]
        frames, skipped = [], []
        for path, d, load in zip(paths, dates, [load_stock_prices, load_bond_prices, load_commodity_prices]):
            excluded = (np.flatnonzero((d > cutoff).to_numpy()) + 1).tolist()
            skipped.append(len(excluded))
            frames.append(load(path, read_csv_kwargs={'skiprows': excluded}))
        aligned, mi = align_markets(*frames)
        missing = val_end-len(aligned)
        if missing < 0:
            raise ValueError('Restricted parser exceeded audited TRAIN/VAL size; STOP')
        if missing == 0:
            break
        # Advancing by the deficit cannot overshoot: each added common date
        # contributes at most one aligned row. Never parse beyond required VAL.
        calendar_position += missing
    common = aligned.index
    # The old full-calendar schema can drop columns for missingness after VAL.
    # Recover ONLY its already-frozen ordering from a TRAIN input cache, never
    # inspect held-out values or invent a fresh node selection for this variant.
    aligned, mi, schema_audit = restore_train_schema(aligned, mi, expected)
    raw = aligned.values
    raw_train, raw_val = raw[:train_end], raw[train_end:]
    _, _, _, norm_stats = normalize_data(raw_train, raw_val, np.empty((0, raw.shape[1]), dtype=raw.dtype))
    prepared = dict(raw_prices_train=raw_train, raw_prices_val=raw_val, norm_stats=norm_stats,
                    n_nodes=raw.shape[1], n_commodities=mi['commodity'][1]-mi['commodity'][0],
                    market_indices=mi, feature_names=aligned.columns.tolist(), train_returns=compute_returns(raw_train))
    data = build_data(SimpleNamespace(batch_size=64, seq_len=20), prepared_train_val=prepared)
    assert set(data['loaders']) == {'train', 'val'} and 'raw_prices_test' not in data
    audit = data_audit(data)
    if not audit['PASS']:
        raise ValueError('Missing/passing historical data audit required')
    for split in ('train', 'val'):
        if audit['split_fingerprint'][split] != expected['split_fingerprint'][split]:
            raise ValueError(f'{split} differs from audited original pipeline; STOP, no TEST fallback')
    canonical = lambda rows: [(v['commodity'], v.get('full_node', v.get('node_index')),
                              v.get('target_output', v.get('target_index'))) for v in rows]
    if canonical(audit['mapping']) != canonical(expected['mapping']):
        raise ValueError('Commodity order mismatch; STOP')
    audit['source'] = dict(reference_audit=str(Path(reference_audit).resolve()), reference_sha256=sha(reference_audit),
                          price_rows_after_val_skipped=skipped, metadata_scope='date column only over full calendar',
                          input_scope='TRAIN/VAL numeric rows only; no TEST features/targets/dataset',
                          train_dates=[str(common[0]), str(common[train_end-1])],
                          val_dates=[str(common[train_end]), str(cutoff)],
                          train_end=train_end, val_end=val_end, schema_recovery=schema_audit)
    return data, audit


def data_loaders(data, seed, full=False):
    if set(data['loaders']) != {'train', 'val'}:
        raise ValueError('This experiment accepts TRAIN/VAL only')
    return {s: DataLoader(data['loaders'][s].dataset, batch_size=64, shuffle=False,
                         drop_last=s == 'train' and not full, num_workers=0,
                         generator=torch.Generator().manual_seed(seed)) for s in ('train', 'val')}


@torch.no_grad()
def evaluate_train_val(model, data, seed, device, out):
    model.eval()
    result = {}
    arrays = {}
    for split, loader in data_loaders(data, seed, full=True).items():
        ps, ys, pis, es, hs = [], [], [], [], []
        for x, y in loader:
            p = model(x.to(device))
            if p.shape != y.shape or p.shape[1:] != (4, 24) or not torch.isfinite(p).all():
                raise ValueError('Invalid prediction')
            c = model.candidate_moe_fusion.last
            ps.append(p.cpu().numpy()); ys.append(y.numpy()); pis.append(c['pi'].cpu().numpy())
            es.append(c['experts'].double().cpu()); hs.append(c['h_moe'].double().cpu())
        p, y, pi = np.concatenate(ps), np.concatenate(ys), np.concatenate(pis)
        e, h = torch.cat(es), torch.cat(hs)
        result[split] = dict(metrics=metrics(p, y), routing=routing(pi),
                            experts=dict(mean_L2=e.norm(dim=-1).mean(0).tolist(), h_moe_mean_L2=float(h.norm(dim=-1).mean()),
                                         mean_absolute_disagreement=float((e[:, 0]-e[:, 1]).abs().mean()),
                                         cosine_similarity=distribution(torch.nn.functional.cosine_similarity(e[:, 0], e[:, 1], dim=-1).numpy())))
        arrays.update({split+'_prediction': p, split+'_target': y, split+'_pi': pi})
    np.savez_compressed(out/'train_val_predictions.npz', **arrays)
    return result


def write_report(r, out):
    lines = ['# Candidate MoE Bottleneck16', '', f"Status: {r['status']}", '',
             'Only expert hidden capacity changes (64 → 16). No scientific outcome is assumed.',
             'Temporal: t + Linear(16,64)(Dropout(ReLU(Linear(64,16)(t)))).',
             'Interaction: Linear(16,64)(Dropout(ReLU(Linear(128,16)([s,t])))); no residual.',
             'Router, raw candidate mixture, shared head and backbones are unchanged.',
             'Objective: sum_h Huber(.02) + native Switch KL. Selection/scheduler/early stopping: prediction-only VAL batch mean.',
             'No TEST evaluation is implemented in this entry point. TRAIN/VAL pooled metrics include zero-target sign agreement.',
             'Original Candidate comparison: PENDING — no audited performance reference was supplied; no historical TEST values substituted.',
             'Future shared-G0/no-microstate controls are not run; use this Full as their own reference.',
             'Single run at the specified seed; no significance or mechanism-restoration claim.', '',
             'Initialization and counts: see initialization_audit.json. Data/order: data_audit.json.',
             f"Checkpoint: {r.get('checkpoint', 'not trained')}",
             f"Training seconds: {r.get('training_seconds', 'N/A')}", '',
             '| Split | Horizon | MAE | MSE | RMSE | Hit% |', '|---|---:|---:|---:|---:|---:|']
    for split, row in r.get('evaluation', {}).items():
        for horizon, m in row['metrics'].items():
            lines.append(f"| {split} | {horizon} | {m['MAE']:.10g} | {m['MSE']:.10g} | {m['RMSE']:.10g} | {m['Hit']:.10g} |")
    lines += ['', 'Capacity interpretation: preserved prediction with stable mechanism deltas supports further study; '
              'near-zero deltas do not establish restored mechanisms. A worse Full with larger deltas is not success. '
              'No ablation conclusions are available from Full alone.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'results.json', r)


def execute(args):
    check_config()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback for formal training')
    data, audit = train_val_data(args.data_audit)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise RuntimeError('Output already contains a run; use --evaluate-completed after completed training, never overwrite')
    model, init = initialization_audit(data, args.seed)
    model.to(device)
    x, y = next(iter(data_loaders(data, args.seed, full=True)['val']))
    structural = sanity(model, x[:2].to(device), y[:2].to(device))
    if not structural['PASS']:
        raise ValueError(f'Preflight failed: {structural}')
    source = source_record()
    r = dict(status='DATA PREFLIGHT PASS', config={**PROTOCOL, 'seed': args.seed}, data=audit,
             source=source, initialization=init, structural_sanity=structural,
             reference_comparison={'status': 'PENDING', 'reason': 'No audited original performance comparison requested/supplied'})
    for filename, value in [('config', r['config']), ('data_audit', audit), ('source_hashes', source),
                            ('initialization_audit', init), ('structural_sanity', structural)]:
        atomic_json(out/(filename+'.json'), value)
    write_report(r, out)
    if not args.run:
        print(f'TRAIN/VAL data preflight PASS: {out}; no training')
        return
    # Persistent per-seed receipt prevents accidental repeated formal training, even
    # if the user supplies a different timestamp/output directory after interruption.
    receipt = DEFAULT_ROOT / f'seed{args.seed}_formal_run.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with (DEFAULT_ROOT/f'seed{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            raise RuntimeError(f'Formal run already reserved: {receipt}; STOP for review, no automatic retry')
        cp = ROOT/'checkpoints/candidate_moe_bottleneck16'/f'seed{args.seed}'/out.name/(VARIANT+'_best.pt')
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.exists():
            raise FileExistsError(cp)
        atomic_json(receipt, dict(status='STARTED', output=str(out), checkpoint=str(cp), seed=args.seed))
        r.update(status='TRAINING', checkpoint=str(cp))
        write_report(r, out)
        # Audit forwards consumed RNG; reconstruct from seed exactly once before fit.
        del model
        model = make_model(data, args.seed).to(device)
        ls = data_loaders(data, args.seed)
        def history_callback(history):
            atomic_json(out/'training_history.json', history)
            atomic_json(out/'routing_history.json', history['candidate_routing_history'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            history = train(model, ls['train'], ls['val'], torch.empty((2, 0), dtype=torch.long, device=device),
                            torch.empty(0, device=device), device, num_epochs=200, lr=1e-4, weight_decay=1e-5,
                            patience=10, checkpoint_path=str(cp),
                            checkpoint_metadata=dict(variant=VARIANT, seed=args.seed, config=r['config'], data=audit, source=source),
                            epoch_history_callback=history_callback)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            r['training_seconds'] = time.perf_counter()-start
            checkpoint = torch.load(cp, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.switching_latent_transformer.set_epoch(checkpoint['best_epoch'])
            checkpoint.update(training_complete=True, training_seconds=r['training_seconds'], history=history)
            temporary = cp.with_suffix('.tmp')
            torch.save(checkpoint, temporary); temporary.replace(cp)
            r.update(status='TRAINING COMPLETE; FROZEN CHECKPOINT', checkpoint_sha256=sha(cp),
                     best_epoch=checkpoint['best_epoch'], best_val_loss=checkpoint['best_val_loss'], training_complete=True)
            history_callback(history)
            atomic_json(out/'best_checkpoint_metadata.json', {k: r[k] for k in
                        ('checkpoint', 'checkpoint_sha256', 'best_epoch', 'best_val_loss', 'training_seconds', 'training_complete')})
            write_report(r, out)
            atomic_json(receipt, dict(status='TRAINING COMPLETE', output=str(out), checkpoint=str(cp), sha256=sha(cp)))
            complete_evaluation(model, data, r, out, device)
        except BaseException as exc:
            r['status'] = 'EVALUATION INTERRUPTED' if r.get('training_complete') else 'INTERRUPTED — REVIEW REQUIRED'
            r['error'] = f'{type(exc).__name__}: {exc}'
            write_report(r, out)
            raise


def complete_evaluation(model, data, r, out, device):
    before = sha(r['checkpoint'])
    r['evaluation'] = evaluate_train_val(model, data, r['config']['seed'], device, out)
    if before != r['checkpoint_sha256'] or sha(r['checkpoint']) != before:
        raise ValueError('Frozen checkpoint changed during evaluation')
    r['status'] = 'COMPLETE — TRAIN/VAL ONLY'
    atomic_json(out/'train_val_metrics.json', {s: v['metrics'] for s, v in r['evaluation'].items()})
    atomic_json(out/'routing_diagnostics.json', {s: v['routing'] for s, v in r['evaluation'].items()})
    atomic_json(out/'expert_diagnostics.json', {s: v['experts'] for s, v in r['evaluation'].items()})
    write_report(r, out)
    print(f"COMPLETE: {out}; no TEST evaluated")


def evaluate_completed(args):
    out = args.evaluate_completed.resolve()
    r = json.loads((out/'results.json').read_text())
    if r['config'] != {**PROTOCOL, 'seed': r['config']['seed']}:
        raise ValueError('Completed run is not this fixed protocol')
    if not r.get('training_complete') or sha(r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Only a completed, identity-verified frozen checkpoint can be evaluated')
    if source_record()['hashes'] != r['source']['hashes']:
        raise ValueError('Source drift; STOP for review')
    data, audit = train_val_data(args.data_audit)
    if audit != r['data']:
        raise ValueError('Data/provenance drift')
    device = torch.device(args.device)
    model = make_model(data, r['config']['seed']).to(device)
    cp = torch.load(r['checkpoint'], map_location=device, weights_only=False)
    if not cp.get('training_complete') or cp['metadata']['config'] != r['config']:
        raise ValueError('Checkpoint protocol mismatch')
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model.switching_latent_transformer.set_epoch(cp['best_epoch'])
    complete_evaluation(model, data, r, out, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--run', action='store_true', help='ONE formal Full training; TRAIN/VAL only')
    group.add_argument('--data-preflight', action='store_true', help='Audit real TRAIN/VAL only, without training')
    group.add_argument('--evaluate-completed', type=Path, help='Re-evaluate completed frozen run on TRAIN/VAL only; never retrain')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--data-audit', type=Path, default=DEFAULT_AUDIT)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.output is None:
        prefix = 'run' if args.run else 'data_preflight' if args.data_preflight else 'synthetic_preflight'
        args.output = DEFAULT_ROOT/f'{prefix}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.evaluate_completed:
        evaluate_completed(args)
    elif args.run or args.data_preflight:
        execute(args)
    else:
        synthetic_preflight(args.seed, args.output)


if __name__ == '__main__':
    main()
