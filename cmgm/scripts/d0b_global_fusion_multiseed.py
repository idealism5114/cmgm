"""Audit and sequentially repeat the existing three global fusions, seeds 43/44.

Uses the unmodified single-experiment entrypoint and trainer. No TEST evaluation.
Default: audit only. --run authorizes exactly six missing fresh runs. Completed
runs are verified and reused; interrupted/inconsistent runs stop for review.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from cmgm.scripts import d0b_node_history_readout as single

VARIANTS = (single.BASE_VARIANT, single.MEAN_FUSION_VARIANT, single.CONCAT_VARIANT)
LABELS = dict(zip(VARIANTS, ('D0B', 'MeanFusion', 'GlobalConcatFusion')))
PARAMS = dict(zip(VARIANTS, (520549, 512293, 520549)))
SEED42_NAMES = (
    'switching_latent_balanced_readout_seed42_20261006_231320_782593',
    'switching_latent_balanced_mean_fusion_seed42_20261006_231354_784110',
    'switching_latent_balanced_concat_fusion_seed42_20261006_231412_785761',
)
METRICS = ('VAL_Huber', 'VAL_MAE_1', 'VAL_MAE_5', 'VAL_MAE_10', 'VAL_MAE_20')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def preserve_rng():
    cpu, py, npstate = torch.get_rng_state(), random.getstate(), np.random.get_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        random.setstate(py)
        np.random.set_state(npstate)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags


def kwargs_from_meta(meta):
    fp = meta['data_fingerprint']
    markets = fp['market_indices']
    return dict(num_nodes=fp['node_count'], n_commodities=len(fp['commodity_order']),
                n_stock=markets['stock'][1] - markets['stock'][0],
                n_bond=markets['bond'][1] - markets['bond'][0],
                commodity_indices=tuple(markets['commodity']))


def check_sources(meta):
    mismatches = [p for p, digest in meta['source_hashes'].items()
                  if sha(single.ROOT / p) != digest]
    if mismatches:
        raise RuntimeError(f'Changed seed-42 sources: {mismatches}; STOP')


def protocol(meta):
    return {k: v for k, v in meta['protocol'].items() if k not in ('seed', 'device')}


def validate_run(directory, reference=None):
    directory = Path(directory)
    r = read(directory / 'result.json')
    meta = r['model']
    v, seed = meta['variant'], meta['seed']
    assert v in VARIANTS and seed in (42, 43, 44)
    assert r['status'] == 'COMPLETE' and r['test_evaluated'] is False
    assert set(r['metrics']) == {'train', 'val'}
    assert read(directory / 'config.json') == meta
    assert read(directory / 'data_audit.json') == meta['data_fingerprint']
    assert read(directory / 'metrics.json') == r['metrics']
    assert meta['initialization_audit']['PASS']
    assert meta['initialization_audit']['shared_max_abs_diff'] == 0
    assert meta['parameter_count'] == PARAMS[v]
    assert meta['protocol']['seed'] == seed
    assert protocol(meta) == single.PROTOCOL
    check_sources(meta)
    if reference:
        assert meta['data_fingerprint'] == reference['data_fingerprint']
        assert meta['commodity_order'] == reference['commodity_order']
        assert meta['commodity_mapping'] == reference['commodity_mapping']
        assert protocol(meta) == protocol(reference)
    for split in ('train', 'val'):
        assert set(r['metrics'][split]) == {'1', '5', '10', '20'}
        for m in r['metrics'][split].values():
            assert all(math.isfinite(m[k]) for k in ('MAE', 'MSE', 'RMSE', 'Hit'))
            assert m['MAE'] >= 0 and m['MSE'] >= 0 and 0 <= m['Hit'] <= 1
            assert math.isclose(m['RMSE'], math.sqrt(m['MSE']), rel_tol=1e-12)
    h = read(directory / 'training_history.json')
    assert all(math.isfinite(x) for key in ('train_loss', 'val_loss', 'lr_history', 'switch_beta') for x in h[key])
    best = r['best_epoch']
    assert h['best_epoch'] == best == int(np.argmin(h['val_loss'])) + 1
    assert len(h['val_loss']) <= 200 and len(h['val_loss']) >= best
    assert all(len(h[k]) == len(h['val_loss']) for k in ('train_loss', 'lr_history', 'switch_beta'))
    assert r['best_val_prediction_loss'] == min(h['val_loss'])
    assert math.isfinite(r['training_seconds']) and r['training_seconds'] > 0
    cp_path = Path(r['checkpoint'])
    cp_hash = sha(cp_path)
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    assert cp['best_epoch'] == best and cp['best_val_loss'] == r['best_val_prediction_loss']
    assert cp['metadata'] == meta
    # train_time is appended after serialization by the unchanged trainer.
    for key in ('train_loss', 'val_loss', 'lr_history', 'switch_beta', 'best_epoch'):
        assert cp['history'][key] == h[key]
    with preserve_rng():
        model = single._make_model(v, kwargs_from_meta(meta))
        model.load_state_dict(cp['model_state_dict'], strict=True)
        assert sum(p.numel() for p in model.parameters()) == PARAMS[v]
        assert all(torch.isfinite(p).all() for p in model.parameters())
        branch = model.switching_latent_transformer
        expected = [branch.set_epoch(e) for e in range(1, len(h['val_loss']) + 1)]
        assert np.array_equal(expected, h['switch_beta'])
    return r, dict(PASS=True, variant=v, seed=seed, result=str((directory/'result.json').resolve()),
                   result_sha256=sha(directory/'result.json'), history=str((directory/'training_history.json').resolve()),
                   checkpoint=str(cp_path), checkpoint_sha256=cp_hash, epochs_trained=len(h['val_loss']),
                   best_epoch=best, strict_load=True, minimum_history_matches=True)


def environment():
    return dict(python=platform.python_version(), pytorch=torch.__version__, numpy=np.__version__,
                cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                torch_threads=torch.get_num_threads(), pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=single.ROOT,text=True).strip(),
                batch_source_sha256=sha(__file__), core_source_hashes=single._hashes(),
                seed_policy='Original set_seed for NumPy/Torch/CUDA; additionally seed unused Python random. No change to Torch initialization or determinism policy.',
                seed42_environment_limitation='Original seed42 artifacts did not record software versions/device model; current environment is recorded, not retrospectively asserted as measured seed42 provenance.')


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out, results, audits):
    raw, all_metrics = [], []
    for r in results:
        meta = r['model']; v, s = meta['variant'], meta['seed']
        a = audits[f'{v}:{s}']
        row = dict(model=LABELS[v], variant=v, seed=s, VAL_Huber=r['best_val_prediction_loss'])
        row.update({f'VAL_MAE_{h}': r['metrics']['val'][str(h)]['MAE'] for h in (1,5,10,20)})
        row.update(VAL_RMSE_5=r['metrics']['val']['5']['RMSE'],
                   VAL_Hit5_percent=100*r['metrics']['val']['5']['Hit'],
                   best_epoch=r['best_epoch'], training_seconds=r['training_seconds'],
                   epochs_trained=a['epochs_trained'], parameters=meta['parameter_count'],
                   result=a['result'], history=a['history'], checkpoint=a['checkpoint'], test_evaluated=False)
        raw.append(row)
        for split in ('train','val'):
            for horizon, values in r['metrics'][split].items():
                all_metrics.append(dict(variant=v, seed=s, split=split, horizon=int(horizon), **values))
    raw.sort(key=lambda r:(VARIANTS.index(r['variant']),r['seed']))
    aggregates=[]
    for v in VARIANTS:
        group=[r for r in raw if r['variant']==v]
        for metric in (*METRICS,'VAL_RMSE_5','VAL_Hit5_percent','best_epoch','training_seconds','epochs_trained'):
            values=[r[metric] for r in group]
            aggregates.append(dict(variant=v,model=LABELS[v],metric=metric,n=len(values),
                                   mean=float(np.mean(values)) if values else None,
                                   sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                   minimum=min(values) if values else None,maximum=max(values) if values else None))
    pairs=[]; pair_summary=[]
    for cand,ref in ((VARIANTS[1],VARIANTS[0]),(VARIANTS[2],VARIANTS[0]),(VARIANTS[2],VARIANTS[1])):
        for metric in METRICS:
            block=[]
            for seed in (42,43,44):
                c=next((r for r in raw if r['variant']==cand and r['seed']==seed),None)
                b=next((r for r in raw if r['variant']==ref and r['seed']==seed),None)
                if c is None or b is None: continue
                row=dict(candidate=LABELS[cand],reference=LABELS[ref],metric=metric,seed=seed,
                         candidate_value=c[metric],reference_value=b[metric],
                         delta_candidate_minus_reference=c[metric]-b[metric],
                         improvement_percent=100*(b[metric]-c[metric])/b[metric])
                block.append(row); pairs.append(row)
            if block:
                pair_summary.append(dict(candidate=LABELS[cand],reference=LABELS[ref],metric=metric,
                    n=len(block),improved_seeds=sum(r['improvement_percent']>0 for r in block),
                    mean_paired_delta=float(np.mean([r['delta_candidate_minus_reference'] for r in block])),
                    sample_std_paired_delta=float(np.std([r['delta_candidate_minus_reference'] for r in block],ddof=1)) if len(block)>1 else None,
                    mean_of_seed_improvement_percent=float(np.mean([r['improvement_percent'] for r in block])),
                    relative_improvement_of_model_means_percent=100*(np.mean([r['reference_value'] for r in block])-np.mean([r['candidate_value'] for r in block]))/np.mean([r['reference_value'] for r in block])))
    missing=[f'{v}:{s}' for v in VARIANTS for s in (42,43,44) if not any(r['variant']==v and r['seed']==s for r in raw)]
    summary=dict(status='COMPLETE' if not missing else 'INCOMPLETE',missing=missing,ddof=1,
                 paired_delta_definition='candidate - reference; negative is lower error',
                 improvement_definition='(reference-candidate)/reference*100; positive is lower error',
                 raw_Hit_unit='fraction',summary_Hit_unit='percent; std in percentage points',
                 per_seed=raw,aggregate=aggregates,paired=pairs,paired_summary=pair_summary,
                 audits=audits,test_evaluated=False)
    for name,rows in [('per_seed_results',raw),('raw_metrics',all_metrics),('aggregate_results',aggregates),
                      ('paired_differences',pairs),('paired_summary',pair_summary)]:
        csv_write(out/(name+'.csv'),rows)
    write(out/'results.json',summary); write(out/'file_index.json',audits)
    lines=['# Global fusion three-seed replication','',f'Status: {summary["status"]}; {len(raw)}/9 audited runs. TEST not evaluated.',
           '', 'Seed42 results are reused from the three explicitly identified user runs. No seed42 retraining. Data/order/protocol/source hashes and strict checkpoint loading are checked for every included run.',
           '', '## Per-seed results','', '|Model|Seed|VAL Huber|VAL1 MAE|VAL5 MAE|VAL10 MAE|VAL20 MAE|VAL5 RMSE|Hit %|Best epoch|Training seconds|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in raw:
        lines.append(f"|{r['model']}|{r['seed']}|"+'|'.join(f'{r[k]:.9f}' for k in (*METRICS,'VAL_RMSE_5'))+f"|{r['VAL_Hit5_percent']:.4f}|{r['best_epoch']}|{r['training_seconds']:.2f}|")
    lines += ['', '## Mean ± sample standard deviation (ddof=1)','', '|Model|Metric|n|Mean|Sample std|','|---|---|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"|{r['model']}|{r['metric']}|{r['n']}|{r['mean']}|{r['sample_std']}|")
    lines += ['', '## Paired differences','', 'Delta = candidate − reference (negative improves). Relative improvement = (reference − candidate) / reference × 100%.', '',
              '|Candidate vs reference|Metric|Seed|Absolute delta|Relative improvement %|','|---|---|---:|---:|---:|']
    for r in pairs:
        lines.append(f"|{r['candidate']} vs {r['reference']}|{r['metric']}|{r['seed']}|{r['delta_candidate_minus_reference']:+.10f}|{r['improvement_percent']:+.6f}|")
    lines += ['', '|Candidate vs reference|Metric|Improved seeds|Mean paired delta|Mean seed improvement %|Improvement of model means %|','|---|---|---:|---:|---:|---:|']
    for r in pair_summary:
        lines.append(f"|{r['candidate']} vs {r['reference']}|{r['metric']}|{r['improved_seeds']}/{r['n']}|{r['mean_paired_delta']:+.10f}|{r['mean_of_seed_improvement_percent']:+.6f}|{r['relative_improvement_of_model_means_percent']:+.6f}|")
    lines += ['', '## Audit and interpretation limits','',
              '- Parameters: D0B 520,549; MeanFusion 512,293; GlobalConcatFusion 520,549.',
              '- Selection is prediction-only four-horizon VAL Huber, batch-mean aggregation; VAL5 does not select checkpoints.',
              '- Every restored checkpoint matches the minimum validation objective and best epoch in its complete training history.',
              '- Runtime is descriptive only: shared GPU contention and different epoch counts prevent controlled efficiency ranking.',
              '- The original seed42 artifacts lack contemporaneous software/device version records. Current software/determinism settings are recorded in environment.json; identical key source hashes and the same environment location are verified, but missing historical version provenance is not invented.',
              '- Three seeds provide preliminary initialization/training-randomness evidence, not statistical significance or cross-period generalization.',
              '- See audit.json for initialization/source/data checks and file_index.json for each result, training history, checkpoint and SHA256.',
              '', 'Missing runs: '+(', '.join(missing) if missing else 'none'), '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return summary


def execute(args):
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    references=[]; audits={}; results=[]
    for name in SEED42_NAMES:
        directory=single.EXPERIMENT_ROOT/name
        r,a=validate_run(directory, references[0]['model'] if references else None)
        references.append(r);results.append(r);audits[f"{r['model']['variant']}:42"]=a
    ref=references[0]['model']; kwargs=kwargs_from_meta(ref)
    init={}
    for seed in (42,43,44):
        for variant in VARIANTS:
            with preserve_rng():
                audit=single._audit_seed(variant,kwargs,seed)
            assert audit['PASS'] and audit['shared_max_abs_diff']==0
            init[f'{variant}:{seed}']=audit
    preflight=dict(PASS=True,seed42_sources_unchanged=True,seed42_runs=audits.copy(),
                   initialization=init,parameters=PARAMS,planned_new_runs=6,
                   data_fingerprint_reference=ref['data_fingerprint'],
                   note='Live data fingerprints are rechecked before each training; no TEST evaluation.')
    write(out/'audit.json',preflight)
    print(f'Preflight PASS: 3 seed42 checkpoints audited, shared initialization PASS for seeds42/43/44. Output={out}',flush=True)
    # This output directory is resumable only by skipping fully completed runs.
    state_path=out/'state.json'
    state=read(state_path) if state_path.exists() else {'jobs':{}}
    for seed in (43,44):
        for variant in VARIANTS:
            key=f'{variant}:{seed}'
            candidates=list(single.EXPERIMENT_ROOT.glob(f'{variant}_seed{seed}_*'))
            if len(candidates)>1:
                raise RuntimeError(f'Ambiguous duplicate runs for {key}: {candidates}')
            if candidates:
                r,a=validate_run(candidates[0],ref)
                results.append(r);audits[key]=a
                state['jobs'][key]=dict(status='COMPLETE',result=a['result'],reused=True)
            elif state['jobs'].get(key,{}).get('status') in ('RUNNING','FAILED'):
                raise RuntimeError(f'{key} was interrupted/failed. STOP for review; no weights-only restart.')
    summarize(out,results,audits)
    write(state_path,state)
    if not args.run:
        print('Audit/report only; no training started.',flush=True);return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    # Call the unchanged entrypoint with a cached instance of its own data builder.
    original_builder, original_train=single.build_data,single.train
    data_cache=None
    def audited_data(arg):
        nonlocal data_cache
        check_sources(ref)
        if data_cache is None:
            print('Building the original deterministic data pipeline once...',flush=True)
            data_cache=original_builder(arg)
            # TEST loaders are explicitly unavailable to the batch training/eval path.
            data_cache['loaders'].pop('test',None)
            data_cache.get('semantic_loaders',{}).pop('test',None)
        fp=single._data_fingerprint(data_cache,include_test=False)
        assert fp==ref['data_fingerprint'], 'Live TRAIN/VAL data fingerprint mismatch; STOP'
        return data_cache
    def audited_train(model,train_loader,val_loader,*pos,**kw):
        assert sum(p.numel() for p in model.parameters())==PARAMS[model.variant]
        assert train_loader.drop_last and not val_loader.drop_last
        assert isinstance(train_loader.sampler,SequentialSampler)
        assert train_loader.batch_size==val_loader.batch_size==64
        assert kw['num_epochs']==200 and kw['patience']==10 and kw['lr']==1e-4 and kw['weight_decay']==1e-5
        checkpoint=Path(kw['checkpoint_path'])
        write(checkpoint.parent/'environment.json',environment())
        py_state=random.getstate()
        history=original_train(model,train_loader,val_loader,*pos,**kw)
        assert random.getstate()==py_state, 'Python random unexpectedly used by training'
        return history
    single.build_data, single.train=audited_data,audited_train
    try:
        for seed in (43,44):
            for variant in VARIANTS:
                key=f'{variant}:{seed}'
                if key in audits: continue
                print(f'START {key}; completed new runs={len(results)-3}/6',flush=True)
                state['jobs'][key]=dict(status='RUNNING',pid=os.getpid())
                write(state_path,state)
                log_path=out/f'{LABELS[variant]}_seed{seed}.log'
                before=set(single.EXPERIMENT_ROOT.glob(f'{variant}_seed{seed}_*'))
                try:
                    random.seed(seed)
                    with log_path.open('x',buffering=1) as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        single.run_experiment(SimpleNamespace(variant=variant,seed=seed,device=args.device,run=True,evaluate_test=False))
                    new=set(single.EXPERIMENT_ROOT.glob(f'{variant}_seed{seed}_*'))-before
                    assert len(new)==1
                    directory=new.pop()
                    r,a=validate_run(directory,ref)
                    results.append(r);audits[key]=a
                    env=read(Path(r['checkpoint']).parent/'environment.json')
                    write(directory/'environment.json',env)
                    write(out/'environment.json',env)
                    state['jobs'][key]=dict(status='COMPLETE',result=a['result'],log=str(log_path),reused=False)
                    write(state_path,state)
                    preflight['completed_run_audits']=audits
                    write(out/'audit.json',preflight)
                    summarize(out,results,audits)
                    print(f'COMPLETE {key}; completed new runs={len(results)-3}/6; best_epoch={r["best_epoch"]}; VAL_Huber={r["best_val_prediction_loss"]:.9f}',flush=True)
                except BaseException as exc:
                    state['jobs'][key]=dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),log=str(log_path))
                    write(state_path,state); summarize(out,results,audits)
                    raise
                finally:
                    gc.collect();torch.cuda.empty_cache()
    finally:
        single.build_data,single.train=original_builder,original_train
    summary=summarize(out,results,audits)
    assert summary['status']=='COMPLETE'
    print(f'ALL COMPLETE: 9/9 validated; 6/6 new training runs. Report: {out / "REPORT.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output-dir',type=Path,required=True)
    execute(parser.parse_args())
