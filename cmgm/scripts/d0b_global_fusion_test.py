"""Frozen nine-checkpoint TEST inference; never starts training.

Default audits and freezes a manifest only. --evaluate-test explicitly authorizes
the shared TEST data build and nine sequential evaluations. Existing TRAIN/VAL
artifacts and checkpoints are read-only. New output directory required.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from cmgm.scripts import d0b_node_history_readout as original
from cmgm.scripts.d0b_global_fusion_multiseed import (
    VARIANTS, LABELS, read, write, sha, validate_run, kwargs_from_meta,
    environment, csv_write,
)
from cmgm.training.metric_standard import population_metrics

ROLES = dict(zip(VARIANTS, ('baseline', 'simple fusion control', 'VAL-selected candidate')))
HORIZONS = (1, 5, 10, 20)
HISTORICAL_TEST_EVIDENCE = (
    'experiments/d0b_moe_fusion/20260921_141206/test_metrics.json',
    'experiments/d0b_candidate_2expert_moe/20260921_151711/test_metrics.json',
)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def aggregate(out, results):
    details = []
    for result in results:
        for h in HORIZONS:
            m = result['metrics'][str(h)]
            details.append(dict(model=LABELS[result['variant']], variant=result['variant'],
                seed=result['seed'], split='TEST', horizon=h, MAE=m['MAE'], MSE=m['MSE'],
                RMSE=m['RMSE'], Hit_fraction=m['Hit'], Hit_percent=100*m['Hit'],
                best_epoch=result['best_epoch'], evaluation_seconds=result['evaluation_seconds']))
    details.sort(key=lambda r:(VARIANTS.index(r['variant']), r['seed'], r['horizon']))
    summary = []
    for variant in VARIANTS:
        for horizon in HORIZONS:
            group = [r for r in details if r['variant']==variant and r['horizon']==horizon]
            for metric in ('MAE','MSE','RMSE','Hit_percent'):
                values = [r[metric] for r in group]
                summary.append(dict(model=LABELS[variant],variant=variant,split='TEST',horizon=horizon,
                    metric=metric,n=len(values),mean=float(np.mean(values)) if values else None,
                    sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None))
    paired, paired_summary = [], []
    for cand, ref in ((VARIANTS[2],VARIANTS[0]),(VARIANTS[2],VARIANTS[1]),(VARIANTS[1],VARIANTS[0])):
        for horizon in HORIZONS:
            for metric in ('MAE','RMSE'):
                group = []
                for seed in (42,43,44):
                    c=next((r for r in details if r['variant']==cand and r['horizon']==horizon and r['seed']==seed),None)
                    b=next((r for r in details if r['variant']==ref and r['horizon']==horizon and r['seed']==seed),None)
                    if c is None or b is None: continue
                    row=dict(candidate=LABELS[cand],reference=LABELS[ref],split='TEST',horizon=horizon,
                        metric=metric,seed=seed,candidate_value=c[metric],reference_value=b[metric],
                        delta=c[metric]-b[metric],improvement_percent=100*(b[metric]-c[metric])/b[metric])
                    group.append(row); paired.append(row)
                if group:
                    refmean=float(np.mean([r['reference_value'] for r in group]))
                    candmean=float(np.mean([r['candidate_value'] for r in group]))
                    paired_summary.append(dict(candidate=LABELS[cand],reference=LABELS[ref],split='TEST',
                        horizon=horizon,metric=metric,n=len(group),improved_seeds=sum(r['delta']<0 for r in group),
                        mean_paired_delta=float(np.mean([r['delta'] for r in group])),
                        relative_improvement_of_model_means_percent=100*(refmean-candmean)/refmean,
                        mean_seed_improvement_percent=float(np.mean([r['improvement_percent'] for r in group]))))
    missing=[f'{v}:{s}' for v in VARIANTS for s in (42,43,44)
             if not any(r['variant']==v and r['seed']==s for r in results)]
    payload=dict(status='COMPLETE' if not missing else 'INCOMPLETE',test_evaluated=bool(results),
        missing=missing,n_complete=len(results),primary_metric='TEST MAE@5d',ddof=1,
        prediction_ensembling=False,target_scale='decimal return; no inverse scaling',
        Hit_raw_unit='fraction',Hit_display_unit='percent; std in percentage points',
        per_seed=details,aggregate=summary,paired=paired,paired_summary=paired_summary)
    for name, rows in [('test_per_seed',details),('test_aggregate',summary),
                       ('test_paired_differences',paired),('test_paired_summary',paired_summary)]:
        csv_write(out/(name+'.csv'),rows)
    write(out/'results.json',payload)
    write(out/'file_index.json',{f'{r["variant"]}:{r["seed"]}':dict(
        status=r['status'], result=r['result_file'], predictions=r['prediction_file'],
        prediction_sha256=r['prediction_sha256'], checkpoint=r['checkpoint'],
        checkpoint_sha256=r['checkpoint_sha256'], best_epoch=r['best_epoch']) for r in results})
    return payload


def report(out, summary, manifest, completed):
    rows=summary['aggregate']
    def cell(v,h,m):
        row=next(x for x in rows if x['variant']==v and x['horizon']==h and x['metric']==m)
        if row['n']!=3: return f"PENDING ({row['n']}/3)"
        digits=4 if m=='Hit_percent' else 8
        return f"{row['mean']:.{digits}f} ± {row['sample_std']:.{digits}f}"
    lines=['# Final TEST evaluation — global fusion, seeds 42/43/44','',
        f"Status: {summary['status']}; {summary['n_complete']}/9 checkpoints COMPLETE. Primary: TEST MAE@5d.",'',
        'All results use the frozen VAL-Huber-selected checkpoints. No training, fine-tuning, optimizer, checkpoint search, or prediction ensemble. Return errors remain on the decimal scale. Hit is stored as a fraction and displayed as percent. Sample std uses ddof=1.','',
        '## TEST 5d main table','', '|Model|Role|TEST MAE@5d|TEST MSE@5d|TEST RMSE@5d|TEST Hit@5d (%)|','|---|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        lines.append('|'+LABELS[v]+'|'+ROLES[v]+'|'+'|'.join(cell(v,5,m) for m in ('MAE','MSE','RMSE','Hit_percent'))+'|')
    for metric in ('MAE','RMSE','Hit_percent'):
        lines+=['',f'## TEST multi-horizon {metric}','',f'|Model|TEST {metric}@1d|TEST {metric}@5d|TEST {metric}@10d|TEST {metric}@20d|','|---|---:|---:|---:|---:|']
        for v in VARIANTS:
            lines.append('|'+LABELS[v]+'|'+'|'.join(cell(v,h,metric) for h in HORIZONS)+'|')
    lines+=['','## Checkpoint audit','', '|Model|Seed|Best epoch|Original VAL Huber|Strict load / history / SHA256 / metrics|','|---|---:|---:|---:|---|']
    for r in completed:
        lines.append(f"|{LABELS[r['variant']]}|{r['seed']}|{r['best_epoch']}|{r['original_val_huber']:.10f}|PASS|")
    lines+=['','## Raw TEST metrics','', '|Model|Seed|Horizon|TEST MAE|TEST MSE|TEST RMSE|TEST Hit (%)|','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['per_seed']:
        lines.append(f"|{r['model']}|{r['seed']}|{r['horizon']}d|{r['MAE']:.10f}|{r['MSE']:.10f}|{r['RMSE']:.10f}|{r['Hit_percent']:.5f}|")
    lines+=['','## Paired TEST comparison','',
        'Delta = candidate − reference; negative improves. Improvement = (reference − candidate) / reference × 100%; positive improves.','',
        '|Candidate vs reference|Horizon|Metric|Improved seeds|Mean paired delta|Improvement of model means (%)|Mean seed improvement (%)|','|---|---:|---|---:|---:|---:|---:|']
    for r in summary['paired_summary']:
        lines.append(f"|{r['candidate']} vs {r['reference']}|{r['horizon']}d|TEST {r['metric']}|{r['improved_seeds']}/{r['n']}|{r['mean_paired_delta']:+.10f}|{r['relative_improvement_of_model_means_percent']:+.6f}|{r['mean_seed_improvement_percent']:+.6f}|")
    lines+=['','Per-seed signed differences and improvement rates: `test_paired_differences.csv` (all 72 comparisons).','',
        '## Provenance and limitations','',
        '- `evaluation_manifest.json` was written and hashed before any TEST inference. Roles, seeds, horizons, primary metric and all nine checkpoint hashes were frozen.',
        '- `test_data_audit.json` records the common TEST fingerprint, chronological sample and commodity indices, train-derived preprocessing and target construction. Prediction NPZs retain sample origins, target indices, horizons, commodity order and full labels.',
        '- The same original data builder is used; no preprocessing statistics are fitted on TEST. No new inverse normalization or return conversion is applied.',
        '- Every TEST loader uses sequential order and drop_last=False. Metrics pool all 269 origins × 24 commodities independently at each horizon; zero targets are included in sign agreement.',
        '- Seed42 historical software/device versions were not recorded in the original artifacts; this missing provenance is retained. Current and seed43/44 recorded environments are compared in the manifest.',
        '- This project has previously inspected this TEST set: historical evidence is listed in the manifest. This is a newly frozen evaluation, not the first blind TEST evaluation in the project, nor cross-period generalization evidence.',
        '- Three seeds provide descriptive variability and paired direction, not a forced statistical-significance claim. Default model and checkpoint choices are unchanged.',
        '- `source_protection_audit.json` verifies the original TRAIN/VAL reports, histories, metrics and checkpoint files remain byte-identical.',
        '- Complete file paths and hashes: `file_index.json`. Recompute every metric directly from each run’s `predictions.npz` using population_metrics.', '']
    (out/'TEST_REPORT.md').write_text('\n'.join(lines))


def run(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    completed=[]
    try:
        directory=args.replication_dir.resolve()
        index=read(directory/'file_index.json'); validation=read(directory/'results.json')
        preaudit=read(directory/'audit.json')
        assert validation['status']=='COMPLETE' and not validation['missing'] and preaudit['PASS']
        assert set(index)=={f'{v}:{s}' for v in VARIANTS for s in (42,43,44)}
        original.set_seed(42)
        env=environment()
        env['evaluation_source_sha256']=sha(__file__)
        env['packages']={p:metadata.version(p) for p in ('torch','numpy','pandas','scipy')}
        env['device']=args.device
        env['autocast']=False;env['model_and_input_dtype']='float32';env['mode']='eval + torch.no_grad'
        protection={};entries=[];ref=None;env_comparison={}
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.json','.csv','.md'):
                protection[str(path)]=sha(path)
        for v in VARIANTS:
            for seed in (42,43,44):
                key=f'{v}:{seed}'; entry=index[key]
                assert entry['variant']==v and entry['seed']==seed
                assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
                assert sha(entry['result'])==entry['result_sha256']
                source_dir=Path(entry['result']).parent
                result,audit=validate_run(source_dir,ref)
                meta=result['model']
                assert meta['variant']==v and meta['seed']==seed
                assert result['best_epoch']==entry['best_epoch']
                ref=ref or meta
                for path in source_dir.iterdir():
                    if path.is_file(): protection[str(path)]=sha(path)
                protection[entry['checkpoint']]=sha(entry['checkpoint'])
                historical=source_dir/'environment.json'
                if historical.exists():
                    old=read(historical)
                    fields=('python','pytorch','numpy','cuda_build','cudnn','cuda_devices',
                            'cudnn_deterministic','cudnn_benchmark','deterministic_algorithms',
                            'float32_matmul_precision','matmul_allow_tf32','cudnn_allow_tf32')
                    diffs={k:dict(recorded=old[k],current=env[k]) for k in fields if old[k]!=env[k]}
                    env_comparison[key]=dict(recorded=True,differences=diffs)
                    assert not diffs, f'Inference environment differs for {key}: {diffs}'
                else:
                    env_comparison[key]=dict(recorded=False,note='Historical seed42 environment missing; not reconstructed.')
                entries.append(dict(**audit,role=ROLES[v],original_val_huber=result['best_val_prediction_loss'],
                    training_result=entry['result'],parameter_count=meta['parameter_count'],model_kwargs=kwargs_from_meta(meta)))
        evidence=[dict(path=str(original.ROOT/p),sha256=sha(original.ROOT/p))
                  for p in HISTORICAL_TEST_EVIDENCE if (original.ROOT/p).exists()]
        manifest=dict(frozen_at=datetime.now().astimezone().isoformat(),status='FROZEN_BEFORE_TEST_INFERENCE',
            model_roles=ROLES,seeds=[42,43,44],horizons=list(HORIZONS),checkpoints=entries,
            checkpoint_selection=ref['protocol']['checkpoint_selection'],primary_metric='TEST MAE@5d',
            secondary_metrics=['TEST MAE','TEST MSE','TEST RMSE','TEST Hit'],
            expected_test_origins=269,expected_commodities=24,training_reference_data=ref['data_fingerprint'],
            metric_definition='Pooled all origins x all commodities separately per horizon; RMSE=sqrt(MSE); unmasked sign agreement including zeros',
            target_scale='decimal cumulative return from original clipped target construction; no new scaling',
            raw_Hit_unit='fraction',display_Hit_unit='percent',environment=env,
            environment_comparison=env_comparison,evaluation_source_hashes={**original._hashes(),
                'cmgm/scripts/d0b_global_fusion_multiseed.py':sha(Path(__file__).with_name('d0b_global_fusion_multiseed.py')),
                'cmgm/scripts/d0b_global_fusion_test.py':sha(__file__)},
            previous_TEST_access=True,historical_TEST_evidence=evidence,
            source_replication_index=str(directory/'file_index.json'),source_replication_index_sha256=sha(directory/'file_index.json'))
        write(out/'evaluation_manifest.json',manifest)
        manifest_sha=sha(out/'evaluation_manifest.json')
        (out/'manifest.sha256').write_text(manifest_sha+'  evaluation_manifest.json\n')
        write(out/'audit.json',dict(status='PASS',n_checkpoints=9,checks=entries))
        write(out/'VAL_selection_context.json',dict(split='VAL',source=str(directory/'results.json'),
            source_sha256=sha(directory/'results.json'),paired_summary=validation['paired_summary']))
        print(f'9/9 checkpoint audits PASS. Manifest FROZEN: {out / "evaluation_manifest.json"}',flush=True)
        if not args.evaluate_test:
            print('Audit only; no TEST inference.');return
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        print('Building the original shared data pipeline once...',flush=True)
        data=original.build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=42))
        assert original._data_fingerprint(data,include_test=False)==ref['data_fingerprint']
        fp=original._data_fingerprint(data,include_test=True)
        dataset=data['loaders']['test'].dataset
        assert len(dataset)==269 and dataset.n_commodities==24
        assert dataset.seq_len==20 and tuple(dataset.horizons)==HORIZONS and dataset.target_type=='return'
        loader=DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,num_workers=0)
        assert isinstance(loader.sampler,SequentialSampler) and not loader.drop_last
        test_start=len(data['raw_prices_train'])+len(data['raw_prices_val'])
        sample_index=np.arange(len(dataset),dtype=np.int64)
        input_start=sample_index+test_start
        origins=input_start+dataset.seq_len-1
        target_indices=origins[:,None]+np.array(HORIZONS)[None,:]
        commodity_names=np.asarray(ref['commodity_order'],dtype=str)
        c0,c1=data['market_indices']['commodity']
        assert np.all(target_indices>origins[:,None])
        assert target_indices[-1,-1]==test_start+len(dataset.raw_prices)-1
        data_audit=dict(PASS=True,test=fp['splits']['test'],origins=len(dataset),commodities=24,
            fingerprint=fp,commodity_order=ref['commodity_order'],commodity_mapping=ref['commodity_mapping'],
            train_val_fingerprints_match=True,batch_sizes=[64,64,64,64,13],shuffle=False,drop_last=False,
            global_TEST_start_row=test_start,origin_global_range=[int(origins[0]),int(origins[-1])],
            origin_definition='0-based row of the last observed timestep in the aligned full chronological price matrix',
            dates='Original data builder does not export dates; exact local/global chronological row indices are stored in NPZ.',
            target_definition='clip(P[t+h]/max(abs(P[t]),1e-8)-1, -1, 1), float32; inherited unchanged',
            preprocessing='Original train-statistics pipeline; no TEST fitting or additional inverse transform')
        write(out/'test_data_audit.json',data_audit)
        print('Data audit PASS: common TEST 269 origins x 24 commodities, 5 batches including final 13.',flush=True)
        common_target=None
        for entry in entries:
            v,seed=entry['variant'],entry['seed']
            assert sha(out/'evaluation_manifest.json')==manifest_sha
            for name,digest in manifest['evaluation_source_hashes'].items():
                assert sha(original.ROOT/name)==digest
            assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
            original.set_seed(seed)
            model=original._make_model(v,entry['model_kwargs'])
            checkpoint=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            model.switching_latent_transformer.set_epoch(entry['best_epoch'])
            model.to(torch.device(args.device)).eval()
            weight_hash=state_digest(model)
            predictions=[];targets=[];batches=[]
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.no_grad():
                for x,y in loader:
                    pred=model(x.to(args.device))
                    assert pred.shape==y.shape and pred.shape[1:]==(4,24)
                    assert torch.isfinite(pred).all() and torch.isfinite(y).all()
                    predictions.append(pred.cpu().numpy());targets.append(y.numpy());batches.append(len(y))
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            assert batches==[64,64,64,64,13]
            pred,target=np.concatenate(predictions),np.concatenate(targets)
            assert pred.shape==target.shape==(269,4,24)
            assert state_digest(model)==weight_hash, 'Model state mutated during TEST inference'
            if common_target is None: common_target=target.copy()
            else: assert np.array_equal(common_target,target), 'Labels/order differ between checkpoints'
            run_dir=out/f'{LABELS[v]}_seed{seed}';run_dir.mkdir()
            prediction_path=run_dir/'predictions.npz'
            np.savez_compressed(prediction_path,prediction=pred,target=target,sample_index=sample_index,
                input_start_global_index=input_start,forecast_origin_global_index=origins,
                target_global_indices=target_indices,commodity_names=commodity_names,
                commodity_node_indices=np.arange(c0,c1),horizons=np.asarray(HORIZONS))
            # Metrics are calculated from the saved complete arrays, not batch means.
            with np.load(prediction_path,allow_pickle=False) as saved:
                assert np.array_equal(saved['sample_index'],sample_index)
                assert np.array_equal(saved['commodity_names'],commodity_names)
                assert np.array_equal(saved['prediction'],pred) and np.array_equal(saved['target'],target)
                metrics={str(h):population_metrics(saved['prediction'][:,i],saved['target'][:,i])
                         for i,h in enumerate(HORIZONS)}
            assert all(math.isfinite(x) for m in metrics.values() for x in m.values())
            assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
            result=dict(status='COMPLETE',test_evaluated=True,variant=v,seed=seed,role=entry['role'],
                checkpoint=entry['checkpoint'],checkpoint_sha256=entry['checkpoint_sha256'],
                best_epoch=entry['best_epoch'],original_val_huber=entry['original_val_huber'],
                checkpoint_selection=manifest['checkpoint_selection'],training_result=entry['training_result'],
                test_fingerprint=fp['splits']['test'],origins=269,commodity_order=ref['commodity_order'],
                horizons=list(HORIZONS),target_scale=manifest['target_scale'],Hit_unit='fraction',
                metric_definition=manifest['metric_definition'],environment=env,
                evaluation_source_hashes=manifest['evaluation_source_hashes'],evaluation_seconds=seconds,
                model_state_unchanged=True,strict_load=True,metrics=metrics,
                manifest_sha256=manifest_sha,prediction_file=str(prediction_path),
                prediction_sha256=sha(prediction_path),result_file=str(run_dir/'result.json'))
            write(run_dir/'result.json',result)
            completed.append(result)
            summary=aggregate(out,completed)
            print(f'COMPLETE {len(completed)}/9: {LABELS[v]} seed{seed}; predictions saved and verified.',flush=True)
            del model,checkpoint,predictions,targets
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        differences=[p for p,digest in protection.items() if sha(p)!=digest]
        write(out/'source_protection_audit.json',dict(PASS=not differences,files_checked=len(protection),
                                                    changed= differences,original_sha256=protection))
        assert not differences, 'Original artifacts changed during evaluation'
        assert sha(out/'evaluation_manifest.json')==manifest_sha
        summary=aggregate(out,completed)
        assert summary['status']=='COMPLETE'
        report(out,summary,manifest,completed)
        print(f'ALL COMPLETE: 9/9 TEST evaluations; original artifacts unchanged. {out / "TEST_REPORT.md"}',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAILED',completed=len(completed),error=repr(exc),traceback=traceback.format_exc()))
        aggregate(out,completed)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replication-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--evaluate-test',action='store_true')
    run(parser.parse_args())
