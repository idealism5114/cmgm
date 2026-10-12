"""One fixed seed-42 lag-factor experiment. Default: synthetic preflight only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts import d0b_node_history_readout as old
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
VARIANT = 'switching_latent_balanced_lag_factor_fusion'
RESULT_ROOT = ROOT / 'experiments' / 'd0b_lag_factor_fusion'
CHECKPOINT_ROOT = ROOT / 'checkpoints' / 'd0b_lag_factor_fusion'
REFERENCE_INDEX = ROOT / 'experiments/d0b_global_fusion_test/evaluation_20261007_final/file_index.json'
REFERENCE_RESULTS = ROOT / 'experiments/d0b_global_fusion_test/evaluation_20261007_final/results.json'
SOURCES = (*old.SOURCE_FILES, 'cmgm/models/lag_factor_fusion.py',
           'cmgm/scripts/d0b_lag_factor_fusion.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def count(model):
    return sum(p.numel() for p in model.parameters())


def synthetic_audit(seed=42):
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
                  commodity_indices=(6, 30))
    old.set_seed(seed)
    base = HeteroMixHopCMGM(**kwargs, variant=old.BASE_VARIANT).eval()
    old.set_seed(seed)
    model = HeteroMixHopCMGM(**kwargs, variant=VARIANT).eval()
    common = dict(base.named_parameters())
    current = dict(model.named_parameters())
    matched = [name for name in common if name != 'gate_fc.weight' and name != 'gate_fc.bias']
    mismatch = [name for name in matched if name not in current or not torch.equal(common[name], current[name])]
    assert not mismatch and not hasattr(model, 'gate_fc')
    assert model.switching_latent_transformer.regime_filter.beta_max == 5e-4
    assert model.switching_latent_transformer.regime_filter.warmup_epochs == 20
    x = torch.randn(2, 20, 30, 21)
    with torch.no_grad():
        direct = model.switching_latent_transformer(x)
        t, h, p = model.switching_latent_transformer(
            x, return_memory=True, return_live_probabilities=True)
        torch.testing.assert_close(direct, t, atol=0, rtol=0)
        assert h.shape == (2, 20, 128) and p.shape == (2, 3)
        pred, aux = model.lag_factor_forward(x, return_components=True)
        assert pred.shape == (2, 4, 24) and torch.isfinite(pred).all()
        assert aux['exposure'].shape == (2, 24, 4)
        assert aux['shock'].shape == (2, 4, 4)
        assert aux['lag_weights'].shape == (2, 4, 5)
        torch.testing.assert_close(aux['lag_weights'].sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(aux['exposure'].mean(dim=1), torch.zeros(2, 4), atol=1e-6, rtol=0)
        torch.testing.assert_close(pred, aux['global_prediction']+aux['correction'])
        torch.testing.assert_close(model(x), pred)
        torch.testing.assert_close(model(x.flip(0)).flip(0), pred, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(model(x[:1]), pred[:1], atol=2e-5, rtol=2e-5)
    model.zero_grad(set_to_none=True)
    pred, aux = model.lag_factor_forward(x, return_components=True)
    correction_loss = aux['correction'].square().mean()
    correction_loss.backward(retain_graph=True)
    gradient_paths = {
        'spatial_nodes': 'attn_mixhop2',
        'history': 'switching_latent_transformer.long_memory',
        'state_probability': 'switching_latent_transformer.regime_filter.regime_evidence',
        'exposure': 'lag_factor_fusion.exposure',
        'lag_router': 'lag_factor_fusion.lag_logits',
        'shock': 'lag_factor_fusion.shock',
    }
    grads = {}
    for label, prefix in gradient_paths.items():
        values = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        grads[label] = bool(values and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in values))
    assert all(grads.values()), grads
    model.zero_grad(set_to_none=True)
    prediction_loss = sum(torch.nn.functional.huber_loss(pred[:, i], torch.zeros_like(pred[:, i]), delta=.02)
                          for i in range(4))
    branch = model.switching_latent_transformer
    branch.set_epoch(1)
    first_beta = branch.regime_filter.current_beta
    branch.set_epoch(20)
    full_beta = branch.regime_filter.current_beta
    total = prediction_loss + branch.switch_loss()
    assert torch.isfinite(total) and total >= prediction_loss
    total.backward()
    assert model.head[0].weight.grad is not None and torch.isfinite(model.head[0].weight.grad).all()
    assert first_beta == 0 and abs(full_beta-5e-4)<1e-12
    return dict(PASS=True,seed=seed,common_parameter_tensors=len(matched),
                common_parameter_max_abs_diff=0,mismatches=mismatch,
                base_parameters=count(base),new_parameters=count(model),
                removed_gate_parameters=sum(p.numel() for n,p in base.named_parameters() if n.startswith('gate_fc.')),
                new_fusion_parameters=count(model.lag_factor_fusion),gradient_paths=grads,
                time_readout_equal=True,initial_switch_beta=first_beta,full_switch_beta=full_beta,
                output_shape=[2,4,24],target_scale='decimal return')


def references(data_fingerprint):
    if not REFERENCE_INDEX.exists() or not REFERENCE_RESULTS.exists():
        return dict(PASS=False,reason='Audited nine-checkpoint reference index/results absent')
    index=json.loads(REFERENCE_INDEX.read_text())
    found={}
    for label,variant in [('D0B',old.BASE_VARIANT),
                          ('GlobalConcat','switching_latent_balanced_concat_fusion')]:
        entry=index.get(f'{variant}:42')
        if entry is None:
            return dict(PASS=False,reason=f'{label} seed42 index entry missing')
        result=json.loads(Path(entry['original_training_result']).read_text())
        checkpoint=Path(entry['checkpoint'])
        if (result['model']['variant']!=variant or result['model']['seed']!=42 or
            result['model']['data_fingerprint']['splits']['train']!=data_fingerprint['splits']['train'] or
            result['model']['data_fingerprint']['splits']['val']!=data_fingerprint['splits']['val'] or
            result['model']['commodity_order']!=data_fingerprint['commodity_order'] or
            result['model']['protocol']['checkpoint_selection']!=old.PROTOCOL['checkpoint_selection'] or
            digest(checkpoint)!=entry['checkpoint_sha256'] or
            result['best_epoch']!=entry['best_epoch']):
            return dict(PASS=False,reason=f'{label} provenance mismatch')
        reference_model=old._make_model(variant,old._model_kwargs_from_fingerprint(data_fingerprint)) if hasattr(old,'_model_kwargs_from_fingerprint') else None
        if reference_model is None:
            mapping=data_fingerprint['market_indices']
            kwargs=dict(num_nodes=data_fingerprint['node_count'],n_commodities=len(data_fingerprint['commodity_order']),
                n_stock=mapping['stock'][1]-mapping['stock'][0],n_bond=mapping['bond'][1]-mapping['bond'][0],
                commodity_indices=tuple(mapping['commodity']))
            try: reference_model=old._make_model(variant,kwargs)
            except Exception as exc: return dict(PASS=False,reason=f'{label} exact implementation unavailable: {exc}')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        try: reference_model.load_state_dict(payload['model_state_dict'],strict=True)
        except Exception as exc: return dict(PASS=False,reason=f'{label} strict checkpoint load failed: {exc}')
        if payload['best_epoch']!=entry['best_epoch']:
            return dict(PASS=False,reason=f'{label} checkpoint best epoch mismatch')
        eval_entry=json.loads(Path(entry['result']).read_text())
        if eval_entry['checkpoint_sha256']!=entry['checkpoint_sha256'] or not eval_entry['test_evaluated']:
            return dict(PASS=False,reason=f'{label} TEST provenance mismatch')
        found[label]=dict(variant=variant,seed=42,checkpoint=str(checkpoint),sha256=entry['checkpoint_sha256'],
                          best_epoch=entry['best_epoch'],parameters=count(reference_model),
                          VAL=result['metrics']['val'],TEST=eval_entry['metrics'],
                          training_result=entry['original_training_result'],test_result=entry['result'])
        del reference_model
    return dict(PASS=True,models=found)


@torch.no_grad()
def evaluate(model,data,device,split,diagnostics=False):
    loader=old._loader(data['loaders'][split])
    preds=[];targets=[];component_rows=[]
    model.eval()
    for x,y in loader:
        x=x.to(device)
        if diagnostics:
            pred,aux=model.lag_factor_forward(x,return_components=True)
            component_rows.append({k:v.cpu().numpy() for k,v in aux.items()})
        else: pred=model(x)
        assert pred.shape==y.shape and torch.isfinite(pred).all()
        preds.append(pred.cpu().numpy());targets.append(y.numpy())
    prediction,target=np.concatenate(preds),np.concatenate(targets)
    metrics={str(h):population_metrics(prediction[:,i],target[:,i]) for i,h in enumerate(config.MULTI_HORIZONS)}
    if not diagnostics: return metrics,prediction,target,None
    components={k:np.concatenate([row[k] for row in component_rows]) for k in component_rows[0]}
    return metrics,prediction,target,components


@torch.no_grad()
def interventions(model,data,device,base_prediction,target,components):
    loader=old._loader(data['loaders']['test'])
    chosen={}
    for name,kwargs in [('remove_factor',dict(remove_correction=True)),
                        ('permute_exposure',dict(exposure_permutation=torch.roll(torch.arange(model.n_commodities,device=device),1))),
                        ('uniform_lags',dict(uniform_lags=True))]:
        rows=[]
        for x,_ in loader: rows.append(model.lag_factor_forward(x.to(device),**kwargs).cpu().numpy())
        p=np.concatenate(rows)
        m=population_metrics(p[:,1],target[:,1])
        chosen[name]=dict(TEST5=m,MAE_delta_vs_frozen_original=m['MAE']-population_metrics(base_prediction[:,1],target[:,1])['MAE'])
    base=np.linalg.norm(components['global_prediction'][:,1],axis=-1)
    correction=np.linalg.norm(components['correction'][:,1],axis=-1)
    chosen['correction_amplitude']=dict(mean_correction_L2=float(correction.mean()),
        mean_global_prediction_L2=float(base.mean()),
        mean_samplewise_ratio=float((correction/(base+1e-8)).mean()),
        median_samplewise_ratio=float(np.median(correction/(base+1e-8))))
    chosen['interpretation']='Frozen checkpoint interventions; not retrained structural ablations.'
    return chosen


def run(args):
    audit=synthetic_audit(args.seed)
    print('Synthetic preflight PASS:',json.dumps(audit),flush=True)
    if not args.run:return
    if args.seed!=42:raise ValueError('This formal experiment is fixed to seed 42')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    expected=(20,21,[1,5,10,20],'return',.02,64,1e-4,1e-5,200,10)
    actual=(config.SEQ_LEN,config.FEATURE_DIM,list(config.MULTI_HORIZONS),config.TARGET_TYPE,
            config.HUBER_DELTA,config.BATCH_SIZE,config.LEARNING_RATE,config.WEIGHT_DECAY,
            config.NUM_EPOCHS,config.PATIENCE)
    if actual!=expected:raise RuntimeError(f'Training protocol changed: {actual}')
    print('Building original D0B data pipeline...',flush=True)
    data=old.build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=42))
    fingerprint=old._data_fingerprint(data,include_test=True)
    if [fingerprint['splits'][s]['origins'] for s in ('train','val','test')]!=[1396,268,269]:
        raise RuntimeError('Chronological split counts differ from audited references')
    refs=references(fingerprint)
    if not refs['PASS']:raise RuntimeError('Required D0B/GlobalConcat reference audit failed: '+refs['reason'])
    old.set_seed(42)
    model=old._make_model(VARIANT,old._model_kwargs(data))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    out=RESULT_ROOT/f'run_seed42_{stamp}'
    checkpoint=CHECKPOINT_ROOT/f'run_seed42_{stamp}'/'best.pt'
    out.mkdir(parents=True,exist_ok=False)
    checkpoint.parent.mkdir(parents=True,exist_ok=False)
    metadata=dict(variant=VARIANT,seed=42,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={s:digest(ROOT/s) for s in SOURCES},data_fingerprint=fingerprint,
        protocol={**old.PROTOCOL,'seed':42,'device':args.device},
        metric_definition='pooled origins × commodities; return decimal; Hit fraction',
        parameters=count(model),new_fusion_parameters=count(model.lag_factor_fusion),
        removed_gate_parameters=8256,commodity_order=fingerprint['commodity_order'],
        synthetic_audit=audit,references=refs)
    save(out/'config.json',metadata)
    save(out/'data_audit.json',fingerprint)
    print('Formal training starts. Output:',out,flush=True)
    start=time.perf_counter()
    history=train(model,data['loaders']['train'],data['loaders']['val'],
        torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.device(args.device),
        num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
        checkpoint_path=str(checkpoint),checkpoint_metadata=metadata)
    seconds=time.perf_counter()-start
    payload=torch.load(checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(payload['best_epoch'])
    model.to(args.device).eval()
    save(out/'training_history.json',history)
    if history['best_epoch']!=payload['best_epoch'] or abs(min(history['val_loss'])-payload['best_val_loss'])>1e-10:
        raise RuntimeError('Best checkpoint/VAL history inconsistency')
    train_metrics,_,_,_=evaluate(model,data,args.device,'train')
    val_metrics,_,_,_=evaluate(model,data,args.device,'val')
    # TEST happens only after the VAL-selected best checkpoint is frozen.
    test_metrics,pred,target,components=evaluate(model,data,args.device,'test',diagnostics=True)
    np.savez_compressed(out/'test_predictions.npz',prediction=pred,target=target,
        horizons=np.asarray(config.MULTI_HORIZONS),commodity_names=np.asarray(fingerprint['commodity_order']))
    with np.load(out/'test_predictions.npz',allow_pickle=False) as saved:
        for i,h in enumerate(config.MULTI_HORIZONS):
            assert population_metrics(saved['prediction'][:,i],saved['target'][:,i])==test_metrics[str(h)]
    diagnostic=interventions(model,data,args.device,pred,target,components)
    save(out/'frozen_interventions.json',diagnostic)
    save(out/'metrics.json',dict(TRAIN=train_metrics,VAL=val_metrics,TEST=test_metrics))
    comparisons={}
    for label,reference in refs['models'].items():
        comparisons[label]=dict(VAL5_MAE_delta=val_metrics['5']['MAE']-reference['VAL']['5']['MAE'],
            TEST5_MAE_delta=test_metrics['5']['MAE']-reference['TEST']['5']['MAE'])
    result=dict(status='COMPLETE',variant=VARIANT,seed=42,best_epoch=payload['best_epoch'],
        best_VAL_four_horizon_Huber=payload['best_val_loss'],parameters=count(model),
        new_fusion_parameters=count(model.lag_factor_fusion),train_seconds=seconds,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        metrics=dict(TRAIN=train_metrics,VAL=val_metrics,TEST=test_metrics),
        reference_differences=comparisons,
        below_D0B_seed42_TEST5_threshold=test_metrics['5']['MAE']<0.0219947838,
        frozen_interventions=diagnostic,test_prediction_file=str(out/'test_predictions.npz'))
    save(out/'result.json',result)
    lines=['# D0B Lag Factor Fusion — single seed42 experiment','',
        f'Git commit: `{metadata["git_commit"]}`. One fixed architecture and one run; checkpoint selected by four-horizon prediction-only VAL Huber.',
        f'Data: {fingerprint["splits"]}; 284 nodes, 21 features, 24 commodities, horizons [1,5,10,20], decimal return. Same TRAIN-fitted preprocessing and pooled evaluator as D0B.',
        f'Parameters: D0B {refs["models"]["D0B"]["parameters"]}; new {count(model)}; fusion module {count(model.lag_factor_fusion)}. Best epoch {payload["best_epoch"]}; train {seconds:.1f} s.',
        '', '## Full TEST result','',
        f'TEST MAE@5d = {test_metrics["5"]["MAE"]:.10f}; threshold 0.0219947838: {result["below_D0B_seed42_TEST5_threshold"]}.',
        '', '|Model|VAL MAE 1/5/10/20d|TEST MAE 1/5/10/20d|TEST RMSE@5d|Best epoch|Parameters|',
        '|---|---|---|---:|---:|---:|']
    for label,record in [('D0B',refs['models']['D0B']),('GlobalConcat',refs['models']['GlobalConcat']),
                         ('LagFactor',dict(VAL=val_metrics,TEST=test_metrics,best_epoch=payload['best_epoch'],parameters=count(model)))]:
        val='/'.join(f'{record["VAL"][str(h)]["MAE"]:.8f}' for h in config.MULTI_HORIZONS)
        test='/'.join(f'{record["TEST"][str(h)]["MAE"]:.8f}' for h in config.MULTI_HORIZONS)
        lines.append(f'|{label}|{val}|{test}|{record["TEST"]["5"]["RMSE"]:.8f}|{record["best_epoch"]}|{record["parameters"]}|')
    lines += ['', '## Frozen checkpoint interventions','',
        'Each intervention retains the trained checkpoint and changes only inference inputs to the fusion layer. They measure path usage and are not retrained ablations.',
        '```json',json.dumps(diagnostic,indent=2),'```','',
        '## Provenance and limits','',
        'RDCFM, DeltaLag and RC-Mamba motivated the conditional-factor and lag design; none establishes a CMGM accuracy gain. This design was proposed after earlier TEST observations, so it is post-hoc rather than a blind model selection exercise.',
        'The correction is centered across commodities, so it cannot change the cross-commodity mean prediction at a given horizon. This is a structural limitation if the new signal is mainly market-wide.',
        'Source hashes, reference checkpoint identities, data fingerprint, training history, full metrics and saved TEST predictions are in the sibling files. No parameter search or second seed was run.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'COMPLETE TEST MAE@5d={test_metrics["5"]["MAE"]:.10f}; threshold passed={result["below_D0B_seed42_TEST5_threshold"]}; results={out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    run(parser.parse_args())


if __name__=='__main__':main()
