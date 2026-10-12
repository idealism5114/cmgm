"""Frozen D0B directed relation vs own-history control; --run authorizes fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.models.directed_relation_readout import FrozenD0BDirectedReadout, LAGS, TOP_K
from cmgm.scripts import d0b_node_history_readout as original
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train, _switching_branch

ROOT=Path(__file__).resolve().parents[2]
INDEX=ROOT/'experiments/d0b_global_fusion_test/evaluation_20261007_final/file_index.json'
RESULT_ROOT=ROOT/'experiments/d0b_directed_relation_readout'
CHECKPOINT_ROOT=ROOT/'checkpoints/d0b_directed_relation_readout'
SOURCES=(*original.SOURCE_FILES,'cmgm/models/directed_relation_readout.py',
         'cmgm/scripts/d0b_directed_relation_readout.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def preflight(seed=42):
    from copy import deepcopy
    kwargs=dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,commodity_indices=(6,30))
    original.set_seed(seed)
    backbone=original._make_model(original.BASE_VARIANT,kwargs).eval()
    x=torch.randn(2,20,30,21)
    with torch.no_grad():
        spatial=backbone._temp_weighted_spatial(x)
        spatial2,nodes,sequence=backbone._temp_weighted_spatial(x,return_nodes=True,return_sequence=True)
        torch.testing.assert_close(spatial,spatial2,rtol=0,atol=0)
        assert nodes.shape==(2,30,64) and sequence.shape==(2,20,30,64)
        baseline=backbone(x)
    torch.manual_seed(seed)
    cross=FrozenD0BDirectedReadout(deepcopy(backbone),commodity_indices=(6,30),
                                   stock_count=4,bond_count=2,cross_asset=True).eval()
    torch.manual_seed(seed)
    own=FrozenD0BDirectedReadout(deepcopy(backbone),commodity_indices=(6,30),
                                 stock_count=4,bond_count=2,cross_asset=False).eval()
    for k,v in cross.readout.state_dict().items():
        assert torch.equal(v,own.readout.state_dict()[k])
    assert all(not p.requires_grad for p in cross.backbone.parameters())
    assert _switching_branch(cross) is None
    with torch.no_grad():
        prediction,aux=cross(x,return_selection=True)
        torch.testing.assert_close(prediction,baseline,rtol=0,atol=0)
        assert prediction.shape==(2,4,24) and torch.isfinite(prediction).all()
        assert torch.count_nonzero(aux['correction'])==0
        assert aux['candidate_index'].shape==(2,24,TOP_K)
        assert aux['candidate_index'].max().item()<6*len(LAGS)
        assert aux['asset_index'].min().item()>=0 and aux['asset_index'].max().item()<6
        assert set(aux['lag'].flatten().tolist())<=set(LAGS)
        own_prediction,own_aux=own(x,return_selection=True)
        torch.testing.assert_close(own_prediction,baseline,rtol=0,atol=0)
        assert own_aux['candidate_index'].shape==(2,24,5)
        assert torch.equal(own_aux['asset_index'],torch.arange(6,30)[None,:,None].expand(2,-1,5))
        assert set(own_aux['lag'].flatten().tolist())==set(LAGS)
        torch.testing.assert_close(cross(x.flip(0)).flip(0),prediction,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(cross(x[:1]),prediction[:1],atol=2e-5,rtol=2e-5)
    # Explicitly audit last-known/past indexing, with no possible future index.
    numbered=torch.arange(20*30,dtype=torch.float32).view(1,20,30,1).expand(1,-1,-1,64)
    idx=torch.arange(6)
    difference=cross.readout.differences(numbered,idx)
    for i,lag in enumerate(LAGS):
        torch.testing.assert_close(difference[0,:,i],numbered[0,19-lag,:6]-numbered[0,18-lag,:6])
    # Zero output initialization gives the preceding projections zero gradient
    # on step one. Once the final layer updates, Q/K/V all receive gradients.
    cross.train()
    optimizer=torch.optim.Adam(cross.readout.parameters(),lr=1e-3)
    loss=cross(x).square().mean()
    optimizer.zero_grad();loss.backward()
    final=cross.readout.predictor[-1]
    assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()
    assert final.weight.grad.abs().sum()>0
    optimizer.step()
    optimizer.zero_grad()
    loss=cross(x).square().mean();loss.backward()
    gradients={}
    for name in ('query','key','value','predictor'):
        layer=getattr(cross.readout,name)
        gradients[name]=any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                            for p in layer.parameters())
    assert all(gradients.values()),gradients
    assert all(p.grad is None for p in cross.backbone.parameters())
    assert torch.isfinite(loss)
    return dict(PASS=True,seed=seed,synthetic_shape=[2,20,30,21],
        baseline_zero_correction_exact=True,lag_indices=list(LAGS),top_k=TOP_K,
        cross_candidate_count=30,own_candidate_count=5,
        same_added_parameter_initialization=True,gradients_after_one_synthetic_step=gradients,
        frozen_backbone_gradient_count=0,trainable_parameter_count=parameters(cross.readout))


def load_reference():
    if not INDEX.exists():raise RuntimeError('Audited D0B checkpoint index missing')
    entry=json.loads(INDEX.read_text())[f'{original.BASE_VARIANT}:42']
    prior=json.loads(Path(entry['original_training_result']).read_text())
    evaluated=json.loads(Path(entry['result']).read_text())
    if prior['model']['variant']!=original.BASE_VARIANT or prior['model']['seed']!=42:
        raise RuntimeError('Reference identity mismatch')
    if (sha(entry['checkpoint'])!=entry['checkpoint_sha256'] or
        prior['best_epoch']!=entry['best_epoch'] or
        evaluated['checkpoint_sha256']!=entry['checkpoint_sha256'] or
        not evaluated['test_evaluated']):
        raise RuntimeError('Reference checkpoint or test provenance mismatch')
    return entry,prior,evaluated


@torch.no_grad()
def evaluate(model,loader,device,selection=False):
    model.eval()
    predictions=[];labels=[]
    count_asset=Counter();count_lag=Counter();count_group=Counter()
    nonzero=[]
    for batch in original._loader(loader):
        x,y=batch[0].to(device),batch[1]
        if selection:
            pred,aux=model(x,return_selection=True)
            for asset in aux['asset_index'].cpu().reshape(-1).tolist():
                count_asset[asset]+=1
                count_group['stock' if asset<model.backbone.n_stock else
                            ('bond' if asset<model.backbone.n_stock+model.backbone.n_bond else 'own_commodity')]+=1
            count_lag.update(aux['lag'].cpu().reshape(-1).tolist())
            nonzero.append(float(aux['correction'].abs().mean()))
        else:pred=model(x)
        assert pred.shape==y.shape and torch.isfinite(pred).all()
        predictions.append(pred.cpu().numpy());labels.append(y.numpy())
    prediction,target=np.concatenate(predictions),np.concatenate(labels)
    metrics={str(h):population_metrics(prediction[:,i],target[:,i]) for i,h in enumerate(config.MULTI_HORIZONS)}
    diagnostic=None
    if selection:
        total=sum(count_group.values())
        diagnostic=dict(selected_total=total,by_group=dict(count_group),
            by_group_fraction={str(k):v/total for k,v in count_group.items()},
            by_lag={str(k):count_lag[k] for k in LAGS},
            top_assets=[dict(node_index=k,count=v) for k,v in count_asset.most_common(20)],
            mean_batch_correction_abs=float(np.mean(nonzero)))
    return metrics,prediction,target,diagnostic


def run(args):
    check=preflight(args.seed)
    print('Synthetic preflight PASS:',json.dumps(check),flush=True)
    if not args.run:return
    if args.seed!=42:raise ValueError('Only seed42 is authorized')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    expected=(20,21,(1,5,10,20),'return',.02,64,1e-4,1e-5,200,10)
    actual=(config.SEQ_LEN,config.FEATURE_DIM,tuple(config.MULTI_HORIZONS),config.TARGET_TYPE,
        config.HUBER_DELTA,config.BATCH_SIZE,config.LEARNING_RATE,config.WEIGHT_DECAY,
        config.NUM_EPOCHS,config.PATIENCE)
    if actual!=expected:raise RuntimeError(f'Frozen D0B protocol differs: {actual}')
    entry,prior,prior_test=load_reference()
    print('Building original shared data once...',flush=True)
    data=original.build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=42))
    fp=original._data_fingerprint(data,include_test=False)
    if fp!=prior['model']['data_fingerprint']:
        raise RuntimeError('TRAIN/VAL data/commodity fingerprint differs from frozen D0B')
    kwargs=original._model_kwargs(data)
    if tuple(kwargs['commodity_indices'])!=(260,284) or len(data['loaders']['test'].dataset)!=269:
        raise RuntimeError('Unexpected commodity order or TEST origin count')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    out=RESULT_ROOT/f'run_seed42_{stamp}'
    checkpoints=CHECKPOINT_ROOT/f'run_seed42_{stamp}'
    out.mkdir(parents=True,exist_ok=False);checkpoints.mkdir(parents=True,exist_ok=False)
    metadata=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        source_hashes={s:sha(ROOT/s) for s in SOURCES},
        data_fingerprint=fp,training_protocol=original.PROTOCOL,
        frozen_D0B_checkpoint=entry['checkpoint'],
        frozen_D0B_checkpoint_sha256=entry['checkpoint_sha256'],
        D0B_best_epoch=entry['best_epoch'],
        D0B_VAL=prior['metrics']['val'],
        D0B_TEST5_reference=prior_test['metrics']['5']['MAE'],
        preflight=check,trainable_backbone_parameters=0,
        added_readout_parameters=check['trainable_parameter_count'],
        source_node_range=[0,260],commodity_node_range=[260,284],
        commodity_order=fp['commodity_order'],lag_set=list(LAGS),top_k=TOP_K,
        KL_training_contribution=0.0,
        checkpoint_selection='four-horizon prediction-only VAL Huber, inherited batch-mean aggregation')
    save(out/'config.json',metadata)
    states={}
    device=torch.device(args.device)
    for label,cross in [('directed_cross_asset',True),('own_history_control',False)]:
        original.set_seed(42)
        backbone=original._make_model(original.BASE_VARIANT,kwargs)
        payload=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
        backbone.load_state_dict(payload['model_state_dict'],strict=True)
        if payload['best_epoch']!=entry['best_epoch']:raise RuntimeError('D0B checkpoint epoch mismatch')
        backbone.switching_latent_transformer.set_epoch(payload['best_epoch'])
        model=FrozenD0BDirectedReadout(backbone,commodity_indices=tuple(kwargs['commodity_indices']),
            stock_count=kwargs['n_stock'],bond_count=kwargs['n_bond'],cross_asset=cross)
        assert _switching_branch(model) is None and all(not p.requires_grad for p in backbone.parameters())
        assert parameters(model.readout)==check['trainable_parameter_count']
        model.to(device)
        # Check zero-correction identity with the actual frozen checkpoint.
        x,_=next(iter(original._loader(data['loaders']['val'])))
        with torch.no_grad():
            model.eval();base=model.backbone(x.to(device));pred=model(x.to(device))
            torch.testing.assert_close(pred,base,rtol=0,atol=0)
        checkpoint=checkpoints/(label+'_best.pt')
        print(f'Training {label}; output={out}',flush=True)
        start=time.perf_counter()
        history=train(model,data['loaders']['train'],data['loaders']['val'],
            torch.empty(2,0,dtype=torch.long),torch.empty(0),device,
            num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
            checkpoint_path=str(checkpoint),checkpoint_metadata=dict(metadata,model=label))
        elapsed=time.perf_counter()-start
        best=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(best['model_state_dict'],strict=True)
        model.eval()
        if history['best_epoch']!=best['best_epoch'] or abs(min(history['val_loss'])-best['best_val_loss'])>1e-10:
            raise RuntimeError('Best checkpoint/VAL selection mismatch')
        if sha(entry['checkpoint'])!=entry['checkpoint_sha256']:
            raise RuntimeError('Frozen D0B checkpoint changed')
        val,pred_val,target_val,distribution=evaluate(model,data['loaders']['val'],device,selection=True)
        run=dict(status='VAL_COMPLETE',label=label,cross_asset=cross,seed=42,
            best_epoch=best['best_epoch'],best_VAL_Huber=best['best_val_loss'],
            added_parameters=parameters(model.readout),total_parameters=parameters(model),
            train_seconds=elapsed,VAL=val,selection_distribution=distribution,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            frozen_backbone_sha256=entry['checkpoint_sha256'],test_evaluated=False)
        save(out/(label+'_training_history.json'),history)
        save(out/(label+'_VAL_result.json'),run)
        states[label]=run
        del model,backbone,payload,best
        if torch.cuda.is_available():torch.cuda.empty_cache()
    gate=dict(criterion='main VAL MAE@5d strictly below both D0B and own-history control',
        D0B_VAL5=prior['metrics']['val']['5']['MAE'],
        directed_VAL5=states['directed_cross_asset']['VAL']['5']['MAE'],
        own_VAL5=states['own_history_control']['VAL']['5']['MAE'])
    gate['PASS']=(gate['directed_VAL5']<gate['D0B_VAL5'] and
                  gate['directed_VAL5']<gate['own_VAL5'])
    save(out/'TEST_gate.json',gate)
    report=['# Frozen D0B directed relation readout — seed42','',
        f'Git commit `{metadata["git_commit"]}`; working-tree source hashes and full status in `config.json`.',
        f'Frozen D0B checkpoint: `{entry["checkpoint"]}` (SHA256 `{entry["checkpoint_sha256"]}`).',
        'Train/VAL/TEST origins: 1396/268/269. Window 20, nodes 284, features 21, commodities 24, horizons [1,5,10,20], target decimal return.',
        'Only the new readout is optimized; the original graph, transformer, regime filter, global gate and shared head remain frozen and are each run once per forward. Native Switch KL has no trainable contribution in this frozen-backbone experiment.',
        'Optimizer Adam lr=1e-4 weight_decay=1e-5; batch64, up to200 epochs, patience10, ReduceLROnPlateau factor0.5; original four-horizon Huber and batch-mean VAL checkpoint selection.',
        '', '## VAL results', '',
        '|Model|VAL MAE@1d|VAL MAE@5d|VAL MAE@10d|VAL MAE@20d|Best epoch|Added params|Train s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in [('frozen_D0B',dict(VAL=prior['metrics']['val'],best_epoch=entry['best_epoch'],added_parameters=0,train_seconds=0)),*states.items()]:
        v=r['VAL'];report.append(f'|{label}|'+ '|'.join(f'{v[str(h)]["MAE"]:.10f}' for h in [1,5,10,20])+f'|{r["best_epoch"]}|{r["added_parameters"]}|{r["train_seconds"]:.1f}|')
    report+=['','## Selection distribution (VAL)',
        'Selection counts include all five chosen source asset/lag positions per commodity and origin. See each `*_VAL_result.json` for group fractions, lag counts and top asset indices.']
    for label,r in states.items():
        d=r['selection_distribution'];report.append(f'- {label}: groups {d["by_group"]}; lags {d["by_lag"]}; mean absolute correction {d["mean_batch_correction_abs"]:.8f}.')
    report+=['',f'## TEST gate: {gate["PASS"]}',
        f'Directed VAL MAE@5d {gate["directed_VAL5"]:.10f}; D0B {gate["D0B_VAL5"]:.10f}; own-history {gate["own_VAL5"]:.10f}.',
        'This is a frozen-base test of incremental information retained before spatial time pooling. It does not determine whether every end-to-end directed relation model would fail or succeed. Added parameters and training budget prevent a pure architecture-causality claim.']
    if gate['PASS']:
        original.set_seed(42)
        backbone=original._make_model(original.BASE_VARIANT,kwargs)
        payload=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
        backbone.load_state_dict(payload['model_state_dict'],strict=True)
        model=FrozenD0BDirectedReadout(backbone,commodity_indices=tuple(kwargs['commodity_indices']),
            stock_count=kwargs['n_stock'],bond_count=kwargs['n_bond'],cross_asset=True).to(device)
        best=torch.load(states['directed_cross_asset']['checkpoint'],map_location=device,weights_only=False)
        model.load_state_dict(best['model_state_dict'],strict=True)
        model.eval()
        test,pred,target,_=evaluate(model,data['loaders']['test'],device)
        np.savez_compressed(out/'directed_cross_asset_TEST_predictions.npz',prediction=pred,target=target,
            horizons=np.asarray(config.MULTI_HORIZONS),commodity_names=np.asarray(fp['commodity_order']))
        with np.load(out/'directed_cross_asset_TEST_predictions.npz',allow_pickle=False) as saved:
            assert all(population_metrics(saved['prediction'][:,i],saved['target'][:,i])==test[str(h)]
                       for i,h in enumerate(config.MULTI_HORIZONS))
        test_result=dict(status='COMPLETE',test_evaluated=True,TEST=test,
            threshold=0.0219947838,below_threshold=test['5']['MAE']<0.0219947838,
            checkpoint=states['directed_cross_asset']['checkpoint'],
            checkpoint_sha256=states['directed_cross_asset']['checkpoint_sha256'],
            D0B_seed42_TEST5_from_audited_reference=prior_test['metrics']['5']['MAE'])
        save(out/'directed_cross_asset_TEST_result.json',test_result)
        report+=['', '## Preselected main checkpoint TEST', '',
            f'TEST MAE@5d {test["5"]["MAE"]:.10f}; below D0B seed42 threshold 0.0219947838: {test_result["below_threshold"]}.',
            '|Horizon|TEST MAE|MSE|RMSE|Hit (%)|','|---|---:|---:|---:|---:|']
        for h in config.MULTI_HORIZONS:
            m=test[str(h)];report.append(f'|{h}d|{m["MAE"]:.10f}|{m["MSE"]:.10f}|{m["RMSE"]:.10f}|{100*m["Hit"]:.4f}|')
    else:
        report+=['','No TEST inference was performed because the predeclared VAL gate failed.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    save(out/'results.json',dict(status='COMPLETE',test_evaluated=gate['PASS'],
        metadata=metadata,VAL_results=states,TEST_gate=gate,
        report=str(out/'REPORT.md')))
    print(f'COMPLETE: VAL gate={gate["PASS"]}; results={out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    run(parser.parse_args())


if __name__=='__main__':main()
