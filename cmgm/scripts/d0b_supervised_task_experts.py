"""Frozen independent D0B + directly supervised G/S/L prediction experts.

Default: synthetic preflight and checkpoint metadata audit. Explicit prepare /
train / summarize modes use TRAIN/VAL only. There is no TEST or router entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.frozen_d0b_task_experts import (
    BASE_VARIANT, HORIZONS, ROLES, CRITERIA, EPSILON, MAIN_CHECKPOINTS,
    FrozenD0BFeatures, ResidualPredictionExpert, matched_experts,
    tensor_state_hash, horizon_weights, horizon_losses, weighted_objective,
    combine_strategies,
)
from cmgm.scripts.d0b_short_long_dynamic_moe import seed_all, SOURCE_FILES, full_loader
from cmgm.scripts.d0b_short_long_expert_diagnostics import (
    ROOT, discover_index, load_run, load_validated_data, compare_saved,
    sha256, write_json, write_csv, model_kwargs, metrics, check_protocol,
)

EXPERIMENT_ROOT = ROOT / 'experiments/d0b_supervised_task_experts'
CHECKPOINT_ROOT = ROOT / 'checkpoints/d0b_supervised_task_experts'
NEW_SOURCES = ('cmgm/models/frozen_d0b_task_experts.py', 'cmgm/scripts/d0b_supervised_task_experts.py')
ROLE_CRITERION = dict(zip(ROLES, CRITERIA))


def source_record():
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        git_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest(),
        source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,*NEW_SOURCES,
            'cmgm/scripts/d0b_short_long_expert_diagnostics.py')})


def runtime(device):
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        device=device,cuda=torch.version.cuda,autocast=False,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def audit_base(file_index=None, device='cpu'):
    index_path, index = discover_index(file_index)
    base, cfg, result, provenance = load_run('d0b',index['d0b'],device)
    if cfg['variant'] != BASE_VARIANT or cfg['seed'] != 42:
        raise ValueError('Require independently trained D0B seed42')
    shape = cfg['data_audit']['input_shape']
    if shape[1:] != [20,284,21] or cfg['data_audit']['output_shape'][1:] != [4,24]:
        raise ValueError('D0B audited data dimensions differ from this fixed study')
    match = re.fullmatch(r'sum_h Huber\(delta=([0-9.]+)\)',cfg['protocol']['prediction_loss'])
    if match is None or float(match.group(1)) != .02:
        raise ValueError('Original Huber specification differs')
    features = FrozenD0BFeatures(base)
    protected = {Path(index_path):sha256(index_path)}
    for k in ('checkpoint','config','result','training_history','TRAIN_predictions','VAL_predictions'):
        p=Path(index['d0b'][k]);protected[p]=sha256(p)
    return features,cfg,result,dict(**provenance,file_index=str(index_path.resolve()),
        file_index_sha256=sha256(index_path),data_audit=cfg['data_audit'],
        target_definition='clip(raw_price[origin+h]/max(abs(raw_price[origin]),1e-8)-1,-1,1)',
        target_scale='decimal cumulative return',asset_order=cfg['data_audit']['node_order'],
        commodity_order=cfg['data_audit']['commodity_order'],
        protected_files_sha256={str(p):digest for p,digest in protected.items()}),index['d0b']


def check_unchanged(protected):
    if any(sha256(Path(p)) != digest for p,digest in protected.items()):
        raise ValueError('An original checkpoint, result or config changed; STOP')


def find_data_cache(explicit, expected):
    if explicit:
        return explicit
    candidates=[]
    for path in sorted((ROOT/'experiments/d0b_short_long_dynamic_moe').glob('expert_diagnostics_*/diagnostic_file_index.json')):
        index=json.loads(path.read_text())
        audit_path=index.get('audit.json');cache=index.get('validated_train_val_cache.npz')
        if not audit_path or not cache:
            continue
        audit=json.loads(Path(audit_path).read_text())
        if audit['status']=='COMPLETE' and audit['data_audit']==expected:
            candidates.append(Path(cache))
    if not candidates or len({sha256(p) for p in candidates}) != 1:
        raise ValueError('Missing or conflicting audited TRAIN/VAL cache; supply --data-cache')
    return candidates[0]


@torch.no_grad()
def extract_split(features, loader, dates, commodity_order, device):
    collected={k:[] for k in ('u','y_base','target')}
    for x,y in full_loader(loader):
        frozen=features(x.to(device))
        for k in ('u','y_base'):
            collected[k].append(frozen[k].cpu().numpy())
        collected['target'].append(y.numpy())
    arrays={k:np.concatenate(v) for k,v in collected.items()}
    if len(arrays['target']) != len(dates):
        raise ValueError('Feature/origin count mismatch')
    arrays.update(origin_dates=np.asarray(dates),commodity_order=np.asarray(commodity_order,dtype=str))
    return arrays


def preflight(args):
    out=args.output or EXPERIMENT_ROOT/f'preflight_seed42_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_frozen_d0b_task_experts.py'],cwd=ROOT,check=True)
    audit=dict(status='PASS',synthetic_validation='PASS',formal_training='NOT_RUN',
        real_TRAIN_VAL_feature_extraction='NOT_RUN',test_evaluated=False)
    try:
        features,cfg,_,base_audit,_=audit_base(args.file_index)
    except FileNotFoundError as error:
        audit.update(base_checkpoint_audit='MISSING',missing=str(error))
        write_json(out/'audit.json',audit)
        write_json(out/'source_hashes.json',source_record())
        print(f'SYNTHETIC_PASS; real base missing: {error}; {out}',flush=True)
        return
    experts,initialization=matched_experts(features.input_dim,features.n_commodities)
    frozen=sum(p.numel() for p in features.parameters())
    count=sum(p.numel() for p in experts['G'].parameters())
    config=dict(base_variant=BASE_VARIANT,seed=42,horizons=list(HORIZONS),epsilon=EPSILON,
        input_dim=features.input_dim,expert_structure=[features.input_dim,32,64,4*features.n_commodities],
        base_frozen_parameters=frozen,per_expert_trainable_parameters=count,
        per_online_adapter_total=frozen+count,three_experts_trainable_parameters=3*count,
        base_plus_three_experts_parameters=frozen+3*count,
        train_objectives={r:list(horizon_weights(ROLE_CRITERION[r])) for r in ROLES},
        main_checkpoints=MAIN_CHECKPOINTS,original_protocol=cfg['protocol'],test_evaluated=False)
    check_unchanged(base_audit['protected_files_sha256'])
    audit.update(base_checkpoint_audit='PASS',strict_load=True,
        original_artifacts_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    write_json(out/'config.json',config);write_json(out/'source_hashes.json',source_record())
    write_json(out/'initialization_audit.json',initialization)
    write_json(out/'base_audit.json',base_audit);write_json(out/'audit.json',audit)
    write_json(out/'file_index.json',{p.name:str(p.resolve()) for p in out.iterdir() if p.is_file()})
    print(f'PREFLIGHT_PASS base={frozen} expert={count} directory={out.resolve()}',flush=True)


def verify_study(study):
    config=json.loads((study/'config.json').read_text())
    status=json.loads((study/'PREPARE_STATUS.json').read_text())
    if status['status'] != 'READY' or config['seed'] != 42 or config['epsilon'] != EPSILON or config['base_variant'] != BASE_VARIANT:
        raise ValueError('Study is not an immutable READY seed42 preparation')
    if sha256(study/'config.json') != status['config_sha256']:
        raise ValueError('Prepared study configuration changed')
    if any(sha256(ROOT/p)!=digest for p,digest in config['source']['source_sha256'].items()):
        raise ValueError('Study/training source changed; STOP for review')
    for name in ('frozen_features.npz','expert_initial_state.pt','initialization_audit.json','base_audit.json'):
        if sha256(study/name) != status['prepared_files_sha256'][name]:
            raise ValueError('Prepared input or initialization changed')
    check_unchanged(config['base_audit']['protected_files_sha256'])
    return config


def prepare(args):
    study=args.study
    if study.exists():
        verify_study(study)
        print(f'REUSE_READY {study.resolve()}',flush=True)
        return study
    study.mkdir(parents=True,exist_ok=False)
    write_json(study/'PREPARE_STATUS.json',dict(status='PREPARING',test_evaluated=False))
    started=time.perf_counter()
    try:
        features,cfg,base_result,base_audit,item=audit_base(args.file_index,args.device)
        env=cfg['environment']
        torch.backends.cudnn.deterministic=env['cudnn_deterministic']
        torch.backends.cudnn.benchmark=env['cudnn_benchmark']
        before=tensor_state_hash(features.base.state_dict())
        cache=find_data_cache(args.data_cache,cfg['data_audit'])
        data,fingerprint=load_validated_data(cache,cfg['data_audit'],Path(cfg['reference_config']))
        if (sha256(Path(cfg['reference_config'])) != cfg['reference_config_sha256']
                or json.loads(json.dumps(model_kwargs(data))) != cfg['model_kwargs']):
            raise ValueError('Original data/reference/model mismatch')
        packed={};reproduction={}
        for split in ('train','val'):
            arrays=extract_split(features,data['loaders'][split],data['origin_dates'][split],fingerprint['commodity_order'],args.device)
            score=metrics(arrays['y_base'],arrays['target'],.02)
            comparable=dict(prediction=arrays['y_base'],target=arrays['target'],origin_dates=arrays['origin_dates'],commodity_order=arrays['commodity_order'])
            difference=compare_saved(comparable,Path(item[f'{split.upper()}_predictions']),score['horizons'],base_result['metrics'][split])
            reproduction[split]=dict(PASS=True,max_abs_prediction_difference=difference,origins=len(arrays['target']))
            packed.update({f'{split}_{k}':v for k,v in arrays.items()})
        experts,initialization=matched_experts(features.input_dim,features.n_commodities)
        initial=experts['G'].state_dict()
        torch.save(initial,study/'expert_initial_state.pt')
        np.savez_compressed(study/'frozen_features.npz',**packed)
        write_json(study/'initialization_audit.json',initialization)
        write_json(study/'base_audit.json',base_audit)
        frozen=sum(p.numel() for p in features.parameters());trainable=sum(p.numel() for p in experts['G'].parameters())
        config=dict(schema_version=1,base_variant=BASE_VARIANT,seed=42,epsilon=EPSILON,delta=.02,
            horizons=list(HORIZONS),input_dim=features.input_dim,n_commodities=features.n_commodities,
            frozen_parameters=frozen,trainable_per_expert=trainable,base_audit=base_audit,
            original_protocol=cfg['protocol'],data_audit=fingerprint,calendar_audit=data['calendar_audit'],
            original_reference_config=cfg['reference_config'],data_cache=str(cache.resolve()),data_cache_sha256=sha256(cache),
            feature_mode='One frozen original D0B forward per preparation batch; immutable detached cache during head training',
            original_environment=env,preparation_environment=runtime(args.device),source=source_record(),
            main_checkpoints=MAIN_CHECKPOINTS,training_objective_weights={r:list(horizon_weights(ROLE_CRITERION[r])) for r in ROLES},
            selector_weights={c:list(horizon_weights(c)) for c in CRITERIA},
            monitor='global prediction-only VAL Huber sum, original batch-mean',
            selection_note='Three selectors saved along one trajectory stopped by global VAL Huber; main G-global/S-short/L-long fixed before training',
            switch_loss_in_adaptation=False,normal_reproduction=reproduction,test_evaluated=False)
        after=tensor_state_hash(features.base.state_dict())
        if before != after or any(p.requires_grad or p.grad is not None for p in features.parameters()):
            raise ValueError('Base changed during frozen preparation')
        check_unchanged(base_audit['protected_files_sha256'])
        write_json(study/'config.json',config)
        write_json(study/'source_hashes.json',config['source'])
        write_json(study/'PREPARE_STATUS.json',dict(status='READY',test_evaluated=False,
            original_base_parameters_unchanged=True,base_state_before=before,base_state_after=after,
            config_sha256=sha256(study/'config.json'),
            preparation_seconds=time.perf_counter()-started,
            prepared_files_sha256={name:sha256(study/name) for name in ('frozen_features.npz','expert_initial_state.pt','initialization_audit.json','base_audit.json')}))
        print(f'PREPARED {study.resolve()} TRAIN/VAL only',flush=True)
        return study
    except Exception as error:
        write_json(study/'PREPARE_STATUS.json',dict(status='FAILED_STOP',error=f'{type(error).__name__}: {error}',test_evaluated=False))
        raise


def cached_arrays(study,config):
    with np.load(study/'frozen_features.npz',allow_pickle=False) as z:
        allowed={f'{s}_{k}' for s in ('train','val') for k in ('u','y_base','target','origin_dates','commodity_order')}
        if set(z.files) != allowed:
            raise ValueError('Cache must contain TRAIN/VAL only')
        arrays={s:{k:z[f'{s}_{k}'] for k in ('u','y_base','target','origin_dates','commodity_order')} for s in ('train','val')}
    for split,a in arrays.items():
        n=config['data_audit']['splits'][split]['origins'];c=config['n_commodities']
        if a['u'].shape != (n,config['input_dim']) or a['target'].shape != (n,4,c) or a['y_base'].shape != a['target'].shape:
            raise ValueError('Invalid cached feature/target shapes')
        if a['commodity_order'].tolist()!=config['data_audit']['commodity_order'] or len(a['origin_dates'])!=n:
            raise ValueError('Origin or commodity alignment changed')
        if any(not np.isfinite(a[k]).all() for k in ('u','y_base','target')):
            raise ValueError('Nonfinite cached inputs')
    return arrays


def cached_loader(arrays,batch_size,drop_last=False):
    tensors=tuple(torch.from_numpy(np.ascontiguousarray(arrays[k])) for k in ('u','y_base','target'))
    return DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=False,drop_last=drop_last,num_workers=0)


@torch.no_grad()
def evaluate_expert(expert,loader,delta,device):
    expert.eval()
    preds,targets=[],[]
    horizon_sum=np.zeros(4);selectors={c:0. for c in CRITERIA};samples=0;batches=0
    sample_sum=np.zeros(4)
    for u,y0,y in loader:
        pred=expert.predict(u.to(device),y0.to(device));y=y.to(device)
        terms=horizon_losses(pred,y,delta)
        if not torch.isfinite(pred).all() or not torch.isfinite(terms).all():
            raise FloatingPointError('Nonfinite expert validation')
        values=terms.cpu().numpy().astype(np.float64)
        horizon_sum+=values;sample_sum+=values*len(u);samples+=len(u);batches+=1
        for criterion in CRITERIA:
            selectors[criterion]+=weighted_objective(terms,criterion).item()
        preds.append(pred.cpu().numpy());targets.append(y.cpu().numpy())
    if not batches:
        raise ValueError('Empty expert evaluation loader')
    prediction,target=np.concatenate(preds),np.concatenate(targets)
    if samples != len(loader.dataset) or loader.drop_last:
        raise ValueError('Evaluation must cover the full population')
    scores=metrics(prediction,target,delta)
    scores.update(selector_batch_mean={c:v/batches for c,v in selectors.items()},
        horizon_Huber_batch_mean=(horizon_sum/batches).tolist(),
        horizon_Huber_sample_mean_float32_batch_terms=(sample_sum/samples).tolist(),
        origins=samples,batches=batches)
    return scores,prediction


def save_selector(path,expert,role,criterion,epoch,value,metadata):
    state={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
    payload=dict(schema_version=1,expert_role=role,selection_criterion=criterion,epoch=epoch,
        selection_loss_batch_mean=value,expert_state_dict=state,expert_state_sha256=tensor_state_hash(state),metadata=metadata)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def fit_expert(expert,train_loader,val_loader,role,delta,protocol,checkpoint_directory,metadata,history_path=None,device='cpu'):
    """Direct independent supervision; all three selectors, global stopping.

    Small synthetic tests pass a reduced budget; CLI always uses the full audited
    protocol. No base model or KL term is accepted by this trainer.
    """
    expert.to(device)
    optimizer=torch.optim.Adam(expert.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in expert.parameters()}:
        raise ValueError('Optimizer must contain exactly this expert')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    best={c:dict(loss=float('inf'),epoch=0) for c in CRITERIA}
    history=[];stale=0;started=time.perf_counter()
    for epoch in range(1,protocol['max_epochs']+1):
        expert.train();terms_sum=np.zeros(4);sample_sum=np.zeros(4);loss_sum=0.;samples=0;batches=0
        grad_bias=np.zeros(4);grad_weight=np.zeros(4);first_grad=0.;middle_grad=0.
        lr_before=optimizer.param_groups[0]['lr']
        for u,y0,y in train_loader:
            u,y0,y=u.to(device),y0.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=expert.predict(u,y0);terms=horizon_losses(pred,y,delta)
            loss=weighted_objective(terms,ROLE_CRITERION[role])
            if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite expert training prediction/loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in expert.parameters()):
                raise FloatingPointError('Missing or nonfinite expert gradient')
            output=expert.network[-1]
            grad_bias+=output.bias.grad.reshape(4,-1).norm(dim=1).cpu().numpy()
            grad_weight+=output.weight.grad.reshape(4,expert.n_commodities,-1).norm(dim=(1,2)).cpu().numpy()
            first_grad+=expert.network[0].weight.grad.norm().item()
            middle_grad+=expert.network[2].weight.grad.norm().item()
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in expert.parameters()):
                raise FloatingPointError('Nonfinite expert update')
            values=terms.detach().cpu().numpy().astype(np.float64)
            terms_sum+=values;sample_sum+=values*len(u);loss_sum+=loss.item();samples+=len(u);batches+=1
        if not batches:
            raise ValueError('Empty TRAIN loader after drop_last')
        val,_=evaluate_expert(expert,val_loader,delta,device)
        monitor=val['selector_batch_mean']['global']
        scheduler.step(monitor)
        improved=monitor<best['global']['loss']
        for criterion in CRITERIA:
            value=val['selector_batch_mean'][criterion]
            if value<best[criterion]['loss']:
                best[criterion]=dict(loss=value,epoch=epoch)
                save_selector(checkpoint_directory/f'{criterion}_best.pt',expert,role,criterion,epoch,value,metadata)
        stale=0 if improved else stale+1
        row=dict(epoch=epoch,train_objective=ROLE_CRITERION[role],train_loss_batch_mean=loss_sum/batches,
            train_horizon_Huber_batch_mean=(terms_sum/batches).tolist(),
            train_horizon_Huber_sample_mean_float32_batch_terms=(sample_sum/samples).tolist(),
            train_weighted_horizon_contribution_batch_mean=(terms_sum/batches*np.array(horizon_weights(ROLE_CRITERION[role]))).tolist(),
            train_origins_used=samples,train_batches=batches,val=val,lr_before=lr_before,
            lr_after=optimizer.param_groups[0]['lr'],global_no_improvement_epochs=stale,
            gradient_diagnostics=dict(mean_output_bias_grad_L2_by_horizon=(grad_bias/batches).tolist(),
                mean_output_weight_grad_L2_by_horizon=(grad_weight/batches).tolist(),
                mean_first_layer_grad_L2=first_grad/batches,mean_middle_layer_grad_L2=middle_grad/batches))
        history.append(row)
        if history_path is not None:
            write_json(history_path,dict(status='TRAINING',epochs=history,selectors=best))
        print(f'[{role}] epoch={epoch} train={loss_sum/batches:.9g} VALglobal={monitor:.9g} VAL5={val["horizons"]["5"]["MAE"]:.9g}',flush=True)
        if stale>=protocol['early_stopping_patience']:
            break
    return dict(status='TRAINING_COMPLETE',epochs=history,selectors=best,final_epoch=len(history),
        early_stopping_monitor='global four-horizon VAL prediction Huber batch-mean',
        scheduler_monitor='global four-horizon VAL prediction Huber batch-mean',
        expert_training_seconds=time.perf_counter()-started)


def load_selector(path,config,role,criterion,history,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    chosen=history['selectors'][criterion]
    losses=[r['val']['selector_batch_mean'][criterion] for r in history['epochs']]
    if (payload['expert_role']!=role or payload['selection_criterion']!=criterion
            or payload['epoch']!=chosen['epoch'] or payload['epoch']!=int(np.argmin(losses))+1
            or payload['selection_loss_batch_mean']!=chosen['loss']
            or payload['expert_state_sha256']!=tensor_state_hash(payload['expert_state_dict'])
            or payload['metadata']['base_checkpoint_sha256']!=config['base_audit']['checkpoint_sha256']
            or payload['metadata']['epsilon']!=EPSILON or payload['metadata']['delta']!=config['delta']
            or payload['metadata']['horizons']!=list(HORIZONS)
            or payload['metadata']['data_audit']!=config['data_audit']):
        raise ValueError('Expert checkpoint/history/identity mismatch')
    expert=ResidualPredictionExpert(config['input_dim'],config['n_commodities'])
    expert.load_state_dict(payload['expert_state_dict'],strict=True)
    expert.requires_grad_(False).to(device).eval()
    return expert,payload


def train_role(args):
    study=prepare(args);config=verify_study(study)
    out=study/args.expert
    if out.exists():
        status=json.loads((out/'RUN_STATUS.json').read_text())
        if status['status']!='COMPLETE':
            raise ValueError('Existing expert run is interrupted/invalid; STOP for review, no silent warm start')
        validate_run(study,args.expert,config)
        print(f'SKIP_COMPLETE {out}',flush=True)
        return
    out.mkdir(exist_ok=False)
    write_json(out/'RUN_STATUS.json',dict(status='TRAINING',test_evaluated=False))
    checkpoint_directory=CHECKPOINT_ROOT/study.name/args.expert
    started=time.perf_counter()
    try:
        checkpoint_directory.mkdir(parents=True,exist_ok=False)
        source=source_record();torch.set_num_threads(args.threads)
        env=config['original_environment'];torch.backends.cudnn.deterministic=env['cudnn_deterministic'];torch.backends.cudnn.benchmark=env['cudnn_benchmark']
        seed_all(42)
        expert=ResidualPredictionExpert(config['input_dim'],config['n_commodities'])
        initial=torch.load(study/'expert_initial_state.pt',map_location='cpu',weights_only=True)
        expert.load_state_dict(initial,strict=True)
        if tensor_state_hash(expert.state_dict())!=json.loads((study/'initialization_audit.json').read_text())['initial_state_sha256']:
            raise ValueError('Expert initial state mismatch')
        arrays=cached_arrays(study,config);protocol=config['original_protocol']
        train_loader=cached_loader(arrays['train'],protocol['batch_size'],drop_last=protocol['train_drop_last'])
        val_loader=cached_loader(arrays['val'],protocol['batch_size'])
        metadata=dict(role=args.expert,seed=42,epsilon=EPSILON,delta=config['delta'],horizons=list(HORIZONS),
            base_checkpoint_sha256=config['base_audit']['checkpoint_sha256'],study_config_sha256=sha256(study/'config.json'),
            initial_state_sha256=tensor_state_hash(initial),data_audit=config['data_audit'],
            main_checkpoint_criterion=MAIN_CHECKPOINTS[args.expert],training_objective=ROLE_CRITERION[args.expert],
            source=source,environment=runtime(args.device),original_protocol=protocol,
            KL_in_objective=False,expert_only=True,test_evaluated=False)
        write_json(out/'config.json',metadata)
        # Equal caller seed after all constructors/audits, as in the old protocol.
        seed_all(42)
        history=fit_expert(expert,train_loader,val_loader,args.expert,config['delta'],protocol,
            checkpoint_directory,metadata,out/'training_history.json',args.device)
        write_json(out/'training_history.json',history)
        checkpoints={}
        for criterion in CRITERIA:
            path=checkpoint_directory/f'{criterion}_best.pt'
            frozen,payload=load_selector(path,config,args.expert,criterion,history,args.device)
            scores={};predictions={}
            for split in ('train','val'):
                score,prediction=evaluate_expert(frozen,cached_loader(arrays[split],protocol['batch_size']),config['delta'],args.device)
                scores[split]=score
                # Reproduce selector value at its selected checkpoint.
                if split=='val' and abs(score['selector_batch_mean'][criterion]-payload['selection_loss_batch_mean'])>1e-8:
                    raise ValueError('Restored checkpoint selector loss does not reproduce')
                filename=out/f'{criterion}_{split}_predictions.npz'
                np.savez_compressed(filename,prediction=prediction,target=arrays[split]['target'],
                    origin_dates=arrays[split]['origin_dates'],commodity_order=arrays[split]['commodity_order'])
                predictions[split]=dict(path=str(filename.resolve()),sha256=sha256(filename))
            checkpoints[criterion]=dict(path=str(path.resolve()),sha256=sha256(path),epoch=payload['epoch'],
                criterion=criterion,selection_loss_batch_mean=payload['selection_loss_batch_mean'],metrics=scores,predictions=predictions,
                strict_load=True,main_checkpoint=criterion==MAIN_CHECKPOINTS[args.expert])
        check_unchanged(config['base_audit']['protected_files_sha256'])
        result=dict(status='COMPLETE',training_complete=True,role=args.expert,seed=42,test_evaluated=False,
            trainable_parameters=config['trainable_per_expert'],frozen_parameters=config['frozen_parameters'],
            checkpoints=checkpoints,main_checkpoint=MAIN_CHECKPOINTS[args.expert],
            expert_training_seconds=history['expert_training_seconds'],total_run_seconds=time.perf_counter()-started,
            original_base_unchanged=True,study_config_sha256=sha256(study/'config.json'))
        write_json(out/'result.json',result)
        write_json(out/'RUN_STATUS.json',dict(status='COMPLETE',test_evaluated=False))
        write_json(out/'file_index.json',dict(config=str((out/'config.json').resolve()),training_history=str((out/'training_history.json').resolve()),
            result=str((out/'result.json').resolve()),checkpoints=checkpoints))
        print(f'COMPLETE {out.resolve()}',flush=True)
    except Exception as error:
        write_json(out/'RUN_STATUS.json',dict(status='FAILED_STOP',error=f'{type(error).__name__}: {error}',test_evaluated=False))
        raise


def validate_run(study,role,config):
    out=study/role
    result=json.loads((out/'result.json').read_text());history=json.loads((out/'training_history.json').read_text())
    status=json.loads((out/'RUN_STATUS.json').read_text());metadata=json.loads((out/'config.json').read_text())
    if (status['status']!='COMPLETE' or result['status']!='COMPLETE' or not result['training_complete']
            or result['test_evaluated'] or result['role']!=role or result['seed']!=42
            or result['study_config_sha256']!=sha256(study/'config.json') or metadata['data_audit']!=config['data_audit']
            or metadata['epsilon']!=EPSILON or metadata['delta']!=config['delta']
            or metadata['horizons']!=list(HORIZONS) or metadata['original_protocol']!=config['original_protocol']
            or metadata['source']['source_sha256']!=config['source']['source_sha256']
            or history['status']!='TRAINING_COMPLETE' or history['final_epoch']!=len(history['epochs'])
            or metadata['initial_state_sha256']!=json.loads((study/'initialization_audit.json').read_text())['initial_state_sha256']):
        raise ValueError('Incomplete/conflicting expert run')
    for criterion,checkpoint in result['checkpoints'].items():
        path=Path(checkpoint['path'])
        if sha256(path)!=checkpoint['sha256']:
            raise ValueError('Expert checkpoint hash changed')
        _,payload=load_selector(path,config,role,criterion,history)
        if payload['metadata']!=metadata:
            raise ValueError('Checkpoint and run metadata differ')
        for split,entry in checkpoint['predictions'].items():
            if sha256(Path(entry['path']))!=entry['sha256']:
                raise ValueError('Expert prediction artifact changed')
    return result,history


def summarize(args):
    study=args.study;config=verify_study(study);arrays=cached_arrays(study,config)
    runs={r:validate_run(study,r,config)[0] for r in ROLES}
    out=args.output or study/f'summary_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    candidates={s:{} for s in ('train','val')};capability=[];checkpoint_index={}
    for role,run in runs.items():
        for criterion,ck in run['checkpoints'].items():
            name=f'{role}-{criterion}';checkpoint_index[name]=ck
            for split,entry in ck['predictions'].items():
                with np.load(entry['path'],allow_pickle=False) as z:
                    if any(not np.array_equal(z[k],arrays[split][k]) for k in ('target','origin_dates','commodity_order')):
                        raise ValueError('Expert origin/target/commodity alignment mismatch')
                    prediction=z['prediction']
                score=metrics(prediction,arrays[split]['target'],config['delta'])
                if any(score[k]!=ck['metrics'][split][k] for k in ('group_huber','horizons')):
                    raise ValueError('Saved expert population metrics not reproduced')
                candidates[split][name]=prediction
                for g,group in enumerate(('short','long')):
                    reference=runs['G']['checkpoints']['global']['metrics'][split]['group_huber'][g]
                    row=dict(split=split,expert=role,selection_criterion=criterion,group=group,
                        group_Huber_sample_mean=score['group_huber'][g],
                        improvement_vs_G_global_same_group_percent=100*(reference-score['group_huber'][g])/reference if reference else None,
                        best_epoch=ck['epoch'],trainable_parameters=run['trainable_parameters'],
                        frozen_parameters=run['frozen_parameters'],expert_training_seconds=run['expert_training_seconds'])
                    for h,m in score['horizons'].items():
                        for metric,value in m.items():row[f'{metric}_{h}d']=value
                        row[f'Hit_percent_{h}d']=100*m['Hit']
                    capability.append(row)
    strategy_rows=[];summary={};packed={}
    for split in ('train','val'):
        selected={k:candidates[split][k] for k in ('G-global','G-short','G-long','S-short','L-long')}
        predictions=combine_strategies(arrays[split]['y_base'],selected)
        scores={k:metrics(v,arrays[split]['target'],config['delta']) for k,v in predictions.items()}
        summary[split]=scores
        for name,score in scores.items():
            for g,ids in enumerate(((0,1),(2,3))):
                reference=scores['G_global']['group_huber'][g]
                for i in ids:
                    h=str(HORIZONS[i]);m=score['horizons'][h]
                    strategy_rows.append(dict(split=split,strategy=name,group=('short','long')[g],horizon=HORIZONS[i],
                        group_Huber_sample_mean=score['group_huber'][g],**m,Hit_percent=100*m['Hit'],
                        improvement_vs_G_global_same_group_percent=100*(reference-score['group_huber'][g])/reference if reference else None,
                        delta_MAE_vs_D0B=m['MAE']-scores['D0B']['horizons'][h]['MAE'],
                        delta_MAE_vs_generic_group_control=m['MAE']-scores['generic_group_control']['horizons'][h]['MAE']))
        packed.update({f'{split}_{k}':v for k,v in arrays[split].items() if k!='u'})
        packed.update({f'{split}_prediction_{k}':v for k,v in predictions.items()})
    write_csv(out/'expert_capability.csv',capability);write_csv(out/'strategy_comparison.csv',strategy_rows)
    write_json(out/'metrics.json',summary);write_json(out/'checkpoint_index.json',checkpoint_index)
    np.savez_compressed(out/'strategy_predictions.npz',**packed)
    val=summary['val'];short=val['S_short']['group_huber'][0];long=val['L_long']['group_huber'][1]
    gs=runs['G']['checkpoints']['short']['metrics']['val']['group_huber'][0]
    gl=runs['G']['checkpoints']['long']['metrics']['val']['group_huber'][1]
    main5=val['supervised_task_assignment']['horizons']['5']['MAE'];base5=val['D0B']['horizons']['5']['MAE']
    supported=short<gs and long<gl
    lines=['# 冻结独立 D0B：监督专家 TRAIN/VAL 汇总','',
        f'任务监督专长证据：short S−匹配G Huber={short-gs:+.9g}；long L−匹配G Huber={long-gl:+.9g}。'+
        ('两组均改善，支持本协议下任务监督形成专长。' if supported else '没有两组同时改善的证据；不能仅凭训练权重认定专长。'),
        f'固定任务分配 VAL MAE@5d={main5:.9f}；D0B={base5:.9f}；Δ={main5-base5:+.9f}。'+
        ('本次固定基座适配优于 D0B，尚需独立确认。' if main5<base5 else '不支持替换 D0B 基线。'),'',
        '| VAL策略 | short Huber | long Huber | MAE@1d | MAE@5d | MAE@10d | MAE@20d |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,score in val.items():
        lines.append(f'| {name} | {score["group_huber"][0]:.9g} | {score["group_huber"][1]:.9g} | '+
            ' | '.join(f'{score["horizons"][str(h)]["MAE"]:.9f}' for h in HORIZONS)+' |')
    lines += ['', '## 选择与对照','',
        '三次独立训练共用一份冻结独立 D0B state 和一份显式复制的残差初值。各轨迹按 global 四周期 prediction-only VAL Huber batch-mean 做 scheduler/早停，同时跟踪 global/short-weighted/long-weighted 三个选择器。主 checkpoint 固定 G-global/S-short/L-long；匹配比较固定 G-short vs S-short、G-long vs L-long。没有 MAE 选优。',
        '完整专家×任务组能力矩阵及同组相对 G-global 改善率见 expert_capability.csv；RMSE、Hit 与四周期差值见 strategy_comparison.csv。Hit 存储为比例，展示百分数。',
        '若同一专家四周期均最好，应描述为更强的通用候选；不要求专家故意损害其他任务。', '',
        '## 边界','',
        '这是冻结、已独立训练 D0B 的预测读出适配，不是端到端训练，也不是与从头训练 D0B 相同预算的纯结构/参数量对照。不同 loss 选择器只在同一 global 早停轨迹内选 checkpoint。',
        '所有能力矩阵使用 float64 离线样本等权组内 Huber；历史选择器继续沿用 float32 batch-mean，二者分别命名。收益率标签为小数尺度，商品及周期顺序不变。',
        'VAL 是此前已使用的探索性区间；不作显著性、经济因果或未来确认性声明。没有 router、Oracle、TEST 读取或自动后续实验。']
    (out/'summary_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    check_unchanged(config['base_audit']['protected_files_sha256'])
    write_json(out/'audit.json',dict(status='COMPLETE',test_evaluated=False,original_base_unchanged=True,
        initialized_experts_match=True,all_nine_checkpoints_strict_loaded=True,
        capability_metrics_recomputed=True,source=source_record()))
    write_json(out/'file_index.json',{p.name:dict(path=str(p.resolve()),sha256=sha256(p)) for p in out.iterdir() if p.is_file()})
    print(f'SUMMARY_COMPLETE {out.resolve()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','prepare','train','summarize'),default='preflight')
    parser.add_argument('--expert',choices=ROLES)
    parser.add_argument('--study',type=Path,default=EXPERIMENT_ROOT/'seed42_supervised_v1')
    parser.add_argument('--output',type=Path,help='New preflight/summary output directory')
    parser.add_argument('--file-index',type=Path)
    parser.add_argument('--data-cache',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();check_protocol();torch.set_num_threads(args.threads)
    if args.mode=='preflight':preflight(args)
    elif args.mode=='prepare':prepare(args)
    elif args.mode=='train':
        if args.expert is None:parser.error('--mode train requires --expert G/S/L')
        train_role(args)
    else:summarize(args)


if __name__=='__main__':main()
