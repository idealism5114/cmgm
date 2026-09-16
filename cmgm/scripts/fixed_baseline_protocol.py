"""One immutable configuration per model; no grid and no D0B training."""
import contextlib
import gc
import json
import os
from pathlib import Path
import resource
import threading
import time
import joblib
import numpy as np
import torch
from cmgm.models.formal_baselines_v2 import ORDER,CLASSICAL,NEURAL,KEYS
from cmgm.scripts.formal_v2_protocol import atomic_json,sha

TRAINABLE_NEURAL=tuple(n for n in NEURAL if n!='D0B')
RUN_ORDER=('Ridge Regression','LSTM','TCN','Vanilla Transformer','Random Forest','XGBoost','Graph WaveNet','MTGNN')
CLASS_CONFIG={
    'Ridge Regression':dict(alpha=1.0,fit_intercept=True,solver='lsqr'),
    'Random Forest':dict(n_estimators=300,max_depth=20,max_features='sqrt',min_samples_split=2,min_samples_leaf=1,bootstrap=True,random_state=42),
    'XGBoost':dict(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='reg:squarederror',reg_lambda=1.,reg_alpha=0.,tree_method='hist',random_state=42),
}
DEEP_CONFIG=dict(optimizer='Adam',lr=1e-4,weight_decay=1e-5,batch_size=64,max_epochs=200,patience=10,
    loss='sum four Huber delta=.02',selection='mean of batch-mean multi-horizon VAL Huber',
    scheduler='ReduceLROnPlateau factor=.5 patience5, same VAL Huber',early_stopping='same VAL Huber',
    train_shuffle=False,train_drop_last=True,seed=42)
FIXED={**CLASS_CONFIG,**{n:dict(DEEP_CONFIG) for n in TRAINABLE_NEURAL},'D0B':dict(variant='switching_latent_balanced_readout',frozen=True,source='existing formal checkpoint')}
PROTOCOL=dict(name='FORMAL BASELINE BENCHMARK V2 — FIXED CONFIG SINGLE RUN',models=list(ORDER),seed=42,
    seq_len=20,features=21,horizons=[1,5,10,20],input='original full B,20,N,21; lossless layouts only',
    deep=DEEP_CONFIG,grid=False,multi_seed=False,D0B_retrain=False,
    statement='Single-run controlled comparison, seed=42. No statistical-significance or multi-seed robustness claim.')


def estimator(name,jobs):
    cfg=dict(CLASS_CONFIG[name])
    if name=='Ridge Regression':
        from sklearn.linear_model import Ridge
        return Ridge(**cfg)
    if name=='Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**cfg,n_jobs=jobs)
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    # One parallel layer only; preserve conservative memory usage of existing CPU policy.
    return MultiOutputRegressor(XGBRegressor(**cfg,n_jobs=jobs),n_jobs=1)


def estimator_matches(name,m):
    if name=='Ridge Regression':
        from sklearn.linear_model import Ridge
        if not isinstance(m,Ridge):return False
    elif name=='Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        if not isinstance(m,RandomForestRegressor):return False
    elif name=='XGBoost':
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        if not isinstance(m,MultiOutputRegressor) or not isinstance(m.estimator,XGBRegressor):return False
        return (all(m.estimator.get_params().get(k)==v for k,v in CLASS_CONFIG[name].items())
                and len(m.estimators_)==96 and all(all(e.get_params().get(k)==v for k,v in CLASS_CONFIG[name].items()) for e in m.estimators_))
    return all(m.get_params().get(k)==v for k,v in CLASS_CONFIG[name].items())


def reuse_plan(old,ranges,prov,root):
    """Never consult TEST metrics or rank candidates; inspect fixed config eligibility."""
    plan={};root=Path(root)
    compatible_data=old.get('input_audit',{}).get('data',{}).get('split_fingerprint')==ranges['data']['split_fingerprint']
    compatible_graph=old.get('provenance',{}).get('official')==prov['official']
    for name in RUN_ORDER:
        trials=[v for v in old.get('jobs',{}).values() if v['model']==name and v['status']=='DONE']
        row=dict(old_run_exists=bool(trials),architecture_valid=bool(trials) and compatible_graph,full_input_valid=compatible_data,
            seed42=any(t['seed']==42 for t in trials),fixed_config_matches=False,sanity_PASS=False,action='Retrain' if trials else 'Pending',reason='',source=None)
        if name in TRAINABLE_NEURAL:
            row['reason']='Old V2 selected/stopped by VAL5 MAE and used drop_last=False; fixed protocol requires multi-horizon Huber selection and historical D0B loader. Existing selected checkpoints cannot establish equivalence.'
        else:
            for trial in trials:
                if trial['seed']!=42 or not compatible_data or not compatible_graph:continue
                # Fast, safe rejection before deserializing large nonmatching models.
                if any(k in trial['hp'] and trial['hp'][k]!=v for k,v in CLASS_CONFIG[name].items()):continue
                path=Path(trial['path'])
                if sha(path)!=trial['sha256']:raise ValueError('Old completed artifact checksum changed')
                obj=joblib.load(path);m=obj['estimator'];md=obj['metadata']
                match=estimator_matches(name,m)
                shape=int(getattr(m,'n_features_in_',-1))==int(ranges['models'][name]['expected'])
                outputs=(len(m.estimators_) if name=='XGBoost' else getattr(m,'n_outputs_',np.asarray(getattr(m,'coef_',[])).shape[0] if hasattr(m,'coef_') else -1))==96
                check=bool(obj.get('fitted_sanity',{}).get('PASS')) and bool(trial.get('sanity',{}).get('PASS'))
                # Full-information source path is protected by the original per-job source hash.
                src=md.get('source_hashes',{}).get('cmgm/models/formal_baselines_v2.py')
                architecture=src==sha(root/'cmgm/models/formal_baselines_v2.py')
                if match and shape and outputs and check and architecture and obj.get('training_complete') and md['seed']==42:
                    row.update(fixed_config_matches=True,sanity_PASS=True,architecture_valid=True,action='Reuse',reason='Exact predefined configuration, full input, seed42 and completed sanity verified; no TEST selection.',
                        source=dict(path=str(path),sha256=trial['sha256'],summary=obj['summary'],sanity=obj['fitted_sanity'],metadata=md))
                    del m,obj;gc.collect();break
                del m,obj;gc.collect()
            if not row['reason']:row['reason']='No completed estimator matches exact fixed configuration (including solver/trees/LR).'
        plan[name]=row
    plan['D0B']=dict(old_run_exists=True,architecture_valid=True,full_input_valid=True,seed42='historical metadata may be unavailable',
        fixed_config_matches=True,sanity_PASS=False,action='Reuse',reason='Existing frozen formal D0B checkpoint, evaluator reproduction required.',source=None)
    return plan


@contextlib.contextmanager
def heartbeat(name,out):
    stop=threading.Event();start=time.monotonic()
    def worker():
        while not stop.wait(60):
            value=dict(model=name,elapsed_seconds=time.monotonic()-start,pid=os.getpid(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       note='Fit is active; no per-output completion percentage is available from standard MultiOutputRegressor.')
            atomic_json(Path(out)/'progress.json',value)
            print(f"[Fixed benchmark heartbeat] {name}: elapsed={value['elapsed_seconds']/60:.1f} min, PID={os.getpid()}",flush=True)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join()
