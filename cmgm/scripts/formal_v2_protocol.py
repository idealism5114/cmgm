"""Pre-registered grids, full-information arrays and VAL-only neural training."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from cmgm.config import MULTI_HORIZONS
from cmgm.models.formal_baselines_v2 import CLASSICAL, NEURAL, ORDER, input_view, restore_input
from cmgm.scripts.baseline_protocol import seed_all, prediction_loss, check_metric_identity, data_audit
from cmgm.training.metric_standard import population_metrics

SEEDS = (42,2025,3407)
LR_GRID = (1e-3,3e-4,1e-4)
PROTOCOL = dict(version='FORMAL BASELINE BENCHMARK V2',models=list(ORDER),seq_len=20,features=21,
    horizons=list(MULTI_HORIZONS),seeds=list(SEEDS),tuning_seed=42,batch_size=64,max_epochs=200,patience=10,
    optimizer='Adam',weight_decay=1e-5,loss='sum four Huber .02; D0B additionally retains switch KL',
    selection='pooled VAL5 MAE',early_stopping='pooled VAL5 MAE',scheduler='ReduceLROnPlateau, batch-mean VAL multi-horizon Huber, factor .5, patience5',
    train_shuffle=False,train_drop_last=False,information='lossless full T,N,F, no normalization outside original build_data',
    seed_std_ddof=1,stage_B_seed42='fresh fit, same fixed seed as Stage A; Ridge selected estimator reused',
    test_policy='historical D0B sanity before A; no TEST in A; all HP frozen and all B checkpoints fixed before formal TEST')


def grid(name):
    if name=='Ridge Regression':return [dict(alpha=a) for a in (.01,.1,1.,10.,100.)]
    if name=='Random Forest':
        return [dict(n_estimators=n,max_depth=d,max_features=f) for n,d,f in itertools.product((300,500),(None,20),('sqrt',.3))]
    if name=='XGBoost':return [dict(max_depth=d,learning_rate=lr) for d,lr in itertools.product((4,6),(.03,.1))]
    if name in NEURAL:return [dict(lr=lr) for lr in LR_GRID]
    raise ValueError(name)


def classical_model(name,hp,seed,jobs):
    if hp not in grid(name):raise ValueError('Unregistered hyperparameters')
    if name=='Ridge Regression':
        from sklearn.linear_model import Ridge
        return Ridge(**hp,fit_intercept=True,solver='auto')
    if name=='Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**hp,min_samples_split=2,min_samples_leaf=1,bootstrap=True,n_jobs=jobs,random_state=seed)
    if name=='XGBoost':
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        return MultiOutputRegressor(XGBRegressor(**hp,n_estimators=500,subsample=.8,colsample_bytree=.8,
            objective='reg:squarederror',reg_lambda=1.,reg_alpha=0.,tree_method='hist',n_jobs=jobs,random_state=seed),n_jobs=1)
    raise ValueError(name)


def loader(dataset,seed):
    # V2 uses every TRAIN origin for every method, including the final partial batch.
    return DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(seed))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_audit(data):
    base=data_audit(data);x=data['loaders']['train'].dataset[0][0].unsqueeze(0)
    base['mapping']=[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in base['mapping']]
    rows={}
    for name in ORDER:
        v=input_view(name,x);restored=restore_input(name,v,x.shape)
        rows[name]=dict(expected=x.numel(),observed=v.numel(),missing=max(x.numel()-v.numel(),0),extra=max(v.numel()-x.numel(),0),
            reconstruction_error=float((restored-x).abs().max()),input_shape=list(v.shape),PASS=torch.equal(restored,x) and v.numel()==x.numel())
    offset=0;origins={};input_ranges={}
    for split,l in data['loaders'].items():
        ds=l.dataset;origins[split]=dict(count=len(ds),raw_rows=len(ds.raw_prices),split_global_rows=[offset,offset+len(ds.raw_prices)-1],
            first_forecast_origin_global=offset+19,last_forecast_origin_global=offset+len(ds)-1+19,
            first_observed_rows=[offset,offset+19],last_target_row=offset+len(ds)-1+19+max(MULTI_HORIZONS),
            sampled_windows=[dict(sample=i,observed_start=offset+i,observed_end=offset+i+19,target_rows=[offset+i+19+h for h in MULTI_HORIZONS]) for i in sorted(set((0,len(ds)//2,len(ds)-1)))])
        offset+=len(ds.raw_prices)
        if split!='test':
            values=np.asarray(ds.feature_matrix)
            input_ranges[split]=dict(min=float(values.min()),max=float(values.max()),mean=float(values.mean(dtype=np.float64)),
                std=float(values.std(dtype=np.float64)),finite=bool(np.isfinite(values).all()))
    return dict(models=rows,commodity_mapping=base['mapping'],data=base,origins=origins,
        input_ranges=input_ranges,
        dates='Original builder does not expose calendar dates; exact split/global row boundaries retained.',
        PASS=base['PASS'] and all(v['PASS'] for v in rows.values()))


def arrays(dataset,directory,split):
    """Full lossless float32 memmaps. No feature selection or learned transform."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    x0,y0=dataset[0];n=len(dataset);paths=[directory/f'{split}_{s}.npy' for s in ('x','y')]
    shapes=[(n,x0.numel()),(n,y0.numel())]
    if not all(p.exists() for p in paths):
        for p,shape,which in zip(paths,shapes,(0,1)):
            temp=p.with_suffix('.tmp.npy');a=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=shape)
            for i in range(n):a[i]=dataset[i][which].numpy().reshape(-1)
            a.flush();del a;os.replace(temp,p)
    result=tuple(np.load(p,mmap_mode='r') for p in paths)
    if any(a.shape!=shape or a.dtype!=np.float32 for a,shape in zip(result,shapes)):raise ValueError('Wrong cached input layout')
    for i in sorted(set((0,n//2,n-1))):
        if not all(np.array_equal(a[i],dataset[i][j].numpy().reshape(-1)) for j,a in enumerate(result)):raise ValueError('Input cache differs from model-ready dataset')
    return result


def metrics(p,y):
    if p.shape!=y.shape or p.ndim!=3 or p.shape[1:]!=(4,24):raise ValueError('Expected aligned B,4,24 prediction and target')
    result={str(h):population_metrics(p[:,MULTI_HORIZONS.index(h)],y[:,MULTI_HORIZONS.index(h)]) for h in MULTI_HORIZONS}
    for v in result.values():check_metric_identity(v)
    return result


def distribution(p,y):
    if not np.isfinite(p).all() or not np.isfinite(y).all():raise FloatingPointError('Nonfinite prediction/target')
    return dict(pred_mean=float(np.mean(p)),pred_std=float(np.std(p)),pred_min=float(np.min(p)),pred_max=float(np.max(p)),
        pred_P1=float(np.quantile(p,.01)),pred_P99=float(np.quantile(p,.99)),finite_fraction=float(np.isfinite(p).mean()),
        target_mean=float(np.mean(y)),target_std=float(np.std(y)))


@torch.no_grad()
def neural_predictions(m,source,device):
    m.eval();ps=[];ys=[];losses=[]
    for batch in source:
        x,y=batch[0].to(device),batch[1].to(device);p=m(x)
        loss=prediction_loss(p,y)
        if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite neural evaluation')
        ps.append(p.cpu().numpy());ys.append(y.cpu().numpy());losses.append(float(loss))
    return np.concatenate(ps),np.concatenate(ys),float(np.mean(losses))


def classical_predictions(m,x):
    p=np.concatenate([np.asarray(m.predict(x[i:i+64]),dtype=np.float64) for i in range(0,len(x),64)])
    if p.shape!=(len(x),96) or not np.isfinite(p).all():raise ValueError('Invalid classical prediction shape/values')
    return p.reshape(-1,4,24)


def neural_train(m,train,val,device,lr,path,metadata,max_epochs=200,patience=10):
    """Receives no TEST loader. Epoch selection and stopping use pooled VAL5 MAE."""
    if lr not in LR_GRID:raise ValueError('LR outside registered grid')
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
    branch=getattr(m,'switching_latent_transformer',None)
    best=float('inf');stale=0;history=[];start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,max_epochs+1):
        beta=branch.set_epoch(epoch) if branch is not None else None
        m.train();sums=[]
        for batch in train:
            x,y=batch[0].to(device),batch[1].to(device);opt.zero_grad();p=m(x)
            pred=prediction_loss(p,y);switch=branch.switch_loss() if branch is not None else pred.new_zeros(())
            loss=pred+switch
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('Nonfinite training gradient')
            opt.step();sums.append([float(pred.detach()),float(switch.detach()),float(loss.detach())])
        vp,vy,vhub=neural_predictions(m,val,device);vm=metrics(vp,vy)['5'];sched.step(vhub)
        row=dict(epoch=epoch,prediction_loss=float(np.mean(sums,axis=0)[0]),switch_loss=float(np.mean(sums,axis=0)[1]),
            total_loss=float(np.mean(sums,axis=0)[2]),val_multi_huber=vhub,val5_MAE=vm['MAE'],val5_MSE=vm['MSE'],lr=opt.param_groups[0]['lr'],switch_beta=beta)
        history.append(row)
        if vm['MAE']<best:
            best=vm['MAE'];stale=0
            cp=dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},best_epoch=epoch,best_val5_MAE=best,metadata=metadata)
            temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
        else:stale+=1
        atomic_json(Path(str(path)+'.history.json'),history)
        print(f"[V2 {metadata['model']} {metadata['stage']} seed={metadata['seed']} lr={lr}] epoch={epoch} VAL5_MAE={vm['MAE']:.9g} best={best:.9g}",flush=True)
        if stale>=patience:break
    if device.type=='cuda':torch.cuda.synchronize(device)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    summary=dict(best_epoch=cp['best_epoch'],val5_MAE=best,train_seconds=time.perf_counter()-start,epochs_completed=len(history),
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
        complexity=sum(p.numel() for p in m.parameters()),complexity_unit='trainable parameters')
    cp.update(training_complete=True,summary=summary,history=history)
    temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
    m.load_state_dict(cp['state_dict'])
    if branch is not None:branch.set_epoch(cp['best_epoch'])
    return summary
