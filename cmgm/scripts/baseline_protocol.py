"""Independent fixed-budget baseline training and non-learning sanity checks."""
import copy
import hashlib
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from cmgm.models.comparison_baselines import HORIZONS
from cmgm.training.metric_standard import population_metrics


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def loaders(data,full=False):
    # Match current main_ablation.build_data: chronological, shuffle=False.
    return {s:DataLoader(l.dataset,batch_size=64,shuffle=False,drop_last=s=='train' and not full,
                        generator=torch.Generator().manual_seed(42),num_workers=0) for s,l in data['loaders'].items()}


def parameter_counts(m):
    return dict(trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),nontrainable=sum(p.numel() for p in m.parameters() if not p.requires_grad),
        buffer_elements=sum(b.numel() for b in m.buffers()))


def prediction_loss(p,y):
    if p.shape!=y.shape or p.ndim!=3 or p.shape[1:]!=(4,24):raise ValueError('Expected prediction/target (B,4,24)')
    return sum(nn.functional.huber_loss(p[:,HORIZONS.index(h)],y[:,HORIZONS.index(h)],delta=.02) for h in HORIZONS)


@torch.no_grad()
def validation(m,loader,device):
    m.eval();losses=[];ae=se=0.;count=0;idx=HORIZONS.index(5)
    for batch in loader:
        x,y=batch[0].to(device),batch[1].to(device);p=m(x)
        loss=prediction_loss(p,y)
        if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite validation')
        losses.append(float(loss));e=p[:,idx].double()-y[:,idx].double();ae+=float(e.abs().sum());se+=float(e.square().sum());count+=e.numel()
    if not count:raise ValueError('Empty validation loader')
    return float(np.mean(losses)),dict(MAE=ae/count,MSE=se/count)


def train_one(m,train_loader,val_loader,device,path,metadata,max_epochs=200,patience=10,on_epoch=None):
    """Same Adam/Huber/scheduler/batch-mean monitor, no auxiliary switching loss."""
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=patience//2)
    history=[];best=float('inf');best_epoch=None;stale=0;best5=float('inf');best5epoch=None;start=time.perf_counter()
    for epoch in range(1,max_epochs+1):
        if device.type=='cuda':torch.cuda.synchronize(device)
        epoch_start=time.perf_counter();m.train();losses=[]
        for batch in train_loader:
            x,y=batch[0].to(device),batch[1].to(device);optimizer.zero_grad();p=m(x);loss=prediction_loss(p,y)
            if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite training prediction/loss; run INVALID')
            loss.backward()
            # Numerical guard only, not a reported gradient diagnostic or clipping.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('Nonfinite gradient; run INVALID')
            optimizer.step();losses.append(float(loss.detach()))
        if not losses:raise ValueError('No full training batches at fixed batch=64')
        val,secondary=validation(m,val_loader,device);scheduler.step(val)
        if secondary['MAE']<best5:best5=secondary['MAE'];best5epoch=epoch
        if device.type=='cuda':torch.cuda.synchronize(device)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_objective=val,val5_MAE=secondary['MAE'],val5_MSE=secondary['MSE'],
            lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-epoch_start)
        history.append(row)
        if val<best:
            best=val;best_epoch=epoch;stale=0
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            torch.save(dict(model_state_dict=state,metadata=metadata,best_epoch=epoch,best_val_loss=best),path)
        else:stale+=1
        if on_epoch:on_epoch(history)
        print(f"[{metadata['model']}] epoch={epoch} train={row['train_loss']:.9g} val={val:.9g} secondary_VAL5_MAE={secondary['MAE']:.9g} best={best_epoch}",flush=True)
        if stale>=patience:break
    elapsed=time.perf_counter()-start
    cp=torch.load(path,map_location=device,weights_only=False);m.load_state_dict(cp['model_state_dict'],strict=True)
    summary=dict(best_epoch=best_epoch,best_val_objective=best,train_seconds=elapsed,seconds_per_epoch_mean=float(np.mean([v['seconds'] for v in history])),
        epochs_completed=len(history),secondary_best_val5_epoch=best5epoch,secondary_best_val5_mae=best5,
        val5_at_formal_best=history[best_epoch-1]['val5_MAE'])
    cp.update(history=history,training_summary=summary,training_complete=True);torch.save(cp,path)
    return summary,history


@torch.no_grad()
def evaluate(m,all_loaders,device,names):
    m.eval();metrics={};commodity=[]
    for split,loader in all_loaders.items():
        ps=[];ys=[]
        for batch in loader:
            p=m(batch[0].to(device));y=batch[1]
            if p.shape!=y.shape or p.shape[1:]!=(4,24):raise ValueError('Output/target order or shape failure')
            ps.append(p.cpu().numpy());ys.append(y.numpy())
        p,y=np.concatenate(ps),np.concatenate(ys)
        metrics[split]={str(h):population_metrics(p[:,HORIZONS.index(h)],y[:,HORIZONS.index(h)]) for h in HORIZONS}
        for values in metrics[split].values():
            check_metric_identity(values)
        if split=='test':
            idx=HORIZONS.index(5)
            commodity=[dict(commodity=str(name),**population_metrics(p[:,idx,i],y[:,idx,i])) for i,name in enumerate(names)]
    return dict(metrics=metrics,per_commodity=commodity)


def check_metric_identity(values):
    """Check the float64 sqrt/square round trip relative to the MSE scale."""
    if not all(np.isfinite(v) for v in values.values()):
        raise FloatingPointError('Nonfinite evaluation metric')
    mse,rmse=values['MSE'],values['RMSE']
    if mse<0 or rmse<0:
        raise AssertionError('Negative MSE/RMSE')
    squared=np.float64(rmse)*np.float64(rmse)
    # A fixed absolute threshold falsely rejects large, correctly computed MSE.
    if not np.isclose(squared,mse,rtol=8*np.finfo(np.float64).eps,atol=0.):
        raise AssertionError(f'RMSE/MSE inconsistency: MSE={mse!r}, RMSE={rmse!r}, squared_error={abs(squared-mse)!r}')


def _sanity(m,x):
    with torch.no_grad():
        m.eval();p=m(x);perm=torch.arange(len(x)-1,-1,-1,device=x.device)
        batch=float((m(x[perm])-p[perm]).abs().max());single=float((m(x[:1])-p[:1]).abs().max())
        changed=x.clone();changed[:,10:]=changed[:,10:]*-3+7
        causal={}
        if hasattr(m,'temporal_states'):
            a,b=m.temporal_states(x),m.temporal_states(changed)
            causal['past_state_max_diff']=float((a[:,:10]-b[:,:10]).abs().max())
        future=torch.cat([x,x[:,:5]+100],1);future[:,20:]*=-7
        causal['observed_window_interface_max_diff']=float((m(future[:,:20])-p).abs().max())
        r=dict(OutputShape=list(p.shape),Finite=bool(torch.isfinite(p).all()),BatchPerm=batch,SingleSample=single,Causal=causal,
            output_scale=float(p.abs().max()),causality_scope='GRU/temporal Transformer/MTGNN prefix states are checked. Linear and inverted tokens use exactly the 20 observed points; no temporal-prefix invariance is claimed for inverted full-window tokens.')
        if hasattr(m,'graph'):r['graph_A_finite']=bool(torch.isfinite(m.graph()).all())
        r['PASS']=r['Finite'] and p.shape==(len(x),4,24) and batch<1e-6 and single<1e-6 and max(causal.values())<1e-6 and r.get('graph_A_finite',True)
        return r


def sanity(m,x):
    # Eval-only, no backward or parameter update; preserve model mode and RNG.
    state=torch.get_rng_state();mode=m.training
    try:
        r=_sanity(m,x)
        if not r['PASS'] and r['Finite']:
            copy_model=copy.deepcopy(m).double();audit=_sanity(copy_model,x.double())
            r['float64_roundoff_audit']=audit
            bound=32*torch.finfo(x.dtype).eps*max(r['output_scale'],1.)
            r['fp32_roundoff_bound']=bound
            r['PASS']=audit['PASS'] and max(r['BatchPerm'],r['SingleSample'],*r['Causal'].values())<=bound
            r['precision_note']='Original FP32 errors retained; valid only if independent double audit passes AND raw errors fit a scale-aware 32-epsilon bound. No training precision change.'
        return r
    finally:m.train(mode);torch.set_rng_state(state)


def data_audit(data):
    cs,ce=data['market_indices']['commodity'];names=data['feature_names'];assert ce-cs==24 and ce==data['n_nodes']
    mapping=[dict(commodity=str(names[cs+i]),full_node=cs+i,target_output=i,neutral_node=4+i,variate_indices=list(range((4+i)*21,(5+i)*21))) for i in range(24)]
    fingerprint={};errors={}
    for split,loader in data['loaders'].items():
        ds=loader.dataset;assert ds.seq_len==20 and tuple(ds.horizons)==HORIZONS and (ds.commodity_start,ds.commodity_end)==(cs,ce)
        assert ds.target_type=='return'
        digest=hashlib.sha256()
        for array in (ds.feature_matrix,ds.raw_prices):
            digest.update(np.ascontiguousarray(array).tobytes())
        fingerprint[split]=dict(sha256=digest.hexdigest(),origins=len(ds),timeline=len(ds.raw_prices))
        error=0.
        for index in range(len(ds)):
            end=index+19
            y=np.stack([np.clip(ds.raw_prices[end+h,cs:ce]/np.maximum(np.abs(ds.raw_prices[end,cs:ce]),1e-8)-1,-1,1).astype(np.float32) for h in HORIZONS])
            error=max(error,float(np.max(np.abs(y-ds[index][1].numpy()))))
        errors[split]=error
    return dict(mapping=mapping,split_fingerprint=fingerprint,target_order_max_error=errors,PASS=all(v==0 for v in errors.values()))
