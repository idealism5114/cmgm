"""Static evaluation and read-only Dynamic checkpoint representation intervention."""
import numpy as np
import torch
from cmgm.models.candidate_moe_fusion import VARIANT as DYNAMIC
from cmgm.scripts.d0b_moe_audit import make_model
from cmgm.scripts.baseline_protocol import loaders
from cmgm.scripts.formal_v2_protocol import metrics,sha,atomic_json
from cmgm.scripts.d0b_candidate_moe_diagnostics import routing


@torch.no_grad()
def intervention(data,reference,device,out):
    path=reference['checkpoint'];digest=sha(path)
    if digest!=reference['checkpoint_sha256']:raise ValueError('Dynamic checkpoint hash mismatch')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    model=make_model(data,DYNAMIC).to(device);model.load_state_dict(cp['model_state_dict'],strict=True);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()};sources=loaders(data,full=True)
    train_pi=[]
    for batch in sources['train']:
        model(batch[0].to(device));train_pi.append(model.candidate_moe_fusion.last['pi'].double().cpu())
    train_pi=torch.cat(train_pi);mean=train_pi.mean(0)
    # Mean is frozen before reading TEST labels or predictions.
    weights=mean.to(device=device,dtype=next(model.parameters()).dtype)
    atomic_json(out/'dynamic_train_mean_pi.json',dict(mean_pi_double=mean.tolist(),weights_used=weights.cpu().tolist(),
        source='FULL TRAIN origins only, no labels',count=len(train_pi),checkpoint_sha256=digest))
    ps=[];fs=[];ys=[];pis=[]
    for batch in sources['test']:
        pred=model(batch[0].to(device));c=model.candidate_moe_fusion.last
        fused=(weights.view(1,2,1)*c['experts']).sum(1)
        fixed=model.head(fused).reshape_as(pred)
        if pred.shape!=batch[1].shape or not torch.isfinite(pred).all() or not torch.isfinite(fixed).all():raise ValueError('Invalid intervention output')
        ps.append(pred.cpu().numpy());fs.append(fixed.cpu().numpy());ys.append(batch[1].numpy());pis.append(c['pi'].double().cpu())
    p,f,y=np.concatenate(ps),np.concatenate(fs),np.concatenate(ys)
    dynamic,fixed=metrics(p,y),metrics(f,y)
    errors={h:{k:abs(v-reference['metrics']['test'][h][k]) for k,v in m.items()} for h,m in dynamic.items()}
    tolerances=dict(MAE=1e-8,MSE=1e-9,RMSE=1e-8,Hit=1e-8)
    if any(v>tolerances[k] for h in errors.values() for k,v in h.items()):raise ValueError('Frozen Dynamic metrics do not reproduce formal reference')
    if sha(path)!=digest or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise AssertionError('Dynamic checkpoint/state changed')
    np.savez_compressed(out/'same_checkpoint_predictions.npz',dynamic=p,fixed_mean=f,target=y)
    return dict(dynamic=dynamic,fixed_mean=fixed,weights=weights.cpu().tolist(),train_count=len(train_pi),
        fixed_minus_dynamic={h:{k:fixed[h][k]-v for k,v in m.items()} for h,m in dynamic.items()},
        delta_within=fixed['5']['MAE']-dynamic['5']['MAE'],
        relative_delta_percent=100*(fixed['5']['MAE']-dynamic['5']['MAE'])/fixed['5']['MAE'],
        prediction_mean_abs_change=float(np.abs(f.astype(np.float64)-p).mean()),prediction_max_abs_change=float(np.abs(f.astype(np.float64)-p).max()),
        routing=dict(train=routing(train_pi.numpy()),test=routing(torch.cat(pis).numpy())),
        reproduction_errors=errors,reproduction_tolerances=tolerances,checkpoint_sha256=digest,checkpoint_unchanged=True,
        intervention='Representation mixture before SAME frozen shared head; no prediction-level mixing, no fitting.')


@torch.no_grad()
def evaluate_static(model,data,device,out):
    model.eval();result={};saved={};weights=model.global_mixture_fusion.weight_diagnostics()
    for split,l in loaders(data,full=True).items():
        ps=[];ys=[]
        for batch in l:
            p=model(batch[0].to(device))
            if p.shape!=batch[1].shape or not torch.isfinite(p).all():raise ValueError('Invalid static prediction')
            ps.append(p.cpu().numpy());ys.append(batch[1].numpy())
        p,y=np.concatenate(ps),np.concatenate(ys);result[split]=metrics(p,y)
        saved[split+'_prediction']=p;saved[split+'_target']=y
        assert model.global_mixture_fusion.weight_diagnostics()==weights
    np.savez_compressed(out/'static_predictions.npz',**saved)
    return dict(metrics=result,best_global_weights=weights)
