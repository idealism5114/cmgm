"""Pooled metrics and bounded residual controls; evaluation never selects a model."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from cmgm.config import MULTI_HORIZONS
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0e_diagnostics import diagnostic_context
from cmgm.scripts.d0b_pregnn_diagnostics import local_controls,fixed_permutation


def amplitude_stats(values):
    x=np.asarray(values,dtype=float);a=np.abs(x)
    return dict(mean_abs=float(a.mean()),median_abs=float(np.median(a)),P90_abs=float(np.quantile(a,.9)),
        P95_abs=float(np.quantile(a,.95)),max_abs=float(a.max()),signed_mean=float(x.mean()),std=float(x.std()))


@torch.no_grad()
def collect_with_controls(model,loaders,device):
    output={};order=fixed_permutation(model.n_commodities,device)
    with diagnostic_context(model):
        for split,source in loaders.items():
            rows={k:[] for k in ('target','native','base','residual','shuffle','shuffle_residual','mean','mean_residual')};errors={k:0. for k in ('zero','shuffle','mean')}
            for batch in DataLoader(source.dataset,batch_size=64,shuffle=False,drop_last=False):
                prediction=model(batch[0].to(device));pack=local_controls(model,order)
                assert torch.equal(prediction,pack['native']['prediction'])
                values=dict(target=batch[1],native=prediction,base=pack['zero']['prediction'],residual=pack['native']['residual'],
                    shuffle=pack['shuffle']['prediction'],shuffle_residual=pack['shuffle']['residual'],mean=pack['mean']['prediction'],mean_residual=pack['mean']['residual'])
                for k,v in values.items():rows[k].append(v.cpu().numpy())
                for k in errors:errors[k]=max(errors[k],float((pack[k]['base_pred']-pack['native']['base_pred']).abs().max()))
            output[split]={k:np.concatenate(v) for k,v in rows.items()}
            output[split]['control_base_max_diff']=errors
    return output,order.cpu().tolist()


def primary_arrays(arrays):
    idx=MULTI_HORIZONS.index(5)
    return {s:dict(target=v['target'][:,idx],prediction=v['native'][:,idx]) for s,v in arrays.items()}


def residual_analysis(arrays,names):
    idx=MULTI_HORIZONS.index(5);out={}
    for split,v in arrays.items():
        r=v['residual'];base=v['base'];pred=v['native'];r5=r[:,idx];b5=base[:,idx];p5=pred[:,idx]
        out[split]=dict(primary_5d=amplitude_stats(r5),all_horizons_pooled=amplitude_stats(r),
            mean_abs_residual_over_base_5d=float(np.abs(r5).mean()/(np.abs(b5).mean()+1e-12)),
            per_horizon={str(h):amplitude_stats(r[:,i]) for i,h in enumerate(MULTI_HORIZONS)},
            direction_5d=dict(same_sign_as_base=float(np.mean(np.sign(r5)==np.sign(b5))),
                opposite_sign_to_base=float(np.mean(np.sign(r5)*np.sign(b5)<0)),
                increases_prediction_magnitude=float(np.mean(np.abs(p5)>np.abs(b5))),
                decreases_prediction_magnitude=float(np.mean(np.abs(p5)<np.abs(b5))),
                note='Equality/zero cases are retained; same/opposite or increase/decrease need not sum to one.'),
            per_commodity_5d=[dict(commodity=str(name),**amplitude_stats(r5[:,i])) for i,name in enumerate(names)])
    return out


def ablation_analysis(base,arrays):
    idx=MULTI_HORIZONS.index(5);out={}
    for split,v in arrays.items():
        y=v['target'][:,idx];native=v['native'][:,idx]
        np.testing.assert_array_equal(base[split]['target'],y)
        m={'OriginalD0B':population_metrics(base[split]['prediction'],y)};impacts={}
        for label,key in [('Full','native'),('ZeroResidual','base'),('ShuffledLocal','shuffle'),('MeanLocal','mean')]:
            pred=v[key][:,idx];m[label]=population_metrics(pred,y);difference=np.abs(pred-native)
            rkey={'native':'residual','base':None,'shuffle':'shuffle_residual','mean':'mean_residual'}[key]
            residual=np.zeros_like(v['residual']) if rkey is None else v[rkey]
            impacts[label]=dict(prediction_mean_diff=float(difference.mean()),prediction_max_diff=float(difference.max()),
                residual_mean_diff=float(np.abs(residual[:,idx]-v['residual'][:,idx]).mean()),
                residual_max_diff=float(np.abs(residual[:,idx]-v['residual'][:,idx]).max()),
                mean_abs_control_residual=float(np.abs(residual[:,idx]).mean()),
                base_pred_max_diff=0. if label=='Full' else v['control_base_max_diff'][{'base':'zero','shuffle':'shuffle','mean':'mean'}[key]])
        out[split]=dict(metrics=m,impacts=impacts,
            direct_effect={k:m['Full'][k]-m['ZeroResidual'][k] for k in ('MAE','MSE','RMSE','Hit')},
            representation_training_effect={k:m['ZeroResidual'][k]-m['OriginalD0B'][k] for k in ('MAE','MSE','RMSE','Hit')},
            definition='Negative error delta indicates improvement. Full minus ZeroResidual is direct output effect; ZeroResidual minus original D0B includes independently trained shared representation changes. Local controls reuse identical native base outputs.')
    return out
