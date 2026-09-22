"""Frozen-checkpoint evaluation, TRAIN-mean intervention and post-hoc utility."""
import numpy as np
import torch
from cmgm.config import MULTI_HORIZONS
from cmgm.models.utility_routed_moe import per_sample_huber
from cmgm.scripts.baseline_protocol import loaders
from cmgm.scripts.formal_v2_protocol import metrics,atomic_json
from cmgm.scripts.d0b_candidate_moe_diagnostics import distribution,routing


def correlations(x,y):
    from scipy.stats import spearmanr,pearsonr
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    constant=np.ptp(x)==0 or np.ptp(y)==0
    return dict(Spearman=None if constant else float(spearmanr(x,y).statistic),
                Pearson=None if constant else float(pearsonr(x,y).statistic),
                undefined_reason='Constant input' if constant else None)


def analyze_experts(pi,pred_t,pred_st,pred_final,target):
    """This function is analysis-only, after formal predictions are frozen."""
    pt,pst,p,y=[np.asarray(v,dtype=np.float64) for v in (pred_t,pred_st,pred_final,target)]
    lt=per_sample_huber(torch.from_numpy(pt),torch.from_numpy(y)).numpy()
    lst=per_sample_huber(torch.from_numpy(pst),torch.from_numpy(y)).numpy()
    lm=per_sample_huber(torch.from_numpy(p),torch.from_numpy(y)).numpy()
    advantage=lt-lst;weight=pi[:,1];idx5=MULTI_HORIZONS.index(5)
    mt=np.abs(pt[:,idx5]-y[:,idx5]).mean(-1);mst=np.abs(pst[:,idx5]-y[:,idx5]).mean(-1);mm=np.abs(p[:,idx5]-y[:,idx5]).mean(-1)
    rows=[]
    for label,ix in zip(('Low','Middle','High'),np.array_split(np.argsort(weight,kind='stable'),3)):
        if not len(ix):continue
        rows.append(dict(group=label,sample_count=len(ix),mean_pi_ST=float(weight[ix].mean()),
            mean_expert_T_error=float(lt[ix].mean()),mean_expert_ST_error=float(lst[ix].mean()),mean_A=float(advantage[ix].mean()),
            fraction_ST_better=float((advantage[ix]>0).mean()),final_MoE_error=float(lm[ix].mean()),
            Temporal_5d_MAE=float(mt[ix].mean()),Interaction_5d_MAE=float(mst[ix].mean()),MoE_5d_MAE=float(mm[ix].mean()),
            fraction_ST_better_5d=float((mst[ix]<mt[ix]).mean())))
    use_t=lt<lst;oracle_pred=np.where(use_t[:,None,None],pt,pst)
    expert_errors=dict(Temporal=float(lt.mean()),Interaction=float(lst.mean()),Dynamic=float(lm.mean()))
    oracle_error=float(np.minimum(lt,lst).mean())
    alignment=dict(primary_error='sum four-horizon Huber(.02), mean commodities per origin',
        primary_correlations=correlations(weight,advantage),five_day_MAE_correlations=correlations(weight,mt-mst),
        groups=rows,grouping='Stable pi_ST rank thirds; chronological tie order. Ties do not establish routing variation.',
        fraction_T_better=float((lt<lst).mean()),fraction_ST_better=float((lst<lt).mean()),fraction_ties=float((lt==lst).mean()),
        mean_errors=expert_errors,per_origin=dict(pi_ST=weight.tolist(),loss_T=lt.tolist(),loss_ST=lst.tolist(),A=advantage.tolist(),
            loss_MoE=lm.tolist(),MAE5_T=mt.tolist(),MAE5_ST=mst.tolist(),MAE5_MoE=mm.tolist()))
    oracle=dict(label='EX-POST ORACLE ONLY — USES TEST LABELS — NOT DEPLOYABLE — NOT A MODEL — NOT FOR MODEL SELECTION',
        selection='lower four-horizon per-origin Huber; equal losses select Interaction',uses_TEST_labels=True,deployable=False,
        metrics=metrics(oracle_pred,y),mean_four_horizon_huber=oracle_error,reference_mean_huber=expert_errors,
        huber_improvement_over_experts={k:v-oracle_error for k,v in expert_errors.items() if k!='Dynamic'},
        fraction_temporal=float(use_t.mean()),warning='Oracle minimizes per-origin four-horizon Huber; its 5d MAE is not a 5d-specific oracle bound.')
    return alignment,oracle


@torch.no_grad()
def formal_evaluation(model,data,device,out):
    model.eval();allmetrics={};routes={};saved={};train_mean=None;test_representations=None
    sources=loaders(data,full=True)
    for split in ('train','val','test'):
        batches={k:[] for k in ('prediction','target','pred_T','pred_ST','pi','regime_prob','regime_entropy')}
        representations=[]
        for batch in sources[split]:
            pred=model(batch[0].to(device));y=batch[1];c=model.utility_moe_fusion.last
            if pred.shape!=y.shape or pred.shape[1:]!=(4,24):raise ValueError('Invalid prediction shape')
            values=dict(prediction=pred,target=y,**{k:c[k] for k in ('pred_T','pred_ST','pi','regime_prob','regime_entropy')})
            if any(not bool(torch.isfinite(v).all()) for v in values.values()):raise FloatingPointError('Implementation failure: nonfinite utility evaluation')
            for key,v in values.items():batches[key].append(v.cpu().numpy())
            if split=='test':representations.append(torch.stack([c['e_T'],c['e_ST']],1).double().cpu())
        arrays={k:np.concatenate(v) for k,v in batches.items()}
        allmetrics[split]=metrics(arrays['prediction'],arrays['target']);routes[split]=routing(arrays['pi'])
        saved.update({split+'_'+k:v for k,v in arrays.items()})
        if split=='train':
            # No labels used to derive the intervention weights; include ALL TRAIN origins.
            exact=arrays['pi'].astype(np.float64).mean(0);train_mean=exact.astype(arrays['prediction'].dtype)
            atomic_json(out/'fixed_router_train_mean.json',dict(source='TRAIN only, frozen best checkpoint, all origins',
                count=len(arrays['pi']),mean_pi_double=exact.tolist(),mixture_weights_used=train_mean.tolist()))
        if split=='test':test_representations=torch.cat(representations)
    pt=saved['test_pred_T'];pst=saved['test_pred_ST'];p=saved['test_prediction'];y=saved['test_target']
    fixed=train_mean[0]*pt+train_mean[1]*pst
    saved['test_fixed_prediction']=fixed
    np.savez_compressed(out/'formal_predictions.npz',**saved)
    # All prediction/control outputs now fixed and persisted. Targets enter only analysis.
    expert_metrics=dict(Temporal=metrics(pt,y),Interaction=metrics(pst,y))
    fixed_metrics=metrics(fixed,y)
    fixed_control=dict(source='Full TRAIN mean pi, no VAL/TEST fitting or target selection',weights=train_mean.tolist(),
        metrics=fixed_metrics,dynamic_minus_fixed={h:{k:allmetrics['test'][h][k]-v for k,v in fixed_metrics[h].items()} for h in fixed_metrics},
        prediction_mean_abs_change=float(np.abs(p.astype(np.float64)-fixed).mean()))
    alignment,oracle=analyze_experts(saved['test_pi'],pt,pst,p,y)
    e=test_representations
    representation=dict(mean_L2=e.norm(dim=-1).mean(0).tolist(),mean_absolute_disagreement=float((e[:,0]-e[:,1]).abs().mean()),
        cosine_similarity=distribution(torch.nn.functional.cosine_similarity(e[:,0],e[:,1],dim=-1).numpy()),
        prediction_disagreement={str(h):float(np.abs(pt[:,MULTI_HORIZONS.index(h)].astype(np.float64)-pst[:,MULTI_HORIZONS.index(h)]).mean()) for h in MULTI_HORIZONS})
    regime=saved['test_regime_prob'].argmax(-1);pi=saved['test_pi'];groups=[]
    for k in range(3):
        mask=regime==k
        groups.append(dict(regime=k,count=int(mask.sum()),mean_pi=pi[mask].astype(np.float64).mean(0).tolist() if mask.any() else None))
    regime_diag=dict(groups=groups,entropy_correlation=correlations(pi[:,1],saved['test_regime_entropy'][:,0]),
        causal_attribution=False,source='final observed posterior, detached router context')
    return dict(metrics=allmetrics,routing=routes,expert_metrics=expert_metrics,expert_diagnostics=representation,
        fixed_router=fixed_control,alignment=alignment,regime=regime_diag,oracle=oracle,
        diagnostic_order='Frozen best → TRAIN-mean weights → persisted dynamic/expert/fixed predictions → TEST-label alignment/oracle; no training or selection.')
