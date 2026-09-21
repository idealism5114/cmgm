"""Post-fit descriptive diagnostics. Never imported by model/training code."""
import numpy as np
import torch
from cmgm.config import MULTI_HORIZONS
from cmgm.scripts.baseline_protocol import loaders
from cmgm.scripts.formal_v2_protocol import metrics


def distribution(a):
    a=np.asarray(a,dtype=np.float64)
    return dict(mean=a.mean(0).tolist(),std=a.std(0).tolist(),median=np.median(a,axis=0).tolist(),
                quantiles={str(q):np.quantile(a,q,axis=0).tolist() for q in (.1,.25,.5,.75,.9,.99)},
                min=a.min(0).tolist(),max=a.max(0).tolist())


def routing(p):
    p=np.asarray(p,dtype=np.float64);d=distribution(p)
    d.update(routing_entropy=float(-(p*np.log(p+1e-8)).sum(-1).mean()),
             hard_occupancy=(np.bincount(p.argmax(-1),minlength=2)/len(p)).tolist(),samples=len(p),
             pi_ST_std=float(p[:,1].std()),pi_ST_P90_minus_P10=float(np.quantile(p[:,1],.9)-np.quantile(p[:,1],.1)),
             pi_ST_range=float(np.ptp(p[:,1])),uniform_entropy=float(np.log(2)))
    return d


def expert_utility(pi,temporal,interaction,formal,target):
    """TEST labels are accepted only by this post-hoc analysis function."""
    from scipy.stats import spearmanr,pearsonr
    idx5=MULTI_HORIZONS.index(5)
    temporal,interaction,formal,target=[np.asarray(v,dtype=np.float64) for v in (temporal,interaction,formal,target)]
    if any(v.shape!=target.shape for v in (temporal,interaction,formal)) or target.shape[1:]!=(4,24):
        raise ValueError('Expert utility requires aligned (origins,4,24) arrays')
    lt=np.abs(temporal[:,idx5]-target[:,idx5]).mean(-1)
    lst=np.abs(interaction[:,idx5]-target[:,idx5]).mean(-1)
    lm=np.abs(formal[:,idx5]-target[:,idx5]).mean(-1)
    advantage=lt-lst;weights=np.asarray(pi,dtype=np.float64)[:,1]
    if not all(np.isfinite(v).all() for v in (lt,lst,lm,weights)):raise ValueError('Nonfinite ex-post utility')
    constant=np.ptp(weights)==0 or np.ptp(advantage)==0
    correlations=dict(Spearman=None if constant else float(spearmanr(weights,advantage).statistic),
                      Pearson=None if constant else float(pearsonr(weights,advantage).statistic),
                      undefined_reason='Constant routing or advantage' if constant else None)
    # Deterministic rank thirds. Equal-weight ties retain origin order, so tied
    # groups cannot themselves establish sample-dependent routing.
    groups=[]
    for label,indices in zip(('Low','Middle','High'),np.array_split(np.argsort(weights,kind='stable'),3)):
        if not len(indices):continue
        groups.append(dict(group=label,count=len(indices),mean_pi_ST=float(weights[indices].mean()),
            temporal_error=float(lt[indices].mean()),interaction_error=float(lst[indices].mean()),
            mean_A_ST=float(advantage[indices].mean()),fraction_A_ST_positive=float((advantage[indices]>0).mean()),
            moe_error=float(lm[indices].mean())))
    choose_t=lt<lst
    oracle=np.where(choose_t[:,None,None],temporal,interaction)
    oracle5=metrics(oracle,target)['5']
    references=dict(Temporal=float(lt.mean()),Interaction=float(lst.mean()),LearnedMoE=float(lm.mean()))
    return dict(correlations=correlations,quantile_groups=groups,
                grouping='Stable ascending pi_ST rank thirds; ties retain origin order; no model selection.',
                unique_pi_ST=int(len(np.unique(weights))),expert_TEST5_MAE=references,
                per_origin=dict(pi_ST=weights.tolist(),ell_T=lt.tolist(),ell_ST=lst.tolist(),A_ST=advantage.tolist(),ell_MoE=lm.tolist()),
                analysis_only=True),dict(label='EX-POST ORACLE DIAGNOSTIC — NOT A MODEL — NOT DEPLOYABLE — USES TEST TARGETS',
                primary_horizon=5,metrics=oracle5,fraction_temporal=float(choose_t.mean()),
                reference_MAE=references,oracle_minus_reference_MAE={k:oracle5['MAE']-v for k,v in references.items()},
                uses_TEST_labels=True,diagnostic_upper_bound_only=True,deployable=False)


@torch.no_grad()
def formal_evaluation(model,data,device,out):
    """Finish and persist all formal predictions before any target-aware analysis."""
    model.eval();result={};route={};saved={};test_experts=None;test_fused=None
    for split,loader in loaders(data,full=True).items():
        predictions=[];targets=[];probabilities=[];experts=[];fused=[]
        for batch in loader:
            pred=model(batch[0].to(device));target=batch[1]
            if pred.shape!=target.shape or pred.shape[1:]!=(4,24) or not torch.isfinite(pred).all():
                raise ValueError('Invalid formal predictions')
            predictions.append(pred.cpu().numpy());targets.append(target.numpy())
            c=model.candidate_moe_fusion.last;probabilities.append(c['pi'].cpu().numpy())
            if split=='test':experts.append(c['experts'].cpu());fused.append(c['h_moe'].cpu())
        p=np.concatenate(predictions);y=np.concatenate(targets);pi=np.concatenate(probabilities)
        result[split]=metrics(p,y);route[split]=routing(pi)
        saved[split+'_prediction']=p;saved[split+'_target']=y;saved[split+'_pi']=pi
        if split=='test':test_experts=torch.cat(experts);test_fused=torch.cat(fused)
    np.savez_compressed(out/'formal_predictions.npz',**saved)
    # Only now generate diagnostic expert predictions through the frozen shared head.
    per_expert=[]
    for k in range(2):
        per_expert.append(torch.cat([model.head(v.to(device)).reshape(-1,4,24).cpu() for v in test_experts[:,k].split(64)]).numpy())
    utility,oracle=expert_utility(saved['test_pi'],*per_expert,saved['test_prediction'],saved['test_target'])
    e=test_experts.double();f=test_fused.double()
    expert=dict(mean_L2=e.norm(dim=-1).mean(0).tolist(),h_moe_mean_L2=float(f.norm(dim=-1).mean()),
                mean_absolute_disagreement=float((e[:,0]-e[:,1]).abs().mean()),
                cosine_similarity=distribution(torch.nn.functional.cosine_similarity(e[:,0],e[:,1],dim=-1).numpy()))
    np.savez_compressed(out/'ex_post_expert_predictions.npz',temporal=per_expert[0],interaction=per_expert[1])
    return dict(metrics=result,routing=route,experts=expert,utility=utility,oracle=oracle,
                diagnostic_order='Frozen best checkpoint → persisted formal predictions → ex-post expert utility and oracle. No update or selection.')
