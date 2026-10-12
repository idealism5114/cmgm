"""Costly opt-in graph-source faithfulness operations for frozen D0B."""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from cmgm.interpretability.d0b_reliability import (
    D0BTrace,LAYER_NAMES,attention_source_scores,matched_random_sources,
    source_intervention,target_gradient_scores)


@contextmanager
def replace_source_values_with_train_mean(model, selected, means):
    """Replace selected source V projections; attention Q/K and self residual stay."""
    handles=[];calls={name:0 for name in LAYER_NAMES}
    try:
        for name in LAYER_NAMES:
            module=getattr(model,name).v
            def hook(_mod,_inp,values,layer=name):
                hop=calls[layer];calls[layer]+=1
                if hop>=2:raise RuntimeError("Unexpected MixHop depth")
                output=values.clone()
                idx=torch.as_tensor(selected,dtype=torch.long,device=values.device)
                mean=torch.as_tensor(means[layer][hop],device=values.device,dtype=values.dtype)
                output[:,idx]=mean[idx]
                return output
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:h.remove()


def training_mean_values(model,dataset,device):
    """Mean V per source asset and propagation hop, fitted on TRAIN only."""
    total={name:[None,None] for name in LAYER_NAMES}
    count=0
    for batch in DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False):
        x=batch[0].to(device);captured={name:[] for name in LAYER_NAMES}
        handles=[]
        for name in LAYER_NAMES:
            handles.append(getattr(model,name).v.register_forward_hook(
                lambda _m,_i,value,key=name: captured[key].append(value.detach().sum(0).cpu())))
        try:
            with torch.inference_mode():model(x)
        finally:
            for h in handles:h.remove()
        for name in LAYER_NAMES:
            if len(captured[name])!=2:raise RuntimeError("Unexpected graph hop count")
            for hop in (0,1):
                total[name][hop]=(captured[name][hop] if total[name][hop] is None
                                  else total[name][hop]+captured[name][hop])
        count+=len(x)
    return {k:[_v.numpy()/count for _v in v] for k,v in total.items()}


def graph_study(model,data,raw,mi,split,out,device,case_indices=None,horizon=5):
    """Source ranking and whole-graph edge interventions, no D0B weight update."""
    from cmgm.scripts.d0b_reliability import no_overwrite,atomic_json
    ds=data["loaders"][split].dataset
    cases=(list(range(len(ds))) if case_indices is None else sorted(set(case_indices)))
    if any(i<0 or i>=len(ds) for i in cases):raise ValueError("Case outside split")
    if not cases:raise ValueError("No cases")
    if horizon not in (1,5,10,20):raise ValueError("Unsupported horizon")
    n=model.num_nodes;h5=(1,5,10,20).index(horizon)
    groups=[mi[k] for k in ("stock","bond","commodity")]
    with torch.inference_mode():
        adjacency=model.graph_learner().detach().cpu().numpy()
    static=adjacency.mean(0)
    train_ds=data["loaders"]["train"].dataset
    std=np.asarray(train_ds.feature_matrix,dtype=np.float64).std(axis=0).astype(np.float32)
    if std.shape!=(n,21):raise RuntimeError("TRAIN feature standard deviation shape mismatch")
    std=torch.as_tensor(std,device=device)
    means=training_mean_values(model,train_ds,device)
    rng=np.random.default_rng(42)
    rows=[];features=[]
    full_cases=set(case_indices or [])
    for index in cases:
        sample=ds[index];x=sample[0][None].to(device);target=sample[1][None].to(device)
        # Input gradients activate alpha nodes while ALL model parameters stay frozen.
        with D0BTrace(model,live_attention=True) as cap:
            original=cap.forward(x.clone().requires_grad_())
            baseline=original["prediction"].detach()
            alpha=list(cap.attentions)
            raw_rank=attention_source_scores(alpha)[0].detach().cpu().numpy()
            per_commodity=[target_gradient_scores(original["prediction"],alpha,h5,c)
                           [0].detach().cpu().numpy() for c in range(model.n_commodities)]
            if index in full_cases:
                path=out/f"attention_case_{split}_{horizon}d_{index}.npz";no_overwrite(path)
                np.savez_compressed(path,**{f"{layer}_call{k}":a.detach().cpu().numpy()
                    for k,(layer,a) in enumerate(alpha)})
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen D0B parameter unexpectedly has gradient")
        # Explanation stability: perturb only observed-window features at 1% TRAIN std.
        pert_rank=[];pred_shift=[];jac=[];rho=[]
        k_stable=max(1,int(math.ceil(.1*n)))
        top_ref=set(np.argsort(-raw_rank)[:k_stable])
        for rep in range(10):
            noise=torch.as_tensor(rng.standard_normal(tuple(x.shape)),device=device,dtype=x.dtype)
            xp=x+.01*std[None,None]*noise
            with torch.inference_mode(),D0BTrace(model) as cap2:
                p=cap2.forward(xp)
                score=torch.stack(p["source_summary"]).mean((0,2))[0].cpu().numpy()
            pert_rank.append(score)
            pred_shift.append(float((p["prediction"]-baseline).abs().mean()))
            top=set(np.argsort(-score)[:k_stable])
            jac.append(len(top&top_ref)/len(top|top_ref))
            corr=spearmanr(raw_rank,score).statistic
            rho.append(float(corr) if np.isfinite(corr) else 0.)
        probs=raw_rank/raw_rank.sum()
        concentration=1.-float(-(probs*np.log(np.clip(probs,1e-12,None))).sum()/math.log(n))
        features.append([concentration,1.-float(np.mean(jac))])
        for c in range(model.n_commodities):
            target_value=float(target[0,h5,c]);base_value=float(baseline[0,h5,c])
            methods={"attention":raw_rank,"gradient_attention":per_commodity[c],
                     "static_graph":static}
            for method,scores in methods.items():
                for q in (.05,.10,.20):
                    k=max(1,int(math.ceil(q*n)))
                    selected=np.argsort(-scores)[:k]
                    selections=[("ranked",selected)]+[(f"matched_random_{rep}",
                        matched_random_sources(selected,groups,adjacency,rng))
                        for rep in range(20)]
                    for chunk in (selections[:1],selections[1:]):
                        masks=np.zeros((len(chunk),n),dtype=bool)
                        for row,(_,source) in enumerate(chunk):masks[row,source]=True
                        xb=x.expand(len(chunk),-1,-1,-1)
                        for op in ("remove","retain"):
                            with torch.inference_mode(),source_intervention(
                                    model,masks,keep=op=="retain"):
                                predictions=model(xb)[:,h5,c].cpu().numpy()
                            for (control,_),value in zip(chunk,predictions):
                                value=float(value)
                                rows.append(dict(index=index,commodity=c,method=method,
                                    fraction=q,control=control,operator=op,
                                    prediction=value,baseline=base_value,
                                    target=target_value,
                                    abs_prediction_change=abs(value-base_value),
                                    delta_abs_error=abs(value-target_value)-abs(base_value-target_value),
                                    stability_top10_jaccard=float(np.mean(jac)),
                                    stability_spearman=float(np.mean(rho)),
                                    perturb_prediction_change=float(np.mean(pred_shift))))
                    # Operator sensitivity: TRAIN-mean V replacement.
                    with torch.inference_mode(),replace_source_values_with_train_mean(
                            model,selected,means):
                        value=float(model(x)[0,h5,c])
                    rows.append(dict(index=index,commodity=c,method=method,
                        fraction=q,control="ranked",operator="TRAIN_mean_V",
                        prediction=value,baseline=base_value,target=target_value,
                        abs_prediction_change=abs(value-base_value),
                        delta_abs_error=abs(value-target_value)-abs(base_value-target_value),
                        stability_top10_jaccard=float(np.mean(jac)),
                        stability_spearman=float(np.mean(rho)),
                        perturb_prediction_change=float(np.mean(pred_shift))))
    # One feature row per origin. A is derived without future labels.
    a_path=out/f"graph_features_{split}.npz" if horizon==5 else out/f"graph_features_{split}_{horizon}d.npz"
    no_overwrite(a_path)
    np.savez_compressed(a_path,index=np.asarray(cases),A=np.asarray(features))
    import pandas as pd
    path=out/f"graph_faithfulness_{split}_{horizon}d.csv";no_overwrite(path)
    df=pd.DataFrame(rows)
    df.to_csv(path,index=False)
    summary=(df.groupby(["method","fraction","control","operator"],as_index=False)
             [["abs_prediction_change","delta_abs_error",
               "stability_top10_jaccard","stability_spearman",
               "perturb_prediction_change"]].mean())
    summary_path=out/f"graph_summary_{split}_{horizon}d.csv";no_overwrite(summary_path)
    summary.to_csv(summary_path,index=False)
    compare_rows=[]
    for (method,fraction,op),subset in df.groupby(["method","fraction","operator"]):
        ranked=subset[subset.control=="ranked"]
        random=subset[subset.control.str.startswith("matched_random_")]
        compare_rows.append(dict(method=method,fraction=fraction,operator=op,
            ranked_mean_prediction_change=float(ranked.abs_prediction_change.mean()),
            random_mean_prediction_change=float(random.abs_prediction_change.mean()) if len(random) else None,
            incremental_change_over_random=(float(ranked.abs_prediction_change.mean()-
                random.abs_prediction_change.mean()) if len(random) else None),
            ranked_delta_MAE=float(ranked.delta_abs_error.mean()),
            random_delta_MAE=float(random.delta_abs_error.mean()) if len(random) else None))
    comparison_path=out/f"graph_comparison_{split}_{horizon}d.csv";no_overwrite(comparison_path)
    pd.DataFrame(compare_rows).to_csv(comparison_path,index=False)
    report=dict(status="COMPLETE",split=split,cases=len(cases),commodities=model.n_commodities,
        evaluated_horizon=horizon,main_horizon=5,market_matched_random_repeats=20,
        noise_std_fraction_of_train_feature_std=.01,noise_repeats=10,
        train_mean_operator="replace selected V projections by their TRAIN per-asset/hop means; Q/K unchanged",
        delete_keep_operator="mask source columns across both graph layers and both hops, preserve self edge, renormalize over sources",
        source_direction="alpha[B,receiver_i,source_j,head]",
        ranking_semantics="gradient_attention is |alpha*d y/d alpha| averaged over receivers, heads, hops and graph layers; heuristic, not causal attribution",
        caveat="large prediction change is influence; positive delta_abs_error means deletion worsened accuracy",
        rows=str(path),summary=str(summary_path),comparison=str(comparison_path),
        risk_feature_file=str(a_path),
        test_evaluated=split=="test")
    atomic_json(out/f"graph_{split}_{horizon}d.json",report)
    return report
