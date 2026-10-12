"""Frozen D0B explanation primitives. No model parameters are modified."""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
import math

import numpy as np
import torch

HORIZONS = (1, 5, 10, 20)
LAYER_NAMES = ("attn_mixhop1", "attn_mixhop2")


def entropy(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def regime_summary(p):
    """One terminal posterior per prediction origin, never all overlapping windows."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("Expected terminal regime posterior [origins,3]")
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Regime probabilities do not sum to one")
    hard = p.argmax(1)
    mean_p = p.mean(0)
    marginal_entropy = float(entropy(mean_p))
    runs = {str(k): [] for k in range(3)}
    start = 0
    for i in range(1, len(hard) + 1):
        if i == len(hard) or hard[i] != hard[start]:
            runs[str(int(hard[start]))].append(i-start)
            start = i
    return dict(soft_occupancy=mean_p.tolist(),
                hard_occupancy=(np.bincount(hard, minlength=3)/len(hard)).tolist(),
                mean_sample_entropy=float(entropy(p).mean()),
                effective_state_count=float(np.exp(marginal_entropy)),
                state_distinctness=float(marginal_entropy-entropy(p).mean()),
                switching_rate=float(np.mean(hard[1:] != hard[:-1])) if len(hard)>1 else 0.,
                duration_by_state={k: dict(mean=float(np.mean(v)) if v else None,
                                           median=float(np.median(v)) if v else None,
                                           n_runs=len(v)) for k,v in runs.items()},
                uniform_allocation_warning=bool(np.max(np.abs(mean_p-1/3))<.02 and
                                                entropy(p).mean()>math.log(3)-.02),
                single_state_warning=bool(max(np.bincount(hard,minlength=3))/len(hard)>.98))


def fixed_gate_shapley(head, a, b, *, a_reference=None, b_reference=None):
    """Exact two-player local decomposition of this head, holding gate fixed."""
    a0 = torch.zeros_like(a) if a_reference is None else a_reference.expand_as(a)
    b0 = torch.zeros_like(b) if b_reference is None else b_reference.expand_as(b)
    f00, f10, f01, f11 = [head(z) for z in
                          (a0+b0, a+b0, a0+b, a+b)]
    phi_t = .5*((f10-f00)+(f11-f01))
    phi_s = .5*((f01-f00)+(f11-f10))
    complete = (phi_t+phi_s-(f11-f00)).abs()
    denom = phi_t.abs()+phi_s.abs()+1e-12
    return dict(phi_temporal=phi_t, phi_spatial=phi_s, baseline=f00,
                prediction=f11, completeness_error=complete,
                temporal_abs_share=phi_t.abs()/denom,
                cancellation=(phi_t.abs()+phi_s.abs()-(phi_t+phi_s).abs())/denom)


class D0BTrace:
    """One actual forward with temporary hooks; default stores source summaries only."""
    def __init__(self, model, *, full_attention=False, live_attention=False):
        if model.variant != "switching_latent_balanced_readout":
            raise ValueError("Only the original gated D0B is supported")
        self.model, self.full, self.live = model, full_attention, live_attention
        self.handles, self.tensors, self.attentions = [], {}, []

    def __enter__(self):
        def output(name):
            def fn(_mod, _inp, value): self.tensors[name] = value
            return fn
        for name,module in (("spatial",self.model.type_pool),
                            ("temporal",self.model.switching_latent_transformer),
                            ("s",self.model.gcn_proj),("t",self.model.lstm_proj),
                            ("gate_logits",self.model.gate_fc)):
            self.handles.append(module.register_forward_hook(output(name)))
        for layer_name in LAYER_NAMES:
            module = getattr(self.model,layer_name).attn_drop
            def attention_hook(_mod, _inp, value, layer=layer_name):
                # The output is live (eval dropout is identity). No detached
                # diagnostic cache is ever used for prediction gradients.
                if self.live or self.full:
                    self.attentions.append((layer,value))
                else:
                    self.tensors.setdefault("source_summary",[]).append(
                        value.detach().mean(dim=1).permute(0,2,1))  # [B,head,source]
            self.handles.append(module.register_forward_hook(attention_hook))
        return self

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        self.handles.clear()

    def forward(self, x, *, forced_route=None, zero_component=None):
        self.tensors.clear(); self.attentions.clear()
        self.model.eval()
        if forced_route is None and zero_component is None:
            prediction = self.model(x)
        else:
            prediction = self.model._switching_latent_transformer_forward(
                x,forced_probabilities=forced_route,
                zero_readout_component=zero_component)
        b = self.model.switching_latent_transformer
        gate = torch.sigmoid(self.tensors["gate_logits"])
        s,t = self.tensors["s"],self.tensors["t"]
        self.tensors.update(prediction=prediction,gate=gate,
            a=gate*t,b=(1-gate)*s,
            posterior=b.last_regime_probabilities,
            applied_route=b.last_latent_probabilities,
            prior=b.last_regime_priors)
        return self.tensors


def attention_source_scores(attentions, *, gradient=None):
    """Receiver i, source j, head h: average over all receiving nodes."""
    if not attentions:
        raise ValueError("No live graph attention captured")
    scores = []
    for _,alpha in attentions:
        value = alpha if gradient is None else alpha * gradient(alpha)
        scores.append(value.abs().mean(dim=(1,3)))  # [B, source]
    return torch.stack(scores).mean(0)


def target_gradient_scores(prediction, attentions, horizon_index, commodity_index):
    """Heuristic |alpha*d y/d alpha|, full prediction path, not a causal attribution."""
    values = [a for _,a in attentions]
    grads = torch.autograd.grad(prediction[:,horizon_index,commodity_index].sum(),
                               values,retain_graph=True,allow_unused=False)
    return torch.stack([(a*g).abs().mean(dim=(1,3))
                        for a,g in zip(values,grads)]).mean(0)


def normalize_masked_attention(alpha, source_mask, *, keep=False):
    """Remove/keep source columns, preserve diagonal self-loop; safe all-mask case."""
    if alpha.ndim != 4 or alpha.shape[1] != alpha.shape[2]:
        raise ValueError("Expected [batch,receiver,source,head]")
    n = alpha.shape[1]
    selected = torch.as_tensor(source_mask,dtype=torch.bool,device=alpha.device)
    if selected.shape == (n,):
        selected=selected[None,:].expand(alpha.shape[0],-1)
    elif selected.shape != (alpha.shape[0],n):
        raise ValueError("Source mask must be [N] or [batch,N]")
    eye = torch.eye(n,dtype=torch.bool,device=alpha.device)
    allowed = (selected[:,None,:] if keep else ~selected[:,None,:]) | eye[None]
    masked = alpha * allowed[:,:,:,None]
    mass=masked.sum(dim=2,keepdim=True)
    normalized=masked/mass.clamp_min(1e-12)
    fallback=eye[None,:,:,None].to(alpha.dtype).expand_as(alpha)
    return torch.where(mass>1e-12,normalized,fallback)


@contextmanager
def source_intervention(model, source_mask, *, keep=False):
    """Temporary hook on BOTH graph blocks and every propagation hop."""
    handles=[]
    try:
        for layer in LAYER_NAMES:
            mod=getattr(model,layer).attn_drop
            handles.append(mod.register_forward_hook(
                lambda _m,_i,a: normalize_masked_attention(a,source_mask,keep=keep)))
        yield
    finally:
        for h in handles: h.remove()


def matched_random_sources(selected, market_ranges, adjacency, rng):
    """Same market counts; greedily match source weighted degree."""
    selected=np.asarray(selected,dtype=np.int64)
    degree=np.asarray(adjacency,dtype=np.float64).sum(axis=0)
    forbidden=set(selected.tolist()); used=set(); out=[]
    for j in selected[rng.permutation(len(selected))]:
        group=next((range(a,b) for a,b in market_ranges if a<=j<b),None)
        if group is None: raise ValueError("Source outside market ranges")
        pool=np.array([k for k in group if k not in used and k not in forbidden],dtype=np.int64)
        if not len(pool):
            # If q exhausts this market, exact market matching requires reuse
            # of an original selected asset; retain unique sources where possible.
            pool=np.array([k for k in group if k not in used],dtype=np.int64)
        if not len(pool): raise ValueError("No market-matched source available")
        distances=np.abs(degree[pool]-degree[j])
        ties=pool[np.flatnonzero(distances==distances.min())]
        choice=int(rng.choice(ties));out.append(choice);used.add(choice)
    return np.array(out,dtype=np.int64)


def market_descriptors(price_window, market_ranges):
    """Past-only 20-day raw-price market summaries for one prediction origin."""
    price=np.asarray(price_window,dtype=np.float64)
    if price.ndim!=2 or price.shape[0]!=20: raise ValueError("Expected 20 observed days")
    returns=np.clip(price[1:]/np.maximum(np.abs(price[:-1]),1e-8)-1,-.5,.5)
    means=[];out={}
    names=("stock","bond","commodity")
    for name,(a,b) in zip(names,market_ranges):
        r=returns[:,a:b]; market=r.mean(1);means.append(market)
        out[f"return_{name}_20"]=float(market.sum())
        out[f"vol_{name}_20"]=float(market.std())
        out[f"dispersion_{name}_20"]=float(r.std(1).mean())
    for i,j in combinations(range(3),2):
        x,y=means[i],means[j]
        corr=np.corrcoef(x,y)[0,1] if x.std()>0 and y.std()>0 else 0.
        out[f"corr_{names[i]}_{names[j]}_20"]=float(corr)
    return out


def weighted_regime_profile(probabilities, descriptors):
    p=np.asarray(probabilities,dtype=np.float64)
    d=np.asarray(descriptors,dtype=np.float64)
    if len(p)!=len(d) or p.shape[1]!=3: raise ValueError("Unaligned profiles")
    return (p.T@d/np.maximum(p.sum(0)[:,None],1e-12))


def train_only_state_alignment(reference_train_p, other_train_p, train_descriptors):
    """Hungarian state permutation based exclusively on TRAIN weighted profiles."""
    from scipy.optimize import linear_sum_assignment
    d=np.asarray(train_descriptors,dtype=float)
    scaled=(d-d.mean(0))/(d.std(0)+1e-12)
    ref=weighted_regime_profile(reference_train_p,scaled)
    alt=weighted_regime_profile(other_train_p,scaled)
    cost=np.linalg.norm(ref[:,None,:]-alt[None,:,:],axis=2)
    rows,columns=linear_sum_assignment(cost)
    perm=np.empty(3,dtype=int);perm[rows]=columns
    return dict(permutation=perm.tolist(),cost=cost.tolist(),
                reference_profile=ref.tolist(),other_profile=alt.tolist(),
                alignment_split="TRAIN only")


def block_mismatch(probabilities, block=40, seed=42):
    """Permute whole consecutive blocks, preserving within-block chronology."""
    p=np.asarray(probabilities);rng=np.random.default_rng(seed)
    blocks=[p[i:i+block] for i in range(0,len(p),block)]
    if len(blocks)<2: raise ValueError("Need >=2 blocks")
    # Circular block shift keeps length and each block's internal order.
    return np.concatenate(blocks[1:]+blocks[:1],axis=0)


def paired_moving_block_bootstrap(values, *, block=40, draws=2000, seed=42):
    """First axis is prediction date; all commodities stay in the same draw."""
    x=np.asarray(values,dtype=np.float64)
    n=len(x)
    if n<block: raise ValueError("Need at least one full block")
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(draws):
        starts=rng.integers(0,n-block+1,size=math.ceil(n/block))
        indices=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        estimates.append(x[indices].mean(axis=0))
    z=np.asarray(estimates)
    return dict(estimate=x.mean(axis=0).tolist(),
                low=np.quantile(z,.025,axis=0).tolist(),
                high=np.quantile(z,.975,axis=0).tolist(),
                block=block,draws=draws)


def risk_evaluation(calibration, evaluation, *, seed=42):
    """Eight fixed B+R/G/A logistic models; no evaluation-label-dependent fitting."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y_cal=np.asarray(calibration["error5"],dtype=float)
    y_eval=np.asarray(evaluation["error5"],dtype=float)
    threshold=float(np.quantile(y_cal,.8))
    labels_cal=y_cal>=threshold; labels_eval=y_eval>=threshold
    if np.unique(labels_cal).size<2: raise ValueError("Calibration has one risk class")
    results={}
    for bits in range(8):
        groups=("B",)+tuple(g for i,g in enumerate(("R","G","A")) if bits&(1<<i))
        cols=[np.asarray(calibration[g]) for g in groups]
        other=[np.asarray(evaluation[g]) for g in groups]
        if any(c.ndim!=2 for c in cols+other):raise ValueError("Feature groups must be matrices")
        X=np.concatenate(cols,axis=1);Z=np.concatenate(other,axis=1)
        if not (np.isfinite(X).all() and np.isfinite(Z).all()):
            raise ValueError("Nonfinite risk features; no imputation from evaluation data")
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,
                                                               random_state=seed))
        clf.fit(X,labels_cal)
        risk=clf.predict_proba(Z)[:,1]
        order=np.argsort(risk,kind="stable")
        coverage=np.arange(1,len(order)+1)/len(order)
        selective=np.cumsum(y_eval[order])/np.arange(1,len(order)+1)
        n80=max(1,int(math.floor(.8*len(order))))
        results["+".join(groups)]=dict(
            threshold=threshold,AURC=float(np.trapezoid(selective,coverage)),
            MAE_at_80pct_coverage=float(y_eval[order[:n80]].mean()),
            AUROC=float(roc_auc_score(labels_eval,risk)) if np.unique(labels_eval).size==2 else None,
            AUPRC=float(average_precision_score(labels_eval,risk)),
            Brier=float(brier_score_loss(labels_eval,risk)),
            evaluation_event_rate=float(labels_eval.mean()),
            coverage=coverage.tolist(),selective_MAE=selective.tolist(),
            risk_score=risk.tolist())
    return results


def paired_risk_block_bootstrap(errors, risk_scores, *, block=40, draws=2000, seed=42):
    """Paired date-block CI for AURC/MAE80 changes versus B, fixed fitted monitors."""
    e=np.asarray(errors,dtype=np.float64)
    risks={k:np.asarray(v,dtype=np.float64) for k,v in risk_scores.items()}
    n=len(e)
    if n<block or "B" not in risks or any(len(v)!=n for v in risks.values()):
        raise ValueError("Need aligned B and candidate risk scores by origin date")
    def metrics(index,score):
        order=np.argsort(score[index],kind="stable")
        y=e[index][order]
        cover=np.arange(1,n+1)/n
        curve=np.cumsum(y)/np.arange(1,n+1)
        return np.array([np.trapezoid(curve,cover),y[:max(1,int(.8*n))].mean()])
    rng=np.random.default_rng(seed)
    deltas={k:[] for k in risks if k!="B"}
    for _ in range(draws):
        starts=rng.integers(0,n-block+1,size=math.ceil(n/block))
        index=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        base=metrics(index,risks["B"])
        for k in deltas:deltas[k].append(metrics(index,risks[k])-base)
    return {k:dict(AURC_delta_95CI=np.quantile(v,[.025,.975],axis=0)[:,0].tolist(),
                   MAE80_delta_95CI=np.quantile(v,[.025,.975],axis=0)[:,1].tolist())
            for k,v in deltas.items()}
