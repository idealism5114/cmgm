"""Fixed-batch Q/K geometry diagnostics; no learned probe or optimizer step."""
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F

from cmgm.scripts.d0e_diagnostics import diagnostic_context
from cmgm.scripts.d0b_hybrid_graph_diagnostics import (
    initialization_check as _initialization_check,gradient_diagnostics as _gradients,
    graph_diagnostics,_head_summary,
)

VARIANT='switching_latent_balanced_qknorm_graph_attention'
BASE='switching_latent_balanced_readout'
DISPLAY='D0B-QKNormGraphAttention'
EPS=1e-6


def initialization_check(model,seed=42):
    r=_initialization_check(model,seed)
    r['QKNorm_params']=r.pop('Hybrid_params')
    return r


def stats(x):
    x=x.detach().double().reshape(-1)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite diagnostic values')
    q=torch.quantile(x,x.new_tensor([.01,.05,.5,.95]))
    return dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),min=float(x.min()),max=float(x.max()),
                P1=float(q[0]),P5=float(q[1]),median=float(q[2]),P50=float(q[2]),P95=float(q[3]))


def norm_stats(raw,used,enabled):
    raw_norm=raw.double().norm(dim=-1);used_norm=used.double().norm(dim=-1)
    scale=math.sqrt(raw.shape[-1]);active=raw_norm>=EPS
    max_error=float((used_norm[active]-scale).abs().max()) if active.any() else None
    expected=raw_norm/raw_norm.clamp_min(EPS)*scale if enabled else raw_norm
    expected_error=float((used_norm-expected).abs().max())
    return dict(raw={**stats(raw_norm),'fraction_norm_lt_1e_minus_6':float((raw_norm<1e-6).double().mean()),
                    'fraction_norm_lt_1e_minus_4':float((raw_norm<1e-4).double().mean())},
        used=stats(used_norm),expected_non_degenerate_norm=scale,epsilon_degenerate_count=int((~active).sum()),
        non_degenerate_count=int(active.sum()),non_degenerate_max_deviation=max_error,
        max_deviation_including_degenerate=float((used_norm-scale).abs().max()),
        formula_norm_max_error=expected_error,
        PASS=(not enabled or ((max_error is None or max_error<1e-5) and expected_error<1e-5)),
        epsilon_semantics='F.normalize divides by max(L2 norm, 1e-6); sub-epsilon vectors need not attain sqrt(head_dim).')


def head_diversity(attention):
    vec=attention.permute(3,0,1,2).reshape(attention.shape[-1],-1).double()
    rows=[]
    for i,j in itertools.combinations(range(len(vec)),2):
        a,b=vec[i],vec[j];ca,cb=a-a.mean(),b-b.mean();denom=ca.norm()*cb.norm()
        rows.append(dict(heads=[i,j],cosine=float(F.cosine_similarity(a,b,dim=0)),
            correlation=float(ca@cb/denom) if denom>0 else None))
    cos=[r['cosine'] for r in rows];corr=[r['correlation'] for r in rows if r['correlation'] is not None]
    return dict(pairs=rows,cosine_mean=float(np.mean(cos)),cosine_std=float(np.std(cos)),
        correlation_mean=float(np.mean(corr)) if corr else None,correlation_std=float(np.std(corr)) if corr else None)


def summarize_hop(record,A,native_prior,enabled):
    qk=record['qk'];q=norm_stats(qk['raw_Q'],qk['Q'],enabled);k=norm_stats(qk['raw_K'],qk['K'],enabled)
    attention=record['attention'].double()
    if attention.ndim==3:attention=attention.unsqueeze(0)
    entropy=-(attention*attention.clamp_min(1e-12).log()).sum(2)
    degree=min(10,len(A));mask=torch.zeros_like(A,dtype=torch.bool)
    mask.scatter_(1,A.topk(degree,dim=1).indices,True)
    mass=(attention*mask[None,:,:,None]).sum(2)
    audit=_head_summary(record,A,degree,0,native_prior)
    cosine_error=None
    if enabled:
        Q=F.normalize(qk['raw_Q'].double(),dim=-1,eps=EPS)
        K=F.normalize(qk['raw_K'].double(),dim=-1,eps=EPS)
        eq='bnhd,bmhd->bnmh' if Q.ndim==4 else 'nhd,mhd->nmh'
        expected=torch.einsum(eq,Q,K)*math.sqrt(Q.shape[-1])
        cosine_error=float((record['content'].double()-expected).abs().max())
    geometry=dict(hop=record['hop'],qk_norm_enabled=enabled,Q=q,K=k,
        content_logits=stats(record['content']),prior_bias=stats(record['prior_bias']),final_logits=stats(record['logits']),
        cosine_content_formula_max_error=cosine_error,
        cosine_bound=math.sqrt(qk['Q'].shape[-1]),all_heads_receive_graph_prior=True,
        graph_prior_audit={name:audit[name] for name in ('graph_prior_error','prior_formula_error',
            'exact_native_logit_recomposition_error','float64_bias_subtraction_error','partition_PASS','partition_precision_note')})
    attn=dict(hop=record['hop'],entropy_mean=float(entropy.mean()),entropy_std=float(entropy.std(unbiased=False)),
        uniform_entropy=math.log(attention.shape[2]),entropy_uniform_ratio=float(entropy.mean())/math.log(attention.shape[2]),
        **{f'top{n}_mass':float(attention.topk(min(n,attention.shape[2]),dim=2).values.sum(2).mean()) for n in (1,5,10)},
        graph_topk_mass=float(mass.mean()),head_diversity=head_diversity(attention))
    geometry['PASS']=q['PASS'] and k['PASS'] and audit['partition_PASS'] and (not enabled or cosine_error<1e-5)
    return geometry,attn


def geometry_diagnostics(model,x):
    layers=[model.attn_mixhop1,model.attn_mixhop2];flags=[l.capture_attention for l in layers]
    with diagnostic_context(model),torch.no_grad():
        try:
            for layer in layers:
                assert layer.graph_prior_heads is None and not layer.hard_mask and layer.cross_mask is None
                assert layer.n_heads==8 and layer.prior_scale==.5 and layer.qk_norm_eps==EPS
                layer.capture_attention=True
            model(x);A_native=model.graph_learner().detach();A=A_native.cpu().double()
            geometry={};attention={}
            for i,layer in enumerate(layers,1):
                prior=(layer.prior_scale*torch.log(A_native.clamp_min(0)+1e-6)).cpu()
                rows=[summarize_hop(record,A,prior,layer.qk_norm) for record in layer.last_attention_diagnostics]
                geometry[f'layer{i}']=[row[0] for row in rows]
                attention[f'layer{i}']=dict(hops=[row[1] for row in rows],
                    **{key:float(np.mean([row[1][key] for row in rows])) for key in
                       ('entropy_mean','entropy_std','uniform_entropy','entropy_uniform_ratio','top1_mass','top5_mass','top10_mass','graph_topk_mass')})
            passed=all(row['PASS'] for rows in geometry.values() for row in rows)
            return dict(qk_norm=geometry,attention=attention,PASS=passed,
                definition='Fixed TEST inputs, eval/pre-dropout capture; stats over sample/node/head or sample/query/source/head as appropriate. Layer summary averages the two hop statistics; hop entropy std is over sample/query/head. No targets used.')
        finally:
            for layer,flag in zip(layers,flags):
                layer.capture_attention=flag;layer.last_attention_diagnostics=[]
                for name in ('last_content_logits','last_prior_bias','last_attention_logits'):
                    if hasattr(layer,name):delattr(layer,name)


def gradient_diagnostics(model,batch):
    r=_gradients(model,batch)
    r['ratios']={}
    for objective in ('prediction_only','total_loss'):
        r['ratios'][objective]={}
        for i in (1,2):
            prefix=f'attn_mixhop{i}.';v=r[objective][prefix+'v']['norm']
            r['ratios'][objective][f'layer{i}']=dict(Q_over_V=r[objective][prefix+'q']['norm']/(v+1e-12),
                K_over_V=r[objective][prefix+'k']['norm']/(v+1e-12))
    r['ratio_epsilon']=1e-12
    return r
