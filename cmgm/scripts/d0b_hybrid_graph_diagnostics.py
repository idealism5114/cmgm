"""Read-only mechanism checks for the fixed four-free/four-prior experiment."""
import itertools
import numpy as np
import torch
import torch.nn.functional as F

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0e_diagnostics import diagnostic_context, model_arguments
from cmgm.training.train import make_loss, _prediction_loss

BASE='switching_latent_balanced_readout'
VARIANT='switching_latent_balanced_hybrid_graph_prior'
DISPLAY='D0B-HybridGraphPriorHeads'


def initialization_check(model,seed=42):
    with diagnostic_context(model):
        torch.random.default_generator.manual_seed(seed)
        baseline=HeteroMixHopCMGM(variant=BASE,**model_arguments(model)).to(next(model.parameters()).device)
        left,right=baseline.state_dict(),model.state_dict()
        assert left.keys()==right.keys()
        assert all(left[k].shape==right[k].shape for k in left)
        diffs={k:float((left[k]-right[k]).abs().max()) for k in left}
        counts=[sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        r=dict(D0B_params=counts[0],Hybrid_params=counts[1],delta_params=counts[1]-counts[0],
            shared_parameter_init_max_diff=max(diffs.values()),mismatch_count=sum(v!=0 for v in diffs.values()))
        r['PASS']=r['delta_params']==0 and r['mismatch_count']==0
        assert r['PASS'],r
        return r


def _head_summary(record,A,top_k,n_free,native_prior):
    attn=record['attention'].double()
    if attn.ndim==3:attn=attn.unsqueeze(0)
    n_heads=attn.shape[-1]
    top=torch.zeros_like(A,dtype=torch.bool)
    top.scatter_(1,A.topk(min(top_k,len(A)),dim=1).indices,True)
    entropy=-(attn*attn.clamp_min(1e-12).log()).sum(2)
    mass=(attn*top[None,:,:,None]).sum(2)
    vec=attn.permute(3,0,1,2).reshape(n_heads,-1)
    def group_pairs(kind):
        rows=[]
        for i,j in itertools.combinations(range(n_heads),2):
            label='free-free' if j<n_free else ('graph-graph' if i>=n_free else 'free-graph')
            if label!=kind:continue
            a,b=vec[i],vec[j]
            cosine=float(F.cosine_similarity(a,b,dim=0))
            ca,cb=a-a.mean(),b-b.mean();denom=ca.norm()*cb.norm()
            rows.append(dict(heads=[i,j],cosine=cosine,correlation=float(ca@cb/denom) if denom>0 else None))
        return dict(pairs=rows,cosine_mean=float(np.mean([r['cosine'] for r in rows])) if rows else None,
            correlation_mean=float(np.mean([r['correlation'] for r in rows if r['correlation'] is not None])) if any(r['correlation'] is not None for r in rows) else None)
    prior=record['prior_bias'].double()
    observed=record['logits'].double()-record['content'].double()
    free_error=float(observed[...,:n_free].abs().max()) if n_free else 0.
    graph_error=float((observed[...,n_free:]-prior[...,n_free:]).abs().max()) if n_free<n_heads else 0.
    # Large real-data logits lose low bits in FP32 addition. Preserve the raw
    # subtraction error, verify the ACTUAL addition bit-for-bit, then audit the
    # same saved content/prior in FP64. No model operation/precision is changed.
    content=record['content'];dtype=content.dtype
    expected_prior=torch.zeros_like(record['prior_bias'])
    expected_prior[...,n_free:]=native_prior[...,None]
    prior_formula_error=float((record['prior_bias']-expected_prior).abs().max())
    expected_logits=content+expected_prior
    exact_addition_error=float((record['logits']-expected_logits).abs().max())
    double_delta=(content.double()+expected_prior.double())-content.double()
    double_error=float((double_delta-expected_prior.double()).abs().max())
    # Conservative one-ULP bound for the actual float addition's subtraction.
    rounding_bound=torch.finfo(dtype).eps*(content.double().abs()+expected_prior.double().abs()).clamp_min(1.)
    within_roundoff=bool(((observed-expected_prior.double()).abs()<=rounding_bound).all())
    passed=(free_error<1e-7 and prior_formula_error==0 and exact_addition_error==0
            and double_error<1e-6 and within_roundoff)
    return dict(hop=record['hop'],free_entropy=float(entropy[...,:n_free].mean()) if n_free else None,
        graph_entropy=float(entropy[...,n_free:].mean()) if n_free<n_heads else None,
        free_topk_mass=float(mass[...,:n_free].mean()) if n_free else None,
        graph_topk_mass=float(mass[...,n_free:].mean()) if n_free<n_heads else None,
        free_prior_error=free_error,graph_prior_error=graph_error,
        prior_formula_error=prior_formula_error,exact_native_logit_recomposition_error=exact_addition_error,
        float64_bias_subtraction_error=double_error,float32_subtraction_threshold_PASS=graph_error<1e-6,
        subtraction_within_native_roundoff_bound=within_roundoff,
        partition_PASS=passed,
        partition_precision_note='Raw FP32 subtraction retained, even when >1e-6. PASS requires exact native logits = content + independently recomputed masked A prior, zero free bias, FP64 same-operands subtraction <1e-6, and native error within its rounding bound. No forward precision modification.',
        head_similarity={k:group_pairs(k) for k in ('free-free','graph-graph','free-graph')})


def attention_diagnostics(model,x):
    layers=[model.attn_mixhop1,model.attn_mixhop2]
    old=[l.capture_attention for l in layers]
    with diagnostic_context(model),torch.no_grad():
        try:
            for l in layers:l.capture_attention=True
            model(x)
            A_native=model.graph_learner().detach()
            A=A_native.cpu().double()
            result={}
            for i,l in enumerate(layers,1):
                nfree=0 if l.graph_prior_heads is None else l.n_heads-l.graph_prior_heads
                # Recompute the canonical formula on the ORIGINAL device to
                # avoid comparing CPU log() with CUDA log() rounding.
                native_prior=(l.prior_scale*torch.log(A_native.clamp_min(0)+1e-6)).cpu()
                hops=[_head_summary(r,A,model.graph_learner.top_k,nfree,native_prior) for r in l.last_attention_diagnostics]
                result[f'layer{i}']=dict(n_free=nfree,n_graph=l.n_heads-nfree,hops=hops,
                    **{key:float(np.mean([h[key] for h in hops])) if hops[0][key] is not None else None
                       for key in ('free_entropy','graph_entropy','free_topk_mass','graph_topk_mass')})
            result['PASS']=all(h['partition_PASS'] for l in ('layer1','layer2') for h in result[l]['hops'])
            result['definition']='Pre-dropout attention, query-row entropy/mass averaged over samples, nodes, heads and both MixHop hops; strongest existing A edges (ties resolved by torch.topk).'
            result['independence_scope']='Free logits have zero DIRECT A bias. Later MixHop hops/layers may inherit A through the preceding mixed hidden states; whole free-head trajectories are not claimed independent of A.'
            return result
        finally:
            for l,flag in zip(layers,old):
                l.capture_attention=flag;l.last_attention_diagnostics=[]
                for key in ('last_content_logits','last_prior_bias','last_attention_logits'):
                    if hasattr(l,key):delattr(l,key)


def graph_diagnostics(model):
    with torch.no_grad():
        g=model.graph_learner;A=g().double();s=A.sum(-1,keepdim=True)
        p=A/s.clamp_min(1e-30);entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
        return dict(alpha=float(g.alpha),A_mean=float(A.mean()),A_std=float(A.std(unbiased=False)),
            near_zero_fraction=float((A<=1e-8).double().mean()),
            row_positive_degree_mean=float((A>1e-8).double().sum(-1).mean()),
            row_effective_degree_mean=float(torch.where(s[:,0]>0,entropy.exp(),torch.zeros_like(entropy)).mean()),
            topk_concentration=float((A.topk(min(g.top_k,len(A)),dim=1).values.sum(-1)/s[:,0].clamp_min(1e-30)).mean()),
            top_k=g.top_k,alpha_trainable=g.alpha.requires_grad)


def gradient_diagnostics(model,batch):
    x,y=(v.to(next(model.parameters()).device) for v in batch[:2])
    groups={f'attn_mixhop{i}.{key}':list(getattr(getattr(model,f'attn_mixhop{i}'),key).parameters())
            for i in (1,2) for key in ('q','k','v')}
    for label,name in [('E1','E1'),('E2','E2'),('Theta1','Θ1'),('Theta2','Θ2'),('alpha','alpha')]:
        groups['graph_learner.'+label]=[getattr(model.graph_learner,name)]
    params=list(dict.fromkeys(p for group in groups.values() for p in group));slots={id(p):i for i,p in enumerate(params)}
    with diagnostic_context(model),torch.enable_grad():
        pred=model(x);loss=_prediction_loss(model,pred,y,make_loss())
        total=loss+model.switching_latent_transformer.switch_loss()
        result={}
        for label,value in [('prediction_only',loss),('total_loss',total)]:
            grad=torch.autograd.grad(value,params,allow_unused=True,retain_graph=label=='prediction_only')
            result[label]={name:dict(norm=float(torch.sqrt(sum((grad[slots[id(p)]].detach().double().square().sum() if grad[slots[id(p)]] is not None else p.new_tensor(0.)) for p in group))),
                connected=any(grad[slots[id(p)]] is not None for p in group)) for name,group in groups.items()}
        result['method']='Fixed TRAIN batch, eval, sum of four Huber losses; autograd.grad only, no optimizer step and no .grad mutation.'
        return result


def fixed_sanity(model,x):
    """Exact cutoff 10; full graph relabeling includes commodity head rows."""
    def trajectory(v):
        b=model.switching_latent_transformer;b(v)
        r={k:getattr(b,a).clone() for k,a in dict(E='last_market_tokens',H='last_long_memory',p='last_regime_probabilities',Z='last_latent_states').items()}
        values={k:[] for k in ('h_long','h_micro','h_temporal')}
        for t in range(v.shape[1]):
            temporal=b.readout(r['H'][:,t],r['Z'][:,t])
            values['h_long'].append(b.last_h_long.clone());values['h_micro'].append(b.last_h_micro.clone());values['h_temporal'].append(temporal)
        return {**r,**{k:torch.stack(v,1) for k,v in values.items()}}
    with diagnostic_context(model),torch.no_grad():
        assert x.shape[1]>10
        before=trajectory(x);changed=x.clone();changed[:,10:]=changed[:,10:]*-2+7
        after=trajectory(changed)
        causality={k:float((before[k][:,:10]-after[k][:,:10]).abs().max()) for k in before}
        pred=model(x);order=torch.arange(len(x)-1,-1,-1,device=x.device)
        batch=float((model(x[order])-pred[order]).abs().max());single=float((model(x[:1])-pred[:1]).abs().max())
        markets={};start=0
        for name,size in zip(('stock','bond','commodity'),(model.n_stock,model.n_bond,model.n_commodities)):
            node_order=torch.arange(model.num_nodes,device=x.device);node_order[start:start+size]=node_order[start:start+size].flip(0)
            output_order=torch.arange(model.n_commodities,device=x.device)
            if name=='commodity':output_order=output_order.flip(0)
            relabeled=HeteroMixHopCMGM(variant=model.variant,**model_arguments(model)).to(device=x.device,dtype=x.dtype).eval()
            state={k:v.clone() for k,v in model.state_dict().items()}
            for key in ('graph_learner.E1','graph_learner.E2'):state[key]=state[key][node_order]
            for key in ('head.3.weight','head.3.bias'):
                v=state[key];state[key]=v.reshape(model.n_horizons,model.n_commodities,*v.shape[1:])[:,output_order].reshape_as(v)
            relabeled.load_state_dict(state)
            markets[name]=float((relabeled(x[:,:,node_order])-pred[:,:,output_order]).abs().max())
            start+=size
        result=dict(prefix_cutoff=10,causality=causality,batch_permutation=batch,single_sample=single,
            within_market_relabeling=markets,definition='Graph E1/E2 and commodity output rows relabeled together; no raw-input-only commodity equivariance claim.')
        result['PASS']=max(causality.values())<1e-6 and batch<1e-6 and single<1e-6 and max(markets.values())<1e-6
        return result
