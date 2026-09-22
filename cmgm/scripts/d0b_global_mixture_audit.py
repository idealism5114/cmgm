"""Audit exact Dynamic/Static shared and expert initialization."""
import torch
from cmgm.models.global_mixture_fusion import VARIANT
from cmgm.models.candidate_moe_fusion import VARIANT as DYNAMIC
from cmgm.scripts.d0b_moe_audit import make_model as _make_model
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss


def make_model(data):return _make_model(data,VARIANT)


def initialization(data,device):
    seed_all(42);dynamic=_make_model(data,DYNAMIC).to(device)
    seed_all(42);static=make_model(data).to(device)
    d=dict(dynamic.named_parameters());s=dict(static.named_parameters())
    shared=sorted(d.keys()&s.keys());diffs={n:float((d[n]-s[n]).detach().abs().max()) for n in shared}
    ed={}
    for name in ('temporal_expert','interaction_expert'):
        a=getattr(dynamic.candidate_moe_fusion,name).state_dict();b=getattr(static.global_mixture_fusion,name).state_dict()
        ed[name+'_max_abs_diff']=max(float((a[k]-b[k]).abs().max()) for k in a)
    ed.update(shared_backbone_max_abs_diff=max(v for n,v in diffs.items() if not n.startswith('head.')),
              shared_head_max_abs_diff=max(v for n,v in diffs.items() if n.startswith('head.')))
    ed['PASS']=all(v==0 for v in ed.values())
    global_parameters=dict(static.global_mixture_fusion.named_parameters())
    router_names=[n for n in s if 'router' in n or 'candidate_moe_fusion' in n or 'interaction_head' in n]
    r=dict(dynamic_parameters=sum(p.numel() for p in d.values()),static_parameters=sum(p.numel() for p in s.values()),
        shared_max_abs_diff=max(diffs.values()),mismatch_count=sum(v!=0 for v in diffs.values()),shared_differences=diffs,
        dynamic_router_parameters=sum(p.numel() for p in dynamic.candidate_moe_fusion.router.parameters()),
        static_router_parameters=global_parameters['global_mixture_logits'].numel(),unexpected_router_parameters=router_names,
        parameter_count_note='Only candidate LayerNorm/MLP router removed; two global trainable logits replace it. No capacity matching.')
    r['parameter_delta']=r['static_parameters']-r['dynamic_parameters']
    r['PASS']=ed['PASS'] and not router_names and r['dynamic_router_parameters']==16834 and r['static_router_parameters']==2 and r['parameter_delta']==-16832
    return static,r,ed


def sanity(model,x,y,initial=True):
    model.eval();f=model.global_mixture_fusion;b=model.switching_latent_transformer
    before={k:v.clone() for k,v in model.state_dict().items()};err=lambda a,b:float((a-b).detach().abs().max())
    try:
        with torch.no_grad():
            pred=model(x);c={k:v.clone() for k,v in f.last.items()}
            alpha=f.global_mixture_logits.softmax(0)
            checks=dict(alpha=err(c['alpha'],alpha),temporal_expert=err(c['e_T'],f.temporal_expert(c['t'])),
                interaction_expert=err(c['e_ST'],f.interaction_expert(torch.cat([c['s'],c['t']],-1))),
                projection_s=err(c['s'],model.gcn_proj(c['h_s'])),projection_t=err(c['t'],model.lstm_proj(c['h_t'])),
                fused=err(c['fused'],(alpha.view(1,2,1)*torch.stack([c['e_T'],c['e_ST']],1)).sum(1)),
                shared_head=err(pred,model.head(c['fused']).reshape_as(pred)),normalization=err(alpha.sum(),alpha.new_tensor(1.)))
            initial_alpha_error=err(alpha,torch.full_like(alpha,.5));initial_identity=err(c['fused'],.5*c['e_T']+.5*c['e_ST'])
            perm=torch.arange(len(x)-1,-1,-1,device=x.device);pp=model(x[perm])
            checks.update(batch_prediction=err(pp,pred[perm]),batch_alpha=err(f.last['alpha'],alpha))
            single=model(x[:1]);checks.update(single_prediction=err(single,pred[:1]),single_alpha=err(f.last['alpha'],alpha))
            model(x*2+3);checks['different_input_alpha']=err(f.last['alpha'],alpha)
            model(x);states={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
            future=x.clone();future[:,10:]=future[:,10:]*-3+7;model(future)
            causal={k:err(v[:,:10],getattr(b,k)[:,:10]) for k,v in states.items()}
            H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
            h,z=b.last_long_memory[:,:10],b.last_latent_states[:,:10]
            old=b.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]));long=b.last_h_long.clone();micro=b.last_h_micro.clone()
            new=b.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]))
            causal.update(h_temporal=err(old,new),h_long=err(long,b.last_h_long),h_micro=err(micro,b.last_h_micro))
            shapes={k:list(v.shape) for k,v in c.items()};shapes['prediction']=list(pred.shape)
            shape_ok=all(shapes[k]==[len(x),64] for k in ('s','t','e_T','e_ST','fused')) and shapes['alpha']==shapes['global_logits']==[2] and pred.shape==(len(x),4,24)
        # Gradients measured without optimizer step or .grad mutation.
        model.train();p=model(x);loss=prediction_loss(p,y)
        grad=torch.autograd.grad(loss,f.global_mixture_logits)[0]
        finite=all(bool(torch.isfinite(v).all()) for v in c.values()) and bool(torch.isfinite(grad).all())
        passed=finite and shape_ok and max([*checks.values(),*causal.values()])<1e-6
        if initial:passed=passed and initial_alpha_error<1e-7 and initial_identity<1e-7 and bool(grad.abs().sum()>0)
        return dict(PASS=passed,initial=initial,shapes=shapes,shape_PASS=shape_ok,finite=finite,checks=checks,
            initial_alpha_error=initial_alpha_error,initial_50_50_identity=initial_identity,
            global_logit_prediction_grad_norm=float(grad.norm()),causal_prefix10=causal,
            note='alpha is a single (2,) vector; no batch axis and no sample-conditioned router.')
    finally:
        model.eval()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert all(p.grad is None for p in model.parameters())
