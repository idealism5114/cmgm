"""Shared initialization, prediction mixture, causal context and gradient isolation."""
import torch
from cmgm.models.utility_routed_moe import VARIANT,utility_target
from cmgm.scripts.d0b_moe_audit import make_model as _make_model,BASE
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss


def make_model(data):return _make_model(data,VARIANT)


def initialization(data,device):
    seed_all(42);base=_make_model(data,BASE).to(device)
    seed_all(42);model=make_model(data).to(device)
    a=dict(base.named_parameters());b=dict(model.named_parameters());shared=sorted(a.keys()&b.keys())
    added=sorted(b.keys()-a.keys());removed=sorted(a.keys()-b.keys())
    diffs={n:float((a[n]-b[n]).detach().abs().max()) for n in shared}
    head_a=dict(model.head.named_parameters());head_b=dict(model.interaction_head.named_parameters())
    r=dict(shared_parameter_count=sum(b[n].numel() for n in shared),new_parameter_count=sum(b[n].numel() for n in added),
           shared_max_abs_diff=max(diffs.values()),mismatch_count=sum(v!=0 for v in diffs.values()),
           baseline_parameters=sum(p.numel() for p in a.values()),utility_parameters=sum(p.numel() for p in b.values()),
           shared_differences=diffs,new_parameters=added,removed_parameters=removed,
           head_init_max_diff=max(float((head_a[n]-head_b[n]).detach().abs().max()) for n in head_a),
           heads_independent=all(head_a[n].data_ptr()!=head_b[n].data_ptr() for n in head_a))
    r['delta_parameters']=r['utility_parameters']-r['baseline_parameters']
    r['PASS']=r['shared_max_abs_diff']==0 and r['new_parameter_count']==48418 and r['delta_parameters']==40162 and removed==['gate_fc.bias','gate_fc.weight'] and r['head_init_max_diff']==0 and r['heads_independent']
    return model,r


def sanity(model,x,y,initial=True):
    model.eval();f=model.utility_moe_fusion;b=model.switching_latent_transformer
    state={k:v.clone() for k,v in model.state_dict().items()}
    err=lambda a,b:float((a-b).detach().abs().max())
    try:
        with torch.no_grad():
            pred=model(x);c={k:v.clone() for k,v in f.last.items()}
            p=b.last_regime_probabilities[:,-1];prior=b.last_regime_priors[:,-1]
            u=f.router.temporal_norm(c['e_T']);v=f.router.interaction_norm(c['e_ST'])
            entropy=-(p*(p+1e-8).log()).sum(-1,keepdim=True)
            ri=torch.cat([u,v,(u-v).abs(),u*v,p,(p-prior).abs(),entropy],-1)
            checks=dict(router_input=err(c['router_input'],ri),router_logits=err(c['router_logits'],f.router.network(ri)),
                pi=err(c['pi'],c['router_logits'].softmax(-1)),regime_prob=err(c['regime_prob'],p),regime_prior=err(c['regime_prior'],prior),
                regime_gap=err(c['regime_gap'],(p-prior).abs()),regime_entropy=err(c['regime_entropy'],entropy),
                pred_T=err(c['pred_T'],model.head(c['e_T']).reshape_as(pred)),pred_ST=err(c['pred_ST'],model.interaction_head(c['e_ST']).reshape_as(pred)),
                pred_identity=err(pred,c['pi'][:,0,None,None]*c['pred_T']+c['pi'][:,1,None,None]*c['pred_ST']),
                projection_s=err(c['s'],model.gcn_proj(c['h_s'])),projection_t=err(c['t'],model.lstm_proj(c['h_t'])),
                normalization=err(c['pi'].sum(-1),torch.ones(len(x),device=x.device)))
            initial_error=err(c['pi'],torch.full_like(c['pi'],.5))
            permutation=torch.arange(len(x)-1,-1,-1,device=x.device);pp=model(x[permutation])
            checks.update(batch_prediction=err(pp,pred[permutation]),batch_pi=err(f.last['pi'],c['pi'][permutation]))
            single=model(x[:1]);checks['single_prediction']=err(single,pred[:1])
            for k in ('pi','e_T','e_ST','pred_T','pred_ST'):checks['single_'+k]=err(f.last[k],c[k][:1])
            model(x);states={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_regime_priors','last_latent_states')}
            future=x.clone();future[:,10:]=future[:,10:]*-3+7;model(future)
            causal={k:err(v[:,:10],getattr(b,k)[:,:10]) for k,v in states.items()}
            H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
            h,z=b.last_long_memory[:,:10],b.last_latent_states[:,:10]
            old=b.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]));long=b.last_h_long.clone();micro=b.last_h_micro.clone()
            new=b.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]))
            causal.update(h_temporal=err(old,new),h_long=err(long,b.last_h_long),h_micro=err(micro,b.last_h_micro))
            expected={k:[len(x),64] for k in ('h_s','h_t','s','t','e_T','e_ST','u_T','u_ST')}
            expected.update({k:[len(x),3] for k in ('regime_prob','regime_prior','regime_gap')})
            expected.update(regime_entropy=[len(x),1],router_input=[len(x),263],pi=[len(x),2],pred_T=[len(x),4,24],pred_ST=[len(x),4,24],pred_final=[len(x),4,24])
            shapes={k:list(v.shape) for k,v in c.items()};shape_ok=all(shapes[k]==v for k,v in expected.items())
            zero=torch.zeros_like(y);bad=torch.ones_like(y)*.03
            equal=utility_target(zero,zero,zero)[0];st_better=utility_target(bad,zero,zero)[0];t_better=utility_target(zero,bad,zero)[0]
            target_checks=dict(equal_50_50=err(equal,torch.full_like(equal,.5))==0,ST_better=bool((st_better[:,1]>.5).all()),T_better=bool((t_better[:,0]>.5).all()))
        model.train();prediction=model(x);lp=prediction_loss(prediction,y);lr,diag=model.utility_router_loss(y)
        named=list(model.named_parameters());params=[v for _,v in named]
        gp=torch.autograd.grad(lp,params,retain_graph=True,allow_unused=True)
        gr=torch.autograd.grad(lr,params,allow_unused=True)
        def norm(grads,prefix):return sum(float(g.detach().double().square().sum()) for (n,_),g in zip(named,grads) if n.startswith(prefix) and g is not None)**.5
        gradients=dict(pred_to_router=norm(gp,'utility_moe_fusion.router'),route_to_router=norm(gr,'utility_moe_fusion.router'),
            pred_to_T=norm(gp,'utility_moe_fusion.temporal_expert'),pred_to_ST=norm(gp,'utility_moe_fusion.interaction_expert'),
            route_to_T=norm(gr,'utility_moe_fusion.temporal_expert'),route_to_ST=norm(gr,'utility_moe_fusion.interaction_expert'),
            route_to_temporal_head=norm(gr,'head'),route_to_interaction_head=norm(gr,'interaction_head'))
        leaks=[n for (n,_),g in zip(named,gr) if not n.startswith('utility_moe_fusion.router.') and g is not None]
        checks['router_recompute_value_error']=err(model._last_utility_pi,model._last_route_pi)
        finite=all(bool(torch.isfinite(v).all()) for v in c.values()) and all(g is None or bool(torch.isfinite(g).all()) for g in (*gp,*gr))
        context_detached=not any(f.router.context_requires_grad.values())
        q_detached=not model._last_utility_q.requires_grad;scale_detached=not model._last_route_scale.requires_grad
        passed=finite and shape_ok and max([*checks.values(),*causal.values()])<1e-6 and not leaks and q_detached and scale_detached and context_detached and all(target_checks.values())
        if initial:passed=passed and initial_error<1e-7 and gradients['route_to_router']>0 and gradients['pred_to_T']>0 and gradients['pred_to_ST']>0
        return dict(PASS=passed,initial=initial,finite=finite,shapes=shapes,shape_PASS=shape_ok,checks=checks,causal_prefix10=causal,
            initial_routing_error=initial_error,utility_target_checks=target_checks,q_detached=q_detached,scale_detached=scale_detached,
            regime_context_detached=context_detached,gradient_audit=gradients,non_router_auxiliary_gradient_paths=leaks,
            initial_route_diagnostics=diag,gradient_note='Router final layer receives gradient; preceding Router layers may initially have zero gradient due to zero final weight. Detaching candidates for utility-only router evaluation prevents ALL auxiliary gradients into experts/heads/branches.',
            causality_scope='Observed window only; temporal prefix10 E/H/p/prior/Z and balanced readouts unchanged by future-window perturbation.')
    finally:
        model.eval()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
        assert all(p.grad is None for p in model.parameters())
