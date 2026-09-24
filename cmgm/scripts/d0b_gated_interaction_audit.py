"""Synthetic/read-only checks for temporal base plus gated interaction correction."""
import copy
import torch
from cmgm.models.candidate_moe_fusion import VARIANT as ORIGINAL
from cmgm.models.gated_interaction_residual import VARIANT, CandidateGatedInteractionResidual
from cmgm.scripts.d0b_moe_audit import make_model as native_model
from cmgm.scripts.baseline_protocol import seed_all, prediction_loss


def make_model(data, seed=42, variant=VARIANT):
    seed_all(seed)
    return native_model(data, variant)


def counts(m):
    count = lambda x: sum(p.numel() for p in x.parameters())
    f = m.candidate_moe_fusion
    correction = count(f.interaction_correction) if hasattr(f, 'interaction_correction') else count(f.interaction_expert)
    return dict(total=count(m), temporal_expert=count(f.temporal_expert), interaction=correction,
                temporal_plus_interaction=count(f.temporal_expert)+correction, router=count(f.router))


def initialization(data, seed):
    old, new = make_model(data, seed, ORIGINAL), make_model(data, seed)
    a, b = dict(old.named_parameters()), dict(new.named_parameters())
    shared = sorted(n for n in a if not n.startswith('candidate_moe_fusion.interaction_expert.'))
    diffs = {n: float((a[n]-b[n]).detach().abs().max()) for n in shared}
    mismatch = [n for n, v in diffs.items() if v != 0]
    before, after = counts(old), counts(new)
    r = dict(original=before, new=after, delta=after['total']-before['total'],
             shared_parameter_count=sum(a[n].numel() for n in shared), shared_max_abs_diff=max(diffs.values()),
             mismatch_count=len(mismatch), mismatched_parameters=mismatch, differences=diffs,
             removed_parameters=sorted(a.keys()-b.keys()), added_parameters=sorted(b.keys()-a.keys()),
             seed=seed, initialization='Complete original Candidate initialization first; replace only ST expert, no fitted weights')
    r['PASS'] = (not mismatch and after['temporal_expert']==8320 and after['interaction']==12288
                 and after['router']==16834 and r['delta']==-128
                 and all(n.startswith('candidate_moe_fusion.interaction_correction.') for n in r['added_parameters'])
                 and not hasattr(new.candidate_moe_fusion, 'interaction_expert'))
    if not r['PASS']:
        raise ValueError(f'Initialization audit failed: {r}')
    return new, r


def controlled_gradients():
    """A disposable nondegenerate synthetic fixture, never the training model."""
    with torch.random.fork_rng(devices=[]):
        f = CandidateGatedInteractionResidual().eval()
        with torch.no_grad():
            for layer in (f.interaction_correction.spatial, f.interaction_correction.temporal):
                layer.weight.copy_(torch.eye(64))
            f.interaction_correction.output.weight.copy_(torch.eye(64)*.1)
        s = (1+torch.rand(3,64)).requires_grad_()
        t = (1+torch.rand(3,64)).requires_grad_()
        named = list(f.named_parameters())
        gradients = torch.autograd.grad(f(s,t).square().mean(), [s,t]+[p for _,p in named], allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        norms = {name: float(g.norm()) for name,g in zip(['projected_s','projected_t']+[n for n,_ in named],gradients)}
        for prefix in ('projected_s','projected_t','interaction_correction.spatial','interaction_correction.temporal',
                       'interaction_correction.output','temporal_expert','router.network.2'):
            assert sum(v for k,v in norms.items() if k.startswith(prefix)) > 0
        assert norms['router.network.0.weight']==0
        # Verify both live candidates connect to a nonzero-final-layer router.
        with torch.no_grad():
            f.router.network[-1].weight.normal_(0,.01)
        base = f.temporal_expert(t)
        enhanced = base+f.interaction_correction(s,t)
        pi = f.router(base,enhanced)
        route_grads = torch.autograd.grad(pi[:,1].sum(), [base,enhanced], retain_graph=False)
        assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in route_grads)
        return dict(PASS=True, norms=norms, router_to_candidates=[float(g.norm()) for g in route_grads],
                    fixture='Positive projected inputs, identity interaction inputs; disposable copy only',
                    initial_router_first_zero_is_expected=True)


def sanity(model, x, y):
    state = {k:v.clone() for k,v in model.state_dict().items()}
    model.eval(); f=model.candidate_moe_fusion; branch=model.switching_latent_transformer
    err=lambda a,b:float((a-b).abs().max())
    calls=[]
    hook=f.temporal_expert.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad():
        p=model(x);c={k:v.clone() for k,v in f.last.items()}
    hook.remove()
    with torch.no_grad():
        u=f.router.temporal_norm(c['e_T']);v=f.router.interaction_norm(c['e_ST'])
        checks=dict(single_base_call=len(calls)==1,
                    no_old_ST=not hasattr(f,'interaction_expert'),
                    bias_free=all(m.bias is None for m in f.interaction_correction.modules() if isinstance(m,torch.nn.Linear)))
        errors=dict(delta_identity=err(c['e_ST']-c['e_T'],c['delta_ST']),
                    shared_base=err(c['e_T'],c['e_base']),
                    mixture=err(c['h_moe'],(c['pi'][...,None]*c['experts']).sum(1)),
                    residual_identity=err(c['h_moe'],c['e_base']+c['pi'][:,1:2]*c['delta_ST']),
                    head=err(p,model.head(c['h_moe']).reshape_as(p)),
                    router_input=err(c['router_input'],torch.cat([u,v,(u-v).abs(),u*v],-1)),
                    normalization=err(c['pi'].sum(-1),torch.ones(len(x),device=x.device)),
                    initial_pi=err(c['pi'],torch.full_like(c['pi'],.5)))
        for mode in (False,True):
            f.interaction_correction.train(mode)
            errors[f'zero_spatial_train{mode}']=float(f.interaction_correction(torch.zeros_like(c['s']),c['t']).abs().max())
            errors[f'zero_temporal_train{mode}']=float(f.interaction_correction(c['s'],torch.zeros_like(c['t'])).abs().max())
        model.eval()
        perm=torch.arange(len(x)-1,-1,-1,device=x.device)
        errors['batch_prediction']=err(model(x[perm]),p[perm])
        errors['batch_pi']=err(f.last['pi'],c['pi'][perm])
        errors['single_prediction']=err(model(x[:1]),p[:1])
        for k in ('pi','e_base','delta_ST'):
            errors['single_'+k]=err(f.last[k],c[k][:1])
        model(x)
        states={k:getattr(branch,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
        changed=x.clone();changed[:,10:]=changed[:,10:]*-3+7;model(changed)
        causal={k:err(v[:,:10],getattr(branch,k)[:,:10]) for k,v in states.items()}
    model.train()
    pred=model(x);loss=prediction_loss(pred,y)+branch.switch_loss()
    named=list(model.named_parameters());grads=torch.autograd.grad(loss,[v for _,v in named],allow_unused=True)
    groups={prefix:sum(float(g.detach().square().sum()) for (n,_),g in zip(named,grads) if n.startswith(prefix) and g is not None)**.5
            for prefix in ('head','candidate_moe_fusion.temporal_expert','candidate_moe_fusion.interaction_correction','candidate_moe_fusion.router')}
    gradient_connected=all(g is not None and torch.isfinite(g).all() for (n,_),g in zip(named,grads)
                           if n.startswith(('head','candidate_moe_fusion.')))
    finite=bool(torch.isfinite(loss) and torch.isfinite(pred).all())
    r=dict(shapes={**{k:list(v.shape) for k,v in c.items()},'prediction':list(p.shape)},checks=checks,
           errors=errors,causal_prefix=causal,finite=finite,gradient_connected=bool(gradient_connected),gradient_norms=groups,
           controlled_gradient_check=controlled_gradients(),
           zero_boundary_scope='Projected s/t only; not deletion of whole branches')
    r['PASS']=(finite and gradient_connected and all(checks.values()) and p.shape==(len(x),4,24)
               and c['pi'].shape==(len(x),2) and max(errors.values())<1e-6 and max(causal.values())<1e-6
               and all(v>0 for v in groups.values()))
    model.eval()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all(v.grad is None for v in model.parameters())
    if not r['PASS']:
        raise ValueError(f'Structural sanity failed: {r}')
    return r
