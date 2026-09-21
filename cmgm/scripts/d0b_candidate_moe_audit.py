"""Read-only structural checks, including zero-initialized candidate routing."""
import torch
from cmgm.models.candidate_moe_fusion import VARIANT
from cmgm.scripts.d0b_moe_audit import make_model as _make_model,BASE
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss


def make_model(data):
    return _make_model(data,VARIANT)


def initialization(data,device):
    seed_all(42);base=_make_model(data,BASE).to(device)
    seed_all(42);model=make_model(data).to(device)
    a=dict(base.named_parameters());b=dict(model.named_parameters())
    shared=sorted(a.keys() & b.keys());added=sorted(b.keys()-a.keys());removed=sorted(a.keys()-b.keys())
    diffs={n:float((a[n]-b[n]).detach().abs().max()) for n in shared}
    r=dict(shared_parameter_count=sum(b[n].numel() for n in shared),new_parameter_count=sum(b[n].numel() for n in added),
           shared_max_abs_diff=max(diffs.values()),mismatch_count=sum(v!=0 for v in diffs.values()),
           baseline_parameters=sum(p.numel() for p in a.values()),candidate_parameters=sum(p.numel() for p in b.values()),
           shared_differences=diffs,new_parameters=added,removed_parameters=removed)
    r['delta_parameters']=r['candidate_parameters']-r['baseline_parameters']
    r['PASS']=r['shared_max_abs_diff']==0 and r['new_parameter_count']==37570 and r['delta_parameters']==29314 and removed==['gate_fc.bias','gate_fc.weight']
    return model,r


def sanity(model,x,y,initial=True):
    model.eval();f=model.candidate_moe_fusion;b=model.switching_latent_transformer
    state={k:v.clone() for k,v in model.state_dict().items()}
    try:
        with torch.no_grad():
            pred=model(x);c={k:v.clone() for k,v in f.last.items()}
            u=f.router.temporal_norm(c['e_T']);v=f.router.interaction_norm(c['e_ST'])
            ri=torch.cat([u,v,(u-v).abs(),u*v],-1)
            expected=(c['pi'][...,None]*torch.stack([c['e_T'],c['e_ST']],1)).sum(1)
            err=lambda a,b:float((a-b).abs().max())
            checks=dict(raw_fusion=err(c['h_moe'],expected),router_input=err(c['router_input'],ri),
                router_logits=err(c['router_logits'],f.router.network(ri)),pi=err(c['pi'],c['router_logits'].softmax(-1)),
                u_T=err(c['u_T'],u),u_ST=err(c['u_ST'],v),abs_diff=err(c['abs_diff'],(u-v).abs()),product=err(c['product'],u*v),
                e_T=err(c['e_T'],f.temporal_expert(c['t'])),e_ST=err(c['e_ST'],f.interaction_expert(torch.cat([c['s'],c['t']],-1))),
                s=err(c['s'],model.gcn_proj(c['h_s'])),t=err(c['t'],model.lstm_proj(c['h_t'])),
                shared_head=err(pred,model.head(c['h_moe']).reshape_as(pred)),normalization=err(c['pi'].sum(-1),torch.ones(len(x),device=x.device)))
            perm=torch.arange(len(x)-1,-1,-1,device=x.device);pp=model(x[perm])
            checks['batch_prediction']=err(pp,pred[perm]);checks['batch_pi']=err(f.last['pi'],c['pi'][perm])
            single=model(x[:1]);checks['single_prediction']=err(single,pred[:1])
            for key in ('pi','e_T','e_ST'):checks['single_'+key]=err(f.last[key],c[key][:1])
            model(x);states={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
            future=x.clone();future[:,10:]=future[:,10:]*-3+7;model(future)
            causal={k:err(v[:,:10],getattr(b,k)[:,:10]) for k,v in states.items()}
            H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
            h,z=b.last_long_memory[:,:10],b.last_latent_states[:,:10]
            old=b.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]));long=b.last_h_long.clone();micro=b.last_h_micro.clone()
            new=b.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]))
            causal.update(h_temporal=err(old,new),h_long=err(long,b.last_h_long),h_micro=err(micro,b.last_h_micro))
            shape_expected={key:[len(x),64] for key in ('h_t','h_s','t','s','e_T','e_ST','u_T','u_ST','abs_diff','product','h_moe')}
            shape_expected.update(router_input=[len(x),256],router_logits=[len(x),2],pi=[len(x),2],experts=[len(x),2,64])
            shapes={k:list(v.shape) for k,v in c.items()}
            shapes['prediction']=list(pred.shape)
            shape_ok=all(shapes[k]==v for k,v in shape_expected.items()) and pred.shape==(len(x),4,24)
            initial_error=err(c['pi'],torch.full_like(c['pi'],.5))
            zero_final=bool((f.router.network[-1].weight==0).all() and (f.router.network[-1].bias==0).all())
        # Training-mode connectivity, no optimizer step or .grad mutation.
        model.train();p=model(x);loss=prediction_loss(p,y)+b.switch_loss()
        named=list(model.named_parameters());grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
        groups={'Temporal':'candidate_moe_fusion.temporal_expert','Interaction':'candidate_moe_fusion.interaction_expert',
                'Router':'candidate_moe_fusion.router','SharedHead':'head',
                'RouterFinal':'candidate_moe_fusion.router.network.2','RouterFirst':'candidate_moe_fusion.router.network.0'}
        norms={label:sum(float(g.detach().double().square().sum()) for (n,_),g in zip(named,grads) if n.startswith(prefix) and g is not None)**.5 for label,prefix in groups.items()}
        connected={label:all(g is not None for (n,_),g in zip(named,grads) if n.startswith(prefix)) for label,prefix in groups.items()}
        finite=all(g is None or bool(torch.isfinite(g).all()) for g in grads) and all(bool(torch.isfinite(v).all()) for v in c.values()) and bool(torch.isfinite(pred).all())
        passed=finite and shape_ok and max([*checks.values(),*causal.values()])<1e-6 and all(connected.values())
        if initial:passed=passed and initial_error<1e-7 and zero_final and all(norms[k]>0 for k in ('Temporal','Interaction','Router','SharedHead','RouterFinal'))
        return dict(PASS=passed,initial=initial,finite=finite,shapes=shapes,shape_PASS=shape_ok,
                    formula_and_batch_errors=checks,causal_prefix10=causal,initial_routing_error=initial_error,
                    router_final_zero=zero_final,gradient_norms=norms,gradient_connected=connected,
                    gradient_note='Initial RouterFirst and router LayerNorm gradients are zero by chain rule through zero RouterFinal; RouterFinal must have nonzero gradient.',
                    causality_scope='E/H/p/Z and balanced readouts prefix10; spatial/fusion use the legal complete observed window only.')
    finally:
        model.eval()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
        assert all(p.grad is None for p in model.parameters())
