"""Synthetic preflight; no real datasets, updates or fitted initialization."""
import io
import torch
from cmgm.scripts.d0b_gated_interaction_audit import make_model as factory
from cmgm.scripts.baseline_protocol import prediction_loss
from cmgm.models.candidate_moe_fusion import VARIANT as ORIGINAL
from cmgm.models.four_source_utility_moe import VARIANT, utility_targets


def make_model(data,seed=42):return factory(data,seed,VARIANT)

def counts(m):
    count=lambda module:sum(p.numel() for p in module.parameters())
    f=m.four_source_moe
    return dict(total=count(m),shared_encoding=count(m)-count(f),experts=[count(e) for e in f.experts],
                all_experts=count(f.experts),router=count(f.router))


def initialization(data,seed=42):
    old,new=factory(data,seed,ORIGINAL),make_model(data,seed)
    a,b=dict(old.named_parameters()),dict(new.named_parameters())
    shared=[n for n in b if not n.startswith('four_source_moe.')]
    differences={n:float((a[n]-b[n]).detach().abs().max()) for n in shared}
    f=new.four_source_moe
    for e in f.experts[:3]:
        assert all(torch.equal(v,old.head.state_dict()[k]) for k,v in e.state_dict().items())
    pointers=[p.data_ptr() for e in f.experts for p in e.parameters()]
    assert len(pointers)==len(set(pointers))
    assert not hasattr(new,'head') and not hasattr(new.switching_latent_transformer,'state_readout')
    r=dict(parameters=counts(new),shared_parameter_count=sum(b[n].numel() for n in shared),
           shared_max_abs_diff=max(differences.values()),mismatched_parameters=[n for n,v in differences.items() if v],
           differences=differences,removed_parameters=sorted(a.keys()-b.keys()),
           expert_initialization='Three independent deepcopies of native shared head; Joint new Linear192,64 + copied final Linear; no fitted checkpoint',
           expert_parameter_independence=True)
    r['PASS']=not r['mismatched_parameters']
    assert r['PASS'];return new,r


def sanity(model,x,y):
    state={k:v.clone() for k,v in model.state_dict().items()};model.eval();f=model.four_source_moe
    with torch.no_grad():
        p=model(x);pi=f.pi.clone();preds=f.predictions.clone()
        assert p.shape==y.shape==(len(x),4,24)
        assert torch.isfinite(p).all() and (pi-.25).abs().max()<1e-7
        assert torch.equal(p,(pi[:,:,None,None]*preds).sum(1))
        perm=torch.arange(len(x)-1,-1,-1,device=x.device)
        batch_error=float((model(x[perm])-p[perm]).abs().max())
        single_error=float((model(x[:1])-p[:1]).abs().max())
        assert max(batch_error,single_error)<2e-5
        model(x);b=model.switching_latent_transformer
        cache={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
        future=x.clone();future[:,10:]+=5;model(future)
        causal=max(float((getattr(b,k)[:,:10]-v[:,:10]).abs().max()) for k,v in cache.items());assert causal<1e-6
    # Nonzero router fixture on this disposable audit model; restored below.
    model.train();torch.nn.init.normal_(f.router.network[-1].weight,0,.02)
    b.set_epoch(20);p=model(x)
    parameters=list(model.named_parameters())
    loss,diag=f.utility_loss(y)
    pi_error=float((diag['pi_aux']-f.pi.detach()).abs().max());assert pi_error<1e-7
    gradients=torch.autograd.grad(loss,[v for _,v in parameters],allow_unused=True,retain_graph=True)
    assert all(g is None for (n,_),g in zip(parameters,gradients) if not n.startswith('four_source_moe.router.'))
    assert all(g is not None and torch.isfinite(g).all() for (n,_),g in zip(parameters,gradients) if n.startswith('four_source_moe.router.'))
    assert sum(float(g.abs().sum()) for (n,_),g in zip(parameters,gradients) if n.startswith('four_source_moe.router.'))>0
    pred_loss=prediction_loss(p,y)
    live_grad=torch.autograd.grad(pred_loss,f.components,retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in live_grad)
    grads=torch.autograd.grad(pred_loss+b.switch_loss(),[v for _,v in parameters],allow_unused=True)
    unused=[n for (n,_),g in zip(parameters,grads) if g is None]
    assert not unused,unused
    assert all(torch.isfinite(g).all() for g in grads)
    for i in range(4):
        assert sum(float(g.abs().sum()) for (n,_),g in zip(parameters,grads) if n.startswith(f'four_source_moe.experts.{i}.'))>0
    zeros=torch.zeros_like(y);same=zeros[:,None].expand(-1,4,-1,-1)
    q,scale,error=utility_targets(same,zeros);assert torch.equal(q,torch.full_like(q,.25)) and scale==0
    unequal=same.clone();unequal[:,1:]=.1
    q,_,_=utility_targets(unequal,zeros);assert (q[:,0]>q[:,1]).all() and torch.allclose(q.sum(-1),torch.ones(len(x),device=x.device))
    warm=[b.set_epoch(e) for e in (1,10,20)];assert all(abs(a-c)<1e-15 for a,c in zip(warm,[0.,.0005*9/19,.0005]))
    model.load_state_dict(state,strict=True);b.set_epoch(1);model.eval()
    with torch.no_grad():before=model(x)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    model.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    with torch.no_grad():assert torch.equal(model(x),before)
    return dict(PASS=True,shape=list(p.shape),initial_uniform=True,expert_independence=True,
                shared_live_components=True,utility_router_only_nonzero_fixture=True,utility_target_checks=True,
                pi_aux_max_error=pi_error,unused_parameters=unused,batch_error=batch_error,single_error=single_error,
                causal_prefix_error=causal,checkpoint_exact=True,switch_beta=warm)
