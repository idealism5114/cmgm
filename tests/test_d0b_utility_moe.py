import copy
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.utility_routed_moe import UtilityAwareRouter,utility_target,per_sample_huber
from cmgm.scripts.d0b_utility_moe_audit import make_model,initialization,sanity
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


def test_initialization_and_structural_audit():
    m,a=initialization(DATA,torch.device('cpu'))
    assert a['PASS'] and a['new_parameter_count']==48418 and a['delta_parameters']==40162
    assert a['head_init_max_diff']==0 and a['heads_independent']
    c=sanity(m,torch.randn(2,20,30,21),torch.randn(2,4,24)*.02)
    assert c['PASS'],c
    assert c['q_detached'] and not c['non_router_auxiliary_gradient_paths']
    assert c['gradient_audit']['route_to_router']>0


def test_prediction_level_mixture_and_independent_heads():
    seed_all(42);m=make_model(DATA).eval();x=torch.randn(2,20,30,21)
    p=m(x)
    torch.testing.assert_close(p,.5*m._last_pred_T+.5*m._last_pred_ST,rtol=0,atol=0)
    before=copy.deepcopy(m.head.state_dict())
    with torch.no_grad():m.interaction_head[-1].bias.add_(.2)
    changed=m(x)
    torch.testing.assert_close(changed-p,torch.full_like(p,.1),atol=5e-8,rtol=0)
    assert all(torch.equal(v,m.head.state_dict()[k]) for k,v in before.items())


def test_soft_target_four_horizons_detached_and_zeros():
    target=torch.zeros(2,4,24);p=torch.zeros_like(target,requires_grad=True)
    q,scale,lt,lst=utility_target(p,p,target)
    assert torch.equal(q,torch.full((2,2),.5)) and scale==0
    assert not q.requires_grad and not scale.requires_grad
    bad=torch.zeros_like(target);bad[:,3]=.1  # Only 20d changes: utility must not be 5d-only.
    q,scale,lt,lst=utility_target(bad,p,target)
    assert (q[:,1]>.5).all() and (lt>lst).all()
    expected=sum(torch.nn.functional.huber_loss(bad[:,h],target[:,h],delta=.02) for h in range(4))
    assert lt.mean()==expected
    q2,*_=utility_target(p,bad,target);torch.testing.assert_close(q2,q.flip(-1))


def test_router_context_detached_even_after_nonzero_final_initialization():
    router=UtilityAwareRouter()
    with torch.no_grad():router.network[-1].weight.normal_(std=.02)
    e1=torch.randn(3,64,requires_grad=True);e2=torch.randn(3,64,requires_grad=True)
    p=torch.softmax(torch.randn(3,3),-1).requires_grad_();prior=torch.softmax(torch.randn(3,3),-1).requires_grad_()
    pi=router(e1,e2,p,prior);grads=torch.autograd.grad(pi[:,1].sum(),(e1,e2,p,prior),allow_unused=True)
    assert grads[0].abs().sum()>0 and grads[1].abs().sum()>0
    assert grads[2] is None and grads[3] is None
    assert router.last['router_input'].shape==(3,263)


def test_router_loss_isolation_not_hidden_by_zero_initialization():
    seed_all(42);m=make_model(DATA).train()
    with torch.no_grad():m.utility_moe_fusion.router.network[-1].weight.normal_(std=.03)
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02
    pred=m(x);primary=prediction_loss(pred,y);route,d=m.utility_router_loss(y)
    torch.testing.assert_close(m._last_route_pi,m._last_utility_pi,atol=0,rtol=0)
    params=list(m.named_parameters());gr=torch.autograd.grad(route,[p for _,p in params],retain_graph=True,allow_unused=True)
    assert all(g is None for (n,_),g in zip(params,gr) if not n.startswith('utility_moe_fusion.router.'))
    assert sum(g.abs().sum() for g in gr if g is not None)>0
    gp=torch.autograd.grad(primary,[p for _,p in params],allow_unused=True)
    for prefix in ('head.','interaction_head.','utility_moe_fusion.temporal_expert.','utility_moe_fusion.interaction_expert.','utility_moe_fusion.router.'):
        assert sum(g.abs().sum() for (n,_),g in zip(params,gp) if n.startswith(prefix) and g is not None)>0
    q=m._last_utility_q;scale=m._last_route_scale
    expected=(q*((q+1e-8).log()-(m._last_utility_pi+1e-8).log())).sum(-1).mean()*scale
    assert route.item()==pytest.approx(expected.item(),abs=1e-10)
    assert not q.requires_grad and not scale.requires_grad


def test_training_loss_and_validation_boundary(monkeypatch):
    from cmgm.training.train import train_epoch,validate_epoch
    seed_all(42);m=make_model(DATA);m.switching_latent_transformer.set_epoch(20)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    x,y=torch.randn(2,20,30,21),torch.randn(2,4,24)*.02
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    p=m(x);pl=prediction_loss(p,y);route,_=m.utility_router_loss(y)
    expected=float((pl+m.switching_latent_transformer.switch_loss()+route).detach())
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    edge=torch.empty((2,0),dtype=torch.long);weight=torch.empty(0)
    value=train_epoch(m,loader,edge,weight,opt,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert value==pytest.approx(expected,abs=1e-9)
    assert 'std_q_ST' in m._last_train_utility and 'scaled_route_loss' in m._last_train_utility
    def forbidden(*args):raise AssertionError('VAL must not compute router supervision')
    m.utility_router_loss=forbidden
    val=validate_epoch(m,loader,edge,weight,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert val==pytest.approx(float(pl.detach()),abs=1e-9)
    assert 'P90_pi_ST' in m._last_val_utility and 'mean_q_ST' not in m._last_val_utility


def test_eval_guard_and_nonfinite_stop():
    m=make_model(DATA);x=torch.randn(2,20,30,21);y=torch.zeros(2,4,24)
    m(x);m._last_pred_T=m._last_pred_T*float('nan')
    with pytest.raises(FloatingPointError):m.utility_router_loss(y)
    m.eval();m(x)
    with pytest.raises(ValueError,match='TRAIN-only'):m.utility_router_loss(y)


def test_oracle_primary_criterion_is_four_horizon_huber():
    from cmgm.scripts.d0b_utility_moe_diagnostics import analyze_experts
    y=np.zeros((6,4,24));t=np.zeros_like(y);st=np.zeros_like(y)
    t[:3,3]=.2;st[:3,1]=.02  # ST wins sum Huber even though T wins 5d MAE.
    st[3:,3]=.2;t[3:,1]=.02
    pi=np.stack([np.linspace(.1,.9,6),np.linspace(.9,.1,6)],-1)
    p=pi[:,0,None,None]*t+pi[:,1,None,None]*st
    a,o=analyze_experts(pi,t,st,p,y)
    assert a['fraction_T_better']==.5 and a['fraction_ST_better']==.5
    assert o['fraction_temporal']==.5 and o['metrics']['5']['MAE']==pytest.approx(.02)
    assert 'four-horizon' in o['selection'] and not o['deployable']
    assert a['primary_correlations']['Spearman']>0


def test_fixed_train_mean_no_labels_and_report(tmp_path,monkeypatch):
    import cmgm.scripts.d0b_utility_moe_diagnostics as d
    from cmgm.scripts.d0b_utility_routed_moe import PROTOCOL
    from cmgm.scripts.d0b_utility_moe_report import report
    seed_all(42);m=make_model(DATA).eval()
    with torch.no_grad():m.utility_moe_fusion.router.network[-1].weight.normal_(std=.02)
    before=copy.deepcopy(m.state_dict())
    x,y=torch.randn(4,20,30,21),torch.randn(4,4,24)*.02
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x*(i+1),y),batch_size=2) for i,s in enumerate(('train','val','test'))})
    original=d.analyze_experts
    def checked(*args):
        assert (tmp_path/'formal_predictions.npz').exists()
        return original(*args)
    monkeypatch.setattr(d,'analyze_experts',checked)
    ev=d.formal_evaluation(m,data,torch.device('cpu'),tmp_path)
    arrays=np.load(tmp_path/'formal_predictions.npz')
    expected=arrays['train_pi'].astype(np.float64).mean(0).astype(np.float32)
    np.testing.assert_array_equal(expected,ev['fixed_router']['weights'])
    np.testing.assert_array_equal(arrays['test_fixed_prediction'],expected[0]*arrays['test_pred_T']+expected[1]*arrays['test_pred_ST'])
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert set(ev['expert_metrics']['Temporal'])=={'1','5','10','20'}
    for metric in ev['metrics']['test'].values():assert abs(metric['RMSE']**2-metric['MSE'])<1e-12
    labels=('TemporalOnly','Adaptive Gated Fusion / Full D0B','3-Expert Dense MoE','Candidate-Aware 2-Expert Representation MoE')
    r=dict(config=PROTOCOL,evaluation=ev,controls={n:dict(metrics=ev['metrics']) for n in labels},source_hashes={},initialization={},sanity={},data={},status='TEST FIXTURE')
    report(r,tmp_path)
    text=(tmp_path/'FINAL_REPORT.md').read_text();assert text.count('\n## ')==22 and 'Q12.' in text
    comparison=(tmp_path/'fusion_comparison.csv').read_text()
    assert len(comparison.splitlines())==6 and 'oracle' not in comparison.lower()
    for path in tmp_path.glob('*.json'):json.loads(path.read_text())
