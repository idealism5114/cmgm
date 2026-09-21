import copy
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.candidate_moe_fusion import CandidateAwareTwoExpertFusion
from cmgm.scripts.d0b_candidate_moe_audit import initialization,sanity,make_model
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


def test_shared_initialization_structural_sanity():
    model,a=initialization(DATA,torch.device('cpu'))
    assert a['PASS'] and a['shared_max_abs_diff']==0 and a['new_parameter_count']==37570
    assert not hasattr(model,'gate_fc') and not hasattr(model,'moe_fusion')
    assert not model.attn_mixhop1.qk_norm and model.attn_mixhop1.graph_prior_heads is None
    c=sanity(model,torch.randn(2,20,30,21),torch.randn(2,4,24)*.02)
    assert c['PASS'],c
    assert c['initial_routing_error']==0 and c['gradient_norms']['RouterFirst']==0


def test_candidate_wiring_router_norm_only_and_learning_connectivity():
    seed_all(42);f=CandidateAwareTwoExpertFusion().eval();s=torch.randn(4,64);t=torch.randn(4,64)*3
    out=f(s,t);c=f.last
    torch.testing.assert_close(c['pi'],torch.full((4,2),.5),rtol=0,atol=0)
    torch.testing.assert_close(out,(c['e_T']+c['e_ST'])/2,rtol=0,atol=0)
    assert not torch.allclose(out,(c['u_T']+c['u_ST'])/2)
    expected=torch.cat([c['u_T'],c['u_ST'],(c['u_T']-c['u_ST']).abs(),c['u_T']*c['u_ST']],-1)
    torch.testing.assert_close(c['router_input'],expected,rtol=0,atol=0)
    out.square().mean().backward()
    for mod in (f.temporal_expert,f.interaction_expert,f.router):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())
        assert sum(p.grad.abs().sum() for p in mod.parameters())>0
    assert f.router.network[0].weight.grad.abs().max()==0
    assert f.router.network[-1].weight.grad.abs().max()>0
    # Synthetic single-step final-layer update: no persistent model or formal fit.
    with torch.no_grad():
        for p in f.router.network[-1].parameters():p.sub_(.01*p.grad)
    f(s,t)
    assert (f.last['pi']-.5).abs().max()>0
    assert (f.last['pi'][0]-f.last['pi'][1]).abs().max()>0
    assert not hasattr(f,'epoch') and not hasattr(f,'balance_loss')


def test_training_loss_exactly_prediction_plus_switch(monkeypatch):
    from cmgm.training.train import train_epoch,validate_epoch
    seed_all(42);m=make_model(DATA);m.switching_latent_transformer.set_epoch(20)
    for mod in m.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0.
    def forbidden(*args):raise AssertionError('Candidate must not use old MoE auxiliary loss/warm-up')
    m.moe_balance_loss=forbidden;m.set_moe_epoch=forbidden
    x,y=torch.randn(2,20,30,21),torch.randn(2,4,24)*.02
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    p=m(x);pl=prediction_loss(p,y);sl=m.switching_latent_transformer.switch_loss()
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    edge=torch.empty((2,0),dtype=torch.long);ew=torch.empty(0)
    actual=train_epoch(m,loader,edge,ew,opt,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert actual==pytest.approx(float((pl+sl).detach()),abs=1e-9)
    val=validate_epoch(m,loader,edge,ew,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert val==pytest.approx(float(pl.detach()),abs=1e-9)
    assert 'std_pi_ST' in m._last_train_candidate and 'routing_entropy' in m._last_val_candidate
    assert 'balance_loss' not in m._last_train_candidate


def test_selection_and_checkpoint_roundtrip(tmp_path,monkeypatch):
    import cmgm.training.train as tr
    m=make_model(DATA);values=iter([.4,.2,.3])
    def train_epoch(model,*args,**kw):model._last_train_candidate={'mean_pi_T':.5};return .5
    def validate(model,*args,**kw):
        model._last_val_candidate={'mean_pi_T':.5}
        model._last_val5_diagnostic=dict(MAE=.03,MSE=.002,count=1)
        return next(values)
    def forbidden(*args):raise AssertionError('No candidate warm-up')
    m.set_moe_epoch=forbidden
    monkeypatch.setattr(tr,'train_epoch',train_epoch);monkeypatch.setattr(tr,'validate_epoch',validate)
    path=tmp_path/'candidate.pt'
    h=tr.train(m,None,None,torch.empty((2,0),dtype=torch.long),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(path))
    assert h['best_epoch']==2 and len(h['candidate_routing_history'])==3
    cp=torch.load(path,weights_only=False);assert 'moe_epoch' not in cp
    other=make_model(DATA);other.load_state_dict(cp['model_state_dict'],strict=True)
    m.eval();other.eval();x=torch.randn(2,20,30,21)
    torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)


def test_expost_utility_sign_correlation_and_oracle():
    from cmgm.scripts.d0b_candidate_moe_diagnostics import expert_utility
    target=np.zeros((6,4,24));t=np.ones_like(target)*.04;st=np.ones_like(target)*.04
    t[:3]=0.;st[3:]=0.;p=np.linspace(.1,.9,6);pi=np.stack([1-p,p],1)
    formal=(1-p[:,None,None])*t+p[:,None,None]*st
    utility,oracle=expert_utility(pi,t,st,formal,target)
    assert utility['correlations']['Spearman']>0
    assert np.asarray(utility['per_origin']['A_ST'])[:3].max()<0
    assert np.asarray(utility['per_origin']['A_ST'])[3:].min()>0
    assert oracle['metrics']['MAE']==0 and oracle['uses_TEST_labels'] and not oracle['deployable']
    assert [g['count'] for g in utility['quantile_groups']]==[2,2,2]
    assert utility['quantile_groups'][-1]['fraction_A_ST_positive']==1


def test_constant_routing_is_not_invalid_and_correlations_undefined():
    from cmgm.scripts.d0b_candidate_moe_diagnostics import expert_utility
    pi=np.full((6,2),.5);y=np.zeros((6,4,24))
    u,o=expert_utility(pi,y,y,y,y)
    assert u['correlations']['Spearman'] is None and u['correlations']['Pearson'] is None
    assert u['unique_pi_ST']==1 and o['metrics']['MAE']==0
    json.dumps(u,allow_nan=False)


def test_formal_predictions_precede_expost_and_report(tmp_path,monkeypatch):
    import cmgm.scripts.d0b_candidate_moe_diagnostics as d
    from cmgm.scripts.d0b_candidate_2expert_moe import PROTOCOL
    from cmgm.scripts.d0b_candidate_moe_report import report
    seed_all(42);m=make_model(DATA);before=copy.deepcopy(m.state_dict())
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x,y),batch_size=2) for s in ('train','val','test')})
    original=d.expert_utility
    def checked(*args):
        assert (tmp_path/'formal_predictions.npz').exists()
        return original(*args)
    monkeypatch.setattr(d,'expert_utility',checked)
    ev=d.formal_evaluation(m,data,torch.device('cpu'),tmp_path)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert set(ev['metrics']['test'])=={'1','5','10','20'}
    for metric in ev['metrics']['test'].values():assert abs(metric['RMSE']**2-metric['MSE'])<1e-12
    controls={n:dict(metrics=ev['metrics'],routing={}) for n in ('TemporalOnly','Fixed Equal Fusion','Adaptive Gated Fusion / Full D0B','3-Expert Dense MoE')}
    r=dict(config=PROTOCOL,evaluation=ev,controls=controls,source_hashes={},initialization={},sanity={'PASS':True},data={},status='TEST FIXTURE')
    report(r,tmp_path)
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==18 and 'Q12.' in text
    assert len((tmp_path/'fusion_comparison.csv').read_text().splitlines())==6
    assert 'oracle' not in (tmp_path/'fusion_comparison.csv').read_text().lower()
