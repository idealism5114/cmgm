"""Synthetic tests only; no formal training, real market data or TEST evaluation."""
import copy
import importlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.gated_interaction_residual import CandidateGatedInteractionResidual, VARIANT
from cmgm.scripts import d0b_candidate_gated_interaction as run
from cmgm.scripts.d0b_gated_interaction_audit import initialization,make_model,sanity,controlled_gradients
from cmgm.scripts.baseline_protocol import prediction_loss

DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('seed',[42,7])
def test_shared_initialization(seed):
    model,audit=initialization(DATA,seed)
    assert audit['PASS'] and audit['delta']==-128 and audit['mismatch_count']==0
    assert audit['shared_max_abs_diff']==0 and audit['new']['temporal_plus_interaction']==20608
    assert not hasattr(model.candidate_moe_fusion,'interaction_expert')
    assert len(audit['added_parameters'])==3
    assert model.candidate_moe_fusion.interaction_correction.output.weight.abs().sum()>0


def test_full_structural_sanity():
    model=make_model(DATA)
    r=sanity(model,torch.randn(2,20,30,21),torch.randn(2,4,24)*.02)
    assert r['PASS'] and r['checks']['single_base_call']
    assert r['errors']['initial_pi']==0
    assert max(r['causal_prefix'].values())<1e-6


def test_single_shared_base_even_with_dropout_and_zero_boundaries():
    f=CandidateGatedInteractionResidual().train()
    s,t=torch.randn(5,64),torch.randn(5,64)
    seen=[];hook=f.temporal_expert.register_forward_hook(lambda m,args,out:seen.append(out))
    h=f(s,t);hook.remove();c=f.last
    assert len(seen)==1
    torch.testing.assert_close(c['e_T'],seen[0],rtol=0,atol=0)
    torch.testing.assert_close(c['e_ST'],seen[0]+c['delta_ST'],rtol=0,atol=0)
    torch.testing.assert_close(h,c['e_base']+c['pi'][:,1:2]*c['delta_ST'],rtol=1e-6,atol=1e-6)
    for training in (True,False):
        f.train(training)
        assert torch.count_nonzero(f.interaction_correction(torch.zeros_like(s),t))==0
        assert torch.count_nonzero(f.interaction_correction(s,torch.zeros_like(t)))==0
    assert controlled_gradients()['PASS']


def test_loss_switch_epoch_and_no_auxiliary(monkeypatch):
    tr=importlib.import_module('cmgm.training.train');m=make_model(DATA)
    b=tr._switching_branch(m)
    assert b is m.switching_latent_transformer
    assert b.set_epoch(1)==0 and b.set_epoch(20)==.0005
    assert b.set_epoch(10)==pytest.approx(.0005*9/19)
    b.set_epoch(20)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    def forbidden(*a,**kw):raise AssertionError('Unexpected MoE/utility objective or warmup')
    m.moe_balance_loss=forbidden;m.set_moe_epoch=forbidden;m.utility_router_loss=forbidden
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02;y[-1]+=.2
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    pl=[];kl=[]
    for a,target in loader:
        pl.append(float(prediction_loss(m(a),target).detach()));kl.append(float(b.switch_loss().detach()))
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    edge=torch.empty((2,0),dtype=torch.long);ew=torch.empty(0);criterion=torch.nn.HuberLoss(delta=.02)
    actual=tr.train_epoch(m,loader,edge,ew,opt,criterion,torch.device('cpu'))
    assert actual==pytest.approx(np.mean(pl)+np.mean(kl),abs=1e-8)
    actual=tr.validate_epoch(m,loader,edge,ew,criterion,torch.device('cpu'))
    assert actual==pytest.approx(np.mean(pl),abs=1e-8)
    assert abs(actual-(2*pl[0]+pl[1])/3)>1e-5
    assert 'mean_pi_ST' in m._last_train_candidate and 'mean_pi_ST' in m._last_val_candidate


def test_checkpoint_selection_and_strict_roundtrip(tmp_path,monkeypatch):
    tr=importlib.import_module('cmgm.training.train');m=make_model(DATA)
    vals=iter([.4,.2,.3]);val5=iter([.01,.02,.001])
    def fake_train(model,*a,**kw):model._last_train_candidate={'mean_pi_T':.5};return .5
    def fake_val(model,*a,**kw):
        model._last_val_candidate={'mean_pi_T':.5};model._last_val5_diagnostic={'MAE':next(val5),'MSE':.01,'count':1}
        return next(vals)
    monkeypatch.setattr(tr,'train_epoch',fake_train);monkeypatch.setattr(tr,'validate_epoch',fake_val)
    path=tmp_path/'synthetic.pt'
    h=tr.train(m,None,None,torch.empty((2,0),dtype=torch.long),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(path))
    assert h['best_epoch']==2 and h['best_val5_epoch']==3
    cp=torch.load(path,weights_only=False);assert 'moe_epoch' not in cp
    other=make_model(DATA);other.load_state_dict(cp['model_state_dict'],strict=True)
    other.switching_latent_transformer.set_epoch(cp['best_epoch'])
    assert other.switching_latent_transformer.regime_filter.current_beta==pytest.approx(.0005/19)
    m.eval();other.eval();x=torch.randn(2,20,30,21)
    torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)


@pytest.mark.parametrize('zero',[False,True])
def test_frozen_diagnostics_same_forward_and_no_mutation(tmp_path,zero):
    m=make_model(DATA).eval()
    if zero:
        with torch.no_grad():m.candidate_moe_fusion.interaction_correction.output.weight.zero_()
    state=copy.deepcopy(m.state_dict());calls=[];heads=[]
    def head_hook(*args):
        heads.append(1)
        if len(heads)>2:assert (tmp_path/'formal_train_val_predictions.npz').exists()
    hook=m.candidate_moe_fusion.temporal_expert.register_forward_hook(lambda *a:calls.append(1))
    hh=m.head.register_forward_hook(head_hook)
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x,y)) for s in ('train','val')})
    ev=run.evaluate_frozen(m,data,42,torch.device('cpu'),tmp_path)
    hook.remove();hh.remove()
    assert len(calls)==2 and len(heads)==4
    assert set(ev['formal'])=={'train','val'}
    assert ev['zero_correction']['same_eval_forward_base']
    for split in ('train','val'):
        assert set(ev['formal'][split]['metrics'])=={'1','5','10','20'}
        cos=ev['formal'][split]['correction']['cosine']
        if zero:
            assert cos['undefined_zero_norm_count']==3 and cos['distribution'] is None
            assert ev['formal'][split]['metrics']==ev['zero_correction']['results'][split]['metrics']
        else:assert cos['valid_count']==3
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
    assert all(p.grad is None for p in m.parameters())
    json.dumps(ev,allow_nan=False)


def test_default_never_reads_data_or_trains(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No real data or fit by default')
    monkeypatch.setattr(run.shared,'train_val_data',forbidden);monkeypatch.setattr(run,'train',forbidden)
    called=[];monkeypatch.setattr(run,'preflight',lambda seed,out:called.append((seed,out)))
    monkeypatch.setattr('sys.argv',['gated','--output',str(tmp_path/'preflight')]);run.main()
    assert called==[(42,tmp_path/'preflight')]


def test_reference_missing_or_wrong_variant_is_pending(tmp_path):
    r=run.reference_audit({},42,DATA,tmp_path/'missing.json',tmp_path/'missing.pt')
    assert r['status']=='PENDING' and 'FileNotFoundError' in r['reason']
    report=tmp_path/'bad.json';report.write_text(json.dumps({'config':{'variant':'global'}}))
    cp=tmp_path/'bad.pt';torch.save({'metadata':{'config':{},'variant':'global'},'history':{}},cp)
    r=run.reference_audit({},42,DATA,report,cp)
    assert r['status']=='PENDING'
