"""Synthetic fixtures only; no formal fitting or real dataset access."""
import importlib
import copy
import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.four_source_utility_moe import utility_targets
from cmgm.scripts.d0b_four_source_audit import initialization,make_model,sanity
from cmgm.scripts import d0b_four_source_utility_moe as run
from cmgm.scripts.baseline_protocol import prediction_loss
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def test_initialization_and_structural_sanity():
    m,a=initialization(DATA);assert a['PASS'] and a['shared_max_abs_diff']==0
    assert a['parameters']['experts']==[10400,10400,10400,18592]
    assert a['parameters']['router']==12996
    assert sanity(m,torch.randn(3,20,30,21),torch.randn(3,4,24)*.02)['PASS']

@pytest.mark.parametrize('final_nonzero',[False,True])
def test_router_aux_gradient_isolation(final_nonzero):
    m=make_model(DATA).train();f=m.four_source_moe
    if final_nonzero:torch.nn.init.normal_(f.router.network[-1].weight,0,.05)
    m(torch.randn(3,20,30,21));y=torch.randn(3,4,24)*.02
    loss,d=f.utility_loss(y)
    assert not d['q'].requires_grad
    torch.testing.assert_close(d['pi_aux'],f.pi)
    named=list(m.named_parameters());g=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    for (n,_),v in zip(named,g):
        if 'four_source_moe.router.' in n:assert v is not None and torch.isfinite(v).all()
        else:assert v is None
    if final_nonzero:
        assert sum(float(v.abs().sum()) for (n,_),v in zip(named,g) if 'router.norms' in n)>0


def test_utility_formula():
    y=torch.randn(3,4,24)*.02;p=torch.randn(3,4,4,24)*.03
    q,scale,e=utility_targets(p,y)
    expected=torch.stack([torch.stack([prediction_loss(p[b:b+1,k],y[b:b+1]) for k in range(4)]) for b in range(3)])
    torch.testing.assert_close(e,expected)
    torch.testing.assert_close(q,(-expected/(expected.mean(-1,keepdim=True)+1e-8)).softmax(-1))
    assert scale==e.mean()
    z=torch.zeros_like(p);q,scale,_=utility_targets(z,torch.zeros_like(y));assert scale==0 and (q==.25).all()


def test_native_trainer_train_only_utility_batch_mean(monkeypatch):
    tr=importlib.import_module('cmgm.training.train');m=make_model(DATA);b=tr._switching_branch(m);b.set_epoch(20)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02;y[-1]+=.2
    loader=DataLoader(TensorDataset(x,y),batch_size=2);pl=[];kl=[];ul=[]
    for xx,yy in loader:
        p=m(xx);pl.append(float(prediction_loss(p,yy).detach()));kl.append(float(b.switch_loss().detach()));ul.append(float(m.four_source_moe.utility_loss(yy)[0].detach()))
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:None)
    edge=torch.empty(2,0,dtype=torch.long);ew=torch.empty(0);criterion=torch.nn.HuberLoss(delta=.02)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    actual=tr.train_epoch(m,loader,edge,ew,opt,criterion,torch.device('cpu'))
    assert actual==pytest.approx(np.mean(pl)+np.mean(kl)+np.mean(ul),abs=1e-8)
    assert m._last_train_four_source['utility_applied']
    def forbidden(*a):raise AssertionError('VAL must not use utility loss')
    monkeypatch.setattr(m.four_source_moe,'utility_loss',forbidden)
    actual=tr.validate_epoch(m,loader,edge,ew,criterion,torch.device('cpu'))
    assert actual==pytest.approx(np.mean(pl),abs=1e-8)
    assert not m._last_val_four_source['utility_applied']
    assert abs(actual-(2*pl[0]+pl[1])/3)>1e-5


def test_selection_restore_and_logging(tmp_path,monkeypatch):
    tr=importlib.import_module('cmgm.training.train');m=make_model(DATA)
    vals=iter([.4,.2,.3]);v5=iter([.01,.02,.001])
    def fit(model,*a,**kw):model._last_train_four_source={'pi':{},'q':{}};return .5
    def val(model,*a,**kw):
        model._last_val_four_source={'pi':{}};model._last_val5_diagnostic={'MAE':next(v5),'MSE':.01,'count':1};return next(vals)
    monkeypatch.setattr(tr,'train_epoch',fit);monkeypatch.setattr(tr,'validate_epoch',val)
    cp=tmp_path/'mock.pt'
    h=tr.train(m,None,None,torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(cp))
    assert h['best_epoch']==2 and h['best_val5_epoch']==3 and len(h['four_source_history'])==3
    saved=torch.load(cp,weights_only=False);other=make_model(DATA);other.load_state_dict(saved['model_state_dict'],strict=True)
    m.eval();other.eval();x=torch.randn(2,20,30,21)
    torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)


def test_frozen_train_mean_no_label_dependency(tmp_path):
    m=make_model(DATA).eval();torch.nn.init.normal_(m.four_source_moe.router.network[-1].weight,0,.1)
    state=copy.deepcopy(m.state_dict());x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x,y)) for s in ('train','val')})
    a=run.evaluate_frozen(m,data,42,torch.device('cpu'),tmp_path)
    data['loaders']['train']=DataLoader(TensorDataset(x,y+100))
    b=run.evaluate_frozen(m,data,42,torch.device('cpu'),tmp_path)
    assert a['train_mean_pi']==b['train_mean_pi']
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
    assert all(p.grad is None for p in m.parameters());json.dumps(a,allow_nan=False)
    assert set(a['formal'])=={'train','val'} and len(a['formal']['val']['experts'])==4


def test_default_cli_no_real_data_no_fit(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No real data or training')
    monkeypatch.setattr(run.shared,'train_val_data',forbidden);monkeypatch.setattr(run,'train',forbidden)
    called=[];monkeypatch.setattr(run,'preflight',lambda seed,out:called.append(seed))
    monkeypatch.setattr('sys.argv',['four','--output',str(tmp_path/'preflight')]);run.main();assert called==[42]
