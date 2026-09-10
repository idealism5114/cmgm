"""Protocol and forward tests; no formal training or optimizer parameter updates."""
import copy
import json
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.formal_baselines_v2 import ORDER,NEURAL,CLASSICAL,make_neural,input_view,restore_input
from cmgm.scripts.formal_v2_protocol import grid,classical_model,neural_train,SEEDS
from cmgm.scripts.formal_v2_audit import sanity
from cmgm.scripts.formal_baseline_benchmark_v2 import evaluation_allowed,final_job_keys
from cmgm.scripts.formal_v2_report import aggregate

DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


@pytest.mark.parametrize('name',ORDER)
def test_full_scalar_layout_roundtrip(name):
    x=torch.arange(2*20*284*21,dtype=torch.float32).reshape(2,20,284,21)
    v=input_view(name,x)
    assert torch.equal(restore_input(name,v,x.shape),x)
    assert v.numel()==x.numel()
    if name in CLASSICAL:assert v.shape==(2,119280)
    if name in ('LSTM','TCN','Vanilla Transformer'):assert v.shape==(2,20,5964)


@pytest.mark.parametrize('name',NEURAL)
def test_forward_causal_batch_and_parameters_unchanged(name):
    torch.manual_seed(42);m=make_neural(name,DATA,torch.device('cpu'))
    before={k:v.clone() for k,v in m.state_dict().items()}
    r=sanity(name,m,torch.randn(2,20,30,21))
    assert r['PASS'],r
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())
    if name!='D0B':
        for module in m.modules():assert not any(s in type(module).__name__ for s in ('MarketAware','EdgeAttn','Switching','BalancedReadout'))


def test_official_graph_output_node_mapping():
    for name in ('Graph WaveNet','MTGNN'):
        m=make_neural(name,DATA,torch.device('cpu'))
        class FakeCore(nn.Module):
            def forward(self,x):return torch.arange(30,dtype=x.dtype)[None,None,:,None].expand(len(x),4,30,1)
        m.core=FakeCore()
        p=m(torch.ones(2,20,30,21))
        torch.testing.assert_close(p,torch.arange(6,30,dtype=p.dtype)[None,None,:].expand(2,4,24))


def test_official_provenance_no_core_rewrite():
    from cmgm.scripts.formal_v2_audit import provenance,ROOT
    pytest.importorskip('xgboost')
    p=provenance();assert p['PASS']
    mt=p['official']['mtgnn']['files']['layer.py'];assert mt['local_sha256']==mt['upstream_sha256']
    text=(ROOT/'third_party/baselines/graphwavenet/model.py').read_text()
    import hashlib
    # Exactly three Conv1d→Conv2d compatibility replacements, recover byte-identical upstream.
    patch=(ROOT/'third_party/baselines/graphwavenet/model.py.patch').read_text()
    assert patch.count('-                self.')==3
    for attr in ('gate_convs','residual_convs','skip_convs'):
        text=text.replace(f'self.{attr}.append(nn.Conv2d(',f'self.{attr}.append(nn.Conv1d(')
    assert hashlib.sha256(text.encode()).hexdigest()==p['official']['graphwavenet']['files']['model.py']['upstream_sha256']


def test_registered_grid_and_official_classical_estimators():
    assert [len(grid(n)) for n in ORDER]==[5,8,4,3,3,3,3,3,3]
    ridge=classical_model('Ridge Regression',grid('Ridge Regression')[0],42,2)
    assert ridge.fit_intercept and ridge.solver=='auto'
    rf=classical_model('Random Forest',grid('Random Forest')[0],2025,2)
    assert rf.n_jobs==2 and rf.random_state==2025 and rf.bootstrap
    pytest.importorskip('xgboost')
    xgb=classical_model('XGBoost',grid('XGBoost')[0],3407,2)
    assert xgb.n_jobs==1 and xgb.estimator.n_estimators==500
    assert xgb.estimator.objective=='reg:squarederror' and xgb.estimator.random_state==3407


def test_selection_mae_scheduler_huber_and_no_actual_updates(tmp_path,monkeypatch):
    import cmgm.scripts.formal_v2_protocol as protocol
    class Branch(nn.Module):
        def __init__(self):super().__init__();self.epochs=[];self.calls=0
        def set_epoch(self,e):self.epochs.append(e);return .0005*min(e/20,1)
        def switch_loss(self):self.calls+=1;return torch.tensor(.00001)
    class Fixture(nn.Module):
        def __init__(self):super().__init__();self.p=nn.Parameter(torch.full((4,24),.01));self.switching_latent_transformer=Branch()
        def forward(self,x):return self.p[None].expand(len(x),4,24)
    m=Fixture();before=m.p.detach().clone();v=iter([(.3,1.),(.1,3.),(.2,.5)])
    def pred(*a):
        mae,huber=next(v);return np.full((2,4,24),mae),np.zeros((2,4,24)),huber
    monkeypatch.setattr(protocol,'neural_predictions',pred)
    seen=[]
    class Scheduler:
        def __init__(self,*a,**k):pass
        def step(self,x):seen.append(x)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler)
    def no_update(opt,*a,**k):assert opt.param_groups[0]['lr']==1e-4 and len(opt.param_groups)==1
    monkeypatch.setattr(torch.optim.Adam,'step',no_update)
    ds=DataLoader(TensorDataset(torch.zeros(2,1),torch.zeros(2,4,24)),batch_size=2)
    summary=neural_train(m,ds,ds,torch.device('cpu'),1e-4,tmp_path/'best.pt',dict(model='fixture',stage='unit',seed=42),max_epochs=3)
    assert summary['best_epoch']==2 and seen==[1.,3.,.5]
    assert m.switching_latent_transformer.calls==3 and m.switching_latent_transformer.epochs==[1,2,3,2]
    assert torch.equal(before,m.p)


def test_test_barrier_requires_all_selections_and_all_formal_seeds():
    jobs={k:dict(status='DONE') for k in final_job_keys()}
    assert len(jobs)==25
    r=dict(selection_frozen=False,jobs=jobs);assert not evaluation_allowed(r)
    r['selection_frozen']=True;assert evaluation_allowed(r)
    jobs[next(iter(jobs))]['status']='RUNNING';assert not evaluation_allowed(r)


def test_aggregate_uses_all_seeds_and_ridge_once():
    runs={}
    for name in ORDER:
        for i,s in enumerate((42,) if name=='Ridge Regression' else SEEDS):
            metric=dict(MAE=float(i+1),MSE=float(i+1)**2,RMSE=float(i+1),Hit=.5)
            runs[f'{name}_{s}']=dict(model=name,seed=s,metrics={split:{str(h):metric for h in (1,5,10,20)} for split in ('val','test')})
    a=aggregate(runs);assert len(a)==72
    for row in a:
        if row['model']=='Ridge Regression':assert row['MAE_std'] is None and row['seeds']==1
        else:assert row['MAE_mean']==2. and row['MAE_std']==1. and row['seeds']==3


def test_stage_a_freezes_all_grids_without_accessing_test(tmp_path,monkeypatch):
    import cmgm.scripts.formal_baseline_benchmark_v2 as runner
    from types import SimpleNamespace
    class Forbidden:
        def __getitem__(self,key):pytest.fail('Stage A must not access TEST')
    calls=[]
    def fake_fit(r,out,data,device,args,name,hp,seed,stage,candidate):
        assert seed==42 and stage=='A'
        calls.append((name,hp))
        return dict(hp=hp,summary=dict(val5_MAE=float(candidate+1)),path=f'fixture_{name}_{candidate}',sha256='fixture')
    monkeypatch.setattr(runner,'fit_job',fake_fit);monkeypatch.setattr(runner,'save',lambda *a:None)
    r=dict(selection_frozen=False,selected={})
    runner.tune(r,tmp_path,dict(test=Forbidden()),torch.device('cpu'),SimpleNamespace())
    assert len(calls)==35 and r['selection_frozen']
    assert all(r['selected'][n]['hp']==grid(n)[0] for n in ORDER)
    assert 'TEST' not in json.dumps(r['selected'])


@pytest.mark.parametrize('name',NEURAL)
def test_training_autograd_is_finite_without_optimizer_step(name):
    from cmgm.scripts.formal_v2_protocol import prediction_loss
    torch.manual_seed(42);m=make_neural(name,DATA,torch.device('cpu')).train()
    branch=getattr(m,'switching_latent_transformer',None)
    if branch is not None:branch.set_epoch(1)
    p=m(torch.randn(2,20,30,21));loss=prediction_loss(p,torch.zeros_like(p))
    if branch is not None:loss=loss+branch.switch_loss()
    grads=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g is not None and torch.count_nonzero(g)>0 for g in grads)
    assert all(p.grad is None for p in m.parameters())
