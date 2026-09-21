import copy
import pytest
import torch
from cmgm.models.moe_fusion import SpatialTemporalMoEFusion, VARIANT, BALANCE_COEFFICIENT
from cmgm.scripts.d0b_moe_audit import initialization,sanity,make_model,BASE
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


def test_initialization_and_full_structural_sanity():
    m,a=initialization(DATA,torch.device('cpu'))
    assert a['PASS'] and a['shared_max_abs_diff']==0 and a['delta_parameters']==25027
    c=sanity(m,torch.randn(2,20,30,21),torch.randn(2,4,24)*.02)
    assert c['PASS'],c
    assert not hasattr(m,'gate_fc')
    assert m.switching_latent_transformer.balanced_readout
    assert not m.attn_mixhop1.qk_norm and m.attn_mixhop1.graph_prior_heads is None


def test_raw_router_projected_experts_warmup_and_balance():
    f=SpatialTemporalMoEFusion().eval()
    inputs=[torch.randn(3,64) for _ in range(4)]
    for epoch in (0,1,2,10,20):
        f.set_epoch(epoch);out=f(*inputs)
        p=f.router(torch.cat(inputs[:2],-1)).softmax(-1)
        eff=(1-min(1,epoch/10))/3+min(1,epoch/10)*p
        ex=torch.stack((f.temporal_expert(inputs[3]),f.spatial_expert(inputs[2]),f.interaction_expert(torch.cat(inputs[2:],-1))),1)
        torch.testing.assert_close(out,(eff.unsqueeze(-1)*ex).sum(1),rtol=0,atol=0)
        mean=p.mean(0);expected=(mean*((mean+1e-8)/(1/3)).log()).sum()
        torch.testing.assert_close(f.balance_loss(),expected,rtol=0,atol=0)
    f.set_epoch(0);f(*inputs)
    assert f.balance_loss().item()>1e-6 # Learned pi, not effective uniform, determines balance.
    assert f.last_effective_pi.shape==(3,3)


def test_epoch_checkpoint_restore_before_warmup_finishes():
    f=SpatialTemporalMoEFusion().eval();f.set_epoch(2);inputs=[torch.randn(2,64) for _ in range(4)]
    output=f(*inputs);state=copy.deepcopy(f.state_dict())
    f.set_epoch(99);f.load_state_dict(state)
    assert f.gamma==.2
    torch.testing.assert_close(f(*inputs),output,rtol=0,atol=0)


def test_loss_wiring_and_validation_excludes_auxiliary(monkeypatch):
    from torch.utils.data import DataLoader,TensorDataset
    from cmgm.training.train import train_epoch,validate_epoch
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    seed_all(42);m=make_model(DATA);m.set_moe_epoch(1);m.switching_latent_transformer.set_epoch(20)
    for mod in m.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0.
    x,y=torch.randn(2,20,30,21),torch.randn(2,4,24)*.02
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    p=m(x);predloss=prediction_loss(p,y);switch=m.switching_latent_transformer.switch_loss();balance=m.moe_balance_loss()
    expected=float((predloss+switch+BALANCE_COEFFICIENT*balance).detach())
    edge=torch.empty((2,0),dtype=torch.long);weight=torch.empty(0)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    result=train_epoch(m,loader,edge,weight,opt,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert result==pytest.approx(expected,abs=1e-9)
    assert m._last_train_moe['balance_loss']==pytest.approx(float(balance.detach()))
    actual=validate_epoch(m,loader,edge,weight,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
    assert actual==pytest.approx(float(predloss.detach()),abs=1e-9)
    assert m._last_val_moe['gamma_warmup']==.1


def test_native_gate_formula_unchanged():
    seed_all(42);m=make_model(DATA,BASE).eval();hs,ht=torch.randn(2,64),torch.randn(2,64)
    p=m._market_token_predict(hs,ht)
    gate=torch.sigmoid(m.gate_fc(torch.cat([hs,ht],-1)))
    expected=m.head(gate*m.lstm_proj(ht)+(1-gate)*m.gcn_proj(hs)).reshape(2,4,24)
    torch.testing.assert_close(p,expected,rtol=0,atol=0)
    assert not hasattr(m,'moe_fusion')


def test_training_schedule_and_best_epoch_restore(tmp_path,monkeypatch):
    import cmgm.training.train as tr
    m=make_model(DATA);seen=[];vals=iter([.4,.2,.3])
    def train_epoch(model,*a,**kw):
        seen.append(model.moe_fusion.gamma)
        model._last_train_moe=dict(balance_loss=.01,gamma_warmup=model.moe_fusion.gamma)
        return .5
    def validate(model,*a,**kw):
        model._last_val_moe=dict(balance_loss=.01,gamma_warmup=model.moe_fusion.gamma)
        model._last_val5_diagnostic=dict(MAE=.01,MSE=.001,count=1)
        return next(vals)
    monkeypatch.setattr(tr,'train_epoch',train_epoch);monkeypatch.setattr(tr,'validate_epoch',validate)
    cp=tmp_path/'moe.pt'
    history=tr.train(m,None,None,torch.empty((2,0),dtype=torch.long),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(cp))
    assert seen==[.1,.2,.3] and history['best_epoch']==2 and m.moe_fusion.gamma==.2
    payload=torch.load(cp,weights_only=False)
    assert payload['moe_epoch']==2 and payload['moe_gamma']==.2


def test_evaluation_and_report_artifacts(tmp_path):
    import json
    from torch.utils.data import DataLoader,TensorDataset
    from cmgm.scripts.d0b_moe_fusion import evaluate,PROTOCOL,CONTROLS
    from cmgm.scripts.d0b_moe_report import report
    seed_all(42);m=make_model(DATA);m.set_moe_epoch(10)
    x,y=torch.randn(3,20,30,21),torch.randn(3,4,24)*.02
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x,y),batch_size=2) for s in ('train','val','test')})
    evaluation=evaluate(m,data,torch.device('cpu'))
    for split in ('train','val','test'):
        assert set(evaluation['metrics'][split])=={'1','5','10','20'}
        for metric in evaluation['metrics'][split].values():
            assert abs(metric['RMSE']**2-metric['MSE'])<1e-12
        d=evaluation['diagnostics'][split]
        assert d['samples']==3 and sum(d['hard_occupancy'])==pytest.approx(1.)
        assert d['mean_pi']==pytest.approx(d['mean_effective_pi'])
    assert set(evaluation['diagnostics']['test']['disagreement'])=={'T_S','T_ST','S_ST'}
    r=dict(evaluation=evaluation,config=PROTOCOL,source_hashes={},initialization={},sanity={'PASS':True},
           data={},status='TEST FIXTURE',controls={n:dict(metrics=evaluation['metrics']) for n in CONTROLS})
    report(r,tmp_path)
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==13 and '## 12. Required Answers' in text
    assert len((tmp_path/'fusion_comparison.csv').read_text().splitlines())==5
    for path in tmp_path.glob('*.json'):
        json.loads(path.read_text())
