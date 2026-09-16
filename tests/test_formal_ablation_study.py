import copy
import pytest
import torch
from cmgm.models.formal_d0b_ablation import NAMES,FormalD0BAblation
from cmgm.models.formal_baselines_v2 import make_neural
from cmgm.scripts.formal_ablation_audit import init_audit,sanity
from cmgm.scripts.baseline_protocol import seed_all
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


@pytest.mark.parametrize('name',NAMES)
def test_exact_shared_init_and_single_factor_sanity(name):
    m,init=init_audit(name,DATA,torch.device('cpu'))
    assert init['PASS'] and init['max_abs_diff']==0
    before={n:p.clone() for n,p in m.named_parameters()}
    check=sanity(m,torch.randn(2,20,30,21),torch.randn(2,4,24)*.02)
    assert check['PASS'],check
    assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
    assert all(p.grad is None for p in m.parameters())


def test_full_control_is_native_forward_and_state_dict():
    seed_all(42);full=make_neural('D0B',DATA,torch.device('cpu')).eval()
    seed_all(42);control=FormalD0BAblation(NAMES[0],DATA).eval()
    assert list(full.state_dict())==list(control.state_dict())
    assert all(torch.equal(v,control.state_dict()[n]) for n,v in full.state_dict().items())
    x=torch.randn(2,20,30,21)
    with torch.no_grad():torch.testing.assert_close(full(x),control(x),atol=0,rtol=0)


def test_uniform_switching_does_not_call_markov_evidence(monkeypatch):
    m=FormalD0BAblation('w/o Markov Switching',DATA)
    def fail(*a):pytest.fail('Learnable Markov routing called')
    monkeypatch.setattr(m.switching_latent_transformer.regime_filter,'step',fail)
    monkeypatch.setattr(m.switching_latent_transformer.regime_filter,'transition_matrix',fail)
    m(torch.randn(2,20,30,21))
    assert m.auxiliary_loss().item()==0 and not m.auxiliary_loss().requires_grad


def test_native_train_epoch_switch_exceptions_without_update(monkeypatch):
    from cmgm.training.train import train_epoch
    from torch.utils.data import TensorDataset,DataLoader
    from cmgm.scripts.baseline_protocol import prediction_loss
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:None)
    source=DataLoader(TensorDataset(torch.randn(2,20,30,21),torch.zeros(2,4,24)),batch_size=2)
    for n in ('FullD0B-Control','w/o Temporal Branch','w/o Switch KL','w/o Markov Switching','w/o Microstate'):
        m=FormalD0BAblation(n,DATA);m.switching_latent_transformer.set_epoch(20)
        # Disable stochastic dropout only in this objective wiring fixture.
        for mod in m.modules():
            if isinstance(mod,torch.nn.Dropout):mod.p=0
        x,y=next(iter(source));pred=m(x);expected=float((prediction_loss(pred,y)+m.auxiliary_loss()).detach())
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-4)
        value=train_epoch(m,source,torch.empty(2,0,dtype=torch.long),torch.empty(0),optimizer,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))
        assert value==pytest.approx(expected,abs=1e-8)


def test_runner_selection_and_early_stop_use_huber_only(tmp_path,monkeypatch):
    import cmgm.scripts.formal_ablation_study as runner
    import cmgm.scripts.baseline_protocol as common
    from types import SimpleNamespace
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.ones(1));self.switching_latent_transformer=SimpleNamespace(set_epoch=lambda e:.0005);self.disable_switch_kl=False
    monkeypatch.setattr(runner,'loaders',lambda *a:dict(train=None,val=None))
    monkeypatch.setattr(runner,'train_epoch',lambda *a:1.)
    seq=iter([1.,.5]+[.6]*10)
    monkeypatch.setattr(runner,'validate_epoch',lambda *a:next(seq))
    monkeypatch.setattr(common,'validation',lambda *a:(0.,dict(MAE=.01,MSE=.0001)))
    m=Tiny();before=m.p.clone()
    cp=runner.fit(NAMES[0],m,None,torch.device('cpu'),tmp_path/'fixture.pt',{},tmp_path)
    assert cp['training_complete'] and cp['best_epoch']==2 and len(cp['history'])==12
    assert cp['best_val_objective']==.5 and torch.equal(before,m.p)


def test_report_never_uses_historical_reference_for_deltas(tmp_path):
    import csv
    from cmgm.scripts.formal_ablation_study import PROTOCOL,save
    base=dict(metrics={'test':{'5':dict(MAE=.1,MSE=.02,RMSE=.02**.5,Hit=.5)}},per_commodity=[],mechanism={},runtime={},sanity={})
    ablation=copy.deepcopy(base);ablation['metrics']['test']['5']['MAE']=.09
    r=dict(status='PARTIAL',protocol=PROTOCOL,results={NAMES[0]:base,NAMES[1]:ablation},initialization={},sanity={},masking={},data={},jobs={},reuse_policy='unit',
        historical=dict(metrics={'test':{'5':dict(MAE=.0219947838)}}),existing_audit={n:dict(existing_checkpoint=False,exact_protocol_match=False,reuse=False,need_training=True) for n in NAMES})
    save(r,tmp_path)
    rows=list(csv.DictReader((tmp_path/'multimodal_ablation.csv').open()))
    assert float(rows[0]['DeltaMAE'])==pytest.approx(-.01)
    assert float(rows[0]['RelativeDeltaMAE'])==pytest.approx(-10.)
    assert rows[-1]['Variant']==NAMES[0] and float(rows[-1]['DeltaMAE'])==0
    assert len(list(csv.DictReader((tmp_path/'architecture_ablation.csv').open())))==10
