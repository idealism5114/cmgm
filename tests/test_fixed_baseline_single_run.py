"""Fixed protocol regression checks, never formal fitting."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from cmgm.scripts.fixed_baseline_protocol import FIXED,PROTOCOL,CLASS_CONFIG,TRAINABLE_NEURAL,RUN_ORDER,estimator,estimator_matches
from cmgm.models.formal_baselines_v2 import ORDER


def test_fixed_configurations_and_single_parallel_layer():
    assert len(ORDER)==9 and len(RUN_ORDER)==8 and len(TRAINABLE_NEURAL)==5 and 'D0B' not in RUN_ORDER
    assert not any(PROTOCOL[k] for k in ('grid','multi_seed','D0B_retrain'))
    assert PROTOCOL['seed']==42
    ridge=estimator('Ridge Regression',2)
    assert ridge.alpha==1 and ridge.solver=='lsqr' and ridge.fit_intercept
    assert estimator_matches('Ridge Regression',ridge)
    ridge.solver='auto';assert not estimator_matches('Ridge Regression',ridge)
    rf=estimator('Random Forest',2)
    assert estimator_matches('Random Forest',rf) and rf.n_jobs==2
    rf.max_features=.3;assert not estimator_matches('Random Forest',rf)
    pytest.importorskip('xgboost')
    xgb=estimator('XGBoost',4)
    assert xgb.n_jobs==1 and xgb.estimator.n_jobs==4
    assert xgb.estimator.n_estimators==200 and xgb.estimator.learning_rate==.05
    assert xgb.estimator.max_depth==4 and xgb.estimator.random_state==42


def test_old_cli_is_retired_without_starting_any_job(monkeypatch):
    import cmgm.scripts.formal_baseline_benchmark_v2 as old
    monkeypatch.setattr(old,'tune',lambda *a:pytest.fail('Old grid must not run'))
    with pytest.raises(SystemExit,match='retired'):old.main()


def test_neural_monitor_is_huber_not_secondary_mae(tmp_path,monkeypatch):
    import cmgm.scripts.baseline_protocol as p
    class Model(nn.Module):
        def __init__(self):super().__init__();self.value=nn.Parameter(torch.full((4,24),.01))
        def forward(self,x):return self.value[None].expand(len(x),4,24)
    m=Model();original=m.value.clone();values=iter([(1.,.3),(.5,.4),(.6,.1),(.7,.05)])
    def valid(*a):
        loss,mae=next(values);return loss,dict(MAE=mae,MSE=mae**2)
    monkeypatch.setattr(p,'validation',valid)
    seen=[]
    class Scheduler:
        def __init__(self,*a,**k):pass
        def step(self,v):seen.append(v)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler)
    def no_update(opt,*a,**k):
        assert opt.param_groups[0]['lr']==1e-4 and opt.param_groups[0]['weight_decay']==1e-5
    monkeypatch.setattr(torch.optim.Adam,'step',no_update)
    ds=DataLoader(TensorDataset(torch.zeros(2,1),torch.zeros(2,4,24)),batch_size=2)
    summary,history=p.train_one(m,ds,ds,torch.device('cpu'),tmp_path/'fixture.pt',dict(model='unit fixture'),max_epochs=4,patience=2)
    assert summary['best_epoch']==2 and summary['secondary_best_val5_epoch']==4
    assert seen==[1.,.5,.6,.7] and torch.equal(original,m.value)
    assert torch.load(tmp_path/'fixture.pt',weights_only=False)['training_complete']


def test_no_fit_in_preparation_and_never_train_d0b(tmp_path):
    from cmgm.scripts.formal_baseline_single_run import process_model
    r=dict(models={},plan={n:dict(action='Pending') for n in RUN_ORDER},checkpoint_dir=str(tmp_path))
    for n in RUN_ORDER:process_model(n,r,tmp_path,None,torch.device('cpu'),SimpleNamespace(),False)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='never'):process_model('D0B',r,tmp_path,None,None,None,True)


def test_resume_completed_job_does_not_fit_or_evaluate(tmp_path):
    from cmgm.scripts.formal_baseline_single_run import process_model
    from cmgm.scripts.formal_v2_protocol import sha
    f=tmp_path/'fixed.bin';f.write_bytes(b'completed fixture')
    r=dict(models={'LSTM':dict(path=str(f),sha256=sha(f))})
    process_model('LSTM',r,tmp_path,None,None,None,True)
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):process_model('LSTM',r,tmp_path,None,None,None,True)


def test_partial_report_keeps_nine_rows_and_np_scalars(tmp_path):
    from cmgm.scripts.formal_baseline_single_run import save
    plan={n:dict(old_run_exists=False,architecture_valid=True,full_input_valid=True,seed42=True,fixed_config_matches=True,sanity_PASS=False,action='Pending',reason='unit fixture') for n in ORDER}
    r=dict(status='PREPARED',models={},plan=plan,jobs={},sanity={'TCN':dict(PASS=True,bound=np.float32(.001))},input_audit=dict(PASS=True),
        protocol=PROTOCOL,computation={},parallelism='one level',cpu_jobs=2)
    save(r,tmp_path)
    assert json.loads((tmp_path/'sanity_checks.json').read_text())['models']['TCN']['bound']==float(np.float32(.001))
    import csv
    rows=list(csv.DictReader((tmp_path/'single_run_results.csv').open()))
    assert len(rows)==9 and all(v['TEST5_MAE']=='' for v in rows)
    assert not (tmp_path/'hyperparameter_search.csv').exists()
    text=(tmp_path/'REPORT.md').read_text()
    assert 'Incomplete' in text and 'no multi-seed statistical claim' in text
    assert 'mean ± std' not in text
