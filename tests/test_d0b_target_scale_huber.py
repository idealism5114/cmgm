"""Controlled loss semantics, frozen scales, D0B equivalence, and report regressions."""
import importlib
from types import SimpleNamespace
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset

from cmgm import config
from cmgm.training.target_scale_huber import (
    VARIANT,OBJECTIVE,HORIZONS,REFERENCE_STD,TargetScaleHuber,estimate_training_scales,
    horizon_terms,restore_scale_metadata,
)
from cmgm.training.train import _prediction_loss,make_loss
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_target_scale_huber import (
    BASE_VARIANT,initialization_check,calibration_statistics,gradient_probe,
    checkpoint_report,case_assessment,
)


def make(n=6,nc=2,feat=5):
    torch.manual_seed(42)
    m=HeteroMixHopCMGM(n,nc,n_stock=n-nc-2,n_bond=2,feat_dim=feat,variant=VARIANT)
    m.target_scale_huber=TargetScaleHuber(REFERENCE_STD)
    return m


def batch():
    gen=torch.Generator().manual_seed(81)
    return torch.randn(3,7,6,5,generator=gen),torch.randn(3,4,2,generator=gen)*.03


def test_full_parameter_initialization_trace_equality_and_rng():
    torch.manual_seed(42)
    m=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,feat_dim=21,variant=VARIANT)
    m.target_scale_huber=TargetScaleHuber(REFERENCE_STD)
    x=torch.randn(2,20,284,21);y=torch.randn(2,4,24)*.02
    rng=torch.get_rng_state().clone()
    r=initialization_check(m,(x,y))
    assert r['PASS'] and r['D0B_params']==r['TargetScaleHuber_params']==520549
    assert r['max_abs_diff']==r['mismatch_count']==r['difference']==0
    assert all(v['max']==0 for v in r['forward_differences'].values())
    assert r['five_day_loss_abs_diff']==0
    assert torch.equal(torch.get_rng_state(),rng) and m.training
    assert not any('target_scale' in k for k in m.state_dict())


def test_train_only_scale_population_tail_and_strict_reference_gate():
    class Data:
        horizons=list(HORIZONS)
        def __len__(self):return 1396
        def __getitem__(self,i):return None,np.repeat((np.array(REFERENCE_STD)*(1 if i%2 else -1))[:,None],24,axis=1)
    c,r=estimate_training_scales(Data())
    assert r['PASS'] and r['shape']==[1396,4,24] and r['ddof']==0
    assert c.deltas[1]==.02 and c.weights[1]==1
    assert np.allclose(c.target_scales,REFERENCE_STD,atol=1e-14)
    with pytest.raises(FrozenInstanceError):c.reference_delta=.03
    class Changed(Data):
        def __getitem__(self,i):return None,Data.__getitem__(self,i)[1]*2
    with pytest.raises(ValueError,match='differs from diagnostic'):estimate_training_scales(Changed())
    class Short(Data):
        def __len__(self):return 1344
    with pytest.raises(ValueError,match='window count'):estimate_training_scales(Short())
    with pytest.raises(ValueError):TargetScaleHuber((0,1,2,3))


def test_dynamic_horizon_index_and_exact_anchor_and_gradient_cap():
    c=TargetScaleHuber(REFERENCE_STD);order=[10,1,20,5]
    p=torch.tensor([[[-.2,0.,.2]]*4],dtype=torch.float64,requires_grad=True);y=torch.zeros_like(p)
    raw,weighted=horizon_terms(p,y,c,order)
    expected=make_loss()(p[:,order.index(5)],y[:,order.index(5)])
    torch.testing.assert_close(weighted['5'],expected,rtol=0,atol=0)
    for h in HORIZONS:
        gradient=torch.autograd.grad(weighted[str(h)],p,retain_graph=True)[0]
        i=order.index(h)
        torch.testing.assert_close(gradient[:,i]*3,torch.tensor([[-.02,0,.02]],dtype=torch.float64),rtol=1e-14,atol=1e-15)
        assert torch.count_nonzero(gradient[:,[j for j in range(4) if j!=i]])==0
    assert np.allclose(np.array(c.weights)*c.deltas,.02,rtol=0,atol=1e-15)


@pytest.mark.parametrize('variant',[
    BASE_VARIANT,'switching_latent_dynamic_slope','switching_latent_balanced_transition',
    'switching_latent_learnable_persistence','switching_latent_balanced_readout_no_switch_kl',
    'switching_latent_balanced_readout_5d_only','switching_latent_balanced_readout_5_10_20'])
def test_old_variant_loss_and_gradients_bitwise_unchanged(variant):
    p=torch.randn(2,4,3,requires_grad=True);y=torch.randn_like(p)
    criterion=make_loss();actual=_prediction_loss(SimpleNamespace(variant=variant),p,y,criterion)
    if variant.endswith('_5d_only'):expected=4*criterion(p[:,config.MULTI_HORIZONS.index(5)],y[:,config.MULTI_HORIZONS.index(5)])
    elif variant.endswith('_5_10_20'):expected=(4/3)*sum(criterion(p[:,config.MULTI_HORIZONS.index(h)],y[:,config.MULTI_HORIZONS.index(h)]) for h in (5,10,20))
    else:expected=sum(criterion(p[:,i],y[:,i]) for i in range(4))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0],atol=0,rtol=0)


def test_shared_train_validate_helper_retains_switch_and_logs_raw_weighted(monkeypatch):
    training=importlib.import_module('cmgm.training.train')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.variant=VARIANT;self.graph_learner=None
            self.target_scale_huber=TargetScaleHuber(REFERENCE_STD)
            self.pred=torch.nn.Parameter(torch.full((4,2),.07))
            self.switch=torch.nn.Parameter(torch.tensor(.1))
            self.switching_latent_transformer=SimpleNamespace(null_control=False,switch_loss=lambda:self.switch.square())
        def forward(self,x,**kw):return self.pred[None].expand(len(x),-1,-1)
    m=Model();x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=2)
    before={k:v.clone() for k,v in m.state_dict().items()}
    class NoUpdate:
        def zero_grad(self):m.zero_grad(set_to_none=True)
        def step(self):pass
    calls=[];original=training._prediction_loss
    def spy(*a):calls.append(a[0].training);return original(*a)
    monkeypatch.setattr(training,'_prediction_loss',spy)
    e=torch.empty(2,0,dtype=torch.long);w=torch.zeros(0)
    tr=training.train_epoch(m,loader,e,w,NoUpdate(),make_loss(),'cpu')
    va=training.validate_epoch(m,loader,e,w,make_loss(),'cpu')
    assert calls==[True,True,False,False]
    assert m.switch.grad.item()==pytest.approx(.2)
    assert tr==pytest.approx(va+.01)
    assert set(m._last_train_objective)=={*[f'{kind}_huber_{h}' for kind in ('raw','weighted') for h in HORIZONS],'prediction_loss','switch_loss','total_loss'}
    assert sum(m._last_train_objective[f'weighted_huber_{h}'] for h in HORIZONS)==pytest.approx(va)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    missing=SimpleNamespace(variant=VARIANT)
    with pytest.raises(ValueError):_prediction_loss(missing,torch.zeros(1,4,2),torch.zeros(1,4,2),make_loss())


def test_scheduler_checkpoint_selection_metadata_no_optimizer_updates(tmp_path,monkeypatch):
    training=importlib.import_module('cmgm.training.train');m=make();x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=3)
    epoch=[0];seen=[]
    def tr(m,*a,**kw):
        epoch[0]+=1;m._last_train_objective={f'{k}_huber_{h}':.1 for k in ('raw','weighted') for h in HORIZONS}
        m._last_train_objective.update(prediction_loss=.4,switch_loss=.01,total_loss=.41);return .41
    def va(m,*a,**kw):
        loss=[.4,.2,.3][epoch[0]-1];m._last_val_objective={'prediction_loss':loss,'MAE_5d':[.3,.4,.1][epoch[0]-1],'MSE_5d':.01};return loss
    class Scheduler:
        def __init__(self,opt,**kw):
            assert len(opt.param_groups)==1 and opt.param_groups[0]['lr']==1e-4 and opt.param_groups[0]['weight_decay']==1e-5
        def step(self,v):seen.append(v)
    monkeypatch.setattr(training,'train_epoch',tr);monkeypatch.setattr(training,'validate_epoch',va)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler)
    def forbidden(*a,**kw):raise AssertionError('No training in selection test')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    p=tmp_path/'model.pt'
    hist=training.train(m,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),'cpu',num_epochs=10,patience=1,checkpoint_path=str(p),checkpoint_metadata={'git_sha':'test','seed':42})
    saved=torch.load(p,weights_only=True);meta=saved['metadata']
    assert hist['best_epoch']==2 and hist['final_epoch']==3 and seen==[.4,.2,.3]
    assert meta['best_val_scale_aware_prediction_loss']==.2 and meta['best_val_5d_MAE']==.4 and meta['best_val_5d_MSE']==.01
    assert meta['switch_kl_enabled'] and meta['objective']==OBJECTIVE and meta['seed']==42
    assert restore_scale_metadata(meta)==m.target_scale_huber
    meta['weight_20']=1
    with pytest.raises(ValueError):restore_scale_metadata(meta)


def test_network_weighted_gradient_anchor_and_analytical_caps_no_mutation():
    m=make().eval();x,y=batch();before={k:v.clone() for k,v in m.state_dict().items()}
    r=gradient_probe(m,(x,y))
    c=calibration_statistics(np.ones((5,4,2)),np.zeros((5,4,2)),m.target_scale_huber)
    assert all(v['weighted_gradient']['max_abs']==pytest.approx(.02) for v in c['horizons'].values())
    assert set(r['norms'])=={'1','5','10','20'} and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())


def test_checkpoint_diagnostic_and_report_end_to_end(tmp_path):
    m=make();x,y=batch();initial=initialization_check(m,(x,y))
    base=HeteroMixHopCMGM(6,2,n_stock=2,n_bond=2,feat_dim=5,variant=BASE_VARIANT);base.load_state_dict(m.state_dict())
    old=tmp_path/'base.pt';new=tmp_path/'new.pt'
    torch.save({'model_state_dict':base.state_dict(),'best_epoch':85},old)
    metadata={**m.target_scale_huber.metadata(),'variant':VARIANT,'initialization':initial,'scale_sanity':{'PASS':True},'seed':42}
    saved={'model_state_dict':m.state_dict(),'metadata':metadata,'history':{'objective_history':[]},'best_epoch':85}
    torch.save(saved,new)
    data={'loaders':{k:DataLoader(TensorDataset(x,y),batch_size=2) for k in ('train','val','test')},
          'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},'feature_names':['a','b','c','d','e','f']}
    r=checkpoint_report(m,saved,data,old,tmp_path/'report',new)
    assert r['integrity']['parameters_unchanged_by_diagnostics'] and r['integrity']['checkpoint_files_unchanged']
    report=(tmp_path/'report'/'REPORT.md').read_text()
    assert '20 个问题' in report and 'MSE' in report and 'STOP' in report
    assert set(r['models'])=={'D0B','TargetScaleHuber'}
    assert all(v['sanity']['PASS'] for v in r['models'].values())


@pytest.mark.parametrize('mae,mse,case',[
    ((-.01,-.01),(-.01,-.01),'Case A'),((0.,0.),(0.,0.),'Case B'),
    ((.01,.01),(.01,.01),'Case C'),((-.01,.01),(-.01,.01),'Case D'),
    ((-.01,-.01),(.01,.01),'Case E'),((.01,.01),(-.01,-.01),'Case F')])
def test_directional_case_rules(mae,mse,case):
    c={s:{'5':{'relative_MAE':mae[i],'relative_MSE':mse[i]}} for i,s in enumerate(('VAL','TEST'))}
    assert case_assessment(c)['case']==case


def test_ablation_result_matches_experiment_logger_contract(tmp_path):
    from cmgm.scripts.d0b_target_scale_huber import ablation_result
    from cmgm.training.metric_standard import population_metrics
    directory=tmp_path/'TargetScaleHuber';directory.mkdir()
    gen=np.random.default_rng(42)
    p=gen.normal(0,.02,(3,4,2)).astype(np.float32);y=gen.normal(0,.02,(3,4,2)).astype(np.float32)
    np.savez(directory/'test_predictions.npz',native=p,target=y)
    data={'norm_stats':{'mean':np.zeros(6),'std':np.ones(6)},'raw_prices_test':np.ones((23,6)),
          'market_indices':{'commodity':(4,6)}}
    report={'models':{'TargetScaleHuber':{'params':520549}},'assessment':{'case':'Case D'}}
    result=ablation_result(report,{'history':{'training_elapsed_seconds':1.}},tmp_path,data)
    assert {'mn','mo','MAE','MSE','RMSE','Hit_Ratio','params','time'}<=set(result)
    assert {'MAE','MSE','RMSE','Residual_Mean','Residual_Std','Skewness'}<=set(result['mn'])
    expected=population_metrics(p[:,config.MULTI_HORIZONS.index(5)],y[:,config.MULTI_HORIZONS.index(5)])
    assert result['MSE']==expected['MSE'] and result['Hit_Ratio']==expected['Hit']


def test_precision_audit_preserves_training_state_and_rejects_prefix_failure(monkeypatch):
    import cmgm.scripts.d0e_diagnostics as shared
    from cmgm.scripts.d0b_target_scale_huber import precision_checked_sanity
    m=make();x,_=batch();before={k:v.clone() for k,v in m.state_dict().items()}
    rng=torch.get_rng_state().clone();calls=[]
    def probe(model,x,**kw):
        calls.append(x.dtype)
        high=x.dtype==torch.float64
        return {'PASS':high,'causality':{'E':0.,'H':0.},'batch_permutation_max':0.,'single_sample_max':0.,
                'within_market':{'stock':{'temporal_invariance_max':1e-15 if high else 4e-6}}}
    monkeypatch.setattr(shared,'fixed_sanity',probe)
    r=precision_checked_sanity(m,x)
    assert r['PASS'] and not r['float32_original']['PASS'] and r['float64_max_diff']==1e-15
    assert calls==[torch.float32,torch.float64] and m.training
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,before[k]) and v.dtype==before[k].dtype for k,v in m.state_dict().items())
    def bad(*a,**kw):return {'PASS':False,'causality':{'E':.01},'batch_permutation_max':0.,'single_sample_max':0.}
    monkeypatch.setattr(shared,'fixed_sanity',bad)
    with pytest.raises(AssertionError,match='Prefix/batch'):precision_checked_sanity(m,x)


def test_experiment_logger_persists_mandatory_metrics(tmp_path,monkeypatch):
    import json
    import cmgm.experiment_logger as log
    from cmgm.training.evaluate import compute_metrics
    monkeypatch.setattr(log,'JSONL_FILE',str(tmp_path/'runs.jsonl'))
    monkeypatch.setattr(log,'SUMMARY_FILE',str(tmp_path/'summary.md'))
    p=np.array([[.01,.03],[.02,.04],[.04,.06]])
    y=np.array([[.02,.01],[.03,.07],[.01,.02]])
    metrics=compute_metrics(p,y)
    log.ExperimentLogger(str(tmp_path)).log_run({'horizon':5},[('TargetScaleHuber synthetic',0,metrics,metrics)])
    row=json.loads((tmp_path/'runs.jsonl').read_text())
    assert row['normalized']['MSE']==metrics['MSE'] and row['normalized']['Hit_Ratio']==metrics['Hit_Ratio']
    assert 'Hit% (norm)' in (tmp_path/'summary.md').read_text()


def test_one_run_entry_and_post_training_logging_contract(tmp_path,monkeypatch):
    import cmgm.scripts.d0b_target_scale_huber as workflow
    training=importlib.import_module('cmgm.training.train')
    m=make();x,y=batch();initial=initialization_check(m,(x,y))
    meta={**m.target_scale_huber.metadata(),'variant':VARIANT,'initialization':initial,
          'scale_sanity':{'PASS':True},'seed':42,'parameter_count':sum(p.numel() for p in m.parameters())}
    baseline=tmp_path/'d0b.pt';torch.save({'model_state_dict':m.state_dict(),'best_epoch':85},baseline)
    args=SimpleNamespace(seed=42,batch_size=64,seq_len=20,epochs=200,patience=10,
                         d0b_checkpoint=baseline,checkpoint_dir=tmp_path/'checkpoints',target_scale_report_dir=tmp_path/'report')
    data={'loaders':{s:DataLoader(TensorDataset(x,y),batch_size=2) for s in ('train','val','test')},
          'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},'feature_names':list('abcdef'),
          'norm_stats':{'mean':np.zeros(6),'std':np.ones(6)},'raw_prices_test':np.ones((23,6))}
    monkeypatch.setattr(workflow,'prepare',lambda *a:(m,meta))
    calls=[]
    def train_stub(model,*a,**kw):
        assert kw['num_epochs']==200 and kw['patience']==10
        assert kw['checkpoint_metadata']['D0B_reference']['best_epoch']==85
        calls.append(kw['checkpoint_path'])
        torch.save({'model_state_dict':model.state_dict(),'metadata':kw['checkpoint_metadata'],
                    'history':{'objective_history':[],'training_elapsed_seconds':0.},'best_epoch':1},kw['checkpoint_path'])
    monkeypatch.setattr(training,'train',train_stub)
    result=workflow.run_target_scale_huber(args,torch.device('cpu'),data)
    assert len(calls)==1 and {'mn','mo','MAE','MSE','RMSE','Hit_Ratio'}<=set(result)
    with pytest.raises(FileExistsError):workflow.run_target_scale_huber(args,torch.device('cpu'),data)
    assert len(calls)==1
