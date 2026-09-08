"""Synthetic objective/selection/diagnostic tests; no market training run."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_5d_only_diagnostics import (
    VARIANT, BASE_VARIANT, DISPLAY, assert_backbone, initialization_check,
    horizon_gradients, checkpoint_report, case_assessment, full_reference,
)
from cmgm.training.train import _prediction_loss, make_loss


def make(variant=VARIANT, feat_dim=5):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(6,2,n_stock=2,n_bond=2,feat_dim=feat_dim,variant=variant)


def batch():
    gen=torch.Generator().manual_seed(175)
    return torch.randn(3,7,6,5,generator=gen),torch.randn(3,4,2,generator=gen)*.03


def test_real_dimension_shared_init_all_traces_and_rng():
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
    x=torch.randn(2,20,284,21);y=torch.randn(2,4,24)*.03
    rng=torch.get_rng_state().clone()
    result=initialization_check(model,(x,y))
    assert result['PASS'] and result['mismatch_count']==0
    assert result['D0B_params']==result['5dOnly_params']==520549
    assert result['difference']==0
    assert all(d['max']==0 for d in result['forward_differences'].values())
    assert torch.equal(torch.get_rng_state(),rng) and model.training
    scale=result['loss_scale']
    assert scale['4x_L5']==pytest.approx(4*scale['L5'])
    assert scale['ratio']==pytest.approx(scale['4x_L5']/scale['sum_multi'])
    assert_backbone(model)
    assert not any('sticky_logit' in n for n,_ in model.named_parameters())


def test_objective_gradient_only_five_days_dynamic_index_and_fixed_multiplier(monkeypatch):
    training=importlib.import_module('cmgm.training.train')
    monkeypatch.setattr(training,'MULTI_HORIZONS',[20,1,5,10])
    pred=torch.randn(2,4,3,requires_grad=True);target=torch.randn_like(pred)
    loss=_prediction_loss(SimpleNamespace(variant=VARIANT),pred,target,make_loss())
    expected=4*make_loss()(pred[:,2],target[:,2])
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    grad=torch.autograd.grad(loss,pred)[0]
    assert torch.count_nonzero(grad[:,[0,1,3]])==0
    assert torch.count_nonzero(grad[:,2])>0
    raw=torch.autograd.grad(make_loss()(pred[:,2],target[:,2]),pred)[0]
    torch.testing.assert_close(grad,4*raw,rtol=0,atol=0)
    changed=target.clone();changed[:,[0,1,3]]+=100
    torch.testing.assert_close(_prediction_loss(SimpleNamespace(variant=VARIANT),pred,changed,make_loss()),loss,rtol=0,atol=0)
    with pytest.raises(ValueError):
        _prediction_loss(SimpleNamespace(variant=VARIANT),pred[:,0],target[:,0],make_loss())


@pytest.mark.parametrize('variant',[BASE_VARIANT,'switching_latent_learnable_persistence',
    'switching_latent_balanced_transition','switching_latent_dynamic_slope',
    'market_dispersion_transformer','switching_transformer','switching_filter_rpe',None])
def test_existing_variants_multi_sum_and_single_loss_unchanged(variant):
    prediction=torch.randn(2,4,3,requires_grad=True);target=torch.randn_like(prediction)
    criterion=make_loss();model=SimpleNamespace(variant=variant)
    old=sum(criterion(prediction[:,h,:],target[:,h,:]) for h in range(4))
    new=_prediction_loss(model,prediction,target,criterion)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(new,prediction)[0],torch.autograd.grad(old,prediction)[0],rtol=0,atol=0)
    torch.testing.assert_close(_prediction_loss(model,prediction[:,0],target[:,0],criterion),criterion(prediction[:,0],target[:,0]),rtol=0,atol=0)


class FixedPredictionModel(torch.nn.Module):
    """Exercise train_epoch/backward with an explicit non-updating optimizer."""
    def __init__(self,variant):
        super().__init__()
        self.variant=variant
        self.graph_learner=None
        self.prediction=torch.nn.Parameter(torch.full((4,2),.07))
        self.switch_parameter=torch.nn.Parameter(torch.tensor(.1))
        self.switching_latent_transformer=SimpleNamespace(null_control=False,switch_loss=lambda:self.switch_parameter.square())
    def forward(self,x,**kwargs):
        return self.prediction.unsqueeze(0).expand(len(x),-1,-1)


def test_train_validate_share_objective_switch_is_unchanged_and_aux_not_selected(monkeypatch):
    training=importlib.import_module('cmgm.training.train')
    x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=2)
    model=FixedPredictionModel(VARIANT)
    before={k:v.clone() for k,v in model.state_dict().items()}
    captured=[]
    class NoUpdateOptimizer:
        def zero_grad(self): model.zero_grad(set_to_none=True)
        def step(self): captured.append(model.prediction.grad.clone())  # only inspect; never update
    calls=[];real=training._prediction_loss
    def spy(*a):
        calls.append(a[0].training)
        return real(*a)
    monkeypatch.setattr(training,'_prediction_loss',spy)
    edges=torch.empty(2,0,dtype=torch.long);weights=torch.zeros(0)
    train_loss=training.train_epoch(model,loader,edges,weights,NoUpdateOptimizer(),make_loss(),torch.device('cpu'))
    assert calls==[True,True]
    assert all(torch.count_nonzero(g[[0,2,3]])==0 and torch.count_nonzero(g[1])>0 for g in captured)
    assert model.switch_parameter.grad.item()==pytest.approx(.2)
    stats=model._last_train_objective
    assert stats['switch_loss']==pytest.approx(.01)
    assert stats['total_loss']==pytest.approx(stats['scaled_prediction_loss']+.01)
    assert train_loss==pytest.approx(stats['total_loss'])
    val=training.validate_epoch(model,loader,edges,weights,make_loss(),torch.device('cpu'))
    assert calls==[True,True,False,False]
    assert val==pytest.approx(stats['scaled_prediction_loss'])
    assert val==pytest.approx(4*model._last_val_objective['raw_L5'])
    changed=y.clone();changed[:,[0,2,3]]+=10
    aux_before=model._last_val_objective['aux_multi_horizon_loss']
    val2=training.validate_epoch(model,DataLoader(TensorDataset(x,changed),batch_size=2),edges,weights,make_loss(),torch.device('cpu'))
    assert val2==val
    assert model._last_val_objective['aux_multi_horizon_loss']>aux_before
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_scheduler_selection_and_history_use_scaled_5d_even_when_multi_disagrees(tmp_path,monkeypatch):
    training=importlib.import_module('cmgm.training.train');model=make()
    x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=3)
    epoch=[0];seen=[]
    def fake_epoch(model,*a,**kw):
        epoch[0]+=1
        model._last_train_objective={'raw_L5':.1,'scaled_prediction_loss':.4,'switch_loss':.02,'total_loss':.42}
        return .42
    def fake_val(model,*a,**kw):
        val=[.4,.2,.3][epoch[0]-1];aux=[.9,1.,.1][epoch[0]-1]
        model._last_val_objective={'raw_L5':val/4,'scaled_prediction_loss':val,'aux_multi_horizon_loss':aux,'per_horizon':{}}
        return val
    class Scheduler:
        def __init__(self,optimizer,**kwargs):
            assert kwargs=={'mode':'min','factor':.5,'patience':0}
            assert len(optimizer.param_groups)==1
            assert optimizer.param_groups[0]['lr']==1e-4 and optimizer.param_groups[0]['weight_decay']==1e-5
        def step(self,value):seen.append(value)
    def forbidden(*a,**kw): raise AssertionError('No optimizer step during orchestration test')
    monkeypatch.setattr(training,'train_epoch',fake_epoch)
    monkeypatch.setattr(training,'validate_epoch',fake_val)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    path=tmp_path/'best.pt'
    h=training.train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),torch.device('cpu'),
                     num_epochs=10,patience=1,checkpoint_path=str(path),checkpoint_metadata={'git_sha':'test','seed':42})
    assert seen==[.4,.2,.3] and h['best_epoch']==2 and h['final_epoch']==3
    saved=torch.load(path,weights_only=True);meta=saved['metadata']
    assert meta['objective']=='4x_5d_only'
    assert meta['best_val_scaled_5d_loss']==.2 and meta['raw_val_5d_loss']==.05
    assert saved['best_val_loss']==.2
    assert h['objective_history'][-1]['val']['aux_multi_horizon_loss']==.1
    assert h['val_loss']==seen


def test_gradients_are_raw_L5_and_preserve_rng_weights_and_grads():
    model=make();x,y=batch()
    for p in model.parameters():p.grad=torch.ones_like(p)
    rng=torch.get_rng_state().clone()
    stats=horizon_gradients(model,(x,y))
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    model.eval();prediction=model(x)
    loss=make_loss()(prediction[:,1],y[:,1])
    param=model.switching_latent_transformer.regime_filter.transition_logits
    grad=torch.autograd.grad(loss,param)[0].norm().item()
    assert stats['norms']['5']['transition logits']==pytest.approx(grad)
    assert stats['five_day_multiplier']==1
    assert {'Market Encoder','LongMemory','Base RPE','state readout'}<=stats['norms']['5'].keys()


def test_synthetic_checkpoint_report_complete_and_initially_identical(tmp_path):
    model=make();baseline=make(BASE_VARIANT);fixed=batch()
    initial=initialization_check(model,fixed)
    loaders={s:DataLoader(TensorDataset(*fixed),batch_size=3) for s in ('train','val','test')}
    base_path=tmp_path/'base.pt'
    torch.save({'model_state_dict':baseline.state_dict(),'best_epoch':1},base_path)
    payload={'best_epoch':1,'history':{},'metadata':{'objective':'4x_5d_only'}}
    result=checkpoint_report(model,payload,{'loaders':loaders},base_path,tmp_path/'report',initialization=initial)
    assert all(v['max']==0 for v in result['representation_drift'].values())
    assert result['case_assessment']['case']=='Case C'
    assert result['integrity']['diagnostic_parameters_unchanged']
    for m in result['models'].values():
        assert m['sanity']['PASS']
        assert all(s in m['splits'] for s in ('TRAIN','VAL','TEST'))
        assert all(h in m['splits']['TRAIN']['native_metrics'] for h in ('1','5','10','20'))
        assert 'uniform' in m['splits']['VAL']['modes']
    text=(tmp_path/'report/REPORT.md').read_text()
    assert '十四个问题' in text and 'scale-matched 5d-only diagnostic objective' in text
    assert 'RoutingFraction_5d' in text and 'STOP' in text


@pytest.mark.parametrize('val,test,case',[(-.01,-.02,'Case A'),(-.0001,-.0002,'Case B'),
    (.0,.0,'Case C'),(.01,.02,'Case D'),(-.01,.01,'Case E'),(.01,-.01,'Case C')])
def test_case_rules_reject_test_only_improvement(val,test,case):
    comparison={s:{'5':{'relative_change':v}} for s,v in (('VAL',val),('TEST',test))}
    result=case_assessment(comparison)
    assert result['case']==case
    assert result['next_5_10_20_eligible']==(case=='Case A')


def test_registration_and_entrypoint_without_training(tmp_path,monkeypatch):
    import sys
    from cmgm.scripts.main_ablation import parse_args,run_variant,select_variants,D_SERIES_VARIANTS
    assert select_variants(DISPLAY)==[(DISPLAY,VARIANT)] and VARIANT in D_SERIES_VARIANTS
    monkeypatch.setattr(sys,'argv',['main_ablation','--variants',DISPLAY])
    args=parse_args()
    assert (args.seed,args.epochs,args.patience,args.batch_size,args.seq_len)==(42,200,10,64,20)
    torch.manual_seed(11)
    x=torch.randn(3,20,6,21);y=torch.randn(3,4,2)*.03
    loaders={s:DataLoader(TensorDataset(x,y),batch_size=3) for s in ('train','val','test')}
    data={'n_nodes':6,'n_commodities':2,'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'loaders':loaders,'norm_stats':{'mean':np.ones(6),'std':np.ones(6)},'raw_prices_test':np.ones((50,6))}
    baseline=make(BASE_VARIANT,feat_dim=21)
    args.d0b_checkpoint=tmp_path/'baseline.pt'
    torch.save({'model_state_dict':baseline.state_dict(),'best_epoch':1},args.d0b_checkpoint)
    args.checkpoint_dir=tmp_path/'checkpoints';args.five_day_report_dir=tmp_path/'reports'
    calls=[]
    def fake_train(model,*a,**kw):
        calls.append(model.variant)
        assert kw['num_epochs']==200 and kw['patience']==10
        assert 'lr' not in kw and 'weight_decay' not in kw
        assert_backbone(model)
        torch.save({'model_state_dict':model.state_dict(),'best_epoch':1,'history':{},
                    'metadata':kw['checkpoint_metadata']},kw['checkpoint_path'])
        return {}
    training=importlib.import_module('cmgm.training.train')
    monkeypatch.setattr(training,'train',fake_train)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    def forbidden_rng(*a,**kw):raise AssertionError('CPU setup must not initialize GPU RNG')
    monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden_rng)
    result=run_variant(DISPLAY,VARIANT,args,torch.device('cpu'),data)
    assert calls==[VARIANT] and result['variant']==DISPLAY
    assert (args.checkpoint_dir/(VARIANT+'_best.pt')).is_file()
    assert result['diagnostics']['transition_drift']==0
