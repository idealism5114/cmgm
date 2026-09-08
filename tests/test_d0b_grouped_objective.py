"""Synthetic grouped-objective verification; no real training experiment."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import _prediction_loss, make_loss
from cmgm.scripts.d0b_grouped_diagnostics import (
    VARIANT, DISPLAY, FIVE_VARIANT, BASE_VARIANT, GROUPED_OBJECTIVE,
    initialization_check, extra_fixed_diagnostics, checkpoint_report,
    full_reference, checked_load, case_assessment,
)


def make(variant=VARIANT,feat_dim=5):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(6,2,n_stock=2,n_bond=2,feat_dim=feat_dim,variant=variant)


def batch():
    gen=torch.Generator().manual_seed(176)
    return torch.randn(3,7,6,5,generator=gen),torch.randn(3,4,2,generator=gen)*.03


def test_full_scale_initialization_parameters_all_forward_traces_and_loss_ratio():
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
    x=torch.randn(2,20,284,21);y=torch.randn(2,4,24)*.03
    rng=torch.get_rng_state().clone()
    result=initialization_check(model,(x,y))
    assert result['PASS'] and result['D0B_params']==result['Grouped_params']==520549
    assert result['difference']==0 and result['mismatch_count']==0 and result['max_abs_diff']==0
    assert all(v['max']==0 for v in result['forward_differences'].values())
    assert torch.equal(torch.get_rng_state(),rng) and model.training
    s=result['loss_scale']
    assert s['group_raw']==pytest.approx(s['L5']+s['L10']+s['L20'])
    assert s['group_scaled']==pytest.approx((4/3)*s['group_raw'])
    assert s['ratio']==pytest.approx(s['group_scaled']/s['sum_multi'])
    b=model.switching_latent_transformer
    assert b.balanced_readout and b.regime_filter.sticky_alpha_value()==.5
    assert not any((b.use_dynamic_slope,b.use_latent_memory,b.use_balanced_transition_input,b.use_regime_relative_memory,b.regime_filter.learnable_sticky_alpha))


def test_dynamic_indices_only_drop_1d_and_fixed_scale(monkeypatch):
    training=importlib.import_module('cmgm.training.train')
    monkeypatch.setattr(training,'MULTI_HORIZONS',[10,1,20,5])
    pred=torch.randn(2,4,3,requires_grad=True);target=torch.randn_like(pred)
    model=SimpleNamespace(variant=VARIANT)
    raw=sum(make_loss()(pred[:,i],target[:,i]) for i in (3,0,2))
    actual=_prediction_loss(model,pred,target,make_loss())
    torch.testing.assert_close(actual,(4/3)*raw,rtol=0,atol=0)
    g=torch.autograd.grad(actual,pred,retain_graph=True)[0]
    torch.testing.assert_close(g,torch.autograd.grad((4/3)*raw,pred)[0],rtol=0,atol=0)
    assert torch.count_nonzero(g[:,1])==0
    assert all(torch.count_nonzero(g[:,i])>0 for i in (0,2,3))
    changed=target.clone();changed[:,1]+=100
    torch.testing.assert_close(_prediction_loss(model,pred,changed,make_loss()),actual,rtol=0,atol=0)
    for variant in (BASE_VARIANT,FIVE_VARIANT,'switching_latent_learnable_persistence'):
        value=_prediction_loss(SimpleNamespace(variant=variant),pred,target,make_loss())
        expected=4*make_loss()(pred[:,3],target[:,3]) if variant==FIVE_VARIANT else sum(make_loss()(pred[:,i],target[:,i]) for i in range(4))
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
    with pytest.raises(ValueError):_prediction_loss(model,pred[:,:3],target[:,:3],make_loss())


def test_train_validate_loss_and_gradients_keep_switch_drop_1d_only(monkeypatch):
    training=importlib.import_module('cmgm.training.train')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.variant=VARIANT;self.graph_learner=None
            self.pred=torch.nn.Parameter(torch.full((4,2),.07))
            self.switch=torch.nn.Parameter(torch.tensor(.1))
            self.switching_latent_transformer=SimpleNamespace(null_control=False,switch_loss=lambda:self.switch.square())
        def forward(self,x,**kw):return self.pred[None].expand(len(x),-1,-1)
    model=Model();grads=[]
    class NoUpdate:
        def zero_grad(self):model.zero_grad(set_to_none=True)
        def step(self):grads.append(model.pred.grad.clone())
    x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    calls=[];original=training._prediction_loss
    def spy(*a):calls.append(a[0].training);return original(*a)
    monkeypatch.setattr(training,'_prediction_loss',spy)
    edges=torch.empty(2,0,dtype=torch.long);weights=torch.zeros(0)
    loss=training.train_epoch(model,loader,edges,weights,NoUpdate(),make_loss(),torch.device('cpu'))
    assert calls==[True,True]
    assert all(torch.count_nonzero(g[0])==0 and all(torch.count_nonzero(g[i])>0 for i in (1,2,3)) for g in grads)
    assert model.switch.grad.item()==pytest.approx(.2)
    train_stats=model._last_train_objective
    assert train_stats['group_raw']==pytest.approx(sum(train_stats[f'raw_L{h}'] for h in (5,10,20)))
    assert train_stats['group_scaled']==pytest.approx((4/3)*train_stats['group_raw'])
    assert loss==pytest.approx(train_stats['group_scaled']+.01)
    val=training.validate_epoch(model,loader,edges,weights,make_loss(),torch.device('cpu'))
    assert calls==[True,True,False,False]
    assert val==pytest.approx(train_stats['group_scaled'])
    changed=y.clone();changed[:,0]+=100
    raw_before=model._last_val_objective['raw_L1']
    other=DataLoader(TensorDataset(x,changed),batch_size=2)
    assert training.validate_epoch(model,other,edges,weights,make_loss(),torch.device('cpu'))==val
    assert model._last_val_objective['raw_L1']>raw_before
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_grouped_scheduler_best_selection_raw_val5_metadata_and_time(tmp_path,monkeypatch):
    training=importlib.import_module('cmgm.training.train');model=make();x,y=batch()
    loader=DataLoader(TensorDataset(x,y),batch_size=3);epoch=[0];seen=[]
    def fake_train(m,*a,**kw):
        epoch[0]+=1
        m._last_train_objective={'raw_L1':.7,'raw_L5':.1,'raw_L10':.2,'raw_L20':.3,
                                 'group_raw':.6,'group_scaled':.8,'switch_loss':.01,'total_loss':.81}
        return .81
    def fake_val(m,*a,**kw):
        v=[.8,.4,.6][epoch[0]-1];five=[.1,.2,.03][epoch[0]-1]
        m._last_val_objective={'raw_L1':10.,'raw_L5':five,'raw_L10':.03,'raw_L20':.07,
                              'group_raw':v/(4/3),'group_scaled':v,'aux_multi_horizon_loss':[11.,12.,1.][epoch[0]-1]}
        return v
    class Scheduler:
        def __init__(self,optimizer,**kw):
            assert len(optimizer.param_groups)==1
            assert optimizer.param_groups[0]['lr']==1e-4 and optimizer.param_groups[0]['weight_decay']==1e-5
            assert kw=={'mode':'min','factor':.5,'patience':0}
        def step(self,value):seen.append(value)
    def forbidden(*a,**kw):raise AssertionError('Synthetic selection test must not optimize')
    monkeypatch.setattr(training,'train_epoch',fake_train);monkeypatch.setattr(training,'validate_epoch',fake_val)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler);monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    path=tmp_path/'best.pt'
    h=training.train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),torch.device('cpu'),
                     num_epochs=10,patience=1,checkpoint_path=str(path),checkpoint_metadata={'git_sha':'synthetic','seed':42})
    assert h['best_epoch']==2 and h['final_epoch']==3 and seen==[.8,.4,.6]
    saved=torch.load(path,weights_only=True);meta=saved['metadata']
    assert meta['best_grouped_val_loss']==.4 and meta['raw_val_5d_loss']==.2
    assert meta['objective']==GROUPED_OBJECTIVE and meta['objective_multiplier']==4/3
    assert meta['parameter_count']==sum(p.numel() for p in model.parameters())
    assert meta['training_elapsed_seconds']>=0 and 'training_elapsed_seconds' in saved['history']


def test_base_rpe_measurement_masks_future_entries_and_readout_weights():
    m=make().eval();x,y=batch()
    with torch.no_grad():
        trace=full_reference(m,x);before=extra_fixed_diagnostics(m,trace)
        memory=m.switching_latent_transformer.long_memory
        future=torch.ones((7,7),dtype=torch.bool).triu(1)
        for layer in memory.layers:layer.attention.last_qk_logits[...,future]=1e6
        memory.last_base_relative_bias=memory.last_base_relative_bias.clone()
        memory.last_base_relative_bias[...,future]=1e6
        after=extra_fixed_diagnostics(m,trace)
    assert before==after
    assert before['readout_weights']['W_long_norm']==pytest.approx(m.switching_latent_transformer.long_memory_readout.weight.norm().item())
    assert before['base_rpe']['base_QK_ratio']>0


def comparison_for(base_change,only_change,aux_change=0.):
    result={}
    for s,i in (('VAL',0),('TEST',1)):
        result[s]={}
        for h in (1,5,10,20):
            m=1.+(base_change[i] if h==5 else aux_change)
            result[s][str(h)]={'grouped_vs':{'D0B':{'relative_change':base_change[i] if h==5 else aux_change},
                                             '5dOnly':{'relative_change':only_change[i]}},
                              'metrics':{'D0B':{'MAE':1.,'RMSE':1.},'Grouped':{'MAE':m,'RMSE':m}}}
    return result


@pytest.mark.parametrize('base,only,aux,case',[
    ((-.01,-.02),(-.1,-.1),0.,'Case A'),((-.0001,-.0002),(-.1,-.1),0.,'Case B'),
    ((.01,.01),(0.,0.),0.,'Case C'),((-.01,.01),(0.,0.),0.,'Case D'),
    ((0.,0.),(0.,0.),0.,'Case E'),((-.01,-.01),(.01,.01),0.,None),
    ((-.01,-.01),(-.1,-.1),.1,None),((.01,-.01),(-.1,-.1),0.,'Case D')])
def test_case_a_requires_both_references_both_splits_and_healthy_aux(base,only,aux,case):
    result=case_assessment(comparison_for(base,only,aux))
    assert result['case']==case and result['candidate_objective_justified']==(case=='Case A')


def test_three_checkpoint_report_synthetic(tmp_path):
    models=[make(v) for v in (BASE_VARIANT,FIVE_VARIANT,VARIANT)];fixed=batch()
    paths=[tmp_path/f'{i}.pt' for i in range(3)]
    for m,p,obj in zip(models,paths,(None,'4x_5d_only',GROUPED_OBJECTIVE)):
        torch.save({'model_state_dict':m.state_dict(),'best_epoch':1,'history':{},'metadata':{'variant':m.variant,'objective':obj}},p)
    with pytest.raises(ValueError):checked_load(models[2],paths[0],torch.device('cpu'),GROUPED_OBJECTIVE)
    payload=checked_load(models[2],paths[2],torch.device('cpu'),GROUPED_OBJECTIVE)
    initial=initialization_check(models[2],fixed)
    loaders={s:DataLoader(TensorDataset(*fixed),batch_size=3) for s in ('train','val','test')}
    result=checkpoint_report(models[2],payload,{'loaders':loaders},paths[0],paths[1],tmp_path/'report',initialization=initial,checkpoint_path=paths[2])
    assert result['assessment']['case']=='Case E'
    assert set(result['models'])=={'D0B','5dOnly','Grouped'}
    assert all(v['max']==0 for d in result['representation_drift'].values() for v in d.values())
    assert all(m['sanity']['PASS'] and m['train_time_seconds'] is None for m in result['models'].values())
    assert all('base_rpe' in m['fixed'] and 'Base RPE' in m['gradients']['norms']['5'] for m in result['models'].values())
    text=(tmp_path/'report/REPORT.md').read_text()
    assert '十七个问题' in text and 'Conflict geometry' in text and 'STOP' in text


def test_entrypoint_registers_one_grouped_run_without_training(tmp_path,monkeypatch):
    import sys
    from cmgm.scripts.main_ablation import run_variant,parse_args,select_variants,D_SERIES_VARIANTS
    assert select_variants(DISPLAY)==[(DISPLAY,VARIANT)] and VARIANT in D_SERIES_VARIANTS
    monkeypatch.setattr(sys,'argv',['main_ablation','--variants',DISPLAY]);args=parse_args()
    assert (args.epochs,args.patience,args.seed,args.batch_size,args.seq_len)==(200,10,42,64,20)
    torch.manual_seed(11);x=torch.randn(3,20,6,21);y=torch.randn(3,4,2)*.03
    data={'n_nodes':6,'n_commodities':2,'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'loaders':{s:DataLoader(TensorDataset(x,y),batch_size=3) for s in ('train','val','test')},
          'norm_stats':{'mean':np.ones(6),'std':np.ones(6)},'raw_prices_test':np.ones((50,6))}
    args.d0b_checkpoint=tmp_path/'base.pt';args.five_day_checkpoint=tmp_path/'five.pt'
    args.checkpoint_dir=tmp_path/'checkpoints';args.grouped_report_dir=tmp_path/'reports'
    for variant,path,obj in ((BASE_VARIANT,args.d0b_checkpoint,None),(FIVE_VARIANT,args.five_day_checkpoint,'4x_5d_only')):
        m=make(variant,feat_dim=21)
        torch.save({'model_state_dict':m.state_dict(),'best_epoch':1,'metadata':{'objective':obj}},path)
    calls=[]
    def fake_train(m,*a,**kw):
        calls.append(m.variant)
        assert kw['num_epochs']==200 and kw['patience']==10 and 'lr' not in kw and 'weight_decay' not in kw
        torch.save({'model_state_dict':m.state_dict(),'best_epoch':1,'history':{},'metadata':kw['checkpoint_metadata']},kw['checkpoint_path'])
        return {}
    monkeypatch.setattr(importlib.import_module('cmgm.training.train'),'train',fake_train)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    def forbidden(*a,**kw):raise AssertionError('CPU setup must not initialize CUDA RNG')
    monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden)
    result=run_variant(DISPLAY,VARIANT,args,torch.device('cpu'),data)
    assert calls==[VARIANT] and result['variant']==DISPLAY
    assert result['diagnostics']['transition_drift']==0
    assert (args.checkpoint_dir/(VARIANT+'_best.pt')).is_file()
