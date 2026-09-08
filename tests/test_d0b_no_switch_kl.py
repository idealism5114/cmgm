"""NoSwitchKL synthetic logic tests. No market-data training or optimizer updates."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_no_switch_kl_diagnostics import (
    VARIANT,BASE_VARIANT,DISPLAY,initialization_check,switch_gradient_probe,
    fixed_regime_probe,checkpoint_report,load_no_switch,collapse_flags,assess,
)
from cmgm.training.train import _prediction_loss,_effective_switch_loss,make_loss


def make(variant=VARIANT,feat_dim=5):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(6,2,n_stock=2,n_bond=2,feat_dim=feat_dim,variant=variant)


def batch():
    gen=torch.Generator().manual_seed(177)
    return torch.randn(3,7,6,5,generator=gen),torch.randn(3,4,2,generator=gen)*.03


def test_full_dimensions_shared_init_train_test_and_nontrivial_loss_sanity():
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
    batches={s:(torch.randn(2,20,284,21),torch.randn(2,4,24)*.03) for s in ('TRAIN','TEST')}
    rng=torch.get_rng_state().clone()
    result=initialization_check(model,batches)
    assert result['PASS'] and result['D0B_params']==result['NoSwitchKL_params']==520549
    assert result['difference']==0 and result['mismatch_count']==0 and result['max_abs_diff']==0
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    b=model.switching_latent_transformer
    assert b.regime_filter.current_epoch==1
    assert b.regime_filter.beta_max==5e-4 and b.regime_filter.warmup_epochs==20
    assert b.regime_filter.sticky_alpha_value()==.5 and b.regime_filter.tau==1
    for stats in result['batches'].values():
        assert all(v['max']==0 for v in stats['forward_differences'].values())
        assert stats['prediction_loss_abs_diff']==0
        late=stats['total_loss_sanity']['20']
        assert late['switch_raw_KL']>0 and late['weighted_switch_loss_D0B']>0
        assert late['weighted_switch_loss_NoSwitchKL']==0 and late['sanity_diff']<1e-8


def test_only_actual_kl_disabled_and_all_four_horizon_prediction_gradients_retained():
    model=make();b=model.switching_latent_transformer
    model(batch()[0]);b.set_epoch(20)
    zero=_effective_switch_loss(model,b)
    assert zero.item()==0 and not zero.requires_grad and b.switch_loss().item()>0
    baseline=make(BASE_VARIANT);baseline(batch()[0]);baseline.switching_latent_transformer.set_epoch(20)
    torch.testing.assert_close(_effective_switch_loss(baseline,baseline.switching_latent_transformer),baseline.switching_latent_transformer.switch_loss(),rtol=0,atol=0)
    pred=torch.randn(2,4,3,requires_grad=True);target=torch.randn_like(pred)
    criterion=make_loss();expected=sum(criterion(pred[:,i],target[:,i]) for i in range(4))
    actual=_prediction_loss(model,pred,target,criterion)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    grad=torch.autograd.grad(actual,pred)[0]
    assert all(torch.count_nonzero(grad[:,i])>0 for i in range(4))
    for epoch in (1,5,10,20,200):
        b.set_epoch(epoch);assert _effective_switch_loss(model,b).item()==0


def test_train_backward_equals_prediction_only_and_validation_unchanged(monkeypatch):
    training=importlib.import_module('cmgm.training.train');model=make()
    model.switching_latent_transformer.set_epoch(20)
    x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=3)
    before={k:v.clone() for k,v in model.state_dict().items()};expected=[]
    original=training._prediction_loss
    def spy(m,p,t,c):
        loss=original(m,p,t,c)
        if m.training:
            expected.extend(torch.autograd.grad(loss,list(m.parameters()),retain_graph=True,allow_unused=True))
        return loss
    monkeypatch.setattr(training,'_prediction_loss',spy)
    class NoUpdate:
        def zero_grad(self):model.zero_grad(set_to_none=True)
        def step(self):
            for p,g in zip(model.parameters(),expected):
                if g is None:assert p.grad is None or torch.count_nonzero(p.grad)==0
                else:torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    ei=torch.empty(2,0,dtype=torch.long);ew=torch.zeros(0)
    total=training.train_epoch(model,loader,ei,ew,NoUpdate(),make_loss(),torch.device('cpu'))
    stats=model._last_train_objective
    assert total==stats['prediction_loss']==stats['total_loss']
    assert stats['raw_KL']>0 and stats['weighted_switch_loss']==stats['beta_effective']==0
    actual=training.validate_epoch(model,loader,ei,ew,make_loss(),torch.device('cpu'))
    with torch.no_grad():
        pred=model(x);manual=sum(make_loss()(pred[:,i],y[:,i]) for i in range(4)).item()
    assert actual==manual
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_counterfactual_gradients_present_without_changing_actual_gradient_or_rng():
    for variant in (VARIANT,BASE_VARIANT):
        model=make(variant).train();model.switching_latent_transformer.set_epoch(20)
        for p in model.parameters():p.grad=torch.ones_like(p)
        rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in model.state_dict().items()}
        r=switch_gradient_probe(model,batch())
        assert torch.equal(rng,torch.get_rng_state()) and model.training
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
        for name in ('regime evidence','transition logits'):
            v=r['modules'][name]
            assert v['raw_KL_norm']>0 and v['counterfactual_switch_norm']>0
            assert v['cos_prediction_switch'] is not None
            if variant==VARIANT:assert v['total_prediction_max_diff']==0
            else:assert v['total_prediction_max_diff']>0
        if variant==VARIANT:
            assert r['actual_total_loss']==r['prediction_loss'] and r['actual_weighted_switch_loss']==0
    model=make();model.switching_latent_transformer.set_epoch(1)
    r=switch_gradient_probe(model,batch())
    assert r['modules']['regime evidence']['cos_prediction_switch'] is None
    assert r['modules']['regime evidence']['raw_KL_norm']>0


def test_history_effective_beta_trajectory_callbacks_selection_and_metadata(tmp_path,monkeypatch):
    training=importlib.import_module('cmgm.training.train');model=make();fixed=batch()
    loader=DataLoader(TensorDataset(*fixed),batch_size=3);epochs=[0];seen=[];stages=[]
    def fake_train(m,*a,**kw):
        epochs[0]+=1
        m._last_train_objective={'prediction_loss':.1,'raw_KL':.03,'posterior_entropy':1.,'prior_entropy':1.08,
                                'posterior_prior_L1':.05,'weighted_switch_loss':0.,'total_loss':.1,'beta_effective':0.}
        return .1
    def fake_val(*a,**kw):return abs(epochs[0]-21)+1.
    class Scheduler:
        def __init__(self,opt,**kw):
            assert len(opt.param_groups)==1 and opt.param_groups[0]['lr']==1e-4 and opt.param_groups[0]['weight_decay']==1e-5
            assert kw=={'mode':'min','factor':.5,'patience':1}
        def step(self,v):seen.append(v)
    def callback(m,stage):stages.append(stage);return fixed_regime_probe(m,fixed,stage)
    def forbidden(*a,**kw):raise AssertionError('History test must not optimize')
    monkeypatch.setattr(training,'train_epoch',fake_train);monkeypatch.setattr(training,'validate_epoch',fake_val)
    monkeypatch.setattr(torch.optim.lr_scheduler,'ReduceLROnPlateau',Scheduler);monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    path=tmp_path/'best.pt'
    h=training.train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),torch.device('cpu'),num_epochs=30,patience=2,
                     checkpoint_path=str(path),checkpoint_metadata={'git_sha':'synthetic','seed':42},epoch_diagnostic=callback)
    assert h['best_epoch']==21 and h['final_epoch']==23
    assert stages==['epoch1','epoch5','epoch10','epoch20','final(epoch 23)','best(epoch 21)']
    assert all(v==0 for v in h['switch_beta'])
    assert h['objective_history'][19]['reference_schedule_beta']==5e-4
    assert all(v['weighted_switch_loss']==0 for v in h['epoch_diagnostics'].values())
    saved=torch.load(path,weights_only=True);meta=saved['metadata']
    assert meta['switch_kl_enabled'] is False and meta['beta_effective']==0
    assert meta['best_val_loss']==min(seen)==1. and meta['objective']=='sum_1d_5d_10d_20d'
    assert meta['seed']==42 and meta['reference_beta_max']==5e-4
    assert meta['parameter_count']==sum(p.numel() for p in model.parameters())


def test_report_synthetic_and_checkpoint_identity(tmp_path):
    model=make();base=make(BASE_VARIANT);fixed=batch()
    initial=initialization_check(model,{'TRAIN':fixed,'TEST':fixed})
    paths=[tmp_path/'base.pt',tmp_path/'new.pt']
    torch.save({'model_state_dict':base.state_dict(),'best_epoch':20},paths[0])
    payload={'model_state_dict':model.state_dict(),'best_epoch':20,'history':{},
             'metadata':{'variant':VARIANT,'switch_kl_enabled':False,'beta_effective':0.}}
    torch.save(payload,paths[1])
    with pytest.raises(ValueError):load_no_switch(model,paths[0],torch.device('cpu'))
    payload=load_no_switch(model,paths[1],torch.device('cpu'))
    data={'loaders':{s:DataLoader(TensorDataset(*fixed),batch_size=3) for s in ('train','val','test')}}
    result=checkpoint_report(model,payload,data,paths[0],tmp_path/'report',initialization=initial,checkpoint_path=paths[1])
    assert result['assessment']['case']=='Case D'
    assert result['integrity']['checkpoint_files_unchanged']
    for m in result['models'].values():
        assert all(v==0 for v in m['sanity']['causality'].values())
        assert all('min_probability' in s['normal_regime'] for s in m['splits'].values())
        assert 'base_rpe' in m['fixed'] and 'switch_gradients' in m
    text=(tmp_path/'report/REPORT.md').read_text()
    assert '十八个问题' in text and 'Counterfactual' in text and 'STOP' in text
    # A requires both split performance and functional-routing evidence.
    import copy
    mods=copy.deepcopy(result['models']);cmp=copy.deepcopy(result['comparison'])
    for s in ('VAL','TEST'):
        cmp[s]['5']['relative_change']=-.01
    assert assess(mods,cmp)['case'] is None  # Equal routing is not Case A.
    for s in ('VAL','TEST'):
        mods['NoSwitchKL']['splits'][s]['modes']['uniform']['impact']['per_horizon']['5']['mean'] += .01
    assert assess(mods,cmp)['case']=='Case A'
    mods['NoSwitchKL']['collapse_checks']['TEST']['hard_posterior_warning']=True
    assert assess(mods,cmp)['case'] is None
    mods['NoSwitchKL']['collapse_checks']['TEST']['hard_posterior_warning']=False
    for s in ('VAL','TEST'):cmp[s]['5']['relative_change']=-.0001
    assert assess(mods,cmp)['case']=='Case B'
    for s in ('VAL','TEST'):cmp[s]['5']['relative_change']=.01
    assert assess(mods,cmp)['case']=='Case C'
    cmp['TEST']['5']['relative_change']=-.01
    assert assess(mods,cmp)['case']=='Case F'
    for s in ('VAL','TEST'):
        cmp[s]['5']['relative_change']=0.
        mods['NoSwitchKL']['splits'][s]['normal_regime']['entropy']-=.1
    assert assess(mods,cmp)['case']=='Case E'


def test_occupancy_warning_alone_does_not_claim_hard_collapse():
    p={'occupancy':[.99,.01,0.],'entropy':1.,'mean_max':.4,'mean':[.4,.3,.3],'min_probability':.2}
    item={'splits':{'TEST':{'normal_regime':p,'candidate_specialization':{'pairwise':{'0-1':{'mean_abs':.1},'0-2':{'mean_abs':.1},'1-2':{'mean_abs':.1}}}}}}
    r=collapse_flags(item)['TEST']
    assert r['argmax_concentration_warning'] and not r['hard_posterior_warning'] and not r['near_identical_candidate_warning']


def test_no_switch_entrypoint_without_actual_training(tmp_path,monkeypatch):
    import sys
    from cmgm.scripts.main_ablation import run_variant,select_variants,parse_args,D_SERIES_VARIANTS
    assert select_variants(DISPLAY)==[(DISPLAY,VARIANT)] and VARIANT in D_SERIES_VARIANTS
    monkeypatch.setattr(sys,'argv',['main_ablation','--variants',DISPLAY]);args=parse_args()
    assert (args.epochs,args.patience,args.seed,args.batch_size,args.seq_len)==(200,10,42,64,20)
    torch.manual_seed(11);x=torch.randn(3,20,6,21);y=torch.randn(3,4,2)*.03
    data={'n_nodes':6,'n_commodities':2,'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'loaders':{s:DataLoader(TensorDataset(x,y),batch_size=3) for s in ('train','val','test')},
          'norm_stats':{'mean':np.ones(6),'std':np.ones(6)},'raw_prices_test':np.ones((50,6))}
    base=make(BASE_VARIANT,feat_dim=21);args.d0b_checkpoint=tmp_path/'base.pt'
    torch.save({'model_state_dict':base.state_dict(),'best_epoch':20},args.d0b_checkpoint)
    args.checkpoint_dir=tmp_path/'checkpoints';args.no_switch_report_dir=tmp_path/'reports'
    calls=[]
    def fake_train(m,*a,**kw):
        calls.append(m.variant)
        assert kw['num_epochs']==200 and kw['patience']==10 and 'lr' not in kw and 'weight_decay' not in kw
        assert m.disable_switch_kl
        torch.save({'model_state_dict':m.state_dict(),'best_epoch':20,'history':{},'metadata':kw['checkpoint_metadata']},kw['checkpoint_path'])
        return {}
    monkeypatch.setattr(importlib.import_module('cmgm.training.train'),'train',fake_train)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    def forbidden(*a,**kw):raise AssertionError('CPU setup must not initialize CUDA RNG')
    monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden)
    result=run_variant(DISPLAY,VARIANT,args,torch.device('cpu'),data)
    assert calls==[VARIANT] and result['variant']==DISPLAY
    assert result['diagnostics']['transition_drift']==0
    assert (args.checkpoint_dir/(VARIANT+'_best.pt')).is_file()
