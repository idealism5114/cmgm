import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from cmgm.models.hetero_mixhop_model import EdgeAttnMixHop,HeteroMixHopCMGM
from cmgm.scripts.d0b_hybrid_graph_diagnostics import (
    BASE,VARIANT,initialization_check,fixed_sanity,attention_diagnostics,gradient_diagnostics,
    _head_summary,
)
from cmgm.scripts.d0b_hybrid_graph_report import build_comparison,improvement_shares,write_report
from cmgm.training.train import validate_epoch,_prediction_loss,make_loss

spec=importlib.util.spec_from_file_location('frozen_edge_attention',Path(__file__).parent/'fixtures/edge_attn_mixhop_before_hybrid.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('hard',[False,True])
@pytest.mark.parametrize('cross',[False,True])
def test_legacy_none_exact_outputs_initialization_and_gradients(batched,hard,cross):
    mask=torch.eye(5,dtype=torch.bool) if cross else None
    kwargs=dict(in_dim=16,out_dim=16,n_heads=8,dropout=.1,hard_mask=hard,prior_scale=.5,cross_mask=mask)
    torch.manual_seed(42);old=legacy.EdgeAttnMixHop(**kwargs)
    torch.manual_seed(42);new=EdgeAttnMixHop(**kwargs)
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    x=torch.randn((2,5,16) if batched else (5,16));A=torch.rand(5,5)
    A[A<.3]=0
    torch.manual_seed(90);y1=old(x,A);g1=torch.autograd.grad(y1.square().mean(),tuple(old.parameters()))
    torch.manual_seed(90);y2=new(x,A);g2=torch.autograd.grad(y2.square().mean(),tuple(new.parameters()))
    torch.testing.assert_close(y1,y2,rtol=0,atol=0)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('n_graph',[0,4,8])
def test_partition_and_first_hop_direct_A_independence(batched,n_graph):
    torch.manual_seed(42)
    layer=EdgeAttnMixHop(16,16,n_heads=8,dropout=0,prior_scale=.5,graph_prior_heads=n_graph)
    layer.capture_attention=True
    x=torch.randn((2,5,16) if batched else (5,16));A=torch.rand(5,5,requires_grad=True)
    pred=layer(x,A);record=layer.last_attention_diagnostics[0]
    nf=8-n_graph;expected=.5*torch.log(A.detach().clamp_min(0)+1e-6)
    if nf:torch.testing.assert_close(record['logits'][...,:nf],record['content'][...,:nf],rtol=0,atol=0)
    if n_graph:
        diff=record['logits'][...,nf:]-record['content'][...,nf:]
        torch.testing.assert_close(diff,expected[...,None].expand_as(diff),rtol=0,atol=1e-6)
    first_free=record['logits'][...,:nf].clone()
    layer(x,torch.ones_like(A)*.1)
    torch.testing.assert_close(layer.last_attention_diagnostics[0]['logits'][...,:nf],first_free,rtol=0,atol=0)
    grads=torch.autograd.grad(pred.square().mean(),tuple(layer.parameters())+(A,),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads[:-1])
    if n_graph:assert grads[-1].abs().sum()>0
    else:assert grads[-1] is None or grads[-1].abs().sum()==0


def test_partition_does_not_modify_hard_mask_path():
    torch.manual_seed(1);a=EdgeAttnMixHop(16,16,n_heads=8,dropout=0,hard_mask=True)
    torch.manual_seed(1);b=EdgeAttnMixHop(16,16,n_heads=8,dropout=0,hard_mask=True,graph_prior_heads=4)
    x=torch.randn(2,5,16);A=torch.eye(5)
    torch.testing.assert_close(a(x,A),b(x,A),rtol=0,atol=0)
    with pytest.raises(ValueError):EdgeAttnMixHop(graph_prior_heads=9)
    with pytest.raises(ValueError):EdgeAttnMixHop(graph_prior_heads=2,cross_mask=torch.eye(5))


def test_large_logits_precision_audit_keeps_raw_error_and_rejects_wrong_bias():
    A=torch.full((3,3),.1);prior=.5*torch.log(A+1e-6)
    content=torch.full((1,3,3,8),1e7);bias=torch.zeros(3,3,8);bias[...,4:]=prior[...,None]
    logits=content+bias
    record=dict(hop=1,content=content,prior_bias=bias,logits=logits,attention=logits.softmax(2))
    result=_head_summary(record,A.double(),2,4,prior)
    assert result['graph_prior_error']>1e-6
    assert not result['float32_subtraction_threshold_PASS']
    assert result['exact_native_logit_recomposition_error']==0
    assert result['float64_bias_subtraction_error']<1e-6 and result['partition_PASS']
    wrong={**record,'logits':logits.clone()};wrong['logits'][...,0]+=2
    assert not _head_summary(wrong,A.double(),2,4,prior)['partition_PASS']


def test_real_parameter_count_and_shared_initialization():
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
    r=initialization_check(model)
    assert r['PASS'] and r['D0B_params']==r['Hybrid_params']==520549
    assert r['shared_parameter_init_max_diff']==0
    b=model.switching_latent_transformer
    assert b.balanced_readout and not b.use_latent_memory and not b.use_dynamic_slope
    assert not b.use_balanced_transition_input and not b.regime_filter.learnable_sticky_alpha
    assert model.attn_mixhop1.cross_mask is None and model.attn_mixhop2.cross_mask is None
    assert model.use_gcn and model.use_lstm and model.use_gate and model.use_edge_attn and not model.use_mixhop


def test_model_sanity_attention_and_graph_gradients_preserve_parameters_rng():
    torch.manual_seed(42);model=HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=VARIANT)
    x=torch.randn(3,20,9,21);y=torch.randn(3,4,3)*.03
    rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in model.state_dict().items()}
    assert fixed_sanity(model,x)['PASS']
    attention=attention_diagnostics(model,x)
    assert attention['PASS'] and attention['layer1']['n_free']==attention['layer1']['n_graph']==4
    grads=gradient_diagnostics(model,(x,y))
    assert grads['prediction_only']['graph_learner.alpha']['connected']
    assert grads['prediction_only']['graph_learner.alpha']['norm']>0
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for k,v in state.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())


class Predictions(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.graph_learner=True
    def forward(self,x,debug=False):return x


def test_secondary_validation_pooled_metrics_do_not_change_original_objective():
    p=torch.arange(40,dtype=torch.float32).view(5,4,2)*.001;y=torch.zeros_like(p)
    loader=DataLoader(TensorDataset(p,y),batch_size=3)
    old=Predictions(BASE);new=Predictions(VARIANT);ei=torch.empty(2,0,dtype=torch.long);ew=torch.empty(0)
    a=validate_epoch(old,loader,ei,ew,make_loss(),torch.device('cpu'))
    b=validate_epoch(new,loader,ei,ew,make_loss(),torch.device('cpu'))
    assert a==b
    expected=p[:,1].double()
    assert new._last_val5_diagnostic['MAE']==pytest.approx(expected.abs().mean().item())
    assert new._last_val5_diagnostic['MSE']==pytest.approx(expected.square().mean().item())
    assert new._last_val5_diagnostic['count']==10
    assert _prediction_loss(new,p,y,make_loss())==_prediction_loss(old,p,y,make_loss())


def test_secondary_best_epoch_cannot_control_formal_checkpoint(tmp_path,monkeypatch):
    import importlib
    mod=importlib.import_module('cmgm.training.train')
    model=nn.Linear(1,1);model.variant=VARIANT
    val=iter([3.,2.,4.]);secondary=iter([.3,.4,.1]);seen=[]
    def fake_validate(model,*args,**kwargs):
        model._last_val5_diagnostic=dict(MAE=next(secondary),MSE=1.,count=10)
        return next(val)
    monkeypatch.setattr(mod,'train_epoch',lambda *a,**k:1.)
    monkeypatch.setattr(mod,'validate_epoch',fake_validate)
    original=mod.optim.lr_scheduler.ReduceLROnPlateau.step
    def step(scheduler,value,*args,**kwargs):
        seen.append(value);return original(scheduler,value,*args,**kwargs)
    monkeypatch.setattr(mod.optim.lr_scheduler.ReduceLROnPlateau,'step',step)
    history=mod.train(model,[],[],torch.empty(2,0),torch.empty(0),torch.device('cpu'),num_epochs=3,
        checkpoint_path=str(tmp_path/'formal.pt'))
    assert seen==[3.,2.,4.] and history['best_epoch']==2
    assert history['best_val5_epoch']==3
    checkpoint=torch.load(tmp_path/'formal.pt',weights_only=False)
    assert checkpoint['best_epoch']==2 and checkpoint['best_val_loss']==2.


def test_error_decomposition_uses_train_thresholds_and_reports_negative_contributions(tmp_path):
    base={};hybrid={}
    for split in ('train','val','test'):
        y=np.array([[.01,-.02],[.08,-.04],[0.,.1]])
        base[split]=dict(target=y,prediction=np.zeros_like(y))
        hybrid[split]=dict(target=y.copy(),prediction=y*.2)
    attention={l:dict(free_entropy=1.,graph_entropy=.5,free_topk_mass=.2,graph_topk_mass=.6) for l in ('layer1','layer2')}
    r=build_comparison(base,hybrid,['焦煤','焦炭'],attention,True,tmp_path)
    assert r['primary_case']=='Case A'
    assert r['train_target_thresholds']['P90']==pytest.approx(np.quantile(np.abs(base['train']['target']),.9))
    shares=improvement_shares([4.,-3.,1.])
    assert shares['net_improvement_sum']==2 and shares['gross_improvements']['largest_share']==.8
    assert shares['largest_positive_share_of_net']==2.
    assert len(pd.read_csv(tmp_path/'overall_metrics.csv'))==12
    for layer in attention.values():
        layer['hops']=[dict(hop=1,free_prior_error=0.,graph_prior_error=0.,prior_formula_error=0.,
            exact_native_logit_recomposition_error=0.,float64_bias_subtraction_error=0.,partition_PASS=True)]
    attention['PASS']=True
    r.update(status='SYNTHETIC TEST',training_executed=True,best_epoch=1,train_time_seconds=0.,best_formal_val_objective=.01,
        history=dict(val5_diagnostic=[dict(MAE=.01)],best_val5_epoch=1,best_val5_mae=.01),
        sanity=dict(PASS=True,shared_init=dict(delta_params=0,shared_parameter_init_max_diff=0)),
        baseline_reference=dict(PASS=True,actual=r['metrics']['D0B']['test']),
        attention={'best':attention},graph={'best':{}},gradients={'best':{}})
    write_report(r,tmp_path)
    report=(tmp_path/'REPORT.md').read_text()
    assert '15. Primary classification: Case A' in report
    assert 'FP64 subtraction error' in report and 'Secondary best-VAL5 epoch' in report
