"""Controlled geometry, frozen legacy, and unchanged selection regression checks."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cmgm.config import MULTI_HORIZONS
from cmgm.models.hetero_mixhop_model import EdgeAttnMixHop,HeteroMixHopCMGM
from cmgm.scripts.d0b_qknorm_diagnostics import BASE,VARIANT,initialization_check,geometry_diagnostics,gradient_diagnostics,norm_stats
from cmgm.scripts.d0b_qknorm_graph_attention import assert_backbone
from cmgm.scripts.d0b_hybrid_graph_diagnostics import fixed_sanity
from cmgm.scripts.d0b_qknorm_report import build_comparison,write_report
from cmgm.training.train import validate_epoch,_prediction_loss,make_loss

spec=importlib.util.spec_from_file_location('before_qknorm',Path(__file__).parent/'fixtures/edge_attn_mixhop_before_qknorm.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('mode',['native','hybrid','hard','cross'])
def test_legacy_false_exact_output_gradients(batched,mode):
    kwargs=dict(in_dim=16,out_dim=16,n_heads=8,dropout=.1,hard_mask=mode=='hard',prior_scale=.5,
        graph_prior_heads=4 if mode=='hybrid' else None,cross_mask=torch.eye(5,dtype=torch.bool) if mode=='cross' else None)
    torch.manual_seed(42);old=legacy.EdgeAttnMixHop(**kwargs)
    torch.manual_seed(42);new=EdgeAttnMixHop(**kwargs,qk_norm=False)
    assert old.state_dict().keys()==new.state_dict().keys()
    for name,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
    x=torch.randn((2,5,16) if batched else (5,16));A=torch.rand(5,5);A[A<.3]=0
    torch.manual_seed(90);a=old(x,A);ga=torch.autograd.grad(a.square().mean(),tuple(old.parameters()))
    torch.manual_seed(90);b=new(x,A);gb=torch.autograd.grad(b.square().mean(),tuple(new.parameters()))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=0,atol=0)

@pytest.mark.parametrize('batched',[False,True])
def test_per_head_geometry_scale_prior_and_unchanged_V(batched):
    torch.manual_seed(42);layer=EdgeAttnMixHop(64,64,n_heads=8,dropout=0,prior_scale=.5,qk_norm=True).eval()
    layer.capture_attention=True
    x=torch.randn((2,5,64) if batched else (5,64));A=torch.rand(5,5)
    v_observations=[]
    handle=layer.v.register_forward_hook(lambda module,args,out:v_observations.append((args[0].detach(),out.detach())))
    output=layer(x,A);handle.remove()
    assert len(layer.last_attention_diagnostics)==2
    for row in layer.last_attention_diagnostics:
        q=row['qk'];scale=math.sqrt(8)
        for key in ('Q','K'):
            torch.testing.assert_close(q[key].norm(dim=-1),torch.full_like(q[key][...,0],scale),atol=1e-6,rtol=0)
        eq='bnhd,bmhd->bnmh' if batched else 'nhd,mhd->nmh'
        cosine=torch.einsum(eq,F.normalize(q['raw_Q'],dim=-1),F.normalize(q['raw_K'],dim=-1))
        torch.testing.assert_close(row['content'],cosine*scale,atol=1e-6,rtol=1e-6)
        assert not torch.allclose(row['content'],cosine/scale)
        expected=row['content']+.5*torch.log(A+1e-6)[...,None]
        torch.testing.assert_close(row['logits'],expected,rtol=0,atol=0)
    for raw,v in v_observations:
        torch.testing.assert_close(v,F.linear(raw,layer.v.weight,layer.v.bias),atol=0,rtol=0)
    # Explicit native MixHop update using captured attention and unnormalized V.
    H=x;expected=layer.Ws[0](x)
    for i in range(2):
        v=layer.v(H).view(*H.shape[:-1],8,8)
        eq='bnmh,bmhd->bnhd' if batched else 'nmh,mhd->nhd'
        agg=torch.einsum(eq,layer.last_attention_diagnostics[i]['attention'],v).reshape_as(H)
        H=layer.beta*x+(1-layer.beta)*layer.out_proj(agg)
        expected=expected+layer.Ws[i+1](H)
    torch.testing.assert_close(output,expected,atol=0,rtol=0)

@pytest.mark.parametrize('value',[0.,1e-12])
def test_zero_or_sub_epsilon_vectors_are_finite(value):
    layer=EdgeAttnMixHop(16,16,n_heads=8,dropout=0,qk_norm=True)
    with torch.no_grad():
        for module in (layer.q,layer.k):module.weight.fill_(value);module.bias.fill_(value)
    layer.capture_attention=True
    output=layer(torch.ones(2,5,16),torch.eye(5))
    assert torch.isfinite(output).all()
    grads=torch.autograd.grad(output.square().mean(),tuple(layer.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    q=layer.last_attention_diagnostics[0]['qk']
    audit=norm_stats(q['raw_Q'],q['Q'],True)
    assert audit['PASS'] and audit['epsilon_degenerate_count']>0

def test_positive_rescaling_raw_qk_does_not_change_first_hop_cosine():
    torch.manual_seed(1);layer=EdgeAttnMixHop(16,16,n_heads=8,dropout=0,qk_norm=True)
    layer.capture_attention=True;x=torch.randn(2,5,16);A=torch.rand(5,5)
    layer(x,A);content=layer.last_attention_diagnostics[0]['content']
    with torch.no_grad():
        layer.q.weight.mul_(4);layer.q.bias.mul_(4);layer.k.weight.mul_(.25);layer.k.bias.mul_(.25)
    layer(x,A)
    torch.testing.assert_close(content,layer.last_attention_diagnostics[0]['content'],rtol=0,atol=0)

def test_real_model_parameter_and_backbone_equality():
    torch.manual_seed(42);model=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
    check=initialization_check(model);assert_backbone(model)
    assert check['PASS'] and check['D0B_params']==check['QKNorm_params']==520549
    assert check['delta_params']==check['shared_parameter_init_max_diff']==check['mismatch_count']==0
    assert not any('qk_norm' in key for key in model.state_dict())

def test_diagnostics_causality_batch_relabeling_and_gradients_no_mutation():
    torch.manual_seed(42);model=HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=VARIANT)
    x=torch.randn(3,20,9,21);y=torch.randn(3,4,3)*.03
    rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in model.state_dict().items()}
    assert fixed_sanity(model,x)['PASS']
    assert geometry_diagnostics(model,x)['PASS']
    g=gradient_diagnostics(model,(x,y))
    for layer in ('attn_mixhop1','attn_mixhop2'):
        for name in ('q','k','v'):assert g['prediction_only'][layer+'.'+name]['norm']>0
    assert g['prediction_only']['graph_learner.alpha']['connected'] and g['prediction_only']['graph_learner.alpha']['norm']>0
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for name,value in state.items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())

class Predictions(nn.Module):
    def __init__(self,variant):super().__init__();self.variant=variant;self.graph_learner=True
    def forward(self,x,debug=False):return x

def test_val5_logging_preserves_multihorizon_objective():
    p=torch.arange(40,dtype=torch.float32).view(5,4,2)*.001;y=torch.zeros_like(p)
    loader=DataLoader(TensorDataset(p,y),batch_size=3);old=Predictions(BASE);new=Predictions(VARIANT)
    args=(loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),make_loss(),torch.device('cpu'))
    assert validate_epoch(old,*args)==validate_epoch(new,*args)
    assert _prediction_loss(old,p,y,make_loss())==_prediction_loss(new,p,y,make_loss())
    e=p[:,MULTI_HORIZONS.index(5)].double()
    assert new._last_val5_diagnostic['MAE']==pytest.approx(e.abs().mean().item())
    assert new._last_val5_diagnostic['MSE']==pytest.approx(e.square().mean().item())

def test_formal_selection_never_uses_secondary_val5(tmp_path,monkeypatch):
    import importlib
    mod=importlib.import_module('cmgm.training.train');model=nn.Linear(1,1);model.variant=VARIANT
    val=iter([3.,2.,4.]);secondary=iter([.3,.4,.1]);seen=[]
    def fake_validate(model,*a,**k):
        model._last_val5_diagnostic=dict(MAE=next(secondary),MSE=1.,count=10);return next(val)
    monkeypatch.setattr(mod,'train_epoch',lambda *a,**k:1.)
    monkeypatch.setattr(mod,'validate_epoch',fake_validate)
    original=mod.optim.lr_scheduler.ReduceLROnPlateau.step
    def step(scheduler,value,*a,**k):seen.append(value);return original(scheduler,value,*a,**k)
    monkeypatch.setattr(mod.optim.lr_scheduler.ReduceLROnPlateau,'step',step)
    history=mod.train(model,[],[],torch.empty(2,0),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(tmp_path/'formal.pt'))
    assert seen==[3.,2.,4.] and history['best_epoch']==2 and history['best_val5_epoch']==3
    assert torch.load(tmp_path/'formal.pt',weights_only=False)['best_epoch']==2


def test_report_and_comparison_with_synthetic_evaluation(tmp_path):
    torch.manual_seed(42);base_model=HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=BASE)
    torch.manual_seed(42);model=HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=VARIANT)
    x=torch.randn(2,20,9,21);y=torch.randn(2,4,3)*.03
    r=dict(status='SYNTHETIC TEST ONLY',training_executed=True,qk_norm={},attention={},gradients={},graph={},
        sanity=dict(PASS=True,legacy_qk_norm_false=True,shared_init=initialization_check(model)))
    for stage,m in [('D0B_best',base_model),('best',model)]:
        d=geometry_diagnostics(m,x);r['qk_norm'][stage]=d['qk_norm'];r['attention'][stage]=d['attention'];r['gradients'][stage]=gradient_diagnostics(m,(x,y))
    base={};new={}
    for split in ('train','val','test'):
        yy=np.array([[.01,-.02,.03],[.08,-.04,0.],[.01,.1,-.06]])
        base[split]=dict(target=yy,prediction=np.zeros_like(yy));new[split]=dict(target=yy.copy(),prediction=yy*.2)
    r.update(build_comparison(base,new,['焦煤','原油','低硫燃料油'],r,tmp_path))
    assert r['primary_case']=='Case A'
    assert r['train_target_thresholds']['P90']==pytest.approx(np.quantile(np.abs(base['train']['target']),.9))
    r.update(best_epoch=1,train_time_seconds=0.,best_formal_val_objective=.01,
        history=dict(val5_diagnostic=[dict(MAE=.01)],best_val5_epoch=1,best_val5_mae=.01),baseline_reference=dict(PASS=True,actual=r['metrics']['D0B']['test']))
    write_report(r,tmp_path)
    text=(tmp_path/'REPORT.md').read_text();assert '16. Primary classification: Case A' in text
    assert 'Secondary best-VAL5 epoch' in text
    assert len(r['commodity_metrics'])==3 and len(r['target_magnitude_groups'])==8
    r.pop('metrics');r.pop('history');write_report(r,tmp_path)
    assert 'No primary case before formal training' in (tmp_path/'REPORT.md').read_text()
