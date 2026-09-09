"""HSR architecture, objective, isolated gradients, immutable diagnostics and reports."""
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.switching_latent_transformer import SwitchingLatentTransformerBranch
from cmgm.training.train import _prediction_loss, make_loss
from cmgm.scripts.d0b_horizon_readout import (
    VARIANT, BASE_VARIANT, NEW, HORIZONS, initialization_check, gradient_probe,
    checkpoint_report, case_assessment, precision_sanity, readout_modules,
)


def make(variant=VARIANT, full=False):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(284 if full else 6,24 if full else 2,n_stock=248 if full else 2,
                           n_bond=12 if full else 2,feat_dim=21 if full else 5,variant=variant)


def batch(full=False):
    g=torch.Generator().manual_seed(82)
    return (torch.randn(2,20,284,21,generator=g) if full else torch.randn(3,7,6,5,generator=g),
            torch.randn(2 if full else 3,4,24 if full else 2,generator=g)*.03)


def test_full_shared_initialization_clone_and_forward_losses():
    m=make(full=True);rng=torch.get_rng_state().clone()
    r=initialization_check(m,batch(full=True))
    assert r['PASS'] and r['D0B_params']==520549 and r['new_params']==545317
    assert r['mismatch_count']==r['max_abs_diff']==0 and r['difference']==24768
    assert m.training and torch.equal(torch.get_rng_state(),rng)
    assert all(r['forward_differences'][f'h_temporal_{h}']['max']==0 for h in HORIZONS)
    base=make(BASE_VARIANT);base_rng=torch.get_rng_state().clone();make()
    assert torch.equal(torch.get_rng_state(),base_rng)
    assert not hasattr(base.switching_latent_transformer,'horizon_state_readouts')


def test_one_dynamics_pass_original_5d_and_old_readout_shape():
    m=make().eval();b=m.switching_latent_transformer;x,y=batch();counts={}
    modules={'market':b.market_encoder,'long':b.long_memory,'evidence':b.regime_filter.regime_evidence,
             **{f'G{i}':g for i,g in enumerate(b.latent_transition.generators)}}
    def hook(name):
        def counted(*args):counts[name]=counts.get(name,0)+1
        return counted
    handles=[mod.register_forward_hook(hook(n)) for n,mod in modules.items()]
    with torch.no_grad():p=m(x)
    for h in handles:h.remove()
    assert counts=={'market':1,'long':1,'evidence':7,'G0':7,'G1':7,'G2':7}
    assert p.shape==(3,4,2)
    assert b.readout(b.last_h_last,b.last_z_last).shape==(3,64)
    assert readout_modules(m)['5'] is b.state_readout
    assert all(readout_modules(m)[h].weight.data_ptr()!=b.state_readout.weight.data_ptr() for h in ('1','10','20'))


def test_select_original_output_rows_and_shared_counterfactual_is_temporary():
    m=make().eval();x,y=batch();b=m.switching_latent_transformer
    with torch.no_grad():
        m(x);H,Z=b.last_h_last,b.last_z_last
        b.horizon_state_readouts['1'].weight.add_(.03*torch.randn_like(b.state_readout.weight))
        state={k:v.clone() for k,v in m.state_dict().items()}
        temporal=b.readout_by_horizon(H,Z);spatial=m._temp_weighted_spatial(x)
        p=m._market_token_predict_by_horizon(spatial,temporal)
        for i,h in enumerate(HORIZONS):
            expected=m._market_token_predict(spatial,temporal[:,i])[:,i]
            torch.testing.assert_close(p[:,i],expected,atol=1e-7,rtol=1e-6)
        shared=b.readout_by_horizon(H,Z,shared_readout=True)
        counter=m._market_token_predict_by_horizon(spatial,shared)
        assert (p[:,0]-counter[:,0]).abs().max()>1e-6
        torch.testing.assert_close(p[:,1:],counter[:,1:],atol=0,rtol=0)
        assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
        # Null micro removes post-LN bias as well, every horizon sees same u.
        captured=[];h=b.state_readout.register_forward_pre_hook(lambda mod,args:captured.append(args[0]))
        b.readout_by_horizon(H,Z,zero_component='Z');h.remove()
        assert torch.count_nonzero(captured[0][:,64:])==0


def test_gradient_isolation_u_geometry_and_no_mutation():
    m=make();x,y=batch();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    for p in m.parameters():p.grad=torch.ones_like(p)*.123
    grads=[p.grad.clone() for p in m.parameters()]
    r=gradient_probe(m,(x,y))
    assert r['readout_isolation_PASS']
    for h,row in r['readout_loss_gradient_matrix'].items():
        assert row[h]>0
        assert all(v==0 for other,v in row.items() if other!=h)
    assert all(v>0 for v in r['u_gradient']['norms'].values())
    assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grads))
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    assert m.training and torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('variant',[BASE_VARIANT,VARIANT,'switching_latent_dynamic_slope',
    'switching_latent_balanced_transition','switching_latent_learnable_persistence'])
def test_original_prediction_loss_and_gradients(variant):
    p=torch.randn(3,4,2,requires_grad=True);y=torch.randn_like(p);criterion=make_loss()
    actual=_prediction_loss(SimpleNamespace(variant=variant),p,y,criterion)
    expected=sum(criterion(p[:,i],y[:,i]) for i in range(4))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    ga=torch.autograd.grad(actual,p,retain_graph=True)[0];ge=torch.autograd.grad(expected,p)[0]
    torch.testing.assert_close(ga,ge,atol=0,rtol=0)


def test_new_branch_rejects_non_d0b_mechanisms():
    kwargs=dict(feat_dim=5,n_stock=2,n_bond=2,n_commodity=2,balanced_readout=True,horizon_specific_state_readout=True)
    for flag in ('learnable_sticky_alpha','use_dynamic_slope','use_balanced_transition_input','use_latent_memory','use_regime_relative_memory'):
        with pytest.raises(ValueError):SwitchingLatentTransformerBranch(**kwargs,**{flag:True})
    with pytest.raises(ValueError):SwitchingLatentTransformerBranch(**kwargs,forecast_horizons=(5,))


def test_causality_all_four_horizons_batch_market_and_rng():
    m=make();x,_=batch();rng=torch.get_rng_state().clone()
    result=precision_sanity(m,x)
    assert result['PASS'] and m.training and torch.equal(torch.get_rng_state(),rng)
    original=result.get('float32',result)
    assert all(f'h_temporal_{h}' in original['causality'] for h in HORIZONS)


def test_training_logging_selection_and_checkpoint_no_updates(tmp_path,monkeypatch):
    t=importlib.import_module('cmgm.training.train')
    m=make();x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=2)
    before=copy.deepcopy(m.state_dict())
    # Exercise actual forward/backward/training plumbing, forbid parameter updates.
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    observed=[];original=t.validate_epoch
    def validation(*a,**kw):
        v=original(*a,**kw);observed.append(v);return v
    monkeypatch.setattr(t,'validate_epoch',validation)
    sched=[];original_step=torch.optim.lr_scheduler.ReduceLROnPlateau.step
    def step(self,metrics,*a,**kw):sched.append(metrics);return original_step(self,metrics,*a,**kw)
    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau,'step',step)
    path=tmp_path/'fixture.pt'
    history=t.train(m,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),'cpu',
                    num_epochs=2,patience=10,checkpoint_path=str(path))
    assert sched==observed
    for row in history['objective_history']:
        assert row['train']['prediction_loss']==pytest.approx(sum(row['train'][f'raw_L{h}'] for h in HORIZONS))
        assert row['train']['total_loss']==pytest.approx(row['train']['prediction_loss']+row['train']['switch_loss'])
        assert row['val']['prediction_loss']==pytest.approx(sum(row['val'][f'raw_L{h}'] for h in HORIZONS))
    saved=torch.load(path,weights_only=True)
    assert saved['best_val_loss']==min(observed)
    assert saved['metadata']['5d_uses_original_state_readout']
    assert saved['metadata']['extra_readout_parameter_count']==24768
    assert saved['metadata']['loss_type']=='huber' and saved['metadata']['switch_beta_max']==5e-4
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())


def test_complete_checkpoint_report_without_training(tmp_path):
    m=make().eval();base=make(BASE_VARIANT).eval();x,y=batch()
    b=m.switching_latent_transformer
    metadata={'variant':VARIANT,'initial_state_readout':{k:v.tolist() for k,v in b.state_readout.state_dict().items()}}
    payload={'model_state_dict':m.state_dict(),'metadata':metadata,'best_epoch':1}
    new_path,base_path=tmp_path/'new.pt',tmp_path/'base.pt'
    torch.save(payload,new_path);torch.save({'model_state_dict':base.state_dict(),'best_epoch':1},base_path)
    loader=DataLoader(TensorDataset(x,y),batch_size=2,drop_last=True)
    data={'loaders':{s:loader for s in ('train','val','test')},'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'feature_names':['a','b','c','d','corn','gold']}
    report=checkpoint_report(m,payload,data,base_path,tmp_path/'report',new_path)
    assert report['integrity']['checkpoint_files_unchanged']
    for name in ('D0B',NEW):
        for split in ('TRAIN','VAL','TEST'):
            r=report['models'][name]['splits'][split]
            assert r['samples']==3  # tail included even if training loader drops it
            assert r['impacts']['shared-readout']['5']['max']==0
            for metrics in r['native_metrics'].values():
                assert metrics['MSE']==pytest.approx(metrics['RMSE']**2)
    md=(tmp_path/'report'/'REPORT.md').read_text()
    assert 'u-gradient' in md and '**STOP**' in md and 'commodity' in md and 'MSE' in md
    assert report['assessment']['case']=='Case E'


@pytest.mark.parametrize('mae,mse,evidence,expected',[
    ((-.01,-.01),(-.01,-.01),True,'Case A'),
    ((-.01,-.01),(-.01,-.01),False,'Case B'),
    ((.01,.01),(.01,.01),True,'Case C'),
    ((-.01,.01),(-.01,.01),True,'Case D'),
    ((0.,0.),(0.,0.),True,'Case E'),
    ((-.01,-.01),(.01,.01),True,'Case F'),
])
def test_case_decisions(mae,mse,evidence,expected):
    comp={s:{'5':{'relative_MAE':mae[i],'relative_MSE':mse[i]}} for i,s in enumerate(('VAL','TEST'))}
    assert case_assessment(comp,evidence)['case']==expected


def test_variant_registration_and_checkpoint_path(tmp_path):
    from cmgm.scripts.main_ablation import select_variants, D_SERIES_VARIANTS, _checkpoint_path_for_variant
    from cmgm.scripts.d0b_horizon_readout import DISPLAY
    assert VARIANT in D_SERIES_VARIANTS
    assert select_variants(DISPLAY)==select_variants(VARIANT)==[(DISPLAY,VARIANT)]
    assert _checkpoint_path_for_variant(VARIANT,tmp_path).name==f'{VARIANT}_best.pt'
