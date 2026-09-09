"""One shared linear residual; strict D0B degeneration and causal node semantics."""
import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import _prediction_loss,make_loss
from cmgm.scripts.d0b_commodity_residual import (
    VARIANT,BASE_VARIANT,NEW,DISPLAY,HORIZONS,initialization_check,reference,
    gradient_probe,commodity_control,commodity_permutation,temporal_control,
    precision_sanity,checkpoint_report,epoch_probe,case_assessment,
)


def make(variant=VARIANT,full=False):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(284 if full else 6,24 if full else 2,n_stock=248 if full else 2,
                           n_bond=12 if full else 2,feat_dim=21 if full else 5,variant=variant)


def batch(full=False):
    g=torch.Generator().manual_seed(85)
    return (torch.randn(2,20,284,21,generator=g) if full else torch.randn(3,7,6,5,generator=g),
            torch.randn(2 if full else 3,4,24 if full else 2,generator=g)*.03)


def test_full_initialization_exact_shared_tensors_and_zero_prediction():
    m=make(full=True);x,y=batch(full=True);rng=torch.get_rng_state().clone()
    result=initialization_check(m,(x,y))
    assert result['PASS'] and result['D0B_params']==520549 and result['new_params']==520806
    assert result['difference']==257 and result['max_abs_diff']==result['mismatch_count']==0
    assert all(v['max']==0 for v in result['forward_differences'].values())
    assert result['initial_residual']['effective_mean_abs']==0
    assert result['initial_residual']['raw_mean_abs']>0
    assert result['gradients']['objectives']['prediction_sum']['norms']['commodity_residual_head.weight']==0
    assert result['gradients']['objectives']['prediction_sum']['signed_alpha_gradient']!=0
    assert m.training and torch.equal(torch.get_rng_state(),rng)
    assert not m.switching_latent_transformer.horizon_specific_state_readout
    assert not hasattr(m.switching_latent_transformer,'horizon_state_readouts')


def test_return_nodes_tap_after_norm_before_pool_and_single_computation():
    m=make().eval();x,_=batch();counts={};captured=[]
    def hook(name):
        def f(*args):counts[name]=counts.get(name,0)+1
        return f
    mods={'spatial1':m.attn_mixhop1,'spatial2':m.attn_mixhop2,'norm':m.gcn_norm,
          'pool':m.type_pool,'temporal':m.switching_latent_transformer,'head':m.head}
    handles=[mod.register_forward_hook(hook(n)) for n,mod in mods.items()]
    handles.append(m.type_pool.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone())))
    with torch.no_grad():p=m(x)
    for h in handles:h.remove()
    assert counts=={k:1 for k in mods}
    torch.testing.assert_close(captured[0][:,4:],m.last_commodity_nodes,rtol=0,atol=0)
    assert captured[0].shape==(3,6,64) and m.last_commodity_nodes.shape==(3,2,64)
    with torch.no_grad():
        old=m._temp_weighted_spatial(x);pooled,nodes=m._temp_weighted_spatial(x,return_nodes=True)
    torch.testing.assert_close(old,pooled,atol=0,rtol=0)
    torch.testing.assert_close(p,m.last_base_pred,atol=0,rtol=0)


def test_initial_head_gradient_zero_then_activation_without_training():
    m=make();x,y=batch();rng=torch.get_rng_state().clone()
    for p in m.parameters():p.grad=torch.ones_like(p)*.2
    grads=[p.grad.clone() for p in m.parameters()]
    first=gradient_probe(m,(x,y))
    assert first['objectives']['prediction_sum']['norms']['commodity_residual_head.weight']==0
    with torch.no_grad():m.commodity_residual_alpha.fill_(.03)
    second=gradient_probe(m,(x,y))
    assert all(v['norms']['commodity_residual_head.weight']>0 for v in second['objectives'].values())
    assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),grads))
    assert torch.equal(torch.get_rng_state(),rng) and m.training


def test_residual_controls_only_change_node_input_and_keep_base_exact():
    m=make().eval();x,_=batch()
    with torch.no_grad():
        m.commodity_residual_alpha.fill_(.1);native=reference(m,x)
        state=copy.deepcopy(m.state_dict());base=m.last_base_pred.clone()
        perm=commodity_permutation(2)
        p0=commodity_control(m,native,'alpha=0',perm)
        shuffled=commodity_control(m,native,'shuffled',perm)
        mean=commodity_control(m,native,'mean-commodity',perm)
        torch.testing.assert_close(p0,base,atol=0,rtol=0)
        torch.testing.assert_close(shuffled,base+m.commodity_residual_alpha*m.commodity_residual_head(native['h_comm'][:,perm]).permute(0,2,1))
        delta=mean-base
        torch.testing.assert_close(delta[:,:,0],delta[:,:,1],atol=2e-8,rtol=1e-6)
        assert (native['prediction']-shuffled).abs().max()>1e-6
        assert torch.equal(base,m.last_base_pred)
        assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())


def test_native_residual_kept_during_HZ_and_uniform_controls():
    m=make().eval();x,_=batch()
    with torch.no_grad():
        m.commodity_residual_alpha.fill_(.03);trace=reference(m,x)
        from cmgm.scripts.d0b_regime_routing_diagnostics import run_intervention
        for mode,spec in [('zero-micro',{'zero_component':'Z'}),('zero-long',{'zero_component':'H'}),('uniform',{'mode':'uniform'})]:
            baseline_control=run_intervention(m,trace['h_spatial'],trace,spec,None,None)
            predicted=temporal_control(m,trace,mode)
            torch.testing.assert_close(predicted,baseline_control['prediction']+trace['effective'],atol=0,rtol=0)
            torch.testing.assert_close(baseline_control['p'],trace['p'],atol=0,rtol=0)


def test_causality_batch_and_legal_commodity_relabeling_with_active_adapter():
    m=make();x,_=batch()
    with torch.no_grad():m.commodity_residual_alpha.fill_(.1)
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    r=precision_sanity(m,x)
    assert r['PASS'] and m.training and torch.equal(torch.get_rng_state(),rng)
    original=r.get('float32',r)
    assert original['residual_shuffle_base_max_diff']==0
    assert max(original['temporal_causality'].values())==0
    assert original['within_market']['commodity']['residual_equivariance_max']<3e-6
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())


@pytest.mark.parametrize('variant',[BASE_VARIANT,VARIANT,'switching_latent_balanced_horizon_readout',
                                  'switching_latent_learnable_persistence','switching_latent_dynamic_slope'])
def test_unchanged_four_horizon_loss_and_gradients(variant):
    p=torch.randn(3,4,2,requires_grad=True);y=torch.randn_like(p);criterion=make_loss()
    actual=_prediction_loss(SimpleNamespace(variant=variant),p,y,criterion)
    expected=sum(criterion(p[:,i],y[:,i]) for i in range(4))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0],atol=0,rtol=0)


def test_training_telemetry_and_selection_without_optimizer_updates(tmp_path,monkeypatch):
    t=importlib.import_module('cmgm.training.train');m=make();x,y=batch();loader=DataLoader(TensorDataset(x,y),batch_size=2)
    initial=m.commodity_residual_head.weight.detach().tolist();state=copy.deepcopy(m.state_dict())
    # Exercise real backward and scheduling with updates disabled, not an experiment.
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:None)
    observed=[];original=t.validate_epoch
    def validation(*a,**kw):v=original(*a,**kw);observed.append(v);return v
    monkeypatch.setattr(t,'validate_epoch',validation)
    scheduler=[];step=torch.optim.lr_scheduler.ReduceLROnPlateau.step
    def spy(self,metric,*a,**kw):scheduler.append(metric);return step(self,metric,*a,**kw)
    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau,'step',spy)
    path=tmp_path/'fixture.pt'
    history=t.train(m,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),'cpu',num_epochs=2,patience=10,
                    checkpoint_path=str(path),epoch_diagnostic=lambda active,stage:epoch_probe(active,(x,y),stage,initial))
    assert scheduler==observed
    assert {'epoch1','best','final'}<=history['epoch_diagnostics'].keys()
    for row in history['objective_history']:
        assert row['train']['prediction_loss']==pytest.approx(sum(row['train'][f'raw_L{h}'] for h in HORIZONS))
        assert row['train']['total_loss']==pytest.approx(row['train']['prediction_loss']+row['train']['switch_loss'])
        assert row['train']['raw_residual_mean_abs']>0 and row['train']['effective_residual_mean_abs']==0
        assert row['alpha']==0 and row['residual_head_distance_from_init']==0
    saved=torch.load(path,weights_only=True)
    assert saved['best_val_loss']==min(observed) and saved['metadata']['commodity_residual_alpha_best']==0
    assert saved['metadata']['commodity_residual_head']=='Linear(64,4,bias=False)'
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())


def test_complete_report_with_active_adapter_and_no_training(tmp_path):
    m=make().eval();baseline=make(BASE_VARIANT).eval();x,y=batch()
    initial=m.commodity_residual_head.weight.detach().tolist()
    with torch.no_grad():m.commodity_residual_alpha.fill_(.03)
    payload={'model_state_dict':m.state_dict(),'best_epoch':5,'metadata':{'variant':VARIANT,'initial_residual_weight':initial}}
    path,basepath=tmp_path/'new.pt',tmp_path/'base.pt'
    torch.save(payload,path);torch.save({'model_state_dict':baseline.state_dict(),'best_epoch':5},basepath)
    loader=DataLoader(TensorDataset(x,y),batch_size=2,drop_last=True)
    data={'loaders':{s:loader for s in ('train','val','test')},'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'feature_names':['a','b','c','d','corn','gold']}
    result=checkpoint_report(m,payload,data,basepath,tmp_path/'report',path)
    assert result['integrity']['checkpoint_files_unchanged']
    for name in ('D0B',NEW):
        for split in ('TRAIN','VAL','TEST'):
            s=result['models'][name]['splits'][split]
            assert s['samples']==3 and s['base_max_diff_during_residual_controls']==0
            for v in s['native_metrics'].values():assert v['MSE']==pytest.approx(v['RMSE']**2)
    assert result['models'][NEW]['splits']['TEST']['impacts']['alpha=0']['5']['mean']>0
    md=(tmp_path/'report'/'REPORT.md').read_text()
    assert '**STOP**' in md and 'RoutingFraction' in md and 'Spearman' in md


@pytest.mark.parametrize('mae,mse,active,rejected,redistribution,expected',[
    ((-.01,-.01),(-.01,-.01),True,False,False,'Case A'),
    ((-.01,-.01),(-.01,-.01),False,True,False,'Case B'),
    ((.01,.01),(.01,.01),True,False,False,'Case C'),
    ((0.,0.),(0.,0.),False,True,False,'Case D'),
    ((-.01,.01),(-.01,.01),True,False,False,'Case E'),
    ((-.01,-.01),(.01,.01),True,False,False,'Case F'),
    ((0.,0.),(0.,0.),True,False,True,'Case G'),
])
def test_case_decisions(mae,mse,active,rejected,redistribution,expected):
    comparison={s:{'5':{'relative_MAE':mae[i],'relative_MSE':mse[i]}} for i,s in enumerate(('VAL','TEST'))}
    commodity={'TEST':{'baseline_error_halves':{'high_error':{'relative_improvement':.1 if redistribution else 0},
                                              'low_error':{'relative_improvement':-.1 if redistribution else 0}}}}
    assert case_assessment(comparison,{'functional_evidence':active,'near_zero_and_tiny':rejected},commodity)['case']==expected


def test_registration_checkpoint_names_and_only_two_extra_tensors(tmp_path):
    from cmgm.scripts.main_ablation import select_variants,D_SERIES_VARIANTS,_checkpoint_path_for_variant
    assert VARIANT in D_SERIES_VARIANTS and select_variants(DISPLAY)==select_variants(VARIANT)==[(DISPLAY,VARIANT)]
    assert _checkpoint_path_for_variant(VARIANT,tmp_path).name==VARIANT+'_best.pt'
    b,m=make(BASE_VARIANT),make()
    assert set(m.state_dict())-set(b.state_dict())=={'commodity_residual_alpha','commodity_residual_head.weight'}
