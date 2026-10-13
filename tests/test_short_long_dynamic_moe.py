"""Synthetic-only static/dynamic matched controls; no market data / TEST reads."""
from unittest.mock import patch
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_dynamic_moe import (
    A_STATIC, A_DYNAMIC, B_STATIC, B_DYNAMIC, DYNAMIC_STUDY_VARIANTS, ShortLongDynamicMoE,
)
from cmgm.models.horizon_tensor_fusion import predict_by_horizon
from cmgm.models.task_conditioned_moe import BASE_VARIANT
from cmgm.scripts.d0b_short_long_dynamic_moe import (
    matched_models, evaluate, frozen_diagnostics, route_statistics, restricted_train_val,
)
from cmgm.training.train import _prediction_loss, _switching_branch, train

KWARGS=dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,commodity_indices=(6,30))


def inputs():
    torch.manual_seed(23)
    return (torch.randn(3,64), torch.randn(3,64), torch.randn(3,20,128),
            torch.randn(3,20,64), torch.randn(3,64))


def model(variant):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(variant=variant,**KWARGS)


def nonzero(p):
    return p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_initialization_explicit_copy_and_rng_preservation():
    torch.manual_seed(13);random.seed(13);np.random.seed(13)
    state=torch.get_rng_state().clone();py=random.getstate();n=np.random.get_state()
    models,audit=matched_models(KWARGS,42)
    assert torch.equal(state,torch.get_rng_state()) and py==random.getstate()
    assert np.array_equal(n[1],np.random.get_state()[1])
    assert audit['PASS']
    assert all(a['shared_max_abs_diff']==0 and a['mismatch_count']==0 for a in audit['shared'].values())
    for a,b in ((A_STATIC,A_DYNAMIC),(B_STATIC,B_DYNAMIC)):
        assert audit['pairs'][b]['PASS']
        assert all(torch.equal(models[a].state_dict()[k],models[b].state_dict()[k])
                   for k in audit['pairs'][b]['copied_keys'])
    assert not hasattr(models[A_STATIC],'gate_fc') and hasattr(models[B_STATIC],'gate_fc')
    x=torch.randn(2,20,30,21)
    base=models[BASE_VARIANT].eval()
    with torch.no_grad():
        hs=base._temp_weighted_spatial(x);ht=base.switching_latent_transformer(x)
        g=torch.sigmoid(base.gate_fc(torch.cat((hs,ht),-1)))
        reference=base.head(g*base.lstm_proj(ht)+(1-g)*base.gcn_proj(hs)).view(2,4,24)
        torch.testing.assert_close(base(x),reference,atol=0,rtol=0)
        for v in (B_STATIC,B_DYNAMIC):
            y=models[v].eval()(x)
            torch.testing.assert_close(y,reference,atol=3e-7,rtol=1e-6)
        clone=model(BASE_VARIANT).eval();clone.load_state_dict(base.state_dict(),strict=True)
        torch.testing.assert_close(clone(x),reference,atol=0,rtol=0)


@pytest.mark.parametrize('variant',DYNAMIC_STUDY_VARIANTS)
def test_routing_formulas_masks_sample_dependence_and_horizon_rows(variant):
    torch.manual_seed(42)
    f=ShortLongDynamicMoE(variant).eval();s,t,h,z,base=inputs()
    states,parts=f(s,t,h,z,base,True);pi=parts['pi']
    assert states.shape==(3,4,64) and pi.shape==(3,2,3)
    assert (pi>=0).all();torch.testing.assert_close(pi.sum(-1),torch.ones(3,2))
    torch.testing.assert_close(states[:,0],states[:,1],atol=0,rtol=0)
    torch.testing.assert_close(states[:,2],states[:,3],atol=0,rtol=0)
    if f.dynamic:
        assert (pi[0]-pi[1]).abs().max()>1e-7
        assert (f.router.short[-1].weight!=0).any()
        torch.testing.assert_close(f.routing_weights(s,t),pi,atol=0,rtol=0)
    else:
        torch.testing.assert_close(pi[0],pi[1],atol=0,rtol=0)
        torch.testing.assert_close(f.routing_weights(s*3,t-1),pi,atol=0,rtol=0)
    expected=[]
    for group,ids in enumerate(((0,1,2),(0,3,4))):
        mix=sum(pi[:,group,k,None]*parts['experts'][:,idx] for k,idx in enumerate(ids))
        expected.append(mix if f.residual else f.fusion_norm(mix))
    torch.testing.assert_close(states,torch.stack((expected[0],expected[0],expected[1],expected[1]),1),atol=5e-7,rtol=1e-6)
    perm=torch.tensor([2,0,1]);xp=f(s[perm],t[perm],h[perm],z[perm],base[perm]);pip=f.last['pi']
    torch.testing.assert_close(xp,states[perm]);torch.testing.assert_close(pip,pi[perm])
    head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,96)).eval()
    y=predict_by_horizon(head,states,24)
    manual=torch.stack([head(states[:,i]).view(3,4,24)[:,i] for i in range(4)],1)
    torch.testing.assert_close(y,manual,atol=3e-7,rtol=1e-6)
    with f.intervention('mask_short_1'):
        alt=f.routing_weights(s,t)
    assert (alt[:,0,1]==0).all()
    torch.testing.assert_close(alt[:,1],pi[:,1],atol=0,rtol=0)
    with f.intervention('shared_only'):
        only=f(s,t,h,z,base)
    if f.residual:
        torch.testing.assert_close(only,base[:,None].expand_as(only),atol=0,rtol=0)
    with pytest.raises(ValueError):
        with f.intervention('override',torch.ones(2,3)):
            f(s,t,h,z,base)


@pytest.mark.parametrize('variant',DYNAMIC_STUDY_VARIANTS)
def test_zero_initialization_learning_and_group_gradients(variant):
    f=ShortLongDynamicMoE(variant);s,t,h,z,base=inputs()
    s.requires_grad_();t.requires_grad_();h.requires_grad_();z.requires_grad_();base.requires_grad_()
    opt=torch.optim.SGD(f.parameters(),lr=.05)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in f.parameters()}
    target=torch.randn(3,4,64)
    for step in range(3):
        opt.zero_grad();states=f(s,t,h,z,base);loss=(states-target).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
        if f.residual and step==0:
            assert all(torch.count_nonzero(p.grad)==0 for p in f.router.parameters())
            assert nonzero(f.short_state.mlp[-1].weight)
            assert torch.count_nonzero(f.short_state.mlp[0].weight.grad)==0
        if step==2:
            assert all(nonzero(p) for p in f.parameters()),[n for n,p in f.named_parameters() if not nonzero(p)]
        before=f.short_state.mlp[-1].weight.detach().clone() if f.residual else None
        opt.step()
        if before is not None:assert not torch.equal(before,f.short_state.mlp[-1].weight)
    # Losses from only one task cannot backpropagate to other-task experts/router.
    for group in (0,1):
        f.zero_grad();states=f(s,t,h,z,base)
        states[:,group*2:group*2+2].square().mean().backward()
        active=('short_state','short_interaction') if group==0 else ('long_state','long_interaction')
        inactive=('long_state','long_interaction') if group==0 else ('short_state','short_interaction')
        assert all(nonzero(p) for name in active for p in getattr(f,name).parameters())
        assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for name in inactive for p in getattr(f,name).parameters())
        if f.dynamic:
            active_router=f.router.short if group==0 else f.router.long
            inactive_router=f.router.long if group==0 else f.router.short
            assert all(nonzero(p) for p in active_router.parameters())
            assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in inactive_router.parameters())
        else:
            assert f.router.logits.grad[group].abs().sum()>0
            assert f.router.logits.grad[1-group].abs().sum()==0


@pytest.mark.parametrize('variant',(A_DYNAMIC,B_DYNAMIC))
def test_full_live_sequences_once_kl_and_roundtrip(variant,tmp_path):
    m=model(variant).eval();x=torch.randn(2,20,30,21);live={};calls=[]
    def hook(module,args,outputs):
        calls.append(1)
        live['h'],live['z']=outputs[1:]
        live['h'].retain_grad();live['z'].retain_grad()
    h=m.switching_latent_transformer.register_forward_hook(hook)
    with patch.object(m,'_temp_weighted_spatial',wraps=m._temp_weighted_spatial) as spatial:
        y=m(x)
    h.remove()
    assert spatial.call_count==1 and len(calls)==1
    assert live['h'].requires_grad and live['z'].requires_grad
    assert _switching_branch(m) is m.switching_latent_transformer
    branch=m.switching_latent_transformer
    assert branch.set_epoch(1)==0.
    assert branch.set_epoch(2)==pytest.approx(5e-4/19)
    assert branch.set_epoch(20)==pytest.approx(5e-4)
    target=torch.randn_like(y)*.01
    pred_loss=_prediction_loss(m,y,target,nn.HuberLoss(delta=.02))
    expected=sum(nn.HuberLoss(delta=.02)(y[:,i],target[:,i]) for i in range(4))
    torch.testing.assert_close(pred_loss,expected)
    kl=branch.switch_loss();assert torch.isfinite(kl) and kl>=0
    (pred_loss+kl).backward()
    assert live['h'].grad.abs().sum()>0 and live['z'].grad.abs().sum()>0
    assert nonzero(m.gcn_proj.weight) and nonzero(m.lstm_proj.weight) and nonzero(m.head[0].weight)
    with torch.no_grad():
        original=m(x);pi=m.short_long_dynamic_moe.last['pi'].clone()
        torch.testing.assert_close(m(x.flip(0)).flip(0),original,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(m.short_long_dynamic_moe.last['pi'].flip(0),pi,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(m(x[:1]),original[:1],atol=1e-6,rtol=1e-5)
    path=tmp_path/'best.pt';torch.save(m.state_dict(),path)
    clone=model(variant).eval();clone.load_state_dict(torch.load(path,weights_only=True),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(clone(x),original,atol=0,rtol=0)
        torch.testing.assert_close(clone.short_long_dynamic_moe.last['pi'],pi,atol=0,rtol=0)


def test_native_train_checkpoint_one_synthetic_step_and_diagnostics(tmp_path):
    m=model(B_DYNAMIC);x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.01
    loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=False)
    before=m.short_long_dynamic_moe.short_state.mlp[-1].weight.detach().clone()
    path=tmp_path/'best.pt'
    history=train(m,loader,loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.device('cpu'),
                  num_epochs=1,lr=1e-4,weight_decay=1e-5,patience=10,checkpoint_path=str(path),
                  checkpoint_metadata={'variant':B_DYNAMIC,'seed':42})
    assert history['best_epoch']==1 and history['switch_beta']==[0.]
    assert not torch.equal(before,m.short_long_dynamic_moe.short_state.mlp[-1].weight)
    payload=torch.load(path,weights_only=False);assert payload['best_epoch']==1
    m.load_state_dict(payload['model_state_dict'],strict=True)
    metrics,arrays=evaluate(m,loader,'cpu');assert arrays['pi'].shape==(2,2,3)
    stats=route_statistics(arrays['pi']);assert len(stats['mean_sample_entropy_nats'])==2
    diagnosis=frozen_diagnostics(m,loader,'cpu',arrays['pi'],arrays['pi'],metrics,42)
    assert len(diagnosis['interventions'])==7
    assert np.array_equal(np.sort(diagnosis['permutation']),np.arange(2))
    assert m.short_long_dynamic_moe._intervention is None
    assert payload['best_val_loss']==pytest.approx(history['val_loss'][0])


@pytest.mark.parametrize('variant',DYNAMIC_STUDY_VARIANTS)
def test_gate_diagnostic_uses_switching_path_and_preserves_modes_rng(variant):
    m=model(variant).train()
    # Preserve deliberately mixed per-module modes as well as model.train().
    m.head[2].eval()
    modes=[(module,module.training) for module in m.modules()]
    before={k:v.clone() for k,v in m.state_dict().items()}
    x=torch.randn(2,20,30,21)
    rng=torch.get_rng_state().clone()
    with patch.object(m,'_temp_weighted_spatial',wraps=m._temp_weighted_spatial) as spatial:
        with patch.object(m.switching_latent_transformer,'forward',wraps=m.switching_latent_transformer.forward) as temporal:
            report=m.get_gate_stats(x)
    assert spatial.call_count==temporal.call_count==1
    assert report['group_order']==['short','long']
    assert np.asarray(report['routing_mean']).shape==(2,3)
    assert all(module.training==mode for module,mode in modes)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    with patch.object(m,'forward',side_effect=RuntimeError('synthetic diagnostic failure')):
        with pytest.raises(RuntimeError,match='synthetic diagnostic failure'):
            m.get_gate_stats(x)
    assert all(module.training==mode for module,mode in modes)


def test_native_train_epoch_exact_objective_and_all_optimizer_parameters():
    from cmgm.training.train import train_epoch,validate_epoch,make_loss
    m=model(B_DYNAMIC)
    m.switching_latent_transformer.set_epoch(20)
    loader=DataLoader(TensorDataset(torch.randn(2,20,30,21),torch.randn(2,4,24)*.01),batch_size=2)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters()}
    # Capture the actual train forward; dropout need not be replayed to prove
    # the reported loss is exactly prediction Huber plus native Switch KL.
    terms=[]
    def capture(module,args,pred):
        target=loader.dataset.tensors[1]
        prediction=sum(nn.HuberLoss(delta=.02)(pred[:,h],target[:,h]) for h in range(4))
        terms.append((prediction.detach(),m.switching_latent_transformer.switch_loss().detach()))
    hook=m.register_forward_hook(capture)
    tracked=m.short_long_dynamic_moe.short_state.mlp[-1].weight.detach().clone()
    loss=train_epoch(m,loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),opt,make_loss(),torch.device('cpu'))
    hook.remove()
    assert len(terms)==1 and terms[0][1]>0
    assert loss==pytest.approx((terms[0][0]+terms[0][1]).item(),abs=1e-10)
    assert not torch.equal(tracked,m.short_long_dynamic_moe.short_state.mlp[-1].weight)
    assert nonzero(m.gate_fc.weight) and nonzero(m.gcn_proj.weight) and nonzero(m.lstm_proj.weight)
    assert not hasattr(m,'moe_fusion') and m.regime_diversity_loss().item()==0
    terms=[];hook=m.register_forward_hook(capture)
    val=validate_epoch(m,loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),make_loss(),torch.device('cpu'))
    hook.remove()
    assert val==pytest.approx(terms[0][0].item(),abs=1e-10)


def test_live_temporal_prefix_causality_and_cache_independence():
    m=model(B_DYNAMIC).eval();x=torch.randn(2,20,30,21)
    future=x.clone();future[:,10:]=future[:,10:]*3+2
    branch=m.switching_latent_transformer
    with torch.no_grad():
        regular=branch(x)
        live,h,z=branch(x,return_live_sequences=True)
        torch.testing.assert_close(live,regular,atol=0,rtol=0)
        _,future_h,future_z=branch(future,return_live_sequences=True)
        torch.testing.assert_close(h[:,:10],future_h[:,:10],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(z[:,:10],future_z[:,:10],atol=1e-6,rtol=1e-6)
        pred=m(x)
        # Diagnostic tensors may be overwritten; the next actual forward must
        # recompute live sequences rather than use detached cached values.
        branch.last_long_memory=torch.full_like(h,float('nan'))
        branch.last_h_temporal=torch.full_like(live,float('nan'))
        m.short_long_dynamic_moe.last={'pi':torch.full((2,2,3),float('nan'))}
        torch.testing.assert_close(m(x),pred,atol=0,rtol=0)


def test_train_mean_routes_use_float64_without_relaxing_simplex_validation():
    m=model(B_DYNAMIC).eval()
    loader=DataLoader(TensorDataset(torch.randn(3,20,30,21),torch.randn(3,4,24)*.01),batch_size=2)
    metrics,arrays=evaluate(m,loader,'cpu')
    # 1396 valid float32 rows reproduce the production accumulation failure.
    train_pi=np.full((1396,2,3),np.float32(1/3),dtype=np.float32)
    assert np.max(np.abs(train_pi.mean(0).sum(-1)-1))>1e-6
    diagnosis=frozen_diagnostics(m,loader,'cpu',train_pi,arrays['pi'],metrics,42)
    assert np.max(np.abs(np.asarray(diagnosis['train_mean']).sum(-1)-1))<1e-6
    assert len(diagnosis['interventions'])==7
    # The last partial batch must get its matching permuted route row.
    assert len(diagnosis['permutation'])==3
    for corrupt in (train_pi*2, np.full_like(train_pi,np.nan)):
        with pytest.raises(ValueError,match='TRAIN routes'):
            frozen_diagnostics(m,loader,'cpu',corrupt,arrays['pi'],metrics,42)
    with pytest.raises(ValueError,match='VAL routes'):
        frozen_diagnostics(m,loader,'cpu',train_pi,arrays['pi'][:-1],metrics,42)
    with pytest.raises(ValueError,match='exactly all loader origins'):
        evaluate(m,loader,'cpu',intervention='override',override=arrays['pi'][:-1])


@pytest.mark.parametrize("missing_pivot_dates", [False, True])
def test_restricted_csv_loader_preserves_synthetic_fingerprint_and_excludes_future(tmp_path,monkeypatch,missing_pivot_dates):
    import pandas as pd
    from types import SimpleNamespace
    from cmgm import config
    from cmgm.data.data_loader import (load_stock_prices, load_bond_prices, load_commodity_prices,
                                      align_markets, normalize_data, compute_returns)
    from cmgm.scripts.main_ablation import build_data
    from cmgm.scripts.d0b_task_conditioned_moe import data_audit, PROTOCOL
    dates=pd.date_range('2010-01-01',periods=400)
    values=10+np.arange(400)*.1+np.sin(np.arange(400)*.2)
    stock,bond,commodity=[tmp_path/f'{k}.csv' for k in ('stock','bond','commodity')]
    pd.DataFrame({'date':dates,'stock1':values,'stock2':values*2}).to_csv(stock,index=False)
    bond_values=values*3
    if missing_pivot_dates:
        bond_values[np.r_[0:11,329:332]]=np.nan
    pd.DataFrame({'date':dates,'品种':'bond1','close':bond_values}).to_csv(bond,index=False)
    pd.DataFrame({'date':dates,'品种':'gold','close':values*4}).to_csv(commodity,index=False)
    prices,mi=align_markets(load_stock_prices(stock),load_bond_prices(bond),load_commodity_prices(commodity))
    a=int(len(prices)*config.TRAIN_RATIO);b=a+int(len(prices)*config.VAL_RATIO)
    final_raw_cutoff=dates.get_loc(prices.index[b-1])
    tr,va=prices.values[:a],prices.values[a:b]
    _,_,_,norm=normalize_data(tr,va,np.empty((0,4)))
    prepared=dict(raw_prices_train=tr,raw_prices_val=va,norm_stats=norm,market_indices=mi,n_nodes=4,
                  n_commodities=1,train_returns=compute_returns(tr),feature_names=list(prices.columns))
    reference=build_data(SimpleNamespace(batch_size=64,seq_len=20),prepared_train_val=prepared)
    expected=data_audit(reference);cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'protocol':PROTOCOL,'data_audit':expected}))
    monkeypatch.setattr(config,'STOCK_FILE',stock);monkeypatch.setattr(config,'BOND_FILE',bond)
    monkeypatch.setattr(config,'COMMODITY_FILE',commodity)
    original=pd.read_csv;calls=[]
    def read(*args,**kwargs):
        if kwargs.get('usecols')!=['date']:
            assert 'skiprows' in kwargs
            assert all(kwargs['skiprows'](i+1) for i in range(final_raw_cutoff+1,400))
        calls.append(kwargs)
        return original(*args,**kwargs)
    with patch('pandas.read_csv',side_effect=read):
        data,actual=restricted_train_val(cfg)
    assert len(calls)>=6 and actual==expected and set(data['loaders'])=={'train','val'}
    if missing_pivot_dates:
        assert len(data['calendar_audit']['attempts'])==3
        assert len(data['calendar_audit']['raw_dates_removed'])==14
    assert data['calendar_audit']['val_end_date']==str(prices.index[b-1])
    assert data['origin_dates']['val'][0]==prices.index[a+19].to_numpy()
    expected['splits']['val']['sha256']='wrong'
    cfg.write_text(json.dumps({'protocol':PROTOCOL,'data_audit':expected}))
    with pytest.raises(ValueError,match='fingerprint'):
        restricted_train_val(cfg)


def test_explicit_run_entry_on_two_synthetic_samples(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import cmgm.scripts.d0b_short_long_dynamic_moe as runner
    torch.manual_seed(42)
    loader=DataLoader(TensorDataset(torch.randn(2,20,30,21),torch.randn(2,4,24)*.01),batch_size=2)
    data=dict(loaders={'train':loader,'val':loader},n_nodes=30,n_commodities=24,
              market_indices={'stock':(0,4),'bond':(4,6),'commodity':(6,30)},
              origin_dates={s:np.array(['2010-01-20','2010-01-21'],dtype='datetime64[ns]') for s in ('train','val')})
    fingerprint={'commodity_order':[f'c{i}' for i in range(24)],'synthetic_only':True}
    ref=tmp_path/'reference.json';ref.write_text('{}')
    monkeypatch.setattr(runner,'restricted_train_val',lambda path:(data,fingerprint))
    monkeypatch.setattr(runner,'EXPERIMENT_ROOT',tmp_path/'experiments')
    monkeypatch.setattr(runner,'CHECKPOINT_ROOT',tmp_path/'checkpoints')
    def one_step(*args,**kwargs):
        assert kwargs['num_epochs']==200  # formal entry still requests fixed budget
        kwargs['num_epochs']=1  # synthetic test only
        return train(*args,**kwargs)
    monkeypatch.setattr(runner,'train',one_step)
    runner.run(SimpleNamespace(device='cpu',seed=42,variant=B_DYNAMIC,reference_config=ref,permutation_seed=42))
    p=next((tmp_path/'experiments').glob('*/result.json'))
    record=json.loads(p.read_text());cfg=json.loads((p.parent/'config.json').read_text())
    assert record['status']=='COMPLETE' and record['test_evaluated'] is False
    assert record['best_epoch']==1 and set(record['metrics'])=={'train','val'}
    assert cfg['origin_date_ranges']['train']['count']==2
    arrays=np.load(p.parent/'val_predictions_routes.npz',allow_pickle=False)
    assert arrays['pi'].shape==(2,2,3) and arrays['origin_dates'].dtype.kind=='M'
    assert arrays['commodity_order'].dtype.kind=='U'
    assert set(arrays)>= {'base_norms','residual_norms','weighted_residual_norms','weighted_residual_to_base'}
    assert json.loads((p.parent/'RUN_STATUS.json').read_text())['status']=='COMPLETE'


def test_postprocess_recovery_never_retrains_or_overwrites_originals(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import cmgm.scripts.d0b_short_long_dynamic_moe as runner
    loader=DataLoader(TensorDataset(torch.randn(2,20,30,21),torch.randn(2,4,24)*.01),batch_size=2)
    data=dict(loaders={'train':loader,'val':loader},n_nodes=30,n_commodities=24,
              market_indices={'stock':(0,4),'bond':(4,6),'commodity':(6,30)},
              origin_dates={s:np.array(['2010-01-20','2010-01-21'],dtype='datetime64[ns]') for s in ('train','val')})
    fingerprint={'commodity_order':[f'c{i}' for i in range(24)],'synthetic_only':True}
    ref=tmp_path/'reference.json';ref.write_text('{}')
    monkeypatch.setattr(runner,'restricted_train_val',lambda path:(data,fingerprint))
    monkeypatch.setattr(runner,'EXPERIMENT_ROOT',tmp_path/'experiments')
    monkeypatch.setattr(runner,'CHECKPOINT_ROOT',tmp_path/'checkpoints')
    def one_step(*args,**kwargs):
        kwargs['num_epochs']=1
        return train(*args,**kwargs)
    original_diagnostics=runner.frozen_diagnostics
    def fail(*args,**kwargs):
        raise ValueError('Override must be normalized (2,3) or (B,2,3)')
    monkeypatch.setattr(runner,'train',one_step)
    monkeypatch.setattr(runner,'frozen_diagnostics',fail)
    with pytest.raises(ValueError,match='Override'):
        runner.run(SimpleNamespace(device='cpu',seed=42,variant=B_DYNAMIC,reference_config=ref,permutation_seed=42))
    directory=next((tmp_path/'experiments').iterdir())
    ck=next((tmp_path/'checkpoints').glob('*/best.pt'))
    original={p:p.read_bytes() for p in [ck,*directory.iterdir()] if p.is_file()}
    def forbidden(*args,**kwargs):
        raise AssertionError('Postprocessing must never call train or build an optimizer')
    monkeypatch.setattr(runner,'train',forbidden)
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(runner,'frozen_diagnostics',original_diagnostics)
    runner.finish_runs([directory],'cpu')
    paths=list(directory.glob('postprocess_*/result.json'));assert len(paths)==1
    result=json.loads(paths[0].read_text())
    assert result['status']=='COMPLETE' and result['test_evaluated'] is False
    assert result['best_epoch']==1 and result['parameters']['trainable']>0
    assert all(p.read_bytes()==contents for p,contents in original.items())
    assert json.loads((paths[0].parent/'config.json').read_text())==json.loads((directory/'config.json').read_text())
    runner.finish_runs([directory],'cpu')  # completed recovery is skipped
    assert len(list(directory.glob('postprocess_*/result.json')))==1
    best_path=directory/'best_checkpoint_metadata.json'
    best=json.loads(best_path.read_text());best['sha256']='tampered';best_path.write_text(json.dumps(best))
    with pytest.raises(ValueError,match='hash mismatch'):
        runner.finish_runs([directory],'cpu')


def test_compare_six_synthetic_records_and_reject_mismatched_protocol(tmp_path):
    from cmgm.scripts.d0b_short_long_dynamic_moe import compare,VARIANTS,PROTOCOL,sha256
    paths=[]
    for i,variant in enumerate(VARIANTS):
        out=tmp_path/f'fixture_{i}';out.mkdir();ck=out/'synthetic.pt'
        torch.save({'synthetic_fixture_only':True},ck)
        record=dict(variant=variant,seed=42,status='COMPLETE',test_evaluated=False,
                    checkpoint=str(ck),checkpoint_sha256=sha256(ck),best_epoch=1,
                    best_val_prediction_huber_batch_mean=.001,train_seconds=1.,parameters={'total':10},
                    metrics={'val':{str(h):dict(MAE=.01+i*.001,MSE=.0001,RMSE=.01,Hit=.5) for h in (1,5,10,20)}})
        cfg=dict(variant=variant,seed=42,initialization_audit={'PASS':True},protocol=PROTOCOL,
                 data_audit={'synthetic_fixture_only':True},source_sha256={},rng_policy='synthetic fixture')
        (out/'result.json').write_text(json.dumps(record));(out/'config.json').write_text(json.dumps(cfg))
        (out/'training_history.json').write_text(json.dumps({'best_epoch':1,'val_loss':[.001]}))
        paths.append(out/'result.json')
    compare(paths,tmp_path/'comparison')
    r=json.loads((tmp_path/'comparison/comparison.json').read_text())
    assert len(r['rows'])==6 and r['test_evaluated'] is False
    assert r['rows'][0]['VAL_Hit@5d']==50.
    assert (tmp_path/'comparison/REPORT.md').is_file()
    cfgpath=paths[-1].parent/'config.json';c=json.loads(cfgpath.read_text());c['protocol']={}
    cfgpath.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='protocol'):
        compare(paths,tmp_path/'rejected')
