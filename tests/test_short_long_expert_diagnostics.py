"""Synthetic functionality checks; no market data or TEST evaluation."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_dynamic_moe import A_DYNAMIC,B_DYNAMIC
from cmgm.scripts.d0b_short_long_expert_diagnostics import (
    capture_normal,predict_frozen_inputs,one_hot,state_digest,expert_losses,
    group_losses,best_candidates,choose_predictions,analyze_split,correlation,preflight,
    save_validated_data,load_validated_data,compare_saved,
)


@pytest.mark.parametrize('variant',(A_DYNAMIC,B_DYNAMIC))
def test_actual_model_onehot_group_isolation_and_frozen_state(variant,tmp_path):
    torch.manual_seed(42)
    m=HeteroMixHopCMGM(30,24,n_stock=4,n_bond=2,commodity_indices=(6,30),variant=variant).eval()
    f=m.short_long_dynamic_moe
    if f.residual:
        with torch.no_grad():
            for expert in (f.short_state,f.short_interaction,f.long_state,f.long_interaction):
                expert.mlp[-1].weight.normal_(0,.01)
                expert.mlp[-1].bias.normal_(0,.01)
    m.requires_grad_(False)
    x=torch.randn(3,20,30,21)
    loader=DataLoader(TensorDataset(x,torch.randn(3,4,24)*.01),batch_size=2)
    ck=tmp_path/'fixture.pt';torch.save(m.state_dict(),ck)
    digest=state_digest(m);saved=ck.read_bytes()
    with patch.object(m,'_temp_weighted_spatial',wraps=m._temp_weighted_spatial) as spatial:
        score,arrays,batches=capture_normal(m,loader,'cpu')
    # full_loader uses the frozen batch64 protocol; this fixture is one batch.
    assert spatial.call_count==1
    assert sum(len(a[0]) for a,k in batches)==3
    torch.testing.assert_close(torch.as_tensor(arrays['prediction']),m(x),atol=0,rtol=0)
    for k in range(3):
        forced=predict_frozen_inputs(m,batches,one_hot(k),'cpu')
        with torch.inference_mode(),f.intervention('override',one_hot(k)):
            full=m(x).cpu().numpy()
        np.testing.assert_allclose(forced,full,atol=0,rtol=0)
        for g,ids,other in ((0,(0,1),(2,3)),(1,(2,3),(0,1))):
            weights=arrays['pi'].copy();weights[:,g]=one_hot(k)[g]
            only=predict_frozen_inputs(m,batches,weights,'cpu')
            np.testing.assert_allclose(only[:,ids],forced[:,ids],atol=1e-7,rtol=0)
            np.testing.assert_allclose(only[:,other],arrays['prediction'][:,other],atol=0,rtol=0)
    assert f._intervention is None
    if f.residual:
        args,kwargs=batches[0]
        with torch.inference_mode(),f.intervention('override',one_hot(0)):
            states=f(*args,**kwargs)
        torch.testing.assert_close(states,kwargs['z_base'][:,None].expand(-1,4,-1),atol=0,rtol=0)
    assert state_digest(m)==digest and ck.read_bytes()==saved
    assert all(p.grad is None for p in m.parameters())


def test_group_huber_oracle_not_5d_selection_and_fixed_group_rows():
    p=np.zeros((2,3,4,1));target=np.zeros((2,4,1))
    p[0,0,0]=10;p[0,1,1]=2;p[0,2,1]=4
    p[1,0,2]=10;p[1,1,3]=2;p[1,2,3]=4
    losses=expert_losses(p,target,.02)
    best,tied,margin=best_candidates(losses)
    assert best[0,0]==1 and best[1,1]==1
    assert p[0,0,1,0]==0  # 5d alone would have chosen shared
    selected=choose_predictions(p,best)
    np.testing.assert_equal(selected[0,:2],p[0,1,:2])
    np.testing.assert_equal(selected[1,2:],p[1,1,2:])
    assert group_losses(selected,target,.02)[0,0]==pytest.approx(losses[0,0,1])


def test_ties_correlations_and_origin_alignment_in_statistics():
    p=np.zeros((4,3,4,2));target=np.ones((4,4,2))*.01
    best,tied,margin=best_candidates(expert_losses(p,target,.02))
    assert np.array_equal(best,np.zeros((4,2),dtype=int)) and tied.all() and not margin.any()
    assert correlation(np.ones(4),np.ones(4)) is None
    assert correlation(np.arange(4),np.arange(4))==pytest.approx(1.)
    arrays=dict(expert_predictions=p,target=target,pi=np.full((4,2,3),1/3),prediction=p[:,0],
        train_mean_prediction=p[:,0],shuffle_prediction=p[:,0],origin_dates=np.arange(4))
    summary,rows,derived=analyze_split('fixture','val',arrays,np.array([0,0]),np.array([0,0]),.02)
    assert all(r['hard_router_oracle_agreement_non_ties'] is None for r in rows['alignment'])
    assert all(r['winner_fraction_fractional_ties']==pytest.approx(1/3) for r in rows['competence'])
    assert all(r['oracle_tied'] for r in rows['per_origin'])
    for row in rows['per_origin']:
        assert row['date']==str(row['origin'])
        g=0 if row['group']=='short' else 1
        assert row['loss_state']==derived['expert_losses'][row['origin'],g,1]
    # Unequal final-batch size must not bias diagnostic Huber aggregation.
    y=np.zeros((3,4,1));q=y.copy();q[2]=1
    assert group_losses(q,y,.02).mean(0)[0]==pytest.approx(.0198/3)


def test_preflight_synthetic_only():
    with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer')):
        preflight()


def test_validated_cache_preserves_inputs_targets_and_rejects_drift(tmp_path):
    from cmgm.data.data_loader import MarketSequenceDataset
    from cmgm.scripts.d0b_short_long_dynamic_moe import data_audit
    rng=np.random.default_rng(42)
    market={'stock':(0,4),'bond':(4,6),'commodity':(6,30)}
    data=dict(loaders={},market_indices=market,n_nodes=30,n_commodities=24,
              feature_names=[str(i) for i in range(30)],origin_dates={},calendar_audit={'synthetic_only':True})
    for split in ('train','val'):
        p=10+rng.random((61,30));f=rng.normal(size=(61,30,21)).astype(np.float32)
        ds=MarketSequenceDataset(p,market,feature_matrix=f,raw_prices=p,target_type='return',horizons=[1,5,10,20])
        data['loaders'][split]=DataLoader(ds,batch_size=64,shuffle=False,drop_last=split=='train')
        data['origin_dates'][split]=np.arange(len(ds)).astype('datetime64[D]')
    expected=data_audit(data);ref=tmp_path/'ref.json';ref.write_text('{}')
    path=tmp_path/'cache.npz';save_validated_data(path,data,expected,ref)
    loaded,actual=load_validated_data(path,expected,ref)
    assert actual==expected and set(loaded['loaders'])=={'train','val'}
    for split in ('train','val'):
        for i in range(len(data['loaders'][split].dataset)):
            a=data['loaders'][split].dataset[i];b=loaded['loaders'][split].dataset[i]
            torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
            torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
    ref.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='provenance mismatch'):
        load_validated_data(path,expected,ref)


def test_float32_reproduction_tolerance_keeps_population_metrics_strict(tmp_path):
    from cmgm.training.metric_standard import population_metrics
    p=np.full((268,4,24),.07,dtype=np.float32);y=np.zeros_like(p)
    path=tmp_path/'saved.npz';np.savez(path,prediction=p,target=y)
    q=p.copy();q[243,1,21]+=2.2e-6
    metric=lambda v:{str(h):population_metrics(v[:,i],y[:,i]) for i,h in enumerate([1,5,10,20])}
    assert compare_saved({'prediction':q,'target':y},path,metric(q),metric(p))>1e-6
    bad=q.copy();bad[0,0,0]+=.001
    with pytest.raises(ValueError,match='Normal reproduction mismatch'):
        compare_saved({'prediction':bad,'target':y},path,metric(bad),metric(p))
    altered=metric(q);altered['5']['Hit']=.01
    with pytest.raises(ValueError,match='Normal metric reproduction mismatch'):
        compare_saved({'prediction':q,'target':y},path,altered,metric(p))
