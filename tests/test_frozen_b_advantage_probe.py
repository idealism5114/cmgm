"""Synthetic-only tests: no experiment data, checkpoints or TEST are read."""
import inspect
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_dynamic_moe import B_DYNAMIC
from cmgm.scripts.d0b_frozen_b_advantage_probe import (
    LinearProbe, temporal_folds, fit_train_probes, advantages, hard_choices,
    soft_weights, choice_weights, capture_router_input, prediction_fit_rows,
)
from cmgm.scripts.d0b_short_long_expert_diagnostics import (
    choose_predictions, expert_losses, state_digest, predict_frozen_inputs,
)


def test_advantage_sign_group_choices_and_deterministic_ties():
    loss = np.array([[[3., 1., 4.], [1., 3., 0.]], [[2., 2., 2.], [2., 2., 2.]]])
    d = advantages(loss)
    np.testing.assert_array_equal(d[0], [[-2, 1], [2, -1]])
    choice = hard_choices(d)
    np.testing.assert_array_equal(choice, [[1, 2], [0, 0]])
    p = np.arange(2*3*4*2).reshape(2,3,4,2)
    selected = choose_predictions(p, choice)
    np.testing.assert_array_equal(selected[0,:2],p[0,1,:2])
    np.testing.assert_array_equal(selected[0,2:],p[0,2,2:])
    np.testing.assert_array_equal(selected[1],p[1,0])


def test_purge_trading_rows_not_calendar_days():
    # Irregular dates with weekends and a long holiday still follow row horizons.
    dates = np.datetime64('2020-01-01') + np.cumsum(np.resize([1,1,3,9],100)).astype('timedelta64[D]')
    rows = np.arange(100) + 19
    for horizon in (5,20):
        folds = temporal_folds(rows,dates,horizon)
        assert [f['validation_first_origin_row'] for f in folds] == [69,85,102]
        for f in folds:
            tr,va,removed = map(np.array,(f['train_indices'],f['validation_indices'],f['purged_indices']))
            assert np.all(rows[tr]+horizon < rows[va[0]])
            assert np.all(rows[removed]+horizon >= rows[va[0]])
            assert len(removed)==horizon
            assert f['last_retained_train_label_end_date']==str(dates[va[0]-1].astype('datetime64[ns]'))


def test_fold_scalers_fit_retained_training_only_and_restore(tmp_path):
    rng = np.random.default_rng(5)
    x=rng.normal(size=(100,6));x[:,5]=1
    x[50:]+=1000
    d=rng.normal(size=(100,2,2))*.001
    dates=np.datetime64('2020-01-01')+np.arange(100).astype('timedelta64[D]')
    models, records=fit_train_probes(x,d,np.arange(100)+19,dates)
    assert 'val' not in inspect.signature(fit_train_probes).parameters
    for g,group in enumerate(('short','long')):
        fold=records[group]['folds'][0]
        retained=fold['train_indices']
        details=records[group]['candidates'][0]['folds'][0]
        np.testing.assert_allclose(details['x_mean'],x[retained].mean(0))
        np.testing.assert_allclose(details['y_mean'],d[retained,g].mean(0))
        assert details['scaler_fit_origins']==len(retained)
        assert np.max(np.abs(np.array(details['x_mean'])-x.mean(0)))>400
        path=tmp_path/f'{group}.npz';models[g].save(path)
        np.testing.assert_array_equal(models[g].predict(x),LinearProbe.load(path).predict(x))


def test_constant_input_degenerate_tau_and_undefined_r2():
    x=np.ones((40,128));y=np.full((40,2),.003)
    probe=LinearProbe.fit(x,y,1)
    assert not probe.mask.any()
    np.testing.assert_allclose(probe.predict(x),y)
    d=np.zeros((40,2,2))
    pi=soft_weights(d,[0,0])
    np.testing.assert_array_equal(pi,np.full((40,2,3),1/3))
    rows=prediction_fit_rows('val',d,d,d)
    assert all(r['R2'] is None for r in rows)
    biased=np.ones_like(d)*.01
    rows=prediction_fit_rows('val',d+np.linspace(0,.001,40)[:,None,None],biased,d)
    assert all(r['R2']<0 for r in rows if r['predictor']=='ridge')


def test_soft_weights_and_origin_permutation_alignment():
    d=np.array([[[-.1,.2],[.3,-.2]],[[.4,-.1],[-.2,.4]]])
    pi=soft_weights(d,[.1,.2])
    np.testing.assert_allclose(pi.sum(-1),1)
    assert np.all(pi>=0)
    np.testing.assert_array_equal(soft_weights(d[::-1],[.1,.2]),pi[::-1])
    assert pi[0,0,1]>pi[0,0,0]>pi[0,0,2]


def test_actual_router_features_onehot_and_nonlinear_soft_path_frozen(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(30,24,n_stock=4,n_bond=2,commodity_indices=(6,30),variant=B_DYNAMIC).eval()
    f=model.short_long_dynamic_moe
    # Nondegenerate, known candidate corrections straddle the head's ReLU.
    with torch.no_grad():
        for expert,value in ((f.short_state,2.),(f.short_interaction,-2.),(f.long_state,2.),(f.long_interaction,-2.)):
            expert.mlp[-1].bias.fill_(value)
        model.head[0].weight.copy_(torch.eye(64));model.head[0].bias.zero_()
        model.head[-1].weight.fill_(.01);model.head[-1].bias.zero_()
    model.requires_grad_(False)
    x=torch.randn(3,20,30,21)
    target=torch.randn(3,4,24)*.01
    loader=DataLoader(TensorDataset(x,target),batch_size=2,shuffle=False)
    ck=tmp_path/'fixture.pt';torch.save(model.state_dict(),ck)
    checkpoint_before=ck.read_bytes();before=state_digest(model)
    with patch.object(model,'_temp_weighted_spatial',wraps=model._temp_weighted_spatial) as spatial:
        score,arrays,batches=capture_router_input(model,loader,'cpu')
    assert spatial.call_count==1  # Full protocol loader fits 3 examples in batch64.
    assert arrays['router_input'].shape==(3,128)
    args,kwargs=batches[0]
    with torch.inference_mode():
        u=torch.cat((f.router.norm_s(args[0]),f.router.norm_t(args[1])),dim=-1)
        np.testing.assert_array_equal(u.numpy(),arrays['router_input'])
        last,recent=args[3][:,-1],args[3][:,-5:].mean(1)
        full=args[2].mean(1)
        delta=torch.stack((f.short_state(torch.cat((last,recent),-1)),f.short_interaction(args[0],last-recent),
            f.long_state(torch.cat((args[2][:,-1],full),-1)),f.long_interaction(args[0],full)),1).view(3,2,2,64)
    cp=np.stack([predict_frozen_inputs(model,batches,np.eye(3)[k][None].repeat(2,0),'cpu') for k in range(3)],1)
    losses=expert_losses(cp,target.numpy(),.02)
    choice=hard_choices(advantages(losses))
    native=predict_frozen_inputs(model,batches,choice_weights(choice),'cpu')
    np.testing.assert_allclose(native,choose_predictions(cp,choice),atol=1e-6,rtol=0)
    pi=soft_weights(np.arange(12).reshape(3,2,2)*.003,[.04,.03])
    pred=predict_frozen_inputs(model,batches,pi,'cpu')
    with torch.inference_mode():
        weights=torch.as_tensor(pi,dtype=torch.float32)
        groups=kwargs['z_base'][:,None]+(weights[...,1:,None]*delta).sum(2)
        expected=f.predict(model.head,groups[:,(0,0,1,1)],24).numpy()
    np.testing.assert_array_equal(pred,expected)
    # Nonlinear head: representation mixture is not a prediction ensemble.
    prediction_ensemble=np.empty_like(pred)
    for g,ids in enumerate(((0,1),(2,3))):
        for h in ids:
            prediction_ensemble[:,h]=(pi[:,g,:,None]*cp[:,:,h]).sum(1)
    assert np.max(np.abs(pred-prediction_ensemble))>.01
    short_only=pi.copy();short_only[:,0]=[1,0,0]
    only=predict_frozen_inputs(model,batches,short_only,'cpu')
    np.testing.assert_array_equal(only[:,2:],pred[:,2:])
    assert state_digest(model)==before and ck.read_bytes()==checkpoint_before
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
