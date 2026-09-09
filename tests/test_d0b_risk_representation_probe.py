import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cmgm.scripts.d0b_risk_probe_analysis import (
    MAIN, fit_linear, predict_linear, make_features, regression_metrics,
    within_commodity, bootstrap_regression, stability_fit, target_distributions,audit_linear,
)
from cmgm.scripts.d0b_risk_representation_probe import RepresentationCollector


def test_ordinary_lstsq_with_intercept_train_normalization_and_constant_drop():
    x=np.column_stack([np.arange(10.),np.full(10,7.)]);y=.02+.003*x[:,0]
    fit=fit_linear(x,y)
    assert fit['mean']==[4.5,7.]
    assert fit['std'][0]==pytest.approx(np.std(x[:,0],ddof=0))
    assert fit['constant_dimensions']==1 and fit['coefficient_count']==2
    heldout=np.array([[12.,1000.],[-20.,-1000.]])
    np.testing.assert_allclose(predict_linear(heldout,fit),.02+.003*heldout[:,0],atol=1e-14)
    assert predict_linear(heldout,fit)[1]<0  # No nonnegative clipping.
    assert fit_linear(x,y)==fit  # Evaluation cannot change the fitted model.


def test_underdetermined_design_stops_and_rank_deficiency_is_reported():
    with pytest.raises(ValueError,match='STOP'):
        fit_linear(np.ones((2,3)),np.ones(2))
    x=np.column_stack([np.arange(10.),np.arange(10.)])
    fit=fit_linear(x,np.arange(10.))
    assert fit['rank']==2 and fit['numerical_warning']


def test_r2_uses_train_mean_and_preserves_negative_r2():
    y=np.array([3.,4.]);pred=np.array([-1.,-2.]);m=regression_metrics(pred,y,1.)
    assert m['R2']==pytest.approx(1-(16+36)/(4+9))
    assert m['negative_prediction_fraction']==1
    assert m['RiskRMSE']**2==pytest.approx(m['RiskMSE'])
    assert regression_metrics(np.ones(2),y,1.)['R2']==0


def test_feature_expansion_retains_origin_and_commodity_identity():
    f=pd.DataFrame(dict(split=['train']*4,sample_index=[0,0,1,1],
        forecast_origin=['a','a','b','b'],commodity_index=[1,0,0,1],
        commodity_vol20=[.1,.2,.3,.4],commodity_vol5=[.01]*4,
        abs_past1=[1.,2.,3.,4.],abs_past5=[4.,3.,2.,1.]))
    reps={k:np.arange(128.).reshape(2,64) for k in ['h_fused','h_temporal','h_spatial','h_long','h_micro']}
    reps['h_comm']=np.arange(256.).reshape(2,2,64)
    x=make_features(f,reps)
    np.testing.assert_array_equal(x['GlobalFused'],reps['h_fused'][[0,0,1,1]])
    np.testing.assert_array_equal(x['CommodityNode'],reps['h_comm'][[0,0,1,1],[1,0,0,1]])
    np.testing.assert_array_equal(x['CommodityNodePlusGlobal'][:,:64],x['CommodityNode'])
    np.testing.assert_allclose(x['Vol20Only'][:,0],np.log(f.commodity_vol20+1e-8))
    assert x['SimpleRiskFeatures'].shape==(4,4)


def test_within_commodity_constants_undefined_and_small_tails_insufficient():
    n=30;f=pd.DataFrame(dict(commodity_index=np.tile([0,1],n),
        commodity=np.tile(['A','B'],n),target=np.column_stack([np.arange(n),np.zeros(n)]).reshape(-1)))
    pred={'static':np.tile([2.,3.],n),'dynamic':f.target.to_numpy()}
    rows,summaries=within_commodity(f,pred,14.)
    static=next(x for x in summaries if x['probe']=='static')
    dynamic=next(x for x in summaries if x['probe']=='dynamic')
    assert static['valid_correlations']==0 and static['Spearman_mean'] is None
    assert static['AUC_mean']==.5
    assert dynamic['valid_correlations']==1 and dynamic['Spearman_mean']==pytest.approx(1.)
    assert dynamic['valid_AUC_commodities']==1
    assert all(x['AUC_status'].startswith('INSUFFICIENT') for x in rows if x['commodity']=='B')


def test_bootstrap_resamples_origins_and_paired_deltas():
    f=pd.DataFrame(dict(sample_index=[0,0,1,1]));y=np.zeros(4)
    pred={p:np.array([1.,1.,3.,3.]) for p in MAIN}
    pred['CommodityNode']=np.zeros(4)
    rows=bootstrap_regression(f,pred,y,1.)
    row=next(x for x in rows if x['kind']=='paired_delta' and x['comparison']=='CommodityNode - GlobalFused' and x['metric']=='RiskMAE')
    assert row['estimate']==-2 and row['CI95']==[-3.,-1.]
    assert rows==bootstrap_regression(f,pred,y,1.)


def test_half_fits_share_full_train_coordinate_system():
    x=np.arange(40.)[:,None];y=.02+.003*x[:,0];fit=fit_linear(x,y)
    stability=stability_fit(x,y,fit,np.repeat(np.arange(20),2))
    assert stability['first_fit']['mean']==stability['second_fit']['mean']==fit['mean']
    assert stability['first_raw_log_vol_slope']==pytest.approx(.003)
    assert stability['second_raw_log_vol_slope']==pytest.approx(.003)
    assert fit_linear(x,y)==fit


def test_numerical_audit_checks_same_fit_without_refitting(monkeypatch):
    x=np.arange(30.)[:,None];y=.01+.002*x[:,0]
    splits=np.repeat(['train','val','test'],10);train=splits=='train'
    fit=fit_linear(x[train],y[train])
    def forbidden(*args,**kwargs):
        raise AssertionError('audit must not refit')
    monkeypatch.setattr('cmgm.scripts.d0b_risk_probe_analysis.fit_linear',forbidden)
    r=audit_linear(x,y,train,fit,splits)
    assert r['normal_equation_mean_residual_max']<1e-14
    assert r['equivalent_design_max_diff']<1e-14
    assert r['test_standardized_abs_max']>r['train_standardized_abs_max']


def test_target_distribution_is_descriptive_and_has_split_top5():
    f=pd.DataFrame(dict(split=np.repeat(['train','val','test'],20),target=np.tile(np.arange(20.)/100,3)))
    rows=target_distributions(f,.1)
    row=next(x for x in rows if x['split']=='test' and x['target']=='absolute')
    assert row['split_top5_count']==1
    assert row['split_top5_target_mass_share']==pytest.approx(.19/1.9)
    log=next(x for x in rows if x['split']=='test' and x['target']=='log_absolute')
    assert log['split_top5_target_mass_share'] is None


class Branch(nn.Module):
    def forward(self,x):
        self.last_h_long=x[:,0];self.last_h_micro=x[:,1]
        return self.last_h_long+2*self.last_h_micro


class Pool(nn.Module):
    def forward(self,x):
        return x.mean(dim=1)


class ToyNative(nn.Module):
    def __init__(self):
        super().__init__();self.num_nodes=4
        self.type_pool=Pool();self.switching_latent_transformer=Branch()
        self.head=nn.Linear(64,8)

    def forward(self,x):
        return self.head(self.type_pool(x)+self.switching_latent_transformer(x))


def test_hooks_capture_actual_tensors_preserve_batch_and_do_not_modify_forward():
    model=ToyNative().eval();x=torch.arange(3*4*64,dtype=torch.float32).reshape(3,4,64)
    original={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        normal=model(x)
        with RepresentationCollector(model,[1,3]) as c:
            hooked=model(x)
        reps=c.concatenated()
    torch.testing.assert_close(normal,hooked,rtol=0,atol=0)
    np.testing.assert_array_equal(reps['h_comm'],x[:,[1,3]].numpy())
    np.testing.assert_array_equal(reps['h_fused'],(x.mean(1)+x[:,0]+2*x[:,1]).numpy())
    assert reps['h_comm'].shape==(3,2,64)
    assert not model.type_pool._forward_pre_hooks and not model.head._forward_pre_hooks
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,original[k],rtol=0,atol=0)
