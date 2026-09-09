import numpy as np
import pandas as pd
import pytest

from cmgm.scripts.d0b_volatility_conditional_analysis import (
    EPS_VOL, fit_train, least_squares, apply_frozen, bootstrap_comparisons,
    analyze, write_report,
)


def records(n=24):
    rng=np.random.default_rng(42)
    frames=[]
    for split in ('train','val','test'):
        p=rng.normal(0,.02,n*3)
        logv=rng.normal(-4.5,.5,n*3)
        vol=np.exp(logv)-EPS_VOL
        y=(2+.3*logv)*p
        frames.append(pd.DataFrame(dict(split=split,sample_index=np.repeat(np.arange(n),3),
            forecast_origin=np.repeat(pd.date_range('2020-01-01',periods=n).astype(str),3),
            commodity_index=np.tile(np.arange(3),n),commodity=np.tile(['A','B','C'],n),
            prediction=p,target=y,commodity_vol20=vol)))
    return pd.concat(frames,ignore_index=True)


def test_lstsq_uses_exactly_two_predictors_without_intercept_or_ridge():
    p=np.array([1.,2.,1.,3.,2.]);z=np.array([-2.,-1.,0.,1.,2.]);y=(.7+.2*z)*p
    fit=least_squares(p,z,y)
    assert fit['a']==pytest.approx(.7)
    assert fit['b']==pytest.approx(.2)
    assert fit['rank']==2
    assert fit['normal_equation_residual_max']<1e-12


def test_rank_deficiency_reported_without_ridge():
    fit=least_squares([1.,2.,3.],[1.,1.,1.],[2.,4.,6.])
    assert fit['rank']==1 and fit['numerical_warning']
    assert fit['a']+fit['b']==pytest.approx(2.)


def test_log_pooled_train_standardization_and_coefficients():
    f=records();tr=f[f.split=='train'];r=fit_train(tr)
    log=np.log(tr.commodity_vol20+EPS_VOL)
    assert r['mu_log_vol']==pytest.approx(log.mean())
    assert r['sigma_log_vol']==pytest.approx(log.std(ddof=0))
    assert r['conditional_fit']['a']==pytest.approx(2+.3*log.mean())
    assert r['conditional_fit']['b']==pytest.approx(.3*(log.std(ddof=0)+EPS_VOL))
    applied=apply_frozen(f,r)
    np.testing.assert_allclose(applied.Conditional,applied.target,atol=1e-12)
    assert applied[applied.split=='train'].z.mean()==pytest.approx(0,abs=1e-12)
    for _,part in applied[applied.split=='train'].groupby('commodity_index'):
        assert part.within_vol20.mean()==pytest.approx(0,abs=1e-12)


def test_no_val_test_refit_and_eval_targets_cannot_change_predictions():
    f=records();frozen=fit_train(f[f.split=='train'])
    before=apply_frozen(f,frozen)
    f.loc[f.split!='train','target']*=1000
    after=apply_frozen(f,frozen)
    np.testing.assert_array_equal(before.Conditional,after.Conditional)
    assert fit_train(f[f.split=='train'])==frozen
    with pytest.raises(AssertionError):
        fit_train(f)


def test_negative_scale_is_not_clipped_and_zero_target_hit_can_remain_wrong():
    f=records();frozen=fit_train(f[f.split=='train'])
    frozen['conditional_fit'].update(a=-1.,b=0.)
    f['target']=0.
    out=apply_frozen(f,frozen)
    np.testing.assert_array_equal(out.Conditional,-out.Native)
    assert out.sign_flip.all()
    assert (np.sign(out.Native)!=np.sign(out.target)).all()
    assert (np.sign(out.Conditional)!=np.sign(out.target)).all()


def test_stability_halves_share_full_train_z_and_never_rewrite_full_fit():
    f=records();fr=fit_train(f[f.split=='train']);applied=apply_frozen(f,fr)
    st=fr['coefficient_stability']
    assert st['first_half']['origins']==st['second_half']['origins']==12
    assert st['first_half']['a']==pytest.approx(fr['conditional_fit']['a'])
    assert st['second_half']['b']==pytest.approx(fr['conditional_fit']['b'])
    for row in st['scale_function']:
        assert row['full']==pytest.approx(fr['conditional_fit']['a']+fr['conditional_fit']['b']*row['z'])
    np.testing.assert_allclose(applied.Conditional,applied.target,atol=1e-12)


def test_cluster_bootstrap_delta_vs_native_and_global():
    f=pd.DataFrame(dict(sample_index=[0,0,1,1],target=[0.]*4,
        Native=[2.,2.,4.,4.],Global=[1.,1.,3.,3.],Conditional=[0.]*4))
    rows=bootstrap_comparisons(f)
    cg=next(r for r in rows if r['comparison']=='Conditional_minus_Global' and r['metric']=='MAE')
    assert cg['delta']==-2 and cg['CI95']==[-3.,-1.]
    cn=next(r for r in rows if r['comparison']=='Conditional_minus_Native' and r['metric']=='MSE')
    assert cn['delta']==-10 and cn['CI95']==[-16.,-4.]


def test_native_mismatch_stops_before_fitting(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('must not fit after baseline failure')
    monkeypatch.setattr('cmgm.scripts.d0b_volatility_conditional_analysis.fit_train',forbidden)
    r=analyze(records(),tmp_path)
    assert r['status'].startswith('STOPPED')
    write_report(r,tmp_path)
    assert 'no calibration fitted' in (tmp_path/'REPORT.md').read_text()


def test_full_exports_comparisons_and_attribution(tmp_path):
    r=analyze(records(),tmp_path,verify_baseline=False)
    f=pd.read_csv(tmp_path/'observation_conditional_diagnostics.csv')
    fit=r['frozen_train']['conditional_fit']
    np.testing.assert_allclose(f.Conditional,(fit['a']+fit['b']*f.z)*f.Native,atol=1e-14)
    assert len(r['bootstrap'])==8
    assert len(r['risk_decomposition'])==9
    for split in ('train','val','test'):
        total=next(x for x in r['overall'] if x['split']==split)
        assert sum(x['count'] for x in r['volatility_bins'] if x['split']==split)==total['count']
    test=next(x for x in r['overall'] if x['split']=='test')
    assert sum(x['contribution_to_overall_delta_MAE'] for x in r['commodity_error_attribution'])==pytest.approx(test['Conditional_minus_Global']['MAE'])
    write_report(r,tmp_path)
    assert '16. Primary case:' in (tmp_path/'REPORT.md').read_text()
