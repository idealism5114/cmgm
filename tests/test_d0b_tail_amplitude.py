import numpy as np
import pandas as pd
import pytest

from cmgm.scripts.d0b_tail_amplitude_analysis import (
    SIGNALS, freeze_train, scalar_coefficient, ranking_metrics, calibration_bootstrap,
    ranking_bootstrap, cluster_weights, analyze, write_report,
)


def frame(n=24):
    rng=np.random.default_rng(16)
    rows=[]
    for s in ('train','val','test'):
        p=rng.normal(0,.01,n*3)
        y=2*p+rng.normal(0,.002,n*3)
        f=pd.DataFrame(dict(split=s,sample_index=np.repeat(np.arange(n),3),
            forecast_origin=np.repeat(pd.date_range('2020-01-01',periods=n).astype(str),3),
            commodity_index=np.tile(np.arange(3),n),commodity=np.tile(['A','B','C'],n),
            prediction=p,target=y))
        for k in SIGNALS:
            f[k]=rng.uniform(0,1,len(f))
        f['abs_prediction']=np.abs(p)
        rows.append(f)
    return pd.concat(rows,ignore_index=True)


def test_closed_form_no_intercept_and_frozen_training_only():
    p=np.array([1.,2.,3.]);y=3*p
    assert scalar_coefficient(p,y)==pytest.approx(3)
    f=frame()
    original=freeze_train(f[f.split=='train'])
    f.loc[f.split=='test','target']*=100
    f.loc[f.split=='val','prediction']*=100
    assert freeze_train(f[f.split=='train'])==original
    with pytest.raises(AssertionError):
        freeze_train(f)
    assert abs(np.dot(p,scalar_coefficient(p,y)*p-y))<1e-10


def test_orientation_uses_train_tail90_for_both_labels():
    f=frame();tr=f[f.split=='train'].copy()
    tr['entropy']=-tr.target.abs()
    spec=freeze_train(tr)['signal_specs']['entropy']
    assert spec['orientation']==-1
    assert spec['train_tail90_pearson']<0
    # Constants keep a deterministic +1, without a spurious correlation.
    tr['confidence']=1.
    spec=freeze_train(tr)['signal_specs']['confidence']
    assert spec['orientation']==1 and spec['train_tail90_pearson'] is None


def test_auc_average_precision_known_ranking_and_ties():
    r=ranking_metrics([1,0,1,0],[.9,.8,.7,.1])
    assert r['AUROC']==.75
    assert r['AUPRC']==pytest.approx(5/6)
    r=ranking_metrics([1,0,0,0],[1,1,1,1])
    assert r['AUROC']==.5
    assert r['AUPRC']==.25 and r['AUPRC_over_prevalence']==1
    assert ranking_metrics([1,1],[2,1])['AUROC'] is None
    assert ranking_metrics([0,0],[2,1])['AUPRC'] is None


def test_weighted_auc_and_ap_equal_explicit_cluster_repetition():
    y=np.array([1,0,1,0,1,0]);s=np.array([.9,.8,.8,.1,.2,.2]);w=np.array([2,2,0,0,3,3])
    weighted=ranking_metrics(y,s,weights=w)
    explicit=ranking_metrics(np.repeat(y,w),np.repeat(s,w))
    assert weighted==explicit


def test_origin_resampling_keeps_all_commodities_together():
    counts,inverse=cluster_weights([0,0,1,1],draws=1000,seed=42)
    np.testing.assert_array_equal(counts[:,inverse][:,0],counts[:,inverse][:,1])
    np.testing.assert_array_equal(counts[:,inverse][:,2],counts[:,inverse][:,3])
    assert (counts.sum(1)==2).all()
    f=pd.DataFrame(dict(sample_index=[0,0,1,1],prediction=[1.,1.,3.,3.],target=[0.]*4,
                        calibrated_prediction=[0.]*4))
    b=calibration_bootstrap(f)
    assert b['MAE']['delta']==-2
    assert b['MSE']['delta']==-5
    assert b['MAE']['CI95']==[-3,-1]


def test_auc_bootstrap_counts_origin_dependence_and_is_deterministic():
    f=pd.DataFrame(dict(sample_index=[0,0,1,1]))
    y=np.array([1,0,1,0]);s=np.array([4,3,2,1])
    a=ranking_bootstrap(f,y,s)
    assert a==ranking_bootstrap(f,y,s)
    assert a['valid_draws']==1000
    assert a['CI95'][0]>=.5 and a['CI95'][1]<=1


def test_negative_coefficient_stops_without_calibrated_predictions(tmp_path):
    f=frame();f.loc[f.split=='train','target']=-f.loc[f.split=='train','prediction']
    r=analyze(f,tmp_path)
    assert r['frozen_train']['c_train']<0
    assert r['status'].startswith('STOPPED')
    assert not (tmp_path/'calibration_metrics.csv').exists()
    write_report(r,tmp_path)
    assert 'Stopped' in (tmp_path/'REPORT.md').read_text()


def test_full_analysis_export_uses_one_scalar_and_preserves_sign(tmp_path):
    f=frame()
    r=analyze(f,tmp_path)
    obs=pd.read_csv(tmp_path/'observation_tail_diagnostics.csv')
    np.testing.assert_allclose(obs.calibrated_prediction,r['frozen_train']['c_train']*obs.prediction)
    assert (np.sign(obs.calibrated_prediction)==np.sign(obs.prediction)).all()
    for s in ('train','val','test'):
        sr=obs[obs.split==s]
        assert (sr.Tail90==(sr.target.abs()>r['frozen_train']['tail_thresholds']['T90'])).all()
    for row in r['calibration']:
        assert row['delta']['Hit']==0
        if row['count']:
            assert row['calibrated']['RMSE']**2==pytest.approx(row['calibrated']['MSE'])
    for row in r['top_capture']:
        spec=r['frozen_train']['signal_specs'][row['signal']]
        assert row['cutoff']==spec[f"oriented_top{row['top_train_percent']}_cut"]
    assert len(obs)==len(f)
    write_report(r,tmp_path)
    assert '18. Primary case:' in (tmp_path/'REPORT.md').read_text()
    assert r['primary_case']=='Case A'
