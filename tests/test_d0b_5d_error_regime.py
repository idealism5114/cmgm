import numpy as np
import pandas as pd
import pytest
import torch

from cmgm.scripts.d0b_5d_error_regime_analysis import (
    QUANTILES, analyze, assign_groups, bin_values, clustered_bootstrap,
    complementarity, fit_thresholds, metrics, paired_win_stats, save_results, write_report,
)
from cmgm.scripts.d0b_5d_error_regime_diagnostic import (
    KEYS, aligned_merge, causal_timeline, discover_checkpoint, latent_context,
)


def records(n=24, comparator=True):
    rng = np.random.default_rng(27)
    frames = []
    for split in ("train", "val", "test"):
        count = n*3
        y = rng.normal(0, .025, count)
        frame = pd.DataFrame({"split": split, "sample_index": np.repeat(np.arange(n), 3),
            "forecast_origin": np.repeat(pd.date_range('2020-01-01', periods=n).astype(str), 3),
            "commodity_index": np.tile(np.arange(3), n), "commodity": np.tile(['A','B','C'], n),
            "target": y, "D0B": y+rng.normal(0, .02, count), "past5": rng.normal(0, .02, count)})
        if comparator:
            frame["TempWeighted"] = y+rng.normal(0, .02, count)
        for k in set(QUANTILES) | {'margin', 'past5_stock', 'past5_bond', 'past5_commodity'}:
            frame[k] = np.repeat(rng.uniform(.01, .1, n), 3)
        frame['target_magnitude'] = frame.target.abs()
        frame['trend_strength'] = frame.past5.abs()
        p = rng.dirichlet(np.ones(3), n)
        for k in range(3):
            frame[f'p{k}'] = np.repeat(p[:, k], 3)
        frame['state'] = np.repeat(p.argmax(-1), 3)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_pooled_metrics_use_observations_and_unmasked_sign():
    pred, y = np.array([1., 0., -1.]), np.array([0., 0., 1.])
    m = metrics(pred, y)
    assert m['MAE'] == 1
    assert m['MSE'] == pytest.approx(5/3)
    assert m['Hit'] == pytest.approx(1/3)
    assert m['RMSE']**2 == pytest.approx(m['MSE'])


def test_alignment_is_identity_based_not_row_order():
    f = records()
    a = f[KEYS+['target','D0B']]
    b = f[KEYS+['target','TempWeighted']].sample(frac=1, random_state=2)
    got = aligned_merge(a, b)
    np.testing.assert_array_equal(got.TempWeighted, f.TempWeighted)
    bad = b.copy()
    bad.iloc[0, bad.columns.get_loc('target')] += .1
    with pytest.raises(AssertionError):
        aligned_merge(a, bad)
    with pytest.raises(AssertionError):
        aligned_merge(a, b.iloc[1:])
    with pytest.raises(AssertionError):
        aligned_merge(a, pd.concat([b,b.iloc[:1]]))


def test_thresholds_refuse_validation_and_do_not_depend_on_test():
    f = records()
    with pytest.raises(AssertionError):
        fit_thresholds(f, ['D0B'])
    initial = fit_thresholds(f[f.split == 'train'], ['D0B'])
    f.loc[f.split == 'test', 'target_magnitude'] = 1000
    assert fit_thresholds(f[f.split == 'train'], ['D0B']) == initial
    assert initial['transition_score']['unit'] == 'forecast_origin'


def test_ties_in_thresholds_do_not_drop_observations():
    bins = bin_values(np.array([0.,1.,1.5,2.]), [1.,1.,2.], ['a','b','c','d'])
    assert list(bins) == ['a','a','c','c']


def test_zero_and_weak_trend_not_forced_into_reversal():
    f = records()
    t = fit_thresholds(f[f.split == 'train'], ['D0B'])
    f.loc[:4, 'past5'] = [0., 1., .000001, .1, -.1]
    f.loc[:4, 'target'] = [1., 0., -.1, .1, .1]
    t['flat_past_p10'] = .001
    g, _ = assign_groups(f, t, ['D0B'])
    assert list(g['reversal'][:5]) == ['Zero','Zero','Flat/weak','Continuation','Reversal']


def test_volatility_and_cross_market_columns_are_prefix_causal():
    rng = np.random.default_rng(11)
    prices = 100*np.exp(np.cumsum(rng.normal(0,.01,(60,6)),axis=0))
    mi = dict(stock=(0,2), bond=(2,4), commodity=(4,6))
    before, past = causal_timeline(prices, mi)
    prices[35:] *= rng.uniform(.1, 4, (25,6))
    after, future_past = causal_timeline(prices, mi)
    for key in before:
        np.testing.assert_array_equal(before[key][:35], after[key][:35])
    np.testing.assert_array_equal(past[:35], future_past[:35])


def test_latent_movement_uses_last_five_actual_differences():
    from types import SimpleNamespace
    p = torch.tensor([[[.3,.3,.4],[.4,.3,.3],[.5,.3,.2],[.6,.2,.2],[.5,.3,.2],[.4,.4,.2],[.3,.5,.2]]])
    z = torch.arange(7.).reshape(1,7,1)
    branch = SimpleNamespace(last_regime_probabilities=p, last_regime_priors=torch.ones_like(p)/3, last_latent_states=z)
    out = latent_context(branch)
    expected = np.abs(np.diff(p.double().numpy(),axis=1)).sum(-1)[:, -5:].mean(-1)
    np.testing.assert_array_equal(out['transition_score'], expected)
    assert out['delta_z'][0] == 1 and out['mean_delta_z5'][0] == 1


def test_cluster_bootstrap_keeps_date_assets_together():
    f = pd.DataFrame({'sample_index':[0,0,1,1], 'target':[0.]*4,
                      'D0B':[1.,1.,3.,3.], 'TempWeighted':[0.]*4})
    got = clustered_bootstrap(f, [True]*4)
    rng = np.random.default_rng(42)
    ids = rng.integers(0,2,(1000,2))
    expected = np.array([1.,3.])[ids].mean(1)
    assert got['delta_MAE'] == 2
    np.testing.assert_array_equal(got['CI95'], np.quantile(expected,[.025,.975]))
    assert got['warning'] == 'LOW SAMPLE SIZE'
    empty = clustered_bootstrap(f, [False]*4)
    assert empty['CI95'] is None and empty['valid_draws'] == 0


def test_cluster_subgroups_weight_observation_counts():
    f = pd.DataFrame({'sample_index':[0,0,1,1], 'target':[0.]*4,
                      'D0B':[1.,1.,4.,4.], 'TempWeighted':[0.]*4})
    got = clustered_bootstrap(f, [True,False,True,True])
    assert got['delta_MAE'] == 3  # (1 + 4 + 4)/3, not average of 1 and 4


def test_win_tolerance_shared_with_mse():
    result = paired_win_stats(np.array([.1,1,2]), np.array([.1+1e-10,2,1]), np.zeros(3))
    assert result['Tie'] == pytest.approx(1/3)
    assert result['D0B_win'] == result['MSE_D0B_win'] == pytest.approx(1/3)
    assert result['TempWeighted_win'] == result['MSE_TempWeighted_win'] == pytest.approx(1/3)


def test_oracle_uses_same_observation_selection_for_mae_and_mse():
    f = records(n=24)
    thresholds = fit_thresholds(f[f.split == 'train'], ['D0B','TempWeighted'])
    groups,_ = assign_groups(f, thresholds, ['D0B','TempWeighted'])
    f['group_reversal'] = groups['reversal']
    f = f[f.split == 'test'].copy()
    got = complementarity(f, thresholds)
    best_errors = np.minimum((f.D0B-f.target).abs(), (f.TempWeighted-f.target).abs())
    assert got['oracle']['MAE'] == pytest.approx(best_errors.mean())
    assert got['oracle']['MSE'] == pytest.approx((best_errors**2).mean())
    assert got['average']['MSE'] == pytest.approx((((f.D0B+f.TempWeighted)/2-f.target)**2).mean())


@pytest.mark.parametrize('comparator',[False,True])
def test_analysis_exports_complete_records_and_contribution_shares(tmp_path, comparator):
    f = records(n=24, comparator=comparator)
    result = analyze(f, tmp_path)
    models = ['D0B','TempWeighted'] if comparator else ['D0B']
    rows = pd.read_csv(tmp_path/'observation_errors.csv')
    assert len(rows) == len(f)*len(models)
    for split in ('train','val','test'):
        groups = [g for g in result['groups'] if g['split']==split and g['dimension']=='target_magnitude']
        assert sum(g['count'] for g in groups) == len(f[f.split==split])
        for model in models:
            assert sum(g[model+'_MSE_share'] for g in groups) == pytest.approx(1.)
            assert sum(g[model+'_MAE_share'] for g in groups) == pytest.approx(1.)
    assert bool(result['complementarity']) == comparator
    if not comparator:
        assert result['primary_classification'] is None
        assert 'NOT COMPUTABLE' in (tmp_path/'model_complementarity.csv').read_text()
    save_results(result, tmp_path/'results.json')
    assert 'NaN' not in (tmp_path/'results.json').read_text()
    result.update(status='COMPLETE' if comparator else 'INCOMPLETE',
                  diagnostic_git_sha='synthetic-test', fixed_batch_shape=[24,20,6,21], checkpoints={})
    write_report(result, tmp_path)
    report = (tmp_path/'REPORT.md').read_text()
    assert '20. Primary A–G classification' in report
    if not comparator:
        assert 'NOT ASSIGNED' in report and '未保存权重' in report


def test_checkpoint_discovery_requires_actual_matching_weights(tmp_path):
    assert discover_checkpoint('temporal_weighted_graph', tmp_path) is None
    unrelated = tmp_path/'other.pt'
    torch.save({'metadata':{'variant':'unrelated'},'model_state_dict':{}},unrelated)
    assert discover_checkpoint('temporal_weighted_graph',tmp_path) is None
    official = tmp_path/'official.pt'
    torch.save({'metadata':{'variant':'temporal_weighted_graph'},'model_state_dict':{}},official)
    assert discover_checkpoint('temporal_weighted_graph',tmp_path) == official
    named = tmp_path/'temporal_weighted_graph_best.pt'
    torch.save({},named)
    with pytest.raises(ValueError, match='Multiple checkpoints'):
        discover_checkpoint('temporal_weighted_graph',tmp_path)
