"""Synthetic-only checks. No real data, fit, TEST evaluation or persistent training."""
import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.scripts import d0b_candidate_moe_bottleneck16 as experiment
from cmgm.scripts.baseline_protocol import prediction_loss, data_audit
from cmgm.scripts.d0b_candidate_moe_audit import sanity

DATA = dict(n_nodes=30, market_indices=dict(stock=(0, 4), bond=(4, 6), commodity=(6, 30)))


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('seed', [42, 19])
def test_shared_initialization_and_structure(seed):
    model, audit = experiment.initialization_audit(DATA, seed)
    assert audit['PASS'] and audit['shared_max_abs_diff'] == 0 and audit['mismatch_count'] == 0
    assert audit['parameter_delta'] == -15456 and audit['bottleneck16']['experts'] == 5280
    assert audit['bottleneck16']['router'] == 16834
    f = model.candidate_moe_fusion
    assert [(m.in_features, m.out_features) for m in f.temporal_expert.mlp if isinstance(m, torch.nn.Linear)] == [(64, 16), (16, 64)]
    assert [(m.in_features, m.out_features) for m in f.interaction_expert if isinstance(m, torch.nn.Linear)] == [(128, 16), (16, 64)]
    assert not hasattr(model, 'moe_fusion') and not hasattr(model, 'global_mixture_fusion')
    assert not hasattr(f, 'balance_loss') and not hasattr(f, 'epoch')


def test_shapes_raw_mixture_head_causality_and_gradients():
    model = experiment.make_model(DATA)
    result = sanity(model, torch.randn(2, 20, 30, 21), torch.randn(2, 4, 24)*.02)
    assert result['PASS'], result
    assert result['initial_routing_error'] == 0
    assert result['gradient_norms']['RouterFirst'] == 0
    assert result['gradient_norms']['RouterFinal'] > 0


def test_native_prediction_plus_KL_and_batch_mean(monkeypatch):
    tr = importlib.import_module('cmgm.training.train')
    model = experiment.make_model(DATA)
    branch = tr._switching_branch(model)
    assert branch is model.switching_latent_transformer
    assert branch.set_epoch(1) == 0
    assert branch.set_epoch(10) == pytest.approx(5e-4*9/19)
    assert branch.set_epoch(20) == 5e-4
    assert branch.set_epoch(100) == 5e-4
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    def forbidden(*args):
        raise AssertionError('No MoE auxiliary loss or routing warm-up')
    model.moe_balance_loss = forbidden
    model.set_moe_epoch = forbidden
    x, y = torch.randn(3, 20, 30, 21), torch.randn(3, 4, 24)*.02
    y[-1] += .3  # Unequal final batch makes pooled vs batch-mean selection distinguishable.
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    pred_losses, switch_losses = [], []
    model.train()
    for a, b in loader:
        p = model(a)
        pred_losses.append(float(prediction_loss(p, b).detach()))
        switch_losses.append(float(branch.switch_loss().detach()))
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *a, **kw: None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    edges, weights = torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    loss = tr.train_epoch(model, loader, edges, weights, optimizer, torch.nn.HuberLoss(delta=.02), torch.device('cpu'))
    assert loss == pytest.approx(np.mean(pred_losses)+np.mean(switch_losses), abs=1e-8)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    val = tr.validate_epoch(model, loader, edges, weights, torch.nn.HuberLoss(delta=.02), torch.device('cpu'))
    assert val == pytest.approx(np.mean(pred_losses), abs=1e-8)
    assert abs(val-(pred_losses[0]*2+pred_losses[1])/3) > 1e-4
    assert 'std_pi_ST' in model._last_train_candidate and 'routing_entropy' in model._last_val_candidate


def test_selection_checkpoint_restore_and_original_shape_compatibility(tmp_path, monkeypatch):
    tr = importlib.import_module('cmgm.training.train')
    model = experiment.make_model(DATA)
    objectives = iter([.4, .2, .3])
    diagnostic_mae = iter([.01, .02, .001])
    def fake_train(m, *args, **kw):
        m._last_train_candidate = {'mean_pi_T': .5}
        return .5
    def fake_val(m, *args, **kw):
        m._last_val_candidate = {'mean_pi_T': .5}
        m._last_val5_diagnostic = dict(MAE=next(diagnostic_mae), MSE=.001, count=1)
        return next(objectives)
    monkeypatch.setattr(tr, 'train_epoch', fake_train)
    monkeypatch.setattr(tr, 'validate_epoch', fake_val)
    path = tmp_path/'synthetic.pt'
    history = tr.train(model, None, None, torch.empty((2, 0), dtype=torch.long), torch.empty(0),
                       torch.device('cpu'), num_epochs=3, checkpoint_path=str(path))
    assert history['best_epoch'] == 2 and history['best_val5_epoch'] == 3
    assert len(history['candidate_routing_history']) == 3
    assert history['switch_beta'] == pytest.approx([0, 5e-4/19, 1e-3/19])
    cp = torch.load(path, weights_only=False)
    assert 'moe_epoch' not in cp
    restored = experiment.make_model(DATA)
    restored.load_state_dict(cp['model_state_dict'], strict=True)
    restored.switching_latent_transformer.set_epoch(cp['best_epoch'])
    model.eval(); restored.eval()
    x = torch.randn(2, 20, 30, 21)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
    original = experiment.make_model(DATA, variant=experiment.ORIGINAL)
    original_copy = experiment.make_model(DATA, variant=experiment.ORIGINAL)
    original_copy.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
    assert experiment.counts(original)['experts'] == 20736
    original.eval(); original_copy.eval()
    torch.testing.assert_close(original(x), original_copy(x), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        original.load_state_dict(cp['model_state_dict'], strict=True)


def test_train_val_loader_and_evaluation_boundary(tmp_path):
    x, y = torch.randn(65, 20, 30, 21), torch.randn(65, 4, 24)*.02
    data = dict(DATA, loaders={s: DataLoader(TensorDataset(x, y)) for s in ('train', 'val')})
    ls = experiment.data_loaders(data, 19)
    assert ls['train'].drop_last and not ls['val'].drop_last
    assert ls['train'].generator.initial_seed() == 19
    for loader in experiment.data_loaders(data, 19, full=True).values():
        assert not loader.drop_last
    small = dict(DATA, loaders={s: DataLoader(TensorDataset(x[:3], y[:3])) for s in ('train', 'val')})
    model = experiment.make_model(DATA)
    ev = experiment.evaluate_train_val(model, small, 42, torch.device('cpu'), tmp_path)
    assert set(ev) == {'train', 'val'}
    assert set(ev['val']['metrics']) == {'1', '5', '10', '20'}
    assert ev['train']['routing']['samples'] == 3
    assert all(v['RMSE']**2 == pytest.approx(v['MSE']) for v in ev['val']['metrics'].values())
    small['loaders']['test'] = small['loaders']['val']
    with pytest.raises(ValueError, match='TRAIN/VAL only'):
        experiment.data_loaders(small, 42)


def test_default_cli_never_builds_real_data_or_trains(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Default preflight must not access real data or training')
    monkeypatch.setattr(experiment, 'train_val_data', forbidden)
    monkeypatch.setattr(experiment, 'train', forbidden)
    called = []
    monkeypatch.setattr(experiment, 'synthetic_preflight', lambda seed, out: called.append((seed, out)))
    monkeypatch.setattr('sys.argv', ['bottleneck16', '--output', str(tmp_path/'preflight')])
    experiment.main()
    assert called == [(42, tmp_path/'preflight')]


def test_restricted_pipeline_exactly_matches_original_train_val(tmp_path, monkeypatch):
    """All calendars/values including the held-out tail are synthetic fixtures."""
    from cmgm.data import data_loader as dl
    from cmgm.scripts import main_ablation as main
    dates = pd.date_range('2019-01-01', periods=300)
    rng = np.random.default_rng(4)
    prices = 100*np.exp(np.cumsum(rng.normal(0, .01, (300, 30)), axis=0))
    stock = pd.DataFrame(prices[:, :4], columns=['S'+str(i) for i in range(4)])
    stock.insert(0, 'date', dates)
    # Original full-calendar preprocessing drops this stock; restricted parsing
    # must recover that frozen schema from TRAIN, not retain an extra node.
    stock.loc[299, 'S3'] = np.nan
    paths = [tmp_path/s for s in ('stock.csv', 'bond.csv', 'commodity.csv')]
    stock.to_csv(paths[0], index=False)
    for path, a, label in [(paths[1], prices[:, 4:6], 'B'), (paths[2], prices[:, 6:], 'C')]:
        frame = pd.DataFrame(a, index=dates, columns=[label+str(i) for i in range(a.shape[1])])
        if label == 'B':
            frame.iloc[:2] = np.nan  # Raw dates exist, pivot_table removes them.
        frame.rename_axis('date').reset_index().melt(id_vars='date', var_name='品种', value_name='close').to_csv(path, index=False)
    base = dl.create_data_loaders(*paths, batch_size=64, seq_len=20)
    monkeypatch.setattr(main, 'create_data_loaders', lambda **kwargs: base)
    original = main.build_data(SimpleNamespace(batch_size=64, seq_len=20))
    expected = data_audit(original)
    cache = tmp_path/'train_x.npy'
    ds = original['loaders']['train'].dataset
    np.save(cache, np.stack([ds[i][0].numpy().reshape(-1) for i in range(len(ds))]))
    monkeypatch.setattr(experiment, 'TRAIN_SCHEMA_CACHE', cache)
    manifest = tmp_path/'audit.json'
    manifest.write_text(json.dumps(expected))
    for key, path in zip(('STOCK_FILE', 'BOND_FILE', 'COMMODITY_FILE'), paths):
        monkeypatch.setattr(experiment.config, key, path)
    # Any parsing of tail VALUE cells now would change dtype or fail downstream.
    # Date metadata remains intact; skiprows must exclude these numeric cells.
    for path in paths:
        frame = pd.read_csv(path)
        tail = pd.to_datetime(frame['date']) > dates[254]
        for key in frame.columns:
            if key not in ('date', '品种'):
                frame[key] = frame[key].astype(object)
                frame.loc[tail, key] = 'FORBIDDEN_HELDOUT_VALUE'
        frame.to_csv(path, index=False)
    restricted, audit = experiment.train_val_data(manifest)
    assert audit['PASS'] and set(restricted['loaders']) == {'train', 'val'}
    assert 'raw_prices_test' not in restricted
    assert restricted['n_nodes'] == 29
    assert audit['source']['schema_recovery']['required']
    for split in ('train', 'val'):
        assert audit['split_fingerprint'][split] == expected['split_fingerprint'][split]
        a, b = original['loaders'][split].dataset, restricted['loaders'][split].dataset
        np.testing.assert_array_equal(a.feature_matrix, b.feature_matrix)
        for i in (0, len(a)-1):
            torch.testing.assert_close(a[i][0], b[i][0], rtol=0, atol=0)
            torch.testing.assert_close(a[i][1], b[i][1], rtol=0, atol=0)
    expected['split_fingerprint']['val']['sha256'] = 'bad'
    manifest.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match='differs from audited'):
        experiment.train_val_data(manifest)


def test_runner_lifecycle_with_mocked_fit_never_retrains(tmp_path, monkeypatch):
    """Exercise persistence/resume with zero optimizer steps and synthetic data."""
    x, y = torch.randn(3, 20, 30, 21), torch.randn(3, 4, 24)*.02
    data = dict(DATA, loaders={s: DataLoader(TensorDataset(x, y)) for s in ('train', 'val')})
    audit = dict(PASS=True, synthetic_fixture=True)
    monkeypatch.setattr(experiment, 'train_val_data', lambda _: (data, audit))
    monkeypatch.setattr(experiment, 'DEFAULT_ROOT', tmp_path/'runs')
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'source_record', lambda: {'hashes': {'synthetic': 'unchanged'}})
    calls = []
    def fake_fit(model, *args, **kw):
        calls.append(kw)
        assert kw['num_epochs'] == 200 and kw['lr'] == 1e-4 and kw['patience'] == 10
        history = dict(best_epoch=2, val_loss=[.3, .2], candidate_routing_history=[], train_time=.1)
        kw['epoch_history_callback'](history)
        torch.save(dict(model_state_dict=model.state_dict(), best_epoch=2, best_val_loss=.2,
                        history=history, metadata=kw['checkpoint_metadata']), kw['checkpoint_path'])
        return history
    monkeypatch.setattr(experiment, 'train', fake_fit)
    args = SimpleNamespace(device='cpu', output=tmp_path/'first', seed=42, run=True, data_audit='unused')
    experiment.execute(args)
    r = json.loads((args.output/'results.json').read_text())
    assert r['training_complete'] and r['status'] == 'COMPLETE — TRAIN/VAL ONLY'
    assert r['best_epoch'] == 2 and set(r['evaluation']) == {'train', 'val'}
    assert len(calls) == 1
    for name in ('training_history', 'routing_history', 'best_checkpoint_metadata', 'train_val_metrics', 'expert_diagnostics'):
        assert (args.output/(name+'.json')).exists()
    args.evaluate_completed = args.output
    experiment.evaluate_completed(args)
    assert len(calls) == 1  # Frozen re-evaluation never enters train().
    args.output = tmp_path/'second'
    with pytest.raises(RuntimeError, match='already reserved'):
        experiment.execute(args)
    assert len(calls) == 1
