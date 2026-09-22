import copy
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cmgm.scripts.baseline_runtime import measure_inference, RUNTIME_PROTOCOL


class TimedFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2, 96), nn.Dropout(.3))
        self.calls = []

    def forward(self, x):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append(len(x))
        return self.head(x).reshape(len(x), 4, 24)


def test_timing_counts_tail_units_and_preserves_state(monkeypatch):
    import cmgm.scripts.baseline_runtime as timing
    model = TimedFixture().train()
    model.head[1].eval()  # Preserve mixed submodule modes as well.
    states = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(TensorDataset(torch.randn(67, 2), torch.full((67, 4, 24), float('nan'))),
                        batch_size=64, generator=generator)
    rng, loader_rng = torch.get_rng_state(), generator.get_state()
    ticks = iter(range(20))
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: next(ticks))
    result = measure_inference(model, loader, torch.device('cpu'))
    assert result['origins'] == 67 and result['batch_sizes'] == [64, 3]
    assert result['pass_seconds'] == [1.] * 10
    assert result['test_forward_seconds_mean'] == 1 and result['test_forward_seconds_std'] == 0
    assert result['inference_ms_per_origin'] == 1000 / 67 and result['origins_per_second'] == 67
    assert model.calls == [64, 3, 64] + [64, 3] * 10
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng) and torch.equal(generator.get_state(), loader_rng)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in states.items())
    assert all(p.grad is None for p in model.parameters())


def test_timing_rejects_dropped_origins_and_restores_mode():
    model = TimedFixture().train()
    loader = DataLoader(TensorDataset(torch.zeros(67, 2), torch.zeros(67)), batch_size=64, drop_last=True)
    with pytest.raises(ValueError, match='every origin'):
        measure_inference(model, loader, 'cpu')
    assert model.training and not model.calls


def test_report_runtime_columns_and_historical_na(tmp_path):
    from cmgm.scripts.baseline_report import write_report, OURS
    runtime = dict(inference_ms_per_origin=2., origins_per_second=500., test_forward_seconds_mean=.538,
                   test_forward_seconds_std=.01, environment={'device_name': 'fixture GPU'})
    r = dict(models={OURS: {'runtime': runtime}, 'GRU': {'runtime': runtime, 'training': {
        'train_seconds': 12., 'seconds_per_epoch_mean': 3., 'epochs_completed': 4}}}, sanity={}, model_status={})
    write_report(r, tmp_path)
    rows = r['tables']['overall']
    ours = rows[-1]
    gru = next(row for row in rows if row['Model'] == 'GRU')
    assert ours['TrainSeconds'] is None and ours['InferenceMsPerOrigin'] == 2.
    assert gru['TrainSeconds'] == 12.
    report = (tmp_path / 'FINAL_REPORT.md').read_text()
    assert 'NOT batch1 request latency' in report and 'N/A (historical/unavailable)' in report
