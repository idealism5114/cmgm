"""D0B-TailPredictabilityAmplitudeDiagnostic: inference and closed-form statistics only."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from cmgm.config import MULTI_HORIZONS
from cmgm.scripts.d0b_5d_error_regime_diagnostic import (
    ROOT, VARIANTS, prepare_data, validate_causal_features, discover_checkpoint,
    load_model, collect, sha256, state_checksum,
)
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results


def add_commodity_signals(frame, datasets, prices, mi):
    """Reuse raw feature_builder semantics, indexed by verified forecast origins."""
    raw = pd.DataFrame(prices.values)
    daily = raw.pct_change().fillna(0.0)
    arrays = {"abs_past1": daily.clip(-5, 5).abs().to_numpy(),
        "abs_past5": raw.pct_change(5).clip(-5, 5).fillna(0).abs().to_numpy(),
        **{f"commodity_vol{w}": daily.rolling(w, min_periods=1).std().fillna(0).clip(0, 5).to_numpy() for w in (5, 20)}}
    result = frame.copy()
    offset = 0
    cs, _ = mi['commodity']
    for split in ('train', 'val', 'test'):
        mask = result.split == split
        ds = datasets[split]
        origin = offset + result.loc[mask, 'sample_index'].to_numpy() + ds.seq_len-1
        commodity = result.loc[mask, 'commodity_index'].to_numpy()+cs
        assert np.array_equal(prices.index[origin].strftime('%Y-%m-%d'), result.loc[mask, 'forecast_origin'])
        for key, value in arrays.items():
            result.loc[mask, key] = value[origin, commodity]
        offset += len(ds.raw_prices)
    np.testing.assert_allclose(result.abs_past5, result.past5.abs(), rtol=0, atol=0)
    result['prediction'] = result.pop('D0B').astype(np.float64)
    result['target'] = result.target.astype(np.float64)
    result['abs_prediction'] = result.prediction.abs()
    return result


def run_inference(args):
    path = discover_checkpoint(VARIANTS['D0B'], args.checkpoint_dir, args.checkpoint)
    if path is None:
        raise FileNotFoundError('Official D0B checkpoint not found')
    torch.manual_seed(42)
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    datasets, prices, mi, fingerprint = prepare_data(args.prepared_data)
    sanity = validate_causal_features(datasets, prices, mi)
    model, info = load_model('D0B', path, mi, device)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    frame = collect(model, 'D0B', datasets, prices, mi, device)
    frame = add_commodity_signals(frame, datasets, prices, mi)
    max_diff = max(float((value.detach().cpu()-before[name]).abs().max()) for name, value in model.state_dict().items())
    info.update(file_sha256_after=sha256(path), state_sha256_after=state_checksum(model),
                model_parameter_max_diff=max_diff)
    assert info['state_sha256_before'] == info['state_sha256_after']
    assert info['file_sha256_before'] == info['file_sha256_after']
    assert max_diff == 0
    output = args.output_dir or ROOT/'experiments/d0b_tail_predictability_amplitude'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(checkpoint=info, source_fingerprint=fingerprint, causal_feature_sanity=sanity,
        diagnostic_git_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        device=str(device), primary_horizon=5, primary_index=MULTI_HORIZONS.index(5),
        alignment='Each date/origin/commodity and target reconstructed and asserted against original raw prices',
        fixed_test_batch_shape=[64, datasets['test'].seq_len, len(prices.columns), datasets['test'].feature_matrix.shape[-1]])
    frame.to_csv(output/'inference_records.csv', index=False)
    save_results(meta, output/'inference_metadata.json')
    return frame, meta, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prepared-data', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    frame, metadata, output = run_inference(args)
    from cmgm.scripts.d0b_tail_amplitude_analysis import analyze, write_report
    result = analyze(frame, output)
    result.update(metadata)
    result['implementation_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in (
        Path(__file__), ROOT/'cmgm/scripts/d0b_tail_amplitude_analysis.py',
        ROOT/'cmgm/scripts/d0b_5d_error_regime_diagnostic.py', ROOT/'cmgm/scripts/d0b_5d_error_regime_analysis.py')}
    save_results(result, output/'results.json')
    write_report(result, output)
    print(f"[D0B tail amplitude] Report: {output/'REPORT.md'}", flush=True)


if __name__ == '__main__':
    main()
