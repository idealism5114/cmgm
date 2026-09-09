"""Frozen TRAIN-only two-column least squares; original D0B inference unchanged."""
import argparse
from datetime import datetime
from pathlib import Path

from cmgm.scripts.d0b_tail_predictability_amplitude import run_inference
from cmgm.scripts.d0b_5d_error_regime_diagnostic import ROOT, sha256
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--prepared-data',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-cuda',action='store_true')
    args=parser.parse_args()
    if args.output_dir is None:
        args.output_dir=ROOT/'experiments/d0b_volatility_conditional_shrinkage'/datetime.now().strftime('%Y%m%d_%H%M%S')
    frame,metadata,output=run_inference(args)
    from cmgm.scripts.d0b_volatility_conditional_analysis import analyze, write_report
    result=analyze(frame,output)
    result.update(metadata)
    result['implementation_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in (
        Path(__file__),ROOT/'cmgm/scripts/d0b_volatility_conditional_analysis.py',
        ROOT/'cmgm/scripts/d0b_tail_predictability_amplitude.py',ROOT/'cmgm/scripts/d0b_tail_amplitude_analysis.py',
        ROOT/'cmgm/scripts/d0b_5d_error_regime_diagnostic.py',ROOT/'cmgm/scripts/d0b_5d_error_regime_analysis.py')}
    save_results(result,output/'results.json')
    write_report(result,output)
    print(f"[D0B volatility conditional] {result['status']}; report: {output/'REPORT.md'}",flush=True)


if __name__=='__main__':
    main()
