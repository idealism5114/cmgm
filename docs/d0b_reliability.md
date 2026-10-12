# Frozen D0B interpretation and prediction reliability

This tool evaluates the existing switching_latent_balanced_readout checkpoint. It never trains, fine-tunes or replaces D0B. The default command runs synthetic preflight only. Full-data extraction, graph analysis, TEST inference, logistic risk fit and bootstrap are NOT_RUN in this delivery.

## Calculation semantics

The temporal branch computes long-memory H, a Markov posterior p, and latent state Z. The existing forced_probabilities interface replaces only weights mixed into the full Z recurrence. The original filter posterior and prior remain computed. Each rolling-window origin restarts the recurrence, so a date-level state curve uses each window's terminal posterior once. State 1/2/3 are not stock/bond/commodity regimes.

The graph attention uses einsum bnhd,bmhd->bnmh: i/n receives from source j/m. Softmax is over sources. A commodity node's attention row alone cannot explain its commodity forecast: D0B subsequently pools all nodes by market and its shared head emits all 24 commodity forecasts.

The fusion projects both branches to 64 dimensions and forms a=g*T, b=(1-g)*S, y=f(a+b), with a feature-wise gate. The zero-reference fixed-gate decomposition is

phi_T = 0.5[(f(a)-f(0))+(f(a+b)-f(b))]

phi_S = 0.5[(f(b)-f(0))+(f(a+b)-f(a))]

The terms sum to f(a+b)-f(0) up to floating-point error. The TRAIN-mean-component sensitivity uses the complete two-player formula with TRAIN mean a,b references. These are reference-dependent local explanations of the original head, not unique full-model causal contributions. Removing H/Z from the temporal readout recomputes the gate and prediction; it is a different intervention. A mean gate value is not an output contribution percentage.

## Fields and metrics

extract_SPLIT.npz contains predictions, labels, full-window filter posteriors and applied Z routes, priors, gate, projected T/S, weighted a/b, per-horizon/commodity phi_T and phi_S, f(0), share/cancellation, source attention summaries by layer/hop/head/source as well as an overall source mean, origin/20-day-label-end dates and risk B/R/G features. The source-detail array has shape [origins, 2 graph layers, 2 hops, heads, source nodes]; receiver nodes are averaged, not confused with source nodes. Full attention matrices are written only for explicitly selected case indices during the graph stage. audit.json records commit, worktree status, source hashes, checkpoint SHA256, TRAIN/VAL fingerprint, calendar and asset order.

The state stage reports soft/hard occupancy, entropy, effective count, distinctness, switch rate and run lengths. It distinguishes nearly uniform routing from one-state concentration. TRAIN-weighted market profiles use past-only 20-day descriptors and are checked later with fixed state labels. The mismatch control shifts intact 40-origin blocks. Optional seed43/44 comparison aligns labels with TRAIN profiles only.

The graph stage compares raw attention, the live heuristic abs(alpha * d forecast / d alpha), the D0B learned adjacency (fixed across samples at inference), and market/degree-matched random sources. It intervenes in both graph blocks and both hops, preserves self edges and renormalizes over sources. A zero-mass row falls back to self, preventing NaN. TRAIN-mean-V replacement is an operator-sensitivity control. Large prediction change indicates influence; positive delta MAE means deletion hurt accuracy, negative means the frozen model depended on harmful information. The gradient heuristic itself is not causal attribution.

Risk B uses past commodity volatility and absolute forecast magnitude; R uses terminal state entropy and adjacent-origin state change; G uses fixed-gate temporal share and cancellation; A uses source concentration and 1%-TRAIN-standard-deviation perturbation instability. No risk input uses future labels, realized error or target-informed oracle routing. The calibration segment alone fits the scaler, fixed-C logistic models and 80th-percentile large-error threshold. All eight B+subset(R,G,A) models share evaluation dates and coverage grid. MAE@80% coverage is selective MAE, not the model's full-sample MAE. Optional 2,000 paired moving-block draws resample 40 whole origin dates, never commodities independently.

Prior project work has viewed VAL and TEST, so analyses on those segments are exploratory. Confirmation requires a later locked, unused period. The runner checks that calibration 20-day label end dates precede evaluation origins; it does not change splits to manufacture independence.

## Commands

Run from the repository root. All non-preflight stages require --run. Use one new --output-dir for a study, shared across stages; outputs refuse to replace existing prediction or graph arrays.
Run each command to completion and stop if it exits nonzero. Later stages now check that both the required NPZ and its COMPLETE report exist before loading the full dataset. Extraction prints origin progress; an audit.json alone does not mean extraction completed.
If implementation changes after an audit.json was created, start a new --output-dir. The runner rejects source-hash mismatches rather than mixing artifacts from different code revisions.

Older completed extractions wrote date_last and target_end20 as NumPy object arrays. The explicit migrate-dates stage can create a separate continuation without repeating D0B inference: it reconstructs those two dates from the audited raw calendar, copies every other array exactly, checks stored metrics and commodity order, and records source/destination hashes. It never unpickles the old object arrays or overwrites them. For the 2026-10-08 seed42 continuation, the preserved source is `experiments/d0b_reliability/frozen_seed42_study_fix_20261008/` and the repaired output is `experiments/d0b_reliability/frozen_seed42_study_date_safe_20261008/`. The latter already contains completed TRAIN/VAL extraction and VAL state analysis; use it for remaining stages.

    cd /home/yangxiaotong/projects/myresearch/Commedities
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability
    OUT=experiments/d0b_reliability/frozen_seed42_study
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage extract --split train --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage extract --split val --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage state --split val --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage route --split val --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage branch --split val --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage graph --split val --horizon 5 --run --device cuda:0 --output-dir "$OUT"

The graph stage over every origin and 24 commodity targets is expensive. --case-index 0 --case-index 10 limits an exploratory smoke run and saves those cases' full attention matrices; risk fitting rejects A features that do not cover every origin. Use --horizon 1, 10 or 20 for supplementary graph analysis.

For later exploratory VAL-calibrated TEST risk evaluation, first extract and graph-evaluate TEST explicitly, then fit the monitor:

    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage extract --split test --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage graph --split test --horizon 5 --run --device cuda:0 --output-dir "$OUT"
    ../.venv/bin/python -m cmgm.scripts.d0b_reliability --stage risk --calibration val --evaluation test --run --output-dir "$OUT"

Append --bootstrap to the final command only to run 2,000 paired date-block draws. The risk stage fits only logistic monitors; D0B remains frozen.

For seed43/44 state-repeatability analysis, extract TRAIN and VAL for each audited checkpoint in separate output directories with explicit --seed, --checkpoint and --reference-result. Then run --stage repeatability --other-dir PATH43 --other-dir PATH44 --run on the seed42 output directory. Alignment uses TRAIN only; VAL is comparison only.

Target units are clipped decimal returns. population_metrics pools all origins × 24 commodities per horizon; RMSE is the square root of pooled MSE; Hit includes zero targets. The main horizon is 5 days. No full-data explanation or risk-monitor outcome is claimed by this implementation.
