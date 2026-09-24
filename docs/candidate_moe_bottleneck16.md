# Candidate MoE bottleneck16: fixed Full experiment

Only the two experts' hidden widths change from 64 to 16. The original Candidate
variant/classes remain unchanged. The subclass constructs the original experts
and router in their original RNG order, then replaces the two expert MLPs.
Thus all shared parameters, including router Linear/LN parameters, have identical
initialization at the same seed. The experiment never initializes from fitted weights.

| Component | Original | Bottleneck16 |
|---|---:|---:|
| Temporal expert | 8,320 | 2,128 |
| Interaction expert | 12,416 | 3,152 |
| Experts total | 20,736 | 5,280 |
| Router | 16,834 | 16,834 |
| Entire model at N=284 | 549,863 | 534,407 |

The temporal residual, ST concatenation/no-residual, candidate normalization/router,
raw representation mixture and shared head are unchanged. No MoE auxiliary loss,
warm-up, new router, clipping or hyperparameter search is introduced.

## Commands (repository root)

Use the same environment as the existing Candidate experiment:

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
../.venv/bin/python -m cmgm.scripts.d0b_candidate_moe_bottleneck16
```

Default is CPU **synthetic preflight only**, N=284, B=2. It does not open real price
files, real datasets, historical predictions or checkpoints. It saves parameter,
initialization and structural audit JSONs. `--seed` defaults to 42; there is no seed loop.

Optional server TRAIN/VAL preflight (no fitting):

```bash
../.venv/bin/python -m cmgm.scripts.d0b_candidate_moe_bottleneck16 \
  --data-preflight --device cuda --seed 42 \
  --data-audit experiments/baseline_comparison_deep_v3/20260922_143029/data_audit.json
```

Only an explicit `--run` starts formal training:

```bash
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_candidate_moe_bottleneck16 \
  --run --device cuda --seed 42 \
  --data-audit experiments/baseline_comparison_deep_v3/20260922_143029/data_audit.json \
  > bottleneck16_seed42.log 2>&1 < /dev/null &
```

Do not reuse a preflight output directory for fitting. Defaults create independent
timestamped directories. `--output PATH` is optional and must not contain a prior run.
There is no automatic CPU fallback. One persistent per-seed receipt prevents
accidental repeated formal runs across different output directories. An interrupted
fit requires review; there is no automatic retry or partial optimizer restart.

For a **completed** frozen checkpoint whose report needs re-evaluation:

```bash
../.venv/bin/python -m cmgm.scripts.d0b_candidate_moe_bottleneck16 \
  --evaluate-completed experiments/d0b_candidate_moe_bottleneck16/run_seed42_TIMESTAMP \
  --device cuda
```

This verifies source/data/checkpoint identities, evaluates TRAIN/VAL only, and
never trains. There is deliberately **no TEST option** in this entry point.

## Data boundary and protocol

The existing raw loader receives optional CSV `skiprows`; its old defaults do not
change. The restricted entry reads **date metadata only** across the existing full
calendar to recover the original chronological boundaries, then skips all numeric
rows after the original validation cutoff at the CSV parser. No TEST Dataset,
features, targets or metrics are constructed. TRAIN/VAL values go through the
original `main_ablation.build_data`, feature builder, TRAIN statistics and
`MarketSequenceDataset`; no second standardization or new target construction.

Both TRAIN/VAL numeric fingerprints, origin counts and commodity order must equal
the supplied prior passing data audit. If future-dependent historical bfill or
column selection prevents exact reproduction, execution stops; it never silently
loads TEST to work around that mismatch. The default audit is an existing dataset
identity artifact, **not** an original-model performance reference. Server real-data
reproduction remains a preflight requirement, not a claim from synthetic tests.

Training uses `cmgm.training.train.train`: Adam 1e-4, WD1e-5, batch64, max200,
patience10, ReduceLROnPlateau factor.5/patience5, chronological TRAIN drop_last=True,
full evaluation drop_last=False, equal sum of four Huber(.02), native switching KL.
Native KL schedule: beta=5e-4*clamp((epoch-1)/19,0,1). Epoch1 is zero, epoch20 is
5e-4. Checkpoint/early stopping/scheduler monitor the **prediction-only batch-mean
four-horizon validation loss**. VAL5 is only logged. Routing diagnostics are detached.

## Artifacts

Results:
`experiments/d0b_candidate_moe_bottleneck16/{synthetic_preflight,data_preflight,run}_seedSEED_TIMESTAMP/`

Formal checkpoints:
`checkpoints/candidate_moe_bottleneck16/seedSEED/run_seedSEED_TIMESTAMP/switching_latent_balanced_candidate_2expert_moe_bottleneck16_best.pt`

No historical checkpoint is overwritten. Formal output includes:

- `config.json`: exact variant, width, seed, objective, selection, loader/optimizer protocol.
- `source_hashes.json`: git commit/status, relevant source hashes, package versions.
- `data_audit.json`: TRAIN/VAL fingerprints, split dates/counts, commodity mapping,
  historical audit provenance and restricted-reader scope.
- `initialization_audit.json`: all shared-parameter differences, mismatches and counts.
- `structural_sanity.json`: shapes, raw mixture/head identities, .5/.5 routing,
  finite gradients, permutation/single-sample consistency, temporal-prefix causality.
- `training_history.json`, `routing_history.json`: saved every epoch; losses, LR,
  native KL beta, VAL5 diagnostics, detached TRAIN/VAL routing diagnostics.
- `best_checkpoint_metadata.json`: frozen checkpoint SHA256, best epoch/objective,
  completion flag and synchronized training seconds (excludes preflight/evaluation).
- `train_val_predictions.npz`, `train_val_metrics.json`: full pooled metrics per horizon.
- `routing_diagnostics.json`: mean/std/quantiles/entropy/occupancy for TRAIN and VAL.
- `expert_diagnostics.json`: two expert L2 norms, fused norm, mean absolute
  disagreement and cosine distribution for TRAIN and VAL.
- `results.json`, `REPORT.md`: consolidated status, provenance and TRAIN/VAL reporting.

The original-model performance comparison is marked PENDING until dataset, seed,
protocol, checkpoint identity and metric definitions are audited. Historical TEST
numbers are never substituted and the original model is never retrained here.

## Future controls and interpretation

Only Full is enabled. The centralized model factory leaves a single integration
point for the existing shared-G0 transition and post-normalization null micro-readout
interventions. No formal-ablation queue is changed and neither control is run.
Both must retain native KL (and recurrence for no-microstate), use this bottleneck16
Full as reference, and compare only matched seed/window/definitions to width64.

Preserved prediction with more stable mechanism effects would support further study;
near-zero deltas would not demonstrate restored contributions. A worse Full with
larger deltas is not success. Full alone cannot answer the mechanism hypothesis.
Single-run results do not establish statistical significance.


## Restricted-calendar repair

Raw CSV date rows may include dates removed by the original pivot (all quotes
missing), so date-only intersections cannot define the split sizes. The entry
uses the already-audited TRAIN/VAL timeline lengths and incrementally reads only
the minimum date prefix needed to reproduce those aligned rows. It never changes
the audited split sizes or relaxes the final numeric SHA256 checks.

The historical full-calendar pipeline also established a fixed node schema.
When prefix-only parsing exposes extra columns, the existing formal TRAIN input
cache (`experiments/formal_baseline_benchmark_v2/20260910_131615/arrays/train_x.npy`)
is used to recover that same schema/order from TRAIN price channels. Matching is
only a schema proposal; both complete TRAIN and VAL raw/features fingerprints
must subsequently equal the original audit exactly. Missing/ambiguous cache
identity causes STOP. No TEST values or model performance are used for schema
recovery. This is not a new feature-selection experiment.
