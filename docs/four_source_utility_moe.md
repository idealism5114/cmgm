# Four-source predictive utility MoE

Variant: `switching_latent_balanced_4source_utility_moe`.

This is one combined exploratory design, changing sources, prediction-level fusion and a routing auxiliary objective together. It cannot establish a separate causal effect of expert count or utility supervision, or claim restoration of backbone mechanism contributions.

## Architecture and gradients

Sources, in fixed order: LongMemory, Microstate, SpatialGraph, Joint.
`long=LN_H(W_H H_last)`, `micro=LN_Z(W_Z Z_last)`, `spatial=original spatial branch output`; Joint concatenates all three. `forward_live_components` runs the original market encoder, causal long-memory transformer, K=3 Markov filter and recurrence once and returns live normalized tensors. Detached diagnostic caches are never expert inputs.

Each independent expert is `Linear(input,64)→ReLU→Dropout(.3)→Linear(64,96)` with input widths `[64,64,64,192]`, reshaped to `(B,4,24)`. First three are separate deepcopies of the untrained native head; Joint has a new first layer and a copied final layer. No weight sharing.

Router: three independent LN64 on the sources, concatenate to192, `Linear(192,64)→ReLU→Linear(64,4)→softmax`. No dropout. Last layer weight/bias zero, initial probabilities exactly1/4. All experts execute. `prediction=sum_k pi_k*expert_prediction_k`; there is no head after this mixture.

New variant deletes merged `state_readout`, old gate, branch projections and old shared head. Thus it changes the temporal readout interface; it does not claim to preserve the complete old temporal readout. Shared encoders, long/micro projections and their norms keep exact seed-matched initialization. All old variants remain unchanged.

For TRAIN, using the same forward predictions:

```
ell[b,k] = sum_h mean_c Huber(.02)(expert[b,k,h,c],target[b,h,c])
q = softmax(-detach(ell)/(mean_k(detach(ell))+1e-8)/1.0)
scale = detach(mean(ell))
pi_aux = same_router(detach(long),detach(micro),detach(spatial))
L_route = .1 * scale * mean_b KL(q || pi_aux)
L_total = L_pred + native_switch_KL + L_route
```

Only Router receives auxiliary gradients, including its LayerNorm parameters. Normal prediction gradients still reach all sources, Router and experts. No expert auxiliary prediction loss, balance/entropy loss, warmup or clipping. q is a relative-error soft target, not an oracle or theoretically optimal mixing weights. At initial zero router final layer, some upstream router gradients are correctly zero; gradient isolation is also tested with a nonzero final layer.

## Training and data

Same audited TRAIN/VAL-only pipeline as Bottleneck16; exact data fingerprints and commodity mapping required. No TEST dataset construction or evaluation. Missing historical schema cache/audit or fingerprint mismatch stops the run.

Seed42, T20/F21, horizons1/5/10/20, Adam1e-4/WD1e-5, batch64, max200, patience10, schedulerReduceLROnPlateau(.5,5), shuffleFalse, train drop_lastTrue, full eval drop_lastFalse. Four Huber(.02) means summed, native beta_max.0005 warmup20 (`clamp((epoch-1)/19)`). Original `cmgm.training.train` is reused. Best checkpoint/early stopping/scheduler use prediction-only VAL Huber arithmetic mean across batches; VAL5 only logs. No utility optimization or q supervision on VAL.

## Commands

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
# Default synthetic CPU preflight only: no real data, no fitting.
../.venv/bin/python -m cmgm.scripts.d0b_four_source_utility_moe

# Optional real TRAIN/VAL preflight, still no fitting.
../.venv/bin/python -m cmgm.scripts.d0b_four_source_utility_moe \
  --data-preflight --device cuda --seed 42

# Explicit single formal run, launched by user:
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_four_source_utility_moe \
  --run --device cuda --seed 42 \
  > four_source_utility_seed42.log 2>&1 < /dev/null &

tail -f four_source_utility_seed42.log
```

Outputs: `experiments/d0b_four_source_utility_moe/run_seed42_<timestamp>/`.
Checkpoint: `checkpoints/four_source_utility_moe/seed42/run_seed42_<timestamp>/switching_latent_balanced_4source_utility_moe_best.pt`.

New output directories only; per-seed receipt and file lock prevent duplicate formal runs. Interrupted runs require review, no automatic retry. `--evaluate-completed <run-directory>` restores a completed checkpoint strictly, verifies source/data/checkpoint hashes, and reruns only frozen TRAIN/VAL evaluation. No test option, sweep or ablation queue.

## Artifacts

- config.json, source_hashes.json (git commit/status and code hashes), data_audit.json.
- initialization_audit.json: shared parameter equality, removed parameters, expert independence/counts.
- structural_sanity.json: shapes, finite values, utility-only gradient isolation, live components, batch/single/prefix causality, restore.
- reference_provenance.json: audited original Candidate64 identity/protocol/data/commodity order/checkpoint; otherwise PENDING, no TEST-number fallback or retraining.
- training_history.json/routing_history.json: train prediction/native raw and weighted KL/raw utilityKL/scale/weighted utility/total; pi/q mean/std/entropy; prediction-only VAL, secondary VAL5, LR, switch beta. VAL has no auxiliary objective.
- best_checkpoint_metadata.json: best epoch, selection objective, SHA256, training seconds.
- train_val_metrics.json: pooled mixed-prediction MAE/MSE/RMSE/Hit for four horizons.
- diagnostics.json: each expert metrics, routing distribution, errors, pairwise prediction disagreement; frozen full-TRAIN mean weights and fixed-vs-dynamic metrics/deltas.
- train_val_predictions.npz: mixture, experts, routing, targets, frozen-mean predictions and per-sample changes.
- results.json and REPORT.md; Hit displayed as percentage, raw JSON Hit is a fraction.

TRAIN mean routing uses eval probabilities only, not labels. The frozen vector is applied to the same checkpoint's TRAIN and VAL predictions. Labels serve post-hoc scoring only; no Router update or lambda/tau selection. Routing std or occupancy alone cannot establish success or market-state specialization.
