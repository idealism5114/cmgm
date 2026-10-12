# D0B consensus–discrepancy fusion

This experiment adds two separate fusion variants on the existing
`switching_latent_balanced_readout` backbone:

- `switching_latent_balanced_mean_fusion`: parameter-free mean-fusion control.
- `switching_latent_balanced_consensus_discrepancy_fusion`: mean plus a learned
  discrepancy residual.

The original sigmoid gate remains the default of
`switching_latent_balanced_readout`. The existing
`switching_latent_balanced_horizon_tensor_fusion` remains registered and
unchanged. The variants use the same spatial and switching-latent temporal
branches, `gcn_proj`, `lstm_proj`, and prediction head. The temporal branch is
still the native balanced-readout path, so its native Switch KL and epoch
schedule remain active.

## Fusion equations

For the existing projected states `s = gcn_proj(h_spatial)` and
`t = lstm_proj(h_temporal)`:

```text
c = 0.5 * (s + t)
d = s - t
```

The mean control predicts from `c` directly. The residual variant computes:

```text
context = A_c(c) + b
r       = tanh(context + A_d(d)) - tanh(context)
delta   = B(r)
fused   = c + delta
```

with `A_c, A_d: 64→16` without bias, `b: 16`, and `B: 16→64` without bias.
The fusion layer has 3,088 trainable parameters. `B.weight` and `b` start at
zero, so the full residual variant initially produces exactly the same fused
state as the mean control. For any equal projected inputs `s == t`, `d == 0`
and the correction is exactly zero, including after training.

“Consensus” and “discrepancy” refer only to arithmetic mean and difference
representations; this implementation does not claim statistical independence,
orthogonality, or semantic disentanglement.

## Parameter and initialization checks

At the repository's `N=284` configuration, the synthetic audit confirms:

| Variant | Parameters | Change from gated D0B |
|---|---:|---:|
| `switching_latent_balanced_readout` | 520,549 | — |
| `switching_latent_balanced_mean_fusion` | 512,293 | −8,256 |
| `switching_latent_balanced_consensus_discrepancy_fusion` | 515,381 | −5,168 |

The mean control removes the 8,256-parameter gate and adds no trainable fusion
parameters. The residual variant replaces the gate with 3,088 parameters.
With the same seed, all parameters shared with the original model match exactly.
The new residual correction is initialized to zero, and the gate is absent from
both new variants' active module/state-dict surfaces. The old gated model's
state-dict structure is unchanged.

The read-only
`model.consensus_discrepancy_diagnostic(x, disable_correction=True)` option
evaluates a frozen residual-fusion checkpoint after replacing its fused
representation with `c`. It reports per-sample L2 norms for `c`, `d`, and
`delta`, plus the mean/median/maximum of per-sample
`||delta||₂ / (||c||₂ + 1e-8)`. This frozen intervention measures dependence of
the fitted model; it is not equivalent to independently training the mean
fusion variant.

## Training and output

The opt-in runner uses the existing `main_ablation.build_data` pipeline and
`training.train` loop. It keeps chronological splits, 20-step/21-feature inputs,
horizons `[1,5,10,20]`, the sum of four Huber losses (`delta=.02`), native
Switch KL, and the original batch-mean four-horizon validation selection,
scheduler, and early stopping. The full-population evaluator uses
`drop_last=False`; metrics are pooled over origins and commodities. `Hit` is
stored as a fraction in `[0,1]`. Every invocation writes a unique run directory
under `experiments/d0b_consensus_discrepancy_fusion/` and a separate best
checkpoint under `checkpoints/d0b_consensus_discrepancy_fusion/`.

Default invocation only runs synthetic preflight. No formal run was started for
this implementation. For a future controlled comparison, run one command at a
time on the GPU, using the same seed and protocol:

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
../.venv/bin/python -m cmgm.scripts.d0b_consensus_discrepancy_fusion \
  --preflight
```

Original gated baseline:

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_consensus_discrepancy_fusion \
  --run --variant switching_latent_balanced_readout --seed 42 --device cuda \
  --evaluate-test > consensus_gate_seed42.log 2>&1 < /dev/null &
```

Independently trained mean-fusion control:

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_consensus_discrepancy_fusion \
  --run --variant switching_latent_balanced_mean_fusion --seed 42 --device cuda \
  --evaluate-test > consensus_mean_seed42.log 2>&1 < /dev/null &
```

Full consensus–discrepancy residual fusion:

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_consensus_discrepancy_fusion \
  --run --variant switching_latent_balanced_consensus_discrepancy_fusion \
  --seed 42 --device cuda --evaluate-test \
  > consensus_discrepancy_seed42.log 2>&1 < /dev/null &
```

`--evaluate-test` is a separate explicit opt-in and runs only after the best
validation-selected checkpoint has been restored. Without it, only TRAIN/VAL
metrics are written. The run metadata records split sizes/fingerprints,
commodity order, horizons, metric definitions, seed, source hashes, git
revision, best epoch, selection loss, and runtime. No hyperparameter search or
additional loss is used.

No prediction improvement or methodological originality is claimed; these
variants require future formal training and evaluation.
