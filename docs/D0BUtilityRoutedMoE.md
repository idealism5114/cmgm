# D0B-UtilityRoutedMoE

Variant: `switching_latent_balanced_utility_routed_moe`.
Implementation base: main `add9af16d06f9b8d7a174d9618958b85b647f89f` (MOE),
verified against GitHub main with a read-only `git ls-remote` on 2026-09-21.

Only the fusion/prediction interface changes. Native full-data spatial and
temporal branches, Balanced Readout, projections and Switch KL remain intact.
The original D0B, three-expert and candidate two-expert variants are preserved.

## Predictive experts and router

Temporal expert: residual Linear64/ReLU/Dropout(.1)/Linear64 on `t`.
Interaction expert: Linear128→64/ReLU/Dropout(.1)/Linear64 on `[s,t]`.
The native `head` predicts from Temporal; `interaction_head=deepcopy(head)`
predicts from Interaction. They initialize identically but have independent
parameters. Both are Linear64→64/ReLU/Dropout(.3)/Linear64→96.

Router-only LayerNorm produces `u_T,u_ST`. Router input is
`[u_T,u_ST,abs(u_T-u_ST),u_T*u_ST,p,abs(p-prior),entropy(p)]`, shape B×263.
Regime posterior/prior are the last observed temporal states and are detached.
Router: Linear263→64/ReLU/Linear64→2/softmax; final weight/bias initialize to zero.
The final prediction is the dense weighted sum of two B×4×24 **predictions**.
There is no shared head applied after representation mixing.

All native shared modules are constructed first. Only the new variant removes
gate_fc; its experts/router are added afterwards. Deepcopy does not consume RNG.
Full N=284: D0B 520,549 parameters, Utility MoE 560,711. Shared elements 512,293,
new elements 48,418, net increase 40,162 after removing the old 8,256-element gate.

## TRAIN-only utility supervision

Per-origin expert error is unreduced Huber(.02), mean over24 commodities and sum
over all4 horizons. Set `a=(loss_T-loss_ST)/(loss_T+loss_ST+1e-8)` and
`q_ST=(1+a)/2`, `q_T=1-q_ST`. Both q and mean expert error scale are detached.
Route loss is `scale * mean KL(q || pi)`, with epsilon1e-8 and no extra coefficient.

**Detaching q alone is insufficient to make this Router-only.** Auxiliary loss
re-evaluates the deterministic router on detached candidates/context. Its pi
values match the primary forward exactly, but its gradients reach only Router
parameters, including its LayerNorms. The main prediction loss retains its full
gradient graph through experts/heads/router/shared branches. Tests verify this
with a nonzero router final layer as well as initialization.

Training uses only prediction + unchanged Switch KL + auto-scaled utility loss.
VAL checkpoint/early-stop/scheduler remain prediction-only four-horizon Huber.
No validation routing loss is trained or added to the monitor. Utility gradient
audits also use TRAIN labels. No warm-up, load balance, entropy/diversity objective,
clipping, tuning, extra seed or optimizer group.

Fixed protocol: seed42, Adam1e-4, WD1e-5, batch64, max200, patience10, native
ReduceLROnPlateau; TRAIN shuffle=False/drop_last=True.

## Run

From the repository root with the existing environment activated, verify only:

```bash
python -m cmgm.scripts.d0b_utility_routed_moe --resume latest
```

If there is no preparation artifact yet, omit `--resume latest` the first time.
CPU-only verification is available using `--cpu-check`. Formal training requires
CUDA and never falls back to CPU. Launch the one formal run yourself:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.d0b_utility_routed_moe --resume latest --run >> d0b_utility_routed_moe.log 2>&1 &
```

```bash
tail -f d0b_utility_routed_moe.log
```

The runner protects sources/data/checkpoints with hashes, locks the experiment
directory, refuses a duplicate run, and reuses a completed matching checkpoint.
Interrupted fitting without a completed checkpoint requires review, not an
automatic repeat. History is written each epoch. No existing controls retrain.

## Frozen-checkpoint diagnostics

Dynamic, standalone experts and Fixed TRAIN-Mean predictions use the same best
checkpoint. Fixed weights come from all TRAIN origins in eval mode without
labels; no VAL/TEST tuning. Formal predictions are persisted before ex-post
alignment/oracle analysis. Primary alignment is four-horizon Huber advantage;
5d per-origin MAE alignment is supplementary. Rank thirds use stable chronological
ties. Correlations of constant inputs are null with an explanation.

Oracle selects the lower four-horizon per-origin Huber expert using TEST labels.
Its 5d metrics are therefore not a 5d-specific oracle bound. It is labeled
nondeployable and excluded from the formal comparison. Regime groups and entropy
correlations are descriptive, not causal. Neither routing variance nor occupancy
alone establishes Router success.

Output: `experiments/d0b_utility_routed_moe/<timestamp>`.
Checkpoint: `checkpoints/switching_latent_balanced_utility_routed_moe_best.pt`.
Until the user launches training, results remain explicitly PENDING.
After the one run, unified evaluation, fixed-mixture/oracle controls and report:
STOP. Accept all results without retuning or rerunning.
