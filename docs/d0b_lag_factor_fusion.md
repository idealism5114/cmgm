# D0B Lag Factor Fusion

This is one fixed, post-hoc seed-42 experiment. It keeps D0B's spatial graph
branch, switching latent temporal branch, balanced readout, branch projections,
shared prediction head, four-horizon Huber objective, native Switch KL and
prediction-only validation selection unchanged. The only new computation is
the prediction fusion path.

For each sample, the spatial branch returns its global vector and the graph
propagation output at all nodes. The 24 commodity nodes are selected using the
dataset mapping. The temporal branch runs once and returns its existing 64D
balanced readout, 20 × 128 long-memory history and the live final regime
probability vector. The latter comes from the same regime computation as the
latent recurrence and retains its prediction gradient.

The two projected global vectors are concatenated, mapped by
`Linear(128,64) → GELU`, and passed through the original shared head. The
24 node vectors are mapped by one shared `Linear(64,4)` to commodity factor
exposures; each factor is centered across commodities. The history is averaged
over its latest 1, 3, 5, 10 and 20 states. A shared `Linear(3,20)` maps the
final regime probability to four sets of five softmax lag weights. Each
weighted history, the projected temporal vector and a 16D horizon embedding
are mapped by `Linear(208,4)` to factor shocks. The correction is
`sum_r exposure[b,c,r] × shock[b,h,r] / sqrt(4)`. The shock layer has nonzero
normal initialization (std 0.005), with zero bias; there is no zero gate or
trainable global correction multiplier. Factor correction is centered over the
commodity dimension, so its commodity-mean value is zero for every horizon.

At N=284 and 24 commodity targets, the new fusion module has 9,496
parameters: global projection 8,256; exposure 260; regime-to-lag 80; horizon
embedding 64; shock 836. The previous gate had 8,256 parameters, so the
net increase over the actual D0B configuration is 1,240. The runner records
the measured full-model count rather than assuming any historical count.

`python -m cmgm.scripts.d0b_lag_factor_fusion` runs only a synthetic preflight.
`python -m cmgm.scripts.d0b_lag_factor_fusion --run --seed 42 --device cuda:0`
runs the single authorized formal fit, selects by four-horizon VAL Huber, then
evaluates the frozen checkpoint on TEST once. Results go to a new timestamped
`experiments/d0b_lag_factor_fusion/run_seed42_*` directory and its checkpoint
to `checkpoints/d0b_lag_factor_fusion/run_seed42_*/best.pt`.

Frozen inference interventions remove the correction, rotate exposure-to-
commodity assignments, or replace the five learned lag weights with uniform
weights. They diagnose reliance on these paths. They are not retrained
ablations. The TEST threshold was selected after earlier project TEST use;
passing it would not make this a first blind test.

Motivation came from the conditional factor idea in
[RDCFM](https://link.springer.com/article/10.1007/s00291-024-00771-1),
time-varying lead-lag modeling in
[DeltaLag](https://arxiv.org/html/2511.00390v1), and conditional state
modeling in [RC-Mamba](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1903497/full).
Those works do not establish a CMGM accuracy improvement for this design.
