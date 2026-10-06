# D0B-HorizonTensorFusion

This is a fixed-configuration experiment variant of the maintained
`switching_latent_balanced_readout` model. It tests whether an explicit
low-rank bilinear interaction with a separate kernel for each forecast horizon
is useful in place of the original shared feature-wise gate. It is a hypothesis
under evaluation; tensor fusion and low-rank factorization are established
operator families, not claims of methodological originality.

## Fusion definition

The existing D0B spatial and switching-latent temporal branches produce
`h_spatial, h_temporal ∈ R^(B×64)`. Their existing projections are retained:

```text
s_raw = gcn_proj(h_spatial)      (B,64)
t_raw = lstm_proj(h_temporal)    (B,64)
s = LN_s(s_raw); t = LN_t(t_raw)
u = U_s(s); v = U_t(t)           (B,8)
M[b,a,c] = u[b,a] v[b,c]        (B,8,8)
```

The full 8×8 outer product is retained; no 64×64 product is formed. For the
fixed horizon order `[1,5,10,20]`:

```text
G_h = G0 + E[h,0] G1 + E[h,1] G2
interaction[b,h,d] = sum(a,c) G_h[d,a,c] M[b,a,c]
linear = A_s(s) + A_t(t)
fused[b,h] = GELU(linear[b] + interaction[b,h] + b)
```

The model applies the original shared prediction-head body to each of the four
`(B,64)` fused states. The existing final layer is viewed as four horizon-major
groups of commodity rows; horizon `h` uses only its own `Nc` rows. Output shape
is `(B,4,Nc)`. No new horizon state readout, gate, router, auxiliary head, or
auxiliary loss is introduced.

The gate is instantiated in its original construction position and removed
only after all shared D0B modules have been created. This preserves the shared
initialization stream. The new module then adds 21,832 parameters and replaces
the 8,256-parameter gate, for a net increase of 13,576. At the N=284 project
configuration the audited counts are 520,549 for the original D0B and 534,125
for HorizonTensorFusion.

## Training and evaluation entry

The new entry is synthetic-preflight-only by default. It does not load the
dataset or create an optimizer unless `--run` is supplied. On `--run`, it uses
the existing `main_ablation.build_data` pipeline, original Adam/Huber/Switch-KL
training loop and batch-mean four-horizon validation objective. It records
TRAIN/VAL metrics using full loaders; it does not evaluate TEST or use TEST for
selection. TRAIN/VAL `Hit` values in JSON are fractions in `[0,1]` as emitted by
the standard metric function.

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
../.venv/bin/python -m cmgm.scripts.d0b_horizon_tensor_fusion --preflight
```

To run one future formal fit manually (not run during implementation):

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_horizon_tensor_fusion \
  --run --device cuda --seed 42 \
  > horizon_tensor_fusion_seed42.log 2>&1 < /dev/null &
```

Each invocation creates a unique `experiments/d0b_horizon_tensor_fusion/`
run directory and a matching unique checkpoint directory under
`checkpoints/d0b_horizon_tensor_fusion/`; existing checkpoints are not
overwritten. The output contains configuration/protocol and source hashes,
TRAIN/VAL data fingerprints and commodity order, training history and best
epoch, parameter counts, full-population TRAIN/VAL MAE/MSE/RMSE/Hit for all
four horizons, interaction and first-order term norms, horizon-kernel
differences, and frozen inference interventions with interaction disabled or
`E=0`. Those interventions describe dependence of the fitted checkpoint;
they do not replace retrained structural ablations.

## Future comparison set (not run here)

1. Original `switching_latent_balanced_readout`.
2. Parameter-count-nearby concatenation MLP.
3. Tensor fusion with one shared interaction kernel across horizons.
4. Full horizon-conditioned tensor fusion.
5. Same fitted architecture with the interaction term structurally removed and
   retrained.

The current work performs only synthetic validation. No training run or TEST
evaluation has been performed, so predictive performance has not been
validated and no improvement is claimed.

## Implemented checks

`tests/test_horizon_tensor_fusion.py` checks the independent flattened
matrix-product formula against the interaction tensor, horizon-specific final
row selection, cycle conditioning, zero compressed factors, shared
initialization, output shape, finite values, batch permutation, gradient
connectivity, unchanged original gated forward formula, native Switch-KL
training inclusion and warmup, and strict checkpoint save/restore behavior.
