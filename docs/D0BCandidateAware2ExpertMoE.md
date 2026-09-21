# D0B-CandidateAware2ExpertMoE

Variant: `switching_latent_balanced_candidate_2expert_moe`.

This independent variant changes only native D0B fusion. Native spatial,
temporal, switching, Balanced Readout, branch projections and shared prediction
head keep their definitions and shared initialization. Shared weights train
end-to-end. The original gate and completed three-expert model remain intact.

- T expert: `t + Linear64(ReLU/Dropout(.1)(Linear64(t)))`.
- ST expert: `Linear64(ReLU/Dropout(.1)(Linear128([s,t]))))`, no residual.
- Router-only LayerNorm normalizes each candidate separately.
- Router input: `[u_T,u_ST,abs(u_T-u_ST),u_T*u_ST]`, shape B×256.
- Router: Linear256→64, ReLU, Linear64→2, softmax. Final weight and bias are
  zero-initialized, giving exact initial 50/50 routing.
- Dense fusion weights **raw** candidates; only one original shared head
  produces B×4×24 predictions.

There is **no routing warm-up and no MoE auxiliary loss**. The initial router
final layer receives gradient. Its preceding layer and LayerNorm initially have
zero gradients by the chain rule through the zero final weight; this is expected.

Full N=284 parameter counts: native D0B 520,549; new model 549,863. There are
512,293 shared parameter elements and 37,570 new elements. The 8,256-element
obsolete gate is removed only from the new variant. Net increase: 29,314.

Training remains seed42, Adam 1e-4, WD 1e-5, batch64, max200, patience10,
native ReduceLROnPlateau. Loss: sum four Huber(.02) plus unchanged Switch KL.
Checkpoint, scheduler and early stopping use multi-horizon VAL Huber; VAL5 is
secondary logging. TRAIN stays chronological, shuffle=False/drop_last=True.

## Execution

Activate the existing project environment and run from the repository root.
Verification only (omit `--resume latest` if no preparation artifact exists):

```bash
python -m cmgm.scripts.d0b_candidate_2expert_moe --resume latest
```

CPU verification is available with `--cpu-check`; formal training requires CUDA.
Launch the single formal GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.d0b_candidate_2expert_moe --resume latest --run >> d0b_candidate_2expert_moe.log 2>&1 &
```

```bash
tail -f d0b_candidate_2expert_moe.log
```

The runner checks data/source fingerprints and protects against concurrent or
duplicate runs. It reuses an already completed matching checkpoint. An interrupted
fit without a completed checkpoint requires review instead of silently repeating.
History and routing summaries are saved after every epoch.

## Evaluation boundary

After restoring the formal best checkpoint, the runner saves formal predictions
before generating expert-only predictions. Ex-post predictions use exactly the
trained shared head in eval mode. Per-origin 5d error averages all 24 commodities.
Advantage is `ell_T - ell_ST`; positive means interaction has lower error.

The report includes Pearson/Spearman alignment, stable rank thirds by pi_ST,
representation differences, and an explicitly labeled TEST-label oracle.
Constant correlation inputs yield null with an explanation, not NaN or a
fabricated zero. Rank ties retain chronological origin order; groups created
from tied routing do not prove sample dependence. All standard deviations are
population standard deviations (`ddof=0`).

Oracle is analysis only, not deployable, and never appears in the formal table.
It bounds a selector restricted to these two expert candidates; it does not bound
an arbitrary nonlinear mixture followed by the shared head. Model parameters and
checkpoint hash must remain unchanged throughout post-fit analysis.

Artifacts: `experiments/d0b_candidate_2expert_moe/<timestamp>`.
Checkpoint: `checkpoints/switching_latent_balanced_candidate_2expert_moe_best.pt`.
The four existing controls are read from verified artifacts and never retrained.
Until training/evaluation complete, the report marks all new results PENDING.

One fixed seed42 run. Accept any result. No tuning, added experts or subsequent
experiments. STOP after reporting.
