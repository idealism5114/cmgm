# Final Candidate-MoE main-innovation ablation

The existing entry point remains `cmgm.scripts.formal_main_innovation_ablation`.
Only the original model/audit/runner/report files are adapted; there is no new
ablation runner. Historical directories and checkpoints remain untouched.

Full is now `switching_latent_balanced_candidate_2expert_moe`. The main table is
exactly 3 spatial + 5 temporal + 2 fusion controls + Full (11 rows). Branch-removal,
old adaptive-gate/equal-fusion and supplementary controls are excluded.

From the repository root, using the project environment:

```bash
python -m cmgm.scripts.formal_main_innovation_ablation --cpu-check
```

Once this Candidate-based experiment is prepared, resume it:

```bash
python -m cmgm.scripts.formal_main_innovation_ablation --resume latest --cpu-check
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.formal_main_innovation_ablation --resume latest --run >> formal_main_ablation.log 2>&1 &
tail -f formal_main_ablation.log
```

Only the user starts GPU training. Default preparation creates no optimizer.
`latest` selects this protocol, never an old Adaptive-Gate experiment. A new
experiment timestamp isolates all new checkpoints in the existing checkpoint
root. No historical file is overwritten.

Full Candidate and the completed Global Static control are audited for exact
configuration, seed, data fingerprints, initialization, sanity, completed history,
checkpoint/report hashes and native strict state loading. Every TRAIN/VAL/TEST
metric is freshly evaluated and compared with the audited artifact. A mismatch
stops. Their old checkpoint format has no `training_complete` flag, so completion
is verified using the completed report and matching full early-stopped history;
report-only `train_time` is recorded separately. Neither checkpoint is modified.
Core expert/temporal implementation hashes must match; historical source drift is
recorded together with native forward equivalence and fresh reproduction.

The 8 internal interventions retain the final Candidate MoE and require new runs;
old Adaptive-Gate results cannot be reused by name. The ninth new run removes
experts/router and uses only `Linear(128,64)([s||t])` before the original head.
The routing control uses exactly the original two experts and global trainable
zero-initialized logits. Eligible existing Global Static results are reused.

All shared modules are created as complete Candidate Full before intervention.
Static experts are moved without RNG consumption. Init comparisons require zero
error; simple fusion and ordinary MixHop replacement modules are constructed last.
No changes are made to the training loop: seed42, Adam1e-4, WD1e-5, batch64,
200 epochs/patience10, scheduler factor.5/patience5, original Switch KL (uniform
routing is mathematically zero), and sum of four Huber(.02) prediction losses.
Checkpoint selection/early stopping/scheduler use prediction-only VAL Huber.
No MoE auxiliary losses, routing warmup, clipping, search or performance retries.

Every configuration passes shape/finite, batch/single-sample, prefix10 causality,
mechanism call and shared-initialization checks before formal fitting. Completed
checkpoints resume without retraining; invalid or incomplete runs stop for review.

`FINAL_REPORT.md` (also `REPORT.md`) contains the prescribed13 sections, 11-row
main table, all-horizon metrics, active/instantiated parameter counts, regime,
readout and routing diagnostics, provenance, and11 required answers. Pending
results remain blank. Delta is ablation−Full Candidate MoE. All unfavorable
results remain visible. Near ties (<0.1% relative MAE delta) are descriptive only.
The whole-MoE control also changes capacity; it is not parameter-matched. Single
seed evidence does not establish statistical significance. STOP after this suite.
