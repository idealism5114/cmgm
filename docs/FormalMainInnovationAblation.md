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
error; simple fusion and standard GCN replacement modules are constructed last.
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


The EdgeAttnMixHop control now replaces the complete operator with two single-hop
GCN layers, not ordinary MixHop. With A_hat=A+I and D the row-degree matrix, each
layer computes Linear(D^(-1/2) A_hat D^(-1/2) H). Learned A retains its direction;
it is not symmetrized. ReLU remains between the two layers; the existing final
LayerNorm/type pooling/MoE remain. There is no attention, multi-hop sum or beta
recurrence. The interpretation is the complete EdgeAttnMixHop subsystem versus
standard graph propagation, rather than an isolated attention-only effect.

For a completed previous Candidate suite, the existing runner supports explicit
`--revise-edge-gcn-from <old experiment directory>` (without `--resume`). It audits
all completed artifacts, retains ten unchanged rows and excludes only the old
ordinary-MixHop result. A new timestamp isolates the revised checkpoint/report.
Subsequent `--resume latest --run` trains only the missing GCN control. No old
artifact is overwritten and no other completed ablation is retrained.

## Global-mixture temporal mechanism verification (post-hoc)

The **same runner** now accepts `--mode global-temporal`. The default Candidate
main-ablation mode remains available. This targeted mode constructs the native
`switching_latent_balanced_candidate_2expert_global_mixture` directly, then applies
only uniform regime routing, shared regime transition, or post-normalization
microstate readout nulling. It never constructs a temporary Candidate router;
initial parameters **and RNG state** match native Global construction. Each new
control starts its own trainable global logits at zero. The training function is
unchanged, including prediction-only multi-horizon VAL selection.

The completed native Global Full is strictly audited, loaded and freshly evaluated
without retraining. Only the three missing controls can train. All initial checks
must pass. Completed runs are reused; interrupted/invalid runs STOP for review.
Old Adaptive-Gate and Dynamic-Candidate results are read-only explanatory sources,
not reused as Global ablation results. No new Python runner or model file is added.

From the repository root, preflight only:

```bash
../.venv/bin/python -m cmgm.scripts.formal_main_innovation_ablation --mode global-temporal --cpu-check
```

After preflight, launch the **three** formal GPU runs (Full is reused):

```bash
nohup ../.venv/bin/python -u -m cmgm.scripts.formal_main_innovation_ablation --mode global-temporal --resume latest --run > global_temporal_ablation.log 2>&1 < /dev/null &
```

Output and checkpoint roots are the existing directories with a `global_temporal/`
subdirectory and independent timestamp. This prevents mixing checkpoints with
Candidate-mode controls that have the same display name. `--old-gate-results` and
`--dynamic-results` select audited historical reports; they never select an epoch
or retrain a model. Historical sources and the Full checkpoint are hash protected.
The report has sixteen sections, a four-row main table, four-horizon metrics,
cross-architecture deltas, global-weight compensation and regime diagnostics.
It explicitly labels the study post-hoc and single-seed; no result is treated as
proof or statistical significance. The inherited 0.1% relative-MAE near-tie
convention is descriptive, with exact deltas retained. No further experiment
starts after these three controls complete.

## T+S branch-specific expert mechanism study

`--mode ts-global-expert` uses the unique native variant
`switching_latent_balanced_ts_2expert_global_mixture`. The existing
`global_mixture_fusion.py` adds a Spatial residual expert symmetric to the reused
Temporal residual expert. Each has 8,320 parameters and sees only its own 64D
projected branch. Two zero-initialized global logits mix their representations
before the unchanged shared head. No ST expert, Candidate Router or auxiliary
MoE objective is present.

This mode runs **one new Full followed by exactly four controls**:
uniform regime routing, shared transitions, effective microstate readout nulling,
and ordinary `MixHopPropagation(64,64,K=2,beta=.05)` ×2 on the same learned graph.
The earlier Candidate-mode GCN comparator remains unchanged. Historical Edge
comparisons deliberately use the audited ordinary-MixHop artifacts with the same
mathematical definition, not the later GCN revision.

Preflight only (no training):

```bash
../.venv/bin/python -m cmgm.scripts.formal_main_innovation_ablation --mode ts-global-expert --cpu-check
```

After a prepared preflight, start the five formal GPU runs:

```bash
nohup ../.venv/bin/python -u -m cmgm.scripts.formal_main_innovation_ablation --mode ts-global-expert --resume latest --run > ts_global_expert.log 2>&1 < /dev/null &
```

Results: `experiments/d0b_ts_global_2expert_mechanism/<timestamp>/`.
New Full checkpoint: `checkpoints/switching_latent_balanced_ts_2expert_global_mixture_best.pt`.
Control checkpoints: `checkpoints/formal_main_innovation_ablation/ts_global/<timestamp>/`.
Full is first because all four deltas refer to this new T+S Full. These are fixed
runs, never selected or repeated based on TEST. Existing Full checkpoints from
other architectures are read-only references and are never retrained.

The existing T+ST Global checkpoint is strictly loaded and freshly reproduced.
Adding TS classes changes the fusion file hash, so reuse requires the entire
historical Global file prefix to match its recorded SHA256 byte-for-byte; the
old Global implementation cannot silently change. All old architecture results
have explicit report/checkpoint hashes. The new 19-section report preserves
signed deltas, parameter/runtime counts, four-horizon results, expert/global/
regime diagnostics and all twelve required answers. It labels the evidence a
post-hoc architecture-mechanism study, not statistical significance or causal
proof. Completed matching runs resume without fitting again; interrupted or
invalid runs STOP for review. No extra variant follows these five runs.
