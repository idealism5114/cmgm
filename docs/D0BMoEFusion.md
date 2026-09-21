# D0B-MoEFusion

Independent variant: `switching_latent_balanced_moe_fusion`.
Only fusion changes. Native D0B spatial/temporal structures, Balanced Readout,
branch projections and shared prediction head remain unchanged. Their weights
still train end-to-end; “frozen” refers to architecture and protocol.

The router reads raw `[h_spatial, h_temporal]` and emits three sample-level
softmax weights, ordered Temporal, Spatial, Interaction. Experts read native
projected `t`, `s`, and `[s,t]`. T/S experts use residual 64→64→64 MLPs; ST uses
128→64→64 without a residual. All use ReLU and dropout 0.1. All three execute.
The original shared head receives one weighted 64-dimensional representation.

Native modules are constructed in their original order. The new variant then
removes only its obsolete gate and constructs its experts/router. Full D0B
retains its gate. At N=284: D0B 520,549 parameters, MoE 545,576, delta 25,027
(33,283 new expert/router parameters minus 8,256 gate parameters).

Training is one seed42 run: Adam, LR 1e-4, WD 1e-5, batch64, max200 epochs,
patience10, original ReduceLROnPlateau. TRAIN remains shuffle=False/drop_last=True.
Loss is four-horizon Huber(.02) plus native Switch KL plus 1e-4 times
KL(mean learned routing || uniform). No clipping, searches or additional losses.
Checkpoint selection, early stopping and scheduler retain prediction-only
multi-horizon VAL Huber. VAL5 is logging only.

`set_moe_epoch(e)` sets gamma=min(1,e/10). Epoch1 uses gamma=.1; epoch10 and later
use learned routing fully. Train and validation use the same epoch gamma.
The epoch is a persistent checkpoint buffer: restoring the best checkpoint
restores exactly its routing mixture, including a best epoch before10.
Balance uses learned probabilities, never the warm-up mixture.

## Run

From the repository root with the project Python environment activated, verify
on GPU without training:

```bash
python -m cmgm.scripts.d0b_moe_fusion --resume latest
```

If no preparation artifact exists yet, omit `--resume latest` for that first
verification. CPU-only verification is available with `--cpu-check`; formal
training requires CUDA and never silently falls back to CPU.

Launch the one authorized formal run yourself:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.d0b_moe_fusion --resume latest --run >> d0b_moe_fusion.log 2>&1 &
```

```bash
tail -f d0b_moe_fusion.log
```

The runner locks its output directory, checks source/data fingerprints, refuses
duplicate new experiments, and reuses a completed matching checkpoint.
An interrupted fit without a completed checkpoint stops for review rather than
silently repeating training. History and routing summaries are written each epoch.

Artifacts go to `experiments/d0b_moe_fusion/<timestamp>`; the checkpoint is
`checkpoints/switching_latent_balanced_moe_fusion_best.pt`. Existing TemporalOnly,
Fixed Equal Fusion and Full D0B results are reused from the audited formal
ablation report, with report/checkpoint hashes and protocol provenance. None of
these controls are retrained or overwritten.

`FINAL_REPORT.md` remains explicitly pending until the single run and best-
checkpoint evaluation finish. Router concentration, occupancy and expert
disagreement are descriptive; they do not establish causal attribution or
statistical significance. After evaluation and reporting: STOP.
