# Dynamic Router vs Learned Global Mixture

New variant: `switching_latent_balanced_candidate_2expert_global_mixture`.
Only the candidate router is replaced by two zero-initialized global trainable logits. Temporal/Interaction experts are imported unchanged, constructed in the same order, and fused before the unchanged shared head. No Utility-Routed predictive MoE, auxiliary router loss, warm-up or new optimizer group is used.

The runner audits existing control artifacts and shared/expert initialization, then extracts the frozen Dynamic checkpoint's mean probabilities over **every TRAIN origin** without labels. This vector replaces only sample-dependent weights at TEST inference, before the shared head. Original Dynamic metrics must reproduce and its weights/checkpoint must remain unchanged.

The separate static fit uses seed42, Adam1e-4/WD1e-5, batch64, chronological TRAIN/drop_last=True, epochs200/patience10 and the native scheduler factor.5/patience5. TRAIN loss is four Huber(.02) terms plus native SwitchKL; VAL selection, early stopping and scheduler use prediction-only multi-horizon Huber. No performance-dependent reruns.

Run from the repository root in the project virtual environment. Verification and frozen-checkpoint intervention only:

```bash
python -m cmgm.scripts.d0b_candidate_2expert_global_mixture --cpu-check
```

After preparation, launch the **one** static formal GPU fit yourself:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.d0b_candidate_2expert_global_mixture --resume latest --run >> d0b_global_mixture.log 2>&1 &
tail -f d0b_global_mixture.log
```

`--resume latest` checks immutable source/protocol/data/reference hashes. A completed run is not repeated. An interrupted fit without a matching completed checkpoint stops for review rather than silently restarting. A checkpoint with mismatched metadata is never overwritten. A filesystem lock prevents concurrent launch in the same experiment root. Historical control checkpoints are only read.

Outputs are under `experiments/d0b_candidate_2expert_global_mixture/<timestamp>`. Training history and global weights are written each epoch. TRAIN/VAL weight entries are explicitly the same end-of-epoch snapshot, not separate learned parameters. Formal metrics and the full report remain pending until the user runs training. Tiny differences must be reported in absolute units; no practical equivalence threshold or significance claim is invented. Stop after this one control and one frozen intervention.
