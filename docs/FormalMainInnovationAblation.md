# D0B formal main-innovation ablation — revised

The entry point is `cmgm.scripts.formal_main_innovation_ablation`. The previous 13-control study and its checkpoints are retained as historical artifacts; the revised table contains exactly A1–A11 and Full, in the preregistered order.

## Launch

Run from the repository root with the project Python environment activated. Preparation is read-only model verification/evaluation; it performs no optimizer step:

```bash
python -m cmgm.scripts.formal_main_innovation_ablation
```

If this workspace already contains the prepared revised experiment, use resume rather than creating another experiment. Formal training requires CUDA and is opt-in:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.formal_main_innovation_ablation --resume latest --run >> formal_main_ablation.log 2>&1 &
```

```bash
tail -f formal_main_ablation.log
```

`--cpu-check` permits only read-only preparation, not formal training. No CPU fallback is used by `--run`. An interrupted, incomplete fit stops for review rather than silently training the same configuration again. A completed checkpoint survives an interrupted evaluation and can be resumed without fitting again.

## Reuse and controls

The default source inventory is the completed previous study dated `20260915_153018`. Eight exact mathematical definitions map to its existing checkpoints. The audit checks source SHA256, checkpoint SHA256, complete-run metadata, seed, complete training protocol, data fingerprints, ordering, and sanity. Renaming a row never triggers a refit.

Four new definitions are isolated in `formal_d0b_main_ablation.py`; native D0B and the old study code are unchanged:

- No Adaptive Graph: all-ones adjacency into both original EdgeAttnMixHop layers; no graph learner call. The original `log(A + 1e-6)` yields a constant at A=1, so softmax reduces to content-only attention to numerical tolerance.
- No EdgeAttnMixHop: the same learned adjacency enters two existing standard `MixHopPropagation(64,64,K=2,beta=.05)` layers. The old edge-attention modules remain instantiated but inactive. Replacement modules are constructed last.
- No RPE: zero relative bias, same existing RPE parameter and module order, unchanged causal attention and Transformer layers.
- No Regime-Specific Transitions: generator[0] produces one candidate repeated across three states; posterior routing and native Switch KL remain. G1/G2 are instantiated but unused. This differs from uniform routing with three distinct generators.

Total instantiated parameters and autograd-connected prediction parameters are reported separately. Keeping inactive native modules preserves shared initialization, not active capacity. The training loop is identical to the previous formal loop, including native KL handling and multi-horizon VAL-Huber checkpoint selection.

Optional NoSwitchKL and input-modality sensitivity artifacts are strictly reuse-only and excluded from the 12-row main table. Old NoGraphPropagation is **not** reused as either NoAdaptiveGraph or NoEdgeAttnMixHop.

## Artifacts and interpretation

The revised directory includes checkpoint provenance, initial/best structural sanity, prefix causality, initialization equality, fixed-order main/subtables, pooled all-horizon metrics, commodity metrics, active parameter counts, and minimal mechanism summaries. Missing new runs remain explicitly pending; no result is invented.

The report distinguishes branch-level net value from local subsystem value. Negative ablation-minus-Full deltas are retained and do not trigger another run. Relative MAE differences smaller than 0.1% are marked near ties. No significance or multi-seed robustness claim is made.
