# Deep baseline benchmark V3

Uses the existing `comparison_baselines.py`, `baseline_comparison.py`,
`baseline_protocol.py` and `baseline_report.py`; no independent trainer.

Scope: RNN, GRU, LSTM, VanillaTransformer, GraphWaveNet, MTGNN (project adapted),
MSGNet and CrossGNN. The frozen **D0B-Candidate-Aware 2-Expert MoE** is Ours.
All prior experiments/checkpoints remain outside this suite and are not overwritten.

The original deterministic 28-token neutral adapter is unchanged. Sequence models
receive20×588, graph-temporal models20×28×21, MSGNet/CrossGNN each learn their own
shared21→1 node projection. Ours retains native full-node input; this is explicitly
not an identical uncompressed architectural information representation.

Core fidelity/provenance and necessary per-sample FFT adaptations are documented
in `cmgm/models/baselines/ADAPTATION_NOTES.md` and `provenance.json`. The official
FFT batch-average choice is replaced because it would break sample independence.
No result-dependent changes or hyperparameter search are permitted.

From the repository root with the project virtual environment activated:

```bash
python -m cmgm.scripts.baseline_comparison --sanity-only --no-cuda
```

If a V3 preflight already exists, use:

```bash
python -m cmgm.scripts.baseline_comparison --sanity-only --no-cuda --resume latest
```

Only the user starts formal GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.baseline_comparison --resume latest --run >> baseline_deep_v3.log 2>&1 &
tail -f baseline_deep_v3.log
```

The eight models train sequentially, once each. There is no Ours optimizer.
Default paths are `checkpoints/baselines_deep_v3` and
`experiments/baseline_comparison_deep_v3/<timestamp>`. Ours checkpoint/report can
be specified with `--ours-checkpoint` and `--ours-results`; both are audited,
including COMPLETE status, data fingerprints, variant, SHA and best sanity.

New model checkpoint/scheduler/early stopping use the unchanged mean of
batch-mean multi-horizon VAL Huber. TEST runs only after best checkpoint selection
and sanity. Completed V3 models require matching metadata/source/config/data/SHA
and are reused. Interrupted or invalid runs STOP for review; no automatic retry,
batch reduction, seed change or reuse of unrelated old results. A lock prevents
concurrent execution in the same output root. No process is killed.

`FINAL_REPORT.md` has the prescribed18 sections. Tables retain all9 rows, with
pending metrics explicitly blank until training. ΔMAE = baseline−Ours and
relative delta divides by Ours. Full all-horizon metrics and per-commodity5d
metrics are persisted separately. One seed42, no significance claim. STOP after
this fixed suite.

Runtime comparison is collected automatically by the same `--run` command.
The main table includes training seconds and inference milliseconds per origin.
`training_efficiency.csv` additionally records mean epoch seconds, complete TEST
forward seconds (mean/std over ten repeats), and origins/second. The frozen Ours
checkpoint is timed for inference too; its historical training time remains N/A.

All inference timing uses batch64 (final partial batch retained), the full TEST
input set, eval/inference mode, three warmup batches and ten forward-only passes.
CUDA is synchronized around each pass. Model adapters are included; input loading,
CPU-to-GPU copies, metrics and disk I/O are excluded. This is amortized batch
inference cost, not single-request latency. No labels enter the timing procedure.
Raw timings and hardware/software metadata are in `runtime_measurements.json`;
`runtime_protocol.json` defines the scope. Hardware/protocol mismatches on resume
stop rather than combine incomparable timings. Timing repeats are not training
reruns and do not affect checkpoint selection.
