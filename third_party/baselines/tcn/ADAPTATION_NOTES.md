# Faithful standard TCN repair

Reference: https://github.com/locuslab/TCN at commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`, `TCN/tcn.py`. MIT license retained. `tcn.py.upstream` is byte-identical upstream; `tcn.py.patch` records all local changes; SHA256 values are in `provenance.json`.

The original Chomp1d, TemporalBlock and TemporalConvNet architecture is preserved: two weight-normalized dilated Conv1d → Chomp → ReLU → Dropout stages per block, identity or 1×1 residual downsample, final ReLU. Padding=(kernel−1)*dilation; dilations=1,2,4. Chomp sizes are 2,4,8 (never zero for this task).

## PyTorch compatibility and initialization

PyTorch compatibility substitution only; weight-normalization semantics retained. `torch.nn.utils.parametrizations.weight_norm` replaces the deprecated hook API, retaining per-output-channel normalization `W=g*v/||v||`.

The upstream initializer writes `conv.weight.data.normal_(0,.01)` *after* applying weight normalization. Under current PyTorch this is a transient computed weight: the next forward recomputes weight from the underlying magnitude/direction. We explicitly initialize underlying direction `v ~ N(0,.01)` and magnitude `g=||v||` over input-channel/kernel dimensions, so the effective weight has the requested small-normal initialization both before and after forward. Downsample initialization remains upstream `.normal_(0,.01)`. Bias initialization and task Linear defaults are unchanged. This preserves intended initialization, not the accidental transient-weight behavior of the old API.

## Task interface

`cmgm.models.formal_baselines_v2.TCN` losslessly reshapes B,20,N,21 → B,N*21,20 and feeds it directly into TemporalConvNet(num_inputs=N*21, num_channels=[128,128,128], kernel_size=3, dropout=.1).

There is no standalone input projection. The first block's 1×1 convolution is solely its standard residual/downsample path. The final causal timestep B,128 enters one Linear(128,96), reshaped horizon-major/commodity-minor to B,4,24. Input information, targets, optimizer, loss, selection, loader and evaluator do not change. No clipping, new normalization layer or additional head is added.

Historical `CausalConv` and `TCNBlock` definitions remain available for history, but corrected TCN never uses them. The old TCN artifact is INVALID_FOR_FINAL_BASELINE_TABLE due to architecture fidelity, not invalidated or retried according to TEST performance. Only one corrected seed42 fit is allowed.
