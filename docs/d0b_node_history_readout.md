# D0B node-preserving, target-conditioned history readout

This is a controlled implementation study based on `switching_latent_balanced_readout`. It tests whether retaining commodity-node representations and querying the observed Transformer history with commodity-specific spatial queries is useful. It does not claim a verified performance gain or a novel method.

## Verified tensor path

The spatial encoder's existing `_temp_weighted_spatial(x, return_nodes=True)` returns its unchanged pooled readout `h_s: (B,64)` and the post-graph, post-normalization node states `S: (B,N,64)`. The runner passes the actual `market_indices['commodity']` interval into the node variants; no commodity suffix is guessed. The target mapping is saved as `commodity name / node index / target index / output index`.

The switching-latent encoder's default call still returns its original `(B,64)` balanced readout. Its optional `return_memory=True` call returns `(h_t, H)` from the same market encoding, causal Transformer, regime filter, and latent recurrence, with `H: (B,T,128)`. It does not execute the time encoder a second time and does not change the native Switch KL path.

All variants reuse `gcn_proj`, `lstm_proj`, and the original head. The original head's final linear rows are stored horizon-major then commodity-minor (`row = horizon_index * Nc + commodity_index`); this is checked by an explicit repeated-global-output test.

## Variants and comparisons

- `switching_latent_balanced_concat_fusion`: GELU over a learned projection of `[s_global || t_global]`, followed by the original global head. This asks whether ordinary global concatenation is sufficient.
- `switching_latent_balanced_node_history_mean`: retains each projected commodity node and combines it with a shared uniform mean of projected history, both global projected states, and the original output rows.
- `switching_latent_balanced_node_history_attention`: uses each commodity's projected spatial state as a query over the full observed temporal history; softmax is over the history axis. This compares content-based temporal selection with uniform history readout while using the same node inputs, V/O maps, fusion projection, and prediction head. Q/K add 6,144 parameters versus the mean readout.

The node-mean versus global comparisons change both representation granularity and readout; they do not isolate one factor. The attention versus node-mean comparison is more targeted, but still includes the Q/K parameter increase. A post-fit forced-uniform attention intervention evaluates the trained attention model with its time weights set uniformly; this frozen-model intervention is not equivalent to independently training the node-mean model.

At the current 284-node, 24-commodity configuration, instantiated parameter counts are: original gated D0B `520,549`; existing parameter-free mean-fusion D0B `512,293`; global concatenation `520,549`; node-history mean `534,885`; node-history attention `541,029`. The node-history mean adds `22,592` active parameters after replacing the `8,256`-parameter gate. Attention adds exactly `6,144` Q/K parameters over node-history mean. These were measured from actual model instances, not hard-coded into model construction.

Attention entropy, maximum weight, distance from uniform, and between-commodity attention differences are behavior diagnostics only, not causal explanations. The optional uniform intervention uses the same frozen model and checkpoint.

## Initialization and protocol

Every new module is constructed after the existing D0B shared encoders, branch projections, and head. The node-mean and node-attention variants construct common `W_v`, `W_o`, and fusion projection in the same order; the attention-only Q/K layers follow. The experiment runner audits common initialization in an RNG-isolated context, then reseeds before the actual model construction. Existing variants retain their state-dict keys and default forward path.

The runner reuses the repository training loop: seed 42 by default, Adam at `1e-4`, weight decay `1e-5`, batch 64, at most 200 epochs, early-stop patience 10, ReduceLROnPlateau, four equal Huber horizon terms (`delta=.02`), and the native D0B Switch KL schedule. Checkpoint selection remains the existing batch-mean, prediction-only multi-horizon VAL Huber objective. TEST metrics are not run unless `--evaluate-test` is explicitly supplied; then they are computed only after restoring the best VAL checkpoint.

## Preflight and future runs

From the repository root, use the active project environment's Python. Default invocation is synthetic preflight only:

```bash
../.venv/bin/python -m cmgm.scripts.d0b_node_history_readout
```

One formal training run at a time, each in its own timestamped result/checkpoint directory:

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_node_history_readout --run --variant switching_latent_balanced_readout --seed 42 --device cuda
../.venv/bin/python -u -m cmgm.scripts.d0b_node_history_readout --run --variant switching_latent_balanced_mean_fusion --seed 42 --device cuda
../.venv/bin/python -u -m cmgm.scripts.d0b_node_history_readout --run --variant switching_latent_balanced_concat_fusion --seed 42 --device cuda
../.venv/bin/python -u -m cmgm.scripts.d0b_node_history_readout --run --variant switching_latent_balanced_node_history_mean --seed 42 --device cuda
../.venv/bin/python -u -m cmgm.scripts.d0b_node_history_readout --run --variant switching_latent_balanced_node_history_attention --seed 42 --device cuda
```

For a run whose checkpoint is frozen, explicit TEST evaluation can be requested with `--evaluate-test`; it is intentionally absent from the commands above. Results go to `experiments/d0b_node_history_readout/<variant>_seed<seed>_<timestamp>_<pid>/`, and checkpoints to `checkpoints/d0b_node_history_readout/<same-run-name>/best.pt`. Each run records source hashes, git SHA, TRAIN/VAL data fingerprints, actual commodity ordering, initialization audit, parameter count, best epoch, selection loss, full TRAIN/VAL metrics, timing, and readout diagnostics.

No formal training, TEST evaluation, multi-seed study, or hyperparameter search was run while implementing this feature.
