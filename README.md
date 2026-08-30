# CMGM

This repository contains the current HeteroMixHop model, its comparison
baselines, and the three retained ablation variants.

## Entrypoints

```bash
# Current main model (edge_attn)
python -m cmgm.scripts.main_hetero_mixhop

# Baseline comparison plus the current main model
python -m cmgm.scripts.main_hetero_mixhop --baselines
# Equivalent standalone baseline entrypoint
python -m cmgm.baselines.run

# All retained ablations
python -m cmgm.scripts.main_ablation

# Select retained ablations
python -m cmgm.scripts.main_ablation \
  --variants +TempWeighted,F-RegimeDynamic,F2-RegimeSemantic
```

## Retained model variants

| Display name | Internal variant | Role |
|---|---|---|
| HeteroMixHop | `edge_attn` | Main model |
| +TempWeighted | `temporal_weighted_graph` | Temporal-weighted spatial aggregation |
| F-RegimeDynamic | `regime_dynamic_transformer` | Regime-specific temporal dynamics |
| F2-RegimeSemantic | `regime_dynamic_semantic` | Regime dynamics with semantic-state loss |

The model rejects any other variant name. Generated experiment indexes,
Python bytecode, and model checkpoints are ignored by Git.

## Layout

```text
cmgm/
├── baselines/       # traditional, recurrent, and graph baselines
├── data/            # loading and feature construction
├── graph/           # static and adaptive graph construction
├── models/          # HeteroMixHop, regime branch, and retained CMGM baselines
├── scripts/         # main and ablation entrypoints
└── training/        # training and evaluation
```
