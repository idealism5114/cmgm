"""
Generate hyperparameter table for paper.
"""
import pandas as pd

OUTPUT = "experiments/paper_analysis"

# =====================================================================
# Table A: Dataset
# =====================================================================
dataset_rows = [
    ("Stock market", "CSI 300 constituents", "646 stocks"),
    ("Bond market", "Treasury futures (T, TF, TS, TL) + Bond ETFs + Green bond indices", "15 instruments"),
    ("Commodity market", "24 commodity futures (excl. index futures)", "24 varieties"),
    ("Macro variables", "Interest rate + VIX", "2 variables"),
    ("Total nodes", "Stocks + Bonds + Commodities", "284 nodes"),
    ("Data period", "2017-01-17 to 2025-06-30", "2,050 trading days"),
    ("Train / Val / Test split", "Temporal (70% / 15% / 15%)", "1,415 / 287 / 288"),
    ("Lookback window (SEQ_LEN)", "—", "20 days"),
    ("Normalization", "Per-feature-channel MinMaxScaler [0,1] (fit on train)", "—"),
]

# =====================================================================
# Table B: Features
# =====================================================================
feature_rows = [
    ("[0] price", "Raw closing price", "—", "MinMaxScaled"),
    ("[1] return", "Daily simple return (pct_change on raw prices)", "Clipped to ±20%", "MinMaxScaled"),
    ("[2] ma5_ratio", "Price / 5-day moving average", "Clipped to [0.5, 2.0]", "MinMaxScaled"),
    ("[3] ma20_ratio", "Price / 20-day moving average", "Clipped to [0.5, 2.0]", "MinMaxScaled"),
    ("[4] volatility", "20-day rolling std of daily returns", "Clipped to [0, 0.5]", "MinMaxScaled"),
    ("[5] rsi_14", "14-day Relative Strength Index", "[0, 100]", "MinMaxScaled"),
    ("[6] macd", "MACD (12-day EMA − 26-day EMA)", "—", "MinMaxScaled"),
]

# =====================================================================
# Table C: Model Architecture (HeteroMixHop)
# =====================================================================
arch_rows = [
    ("Spatial branch: Type projection", "Per-market linear layer (stock/bond/commodity): 7 → 64"),
    ("Spatial branch: Graph learner", "AdaptiveGraphLearner: node emb (284×10) → anti-symmetric similarity → tanh → soft top-10"),
    ("Spatial branch: MixHop layers", "2× MixHopPropagation(K=2, β=0.05): 64→64 → 64→64"),
    ("Spatial branch: Per-type pooling", "Mean pool per market → concat → Linear(3×64→64)"),
    ("Temporal branch: LSTM", "2-layer LSTM, input=284×7=1988, hidden=64, dropout=0.3"),
    ("Fusion: Gated mechanism", "σ(W·[gcn‖lstm]) × lstm_proj + (1-σ) × gcn_proj → 64"),
    ("Output: FC", "Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→24)"),
    ("Total parameters", "~727K"),
]

# =====================================================================
# Table D: Training
# =====================================================================
train_rows = [
    ("Batch size", "64"),
    ("Learning rate", "1e-4"),
    ("Optimizer", "Adam"),
    ("Weight decay", "1e-5"),
    ("Max epochs", "200"),
    ("Early stopping patience", "10 (monitor val loss)"),
    ("Loss function", "MSE"),
    ("Hardware", "NVIDIA Tesla V100 (32GB)"),
    ("Framework", "PyTorch 2.x + PyG"),
    ("Random seed", "42"),
]

# =====================================================================
# Table E: Baseline configurations
# =====================================================================
baseline_rows = [
    ("PCA+Ridge", "Flatten → PCA(100) → StandardScaler → Ridge(α=1.0)"),
    ("PCA+SVR", "Flatten → PCA(100) → StandardScaler → LinearSVR(C=1.0)"),
    ("LSTM", "LSTM(1988→64, 2 layers) → FC(64→64→24)"),
    ("BiLSTM", "BiLSTM(1988→64, 2 layers) → FC(64→64→24)"),
    ("GCN-Only", "Pre-defined Pearson graph → 3× GCN(7→64→64→64) → Mean → FC(64→24)"),
    ("GCN+GAT", "Pre-defined Pearson graph → GCN(7→64) + GAT(64→64) → FC..."),
    ("CMGM-Feat", "Same features + Pre-defined graph → GCN||LSTM → Concat → FC"),
    ("HeteroMixHop (ours)", "Learned adaptive graph + Per-type projection + MixHop × 2 + LSTM → Gated fusion → FC"),
]

# =====================================================================
# Write LaTeX and CSV
# =====================================================================

with open(f"{OUTPUT}/hyperparams.md", "w") as f:
    f.write("# Hyperparameter Configuration\n\n")

    f.write("## A. Dataset Configuration\n\n")
    f.write("| Parameter | Description | Value |\n")
    f.write("|-----------|-------------|-------|\n")
    for p, d, v in dataset_rows:
        f.write(f"| {p} | {d} | {v} |\n")

    f.write("\n\n## B. Input Features (7-dim)\n\n")
    f.write("| Index | Feature | Clipping | Post-processing |\n")
    f.write("|-------|---------|----------|-----------------|\n")
    for idx, feat, clip, norm in feature_rows:
        f.write(f"| {idx} | {feat} | {clip} | {norm} |\n")

    f.write("\n\n## C. Model Architecture (HeteroMixHop)\n\n")
    f.write("| Component | Description |\n")
    f.write("|-----------|-------------|\n")
    for comp, desc in arch_rows:
        f.write(f"| {comp} | {desc} |\n")

    f.write("\n\n## D. Training Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    for p, v in train_rows:
        f.write(f"| {p} | {v} |\n")

    f.write("\n\n## E. Baseline Model Summaries\n\n")
    f.write("| Model | Architecture |\n")
    f.write("|-------|-------------|\n")
    for m, desc in baseline_rows:
        f.write(f"| {m} | {desc} |\n")

print("Hyperparameter table written to", f"{OUTPUT}/hyperparams.md")

# Also print to console
print("\n" + "=" * 80)
print("HYPERPARAMETER TABLE — Paper-Ready")
print("=" * 80)

print("\n--- A. Dataset ---")
for p, d, v in dataset_rows:
    print(f"  {p:40s} {v}")

print("\n--- B. Input Features ---")
print(f"  {'Index':10s} {'Feature':20s} {'Clip':20s} {'Norm'}")
for idx, feat, clip, norm in feature_rows:
    print(f"  {idx:10s} {feat:20s} {clip:20s} {norm}")

print("\n--- C. Model Architecture ---")
for comp, desc in arch_rows:
    print(f"  {comp}")

print("\n--- D. Training ---")
for p, v in train_rows:
    print(f"  {p:30s} {v}")

print("\n--- E. Baselines ---")
for m, desc in baseline_rows:
    print(f"  {m:25s} {desc}")
