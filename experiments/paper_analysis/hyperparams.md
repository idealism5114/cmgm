# Hyperparameter Configuration

## A. Dataset Configuration

| Parameter | Description | Value |
|-----------|-------------|-------|
| Stock market | CSI 300 constituents | 646 stocks |
| Bond market | Treasury futures (T, TF, TS, TL) + Bond ETFs + Green bond indices | 15 instruments |
| Commodity market | 24 commodity futures (excl. index futures) | 24 varieties |
| Macro variables | Interest rate + VIX | 2 variables |
| Total nodes | Stocks + Bonds + Commodities | 284 nodes |
| Data period | 2017-01-17 to 2025-06-30 | 2,050 trading days |
| Train / Val / Test split | Temporal (70% / 15% / 15%) | 1,415 / 287 / 288 |
| Lookback window (SEQ_LEN) | — | 20 days |
| Normalization | Per-feature-channel MinMaxScaler [0,1] (fit on train) | — |


## B. Input Features (7-dim)

| Index | Feature | Clipping | Post-processing |
|-------|---------|----------|-----------------|
| [0] price | Raw closing price | — | MinMaxScaled |
| [1] return | Daily simple return (pct_change on raw prices) | Clipped to ±20% | MinMaxScaled |
| [2] ma5_ratio | Price / 5-day moving average | Clipped to [0.5, 2.0] | MinMaxScaled |
| [3] ma20_ratio | Price / 20-day moving average | Clipped to [0.5, 2.0] | MinMaxScaled |
| [4] volatility | 20-day rolling std of daily returns | Clipped to [0, 0.5] | MinMaxScaled |
| [5] rsi_14 | 14-day Relative Strength Index | [0, 100] | MinMaxScaled |
| [6] macd | MACD (12-day EMA − 26-day EMA) | — | MinMaxScaled |


## C. Model Architecture (HeteroMixHop)

| Component | Description |
|-----------|-------------|
| Spatial branch: Type projection | Per-market linear layer (stock/bond/commodity): 7 → 64 |
| Spatial branch: Graph learner | AdaptiveGraphLearner: node emb (284×10) → anti-symmetric similarity → tanh → soft top-10 |
| Spatial branch: MixHop layers | 2× MixHopPropagation(K=2, β=0.05): 64→64 → 64→64 |
| Spatial branch: Per-type pooling | Mean pool per market → concat → Linear(3×64→64) |
| Temporal branch: LSTM | 2-layer LSTM, input=284×7=1988, hidden=64, dropout=0.3 |
| Fusion: Gated mechanism | σ(W·[gcn‖lstm]) × lstm_proj + (1-σ) × gcn_proj → 64 |
| Output: FC | Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→24) |
| Total parameters | ~727K |


## D. Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Weight decay | 1e-5 |
| Max epochs | 200 |
| Early stopping patience | 10 (monitor val loss) |
| Loss function | MSE |
| Hardware | NVIDIA Tesla V100 (32GB) |
| Framework | PyTorch 2.x + PyG |
| Random seed | 42 |


## E. Baseline Model Summaries

| Model | Architecture |
|-------|-------------|
| PCA+Ridge | Flatten → PCA(100) → StandardScaler → Ridge(α=1.0) |
| PCA+SVR | Flatten → PCA(100) → StandardScaler → LinearSVR(C=1.0) |
| LSTM | LSTM(1988→64, 2 layers) → FC(64→64→24) |
| BiLSTM | BiLSTM(1988→64, 2 layers) → FC(64→64→24) |
| GCN-Only | Pre-defined Pearson graph → 3× GCN(7→64→64→64) → Mean → FC(64→24) |
| GCN+GAT | Pre-defined Pearson graph → GCN(7→64) + GAT(64→64) → FC... |
| CMGM-Feat | Same features + Pre-defined graph → GCN||LSTM → Concat → FC |
| HeteroMixHop (ours) | Learned adaptive graph + Per-type projection + MixHop × 2 + LSTM → Gated fusion → FC |
