"""
Configuration file for CMGM model.
Contains all hyperparameters and data paths for the CMGM paper reproduction.

Reference: Ali et al. (2025), "CMGM: A novel cross-market assets and multi-market
modeling graph neural networks for financial market forecasting", AEJ.
"""

from pathlib import Path


# =============================================================================
# Section 4: Experimental Setup — Dataset Configuration
# =============================================================================
DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"

# Market data files
STOCK_FILE = DATA_ROOT / "hs300_data" / "hs300_close.csv"
BOND_FILE = DATA_ROOT / "bond_data" / "全部债券_合并.csv"
COMMODITY_FILE = DATA_ROOT / "futures_data" / "全部品种_合并.csv"

# Target market: commodities (we predict commodity prices)
TARGET_MARKET = "commodity"

# =============================================================================
# Section 3.2: Graph Construction Parameters
# =============================================================================

# Correlation strategy: one of ['pearson', 'volatility_adjusted',
#                               'skewness_kurtosis_adjusted', 'dynamic']
CORRELATION_METHOD = "pearson"

# Rolling window for dynamic correlation and volatility (trading days)
ROLLING_WINDOW = 20

# Top-k edges per node (retain strongest correlations)
TOP_K_EDGES = 10

# Adaptive graph learner (MTGNN-style)
GRAPH_EMBED_DIM = 10   # Node embedding dimension
GRAPH_ALPHA = 0.5      # Initial tanh saturation rate (learnable)
GRAPH_TOP_K = 10       # Neighbors per node in learned graph

# =============================================================================
# Section 3.3 & 3.5: Model Architecture Parameters
# =============================================================================

# Sequence length (lookback window)
SEQ_LEN = 20

# GCN parameters (paper: 3 layers, mean aggregation, concat combination)
GCN_INPUT_DIM = 7       # Legacy; overridden per model by FEATURE_DIM
FEATURE_DIM = 21        # Feature dimension for node inputs (21 = full set)
GCN_OUTPUT_DIM = 10     # Output dimension per node (paper: 10)
GCN_NUM_LAYERS = 3      # Graph convolution layers (paper: 3)

# GCN hidden dim for baseline models (GCN-only, GCN+GAT)
GCN_HIDDEN_DIM = 64     # Hidden dimension for baseline GCN layers

# LSTM parameters (paper: 64 units, ReLU activation)
LSTM_HIDDEN_DIM = 64   # LSTM hidden state dimension (paper: 64)
LSTM_NUM_LAYERS = 2     # Number of LSTM layers
LSTM_DROPOUT = 0.3      # Dropout between LSTM layers

# GCN dropout (applied after each GCN layer)
GCN_DROPOUT = 0.3

# Fully connected layer parameters
FC_HIDDEN_DIM = 64      # Hidden dimension in FC layer

# =============================================================================
# Section 3.4: Training Parameters
# =============================================================================

# Train/val/test split ratios (temporal split)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200
PATIENCE = 10           # Early stopping patience

# Loss function: 'mse' or 'huber'
LOSS_TYPE = "huber"
HUBER_DELTA = 0.02      # Delta for Huber loss (2% daily return → linear above)

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Target Configuration
# =============================================================================
# Target type: 'price', 'return', or 'volatility'
TARGET_TYPE = "return"

# Target horizon: number of steps ahead (primary, for reporting)
TARGET_HORIZON = 5

# Multi-horizon prediction: all horizons the model predicts simultaneously
VOLA_HORIZONS = [1, 5, 10, 20]    # horizons for volatility prediction
# For returns: MULTI_HORIZONS = [1, 5, 10, 20]
MULTI_HORIZONS = [1, 5, 10, 20]   # kept for reference

# =============================================================================
# Normalization Configuration
# =============================================================================
# Epsilon for per-asset z-score to prevent division by near-zero std
ZSCORE_EPS = 1e-8

# Minimum std for feature z-score (features like price can be very large,
# so a higher floor prevents explosion when an asset has near-zero variance)
FEAT_ZSCORE_EPS = 1e-3

# =============================================================================
# Evaluation: Confidence Interval (Section 4.4)
# =============================================================================
CONFIDENCE_LEVEL = 0.95
NUM_BOOTSTRAP_SAMPLES = 1000
