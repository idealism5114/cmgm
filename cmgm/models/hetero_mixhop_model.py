"""
HeteroMixHop — 7-dim features + per-type projection + MixHop + gated fusion.

Architecture:
    7-dim features (price, return, MA, volatility, RSI, MACD)
    → Per-type Input Projection (7→64 per market)
    → AdaptiveGraphLearner → A (N×N)
    → MixHopPropagation × 2 (64→64, 64→64)
    → Per-type mean pooling → type_agg(192→64)
    → Gated fusion with LSTM(284×7→64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgm.config import (
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM, FEATURE_DIM, MULTI_HORIZONS, TARGET_TYPE,
)
from cmgm.models.model import MixHopPropagation
from cmgm.graph.adaptive_graph import AdaptiveGraphLearner


class _TypeInputProjection(nn.Module):
    """Per-type linear projection: FEATURE_DIM → hidden_dim."""

    def __init__(self, hidden_dim: int = LSTM_HIDDEN_DIM):
        super().__init__()
        self.stock_proj  = nn.Linear(FEATURE_DIM, hidden_dim)
        self.bond_proj   = nn.Linear(FEATURE_DIM, hidden_dim)
        self.future_proj = nn.Linear(FEATURE_DIM, hidden_dim)

    def forward(self, x_t: torch.Tensor,
                n_stock: int, n_bond: int) -> torch.Tensor:
        s = self.stock_proj(x_t[:, :n_stock, :])          # (B, n_stock, H)
        b = self.bond_proj(x_t[:, n_stock:n_stock+n_bond, :])
        f = self.future_proj(x_t[:, n_stock+n_bond:, :])
        return torch.cat([s, b, f], dim=1)                 # (B, N, H)


class _TypeMeanPool(nn.Module):
    """Per-type mean pooling → concat → project."""

    def __init__(self, hidden_dim: int = LSTM_HIDDEN_DIM,
                 n_stock: int = 248, n_bond: int = 12):
        super().__init__()
        self.n_stock = n_stock
        self.n_bond = n_bond
        self.type_agg = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        stock_pool  = h[:self.n_stock, :].mean(dim=0)
        bond_pool   = h[self.n_stock:self.n_stock+self.n_bond, :].mean(dim=0)
        future_pool = h[self.n_stock+self.n_bond:, :].mean(dim=0)
        concat = torch.cat([stock_pool, bond_pool, future_pool])
        return self.type_agg(concat)


class HeteroMixHopCMGM(nn.Module):
    """
    Heterogeneous MixHop CMGM with 7-dim features.

    Forward:  (B, T, N, 7) → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 n_stock: int = 248, n_bond: int = 12):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.n_stock = n_stock
        self.n_bond = n_bond

        # ── Learnable graph ──
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes, embed_dim=10, alpha=0.5, top_k=10,
        )

        # ── Per-type projection (7 → 64) ──
        self.type_proj = _TypeInputProjection(LSTM_HIDDEN_DIM)

        # ── MixHop propagation (both layers 64→64) ──
        self.mixhop1 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
        self.mixhop2 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
        self.gcn_norm = nn.LayerNorm(LSTM_HIDDEN_DIM)

        # ── Per-type mean pool ──
        self.type_pool = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)

        # ── LSTM (input = num_nodes * FEATURE_DIM) ──
        self.temporal = nn.LSTM(
            input_size=num_nodes * FEATURE_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )

        # ── Gated fusion ──
        self.n_horizons = len(MULTI_HORIZONS)
        out_dim = self.n_horizons * n_commodities
        self.gate_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)
        self.gcn_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
        self.lstm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
        self.gate_output = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, out_dim),
        )

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_weight=None,
                debug: bool = False) -> torch.Tensor:
        B, T, N, _ = x.shape

        # ── 1. Learn adjacency ──
        A = self.graph_learner()                                   # (N, N)

        # ── 2. MixHop branch ──
        # Average over batch → get per-node 7-dim features
        # x: (B, T, N, 7) → mean over B → (T, N, 7) → permute → (N, T, 7)
        # type_proj projects last dim 7→64 → (N, T, 64) → mean over T → (N, 64)
        x_gcn = x.mean(dim=0)                                      # (T, N, 7)
        x_gcn = x_gcn.permute(1, 0, 2)                             # (N, T, 7)
        x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)  # (N, T, 64)
        x_proj = x_proj.mean(dim=1)                                # (N, 64)

        h1 = F.relu(self.mixhop1(x_proj, A))                       # (N, 64)
        h2 = self.mixhop2(h1, A)                                   # (N, 64)
        h = self.gcn_norm(h2)                                      # (N, 64)

        # Per-type pooling → expand to batch
        gcn_out = self.type_pool(h).unsqueeze(0).expand(B, -1)     # (B, 64)

        # ── 3. LSTM branch (flattened features) ──
        x_seq = x.reshape(B, T, -1)                                # (B, T, N*7)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                         # (B, 64)

        # ── 4. Gated fusion ──
        combined = torch.cat([gcn_out, lstm_out], dim=-1)          # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        gcn_p  = self.gcn_proj(gcn_out)
        lstm_p = self.lstm_proj(lstm_out)
        fused = gate * lstm_p + (1 - gate) * gcn_p
        pred = self.gate_output(fused)                                    # (B, n_horizons * n_comm)
        pred = pred.view(B, self.n_horizons, self.n_commodities)          # (B, n_horizons, n_comm)
        if TARGET_TYPE == "volatility":
            pred = F.softplus(pred) + 1e-6                                # ensure positive

        if debug:
            nz = (A > 0).sum().item()
            gm = gate.mean().item()
            print(f"  Feat(F={FEATURE_DIM}) → TypeProj → MixHop→({list(gcn_out.shape)}) || "
                  f"LSTM(N*F={N*FEATURE_DIM})→({list(lstm_out.shape)}) → "
                  f"Gate(mean={gm:.3f}, A_nz={nz}) → FC → ({list(pred.shape)})")

        return pred

    def get_gate_stats(self, x, edge_index=None, edge_weight=None):
        """Return gating statistics. Works with any FEATURE_DIM."""
        self.eval()
        with torch.no_grad():
            A = self.graph_learner()
            x_gcn = x.mean(dim=0).permute(1, 0, 2)
            x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)
            x_proj = x_proj.mean(dim=1)
            h1 = F.relu(self.mixhop1(x_proj, A))
            h2 = self.mixhop2(h1, A)
            h = self.gcn_norm(h2)
            B = x.size(0)
            gcn_out = self.type_pool(h).unsqueeze(0).expand(B, -1)

            x_seq = x.reshape(B, x.size(1), -1)
            lstm_out, (h_n, _) = self.temporal(x_seq)
            lstm_out = h_n[-1]

            combined = torch.cat([gcn_out, lstm_out], dim=-1)
            gate = torch.sigmoid(self.gate_fc(combined))

        return {
            'gate_mean': gate.mean().item(),
            'gate_std':  gate.std().item(),
            'gate_min':  gate.min().item(),
            'gate_max':  gate.max().item(),
            'mixhop_diff': (h1 - h2).norm().item(),
        }
