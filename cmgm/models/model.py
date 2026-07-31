"""
CMGM Model — Parallel Architecture (GCN || LSTM → Concat → FC)

GCN branch:     per-timestep GCN(mean+concat, 3 layers, 1→10) → mean pool → Linear(2840→64)
LSTM branch:    raw prices → LSTM(284→64)
Fusion:         Concat(GCN_64, LSTM_64) → FC(128→64→N_commodities)

GCN captures cross-market spatial dependencies.
LSTM captures temporal patterns from raw prices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS,
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM, SEQ_LEN,
)


# =============================================================================
# Mix-hop Propagation (MTGNN Section 4.3)
# =============================================================================

class MixHopPropagation(nn.Module):
    """
    MTGNN-style mix-hop propagation layer.

    Two steps per hop k:
      1. Propagation:  H(k) = β·H_in + (1-β)·Â·H(k-1)
      2. Selection:    H_out = Σ_k H(k)·W(k)

    Args:
        in_dim:  Input feature dimension
        out_dim: Output feature dimension
        K:       Number of hops (default: 2)
        beta:    Self-information retention (default: 0.05)
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 2, beta: float = 0.05):
        super().__init__()
        self.K = K
        self.beta = beta
        self.Ws = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(K + 1)
        ])

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_hat = A + torch.eye(A.size(0), device=A.device)
        deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_norm = A_hat / deg

        H = x
        H_in = x
        out = self.Ws[0](H)

        for k in range(1, self.K + 1):
            H = self.beta * H_in + (1 - self.beta) * (A_norm @ H)
            out = out + self.Ws[k](H)

        return out


# =============================================================================
# Spatial: Mean+Concat GCN (per time step)
# =============================================================================

class MeanConcatGCNLayer(nn.Module):
    """
    GCN layer with Mean aggregation and Concat combination.

    For each node v:
      1. h_neighbors = mean(h_u for u in N(v))        # Mean aggregation
      2. h_combined = [h_v || h_neighbors]             # Concat combination
      3. h_v' = ReLU(W * h_combined)                   # Linear transform
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source_idx, target_idx = edge_index[0], edge_index[1]

        # Mean aggregation of neighbor features
        source_x = x[source_idx]
        aggr = scatter(source_x, target_idx, dim=0, dim_size=x.size(0), reduce='mean')

        # Concat combination: [node_features || mean_neighbor_features]
        combined = torch.cat([x, aggr], dim=-1)

        # Linear + ReLU + Dropout
        out = self.lin(combined)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class SpatialGCN(nn.Module):
    """
    Spatial dependency extractor — 3× MeanConcatGCNLayer(1→10, 10→10, 10→10).

    Input per time step:  (B, N, 1)
    Output per time step: (B, N * 10)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            MeanConcatGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[MeanConcatGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t: torch.Tensor, edge_index: torch.Tensor,
                batch_size: int, num_nodes: int) -> torch.Tensor:
        x_flat = x_t.reshape(batch_size * num_nodes, -1)
        batched_edge_index = self._batch_edge_index(
            edge_index, num_nodes, batch_size, x_flat.device
        )
        for layer in self.layers:
            x_flat = layer(x_flat, batched_edge_index)
        return x_flat.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)

    @staticmethod
    def _batch_edge_index(edge_index, num_nodes, batch_size, device):
        offsets = torch.arange(batch_size, device=device) * num_nodes
        batched = edge_index.repeat(1, batch_size)
        offset_repeat = offsets.repeat_interleave(edge_index.shape[1])
        return batched + offset_repeat.unsqueeze(0)


# =============================================================================
# Temporal: LSTM on raw prices
# =============================================================================

class TemporalLSTM(nn.Module):
    """LSTM for raw price sequence. Input: (B, T, N), Output: (B, LSTM_HIDDEN_DIM)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# =============================================================================
# Output: Fusion FC
# =============================================================================

class OutputLayer(nn.Module):
    """Fully connected output: Linear(in→FC_HIDDEN) → ReLU → Dropout → Linear(FC_HIDDEN→N_commodities)"""

    def __init__(self, in_dim: int, n_commodities: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, n_commodities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# CMGM: Parallel GCN || LSTM
# =============================================================================

class CMGM(nn.Module):
    """
    Parallel architecture:
        GCN(per-timestep, mean+concat ×3) → mean pool → Linear(2840→64)  ──┐
        LSTM(raw prices, 284→64)                                          ──┤── Fusion → FC → (B, 24)

    Fusion modes:
        'concat' (default, original): Concat(128) → OutputLayer(128→64→24)
        'gate'   (gated fusion):      gate * lstm_proj + (1-gate) * gcn_proj → GateOutput(64→64→24)
        'mixhop' (MTGNN MixHop + gate):  MixHopPropagation replaces GCN + gated fusion → GateOutput
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 fusion_mode: str = 'concat'):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.fusion_mode = fusion_mode

        # GCN branch: spatial features
        self.spatial = SpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM  # 284 * 10 = 2840
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)  # 2840 → 64

        # LSTM branch: temporal features from raw prices
        self.temporal = TemporalLSTM(num_nodes)  # 284 → 64

        # Fusion: concat GCN(64) + LSTM(64) → FC(128 → 64 → 24)  [original]
        fusion_dim = LSTM_HIDDEN_DIM * 2  # 128
        self.output = OutputLayer(fusion_dim, n_commodities)

        # Fusion: gated fusion (optional, activated by fusion_mode='gate')
        if fusion_mode == 'gate':
            self.gate_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)
            self.gcn_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.lstm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.gate_output = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(GCN_DROPOUT),
                nn.Linear(FC_HIDDEN_DIM, n_commodities),
            )

        # MixHop propagation + gated fusion (activated by fusion_mode='mixhop')
        if fusion_mode == 'mixhop':
            from cmgm.graph.adaptive_graph import AdaptiveGraphLearner
            self.graph_learner = AdaptiveGraphLearner(
                num_nodes, embed_dim=10, alpha=0.5, top_k=10,
            )
            self.mixhop1 = MixHopPropagation(SEQ_LEN, LSTM_HIDDEN_DIM, K=2, beta=0.05)
            self.mixhop2 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
            self.gcn_norm = nn.LayerNorm(LSTM_HIDDEN_DIM)
            self.gate_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)
            self.gcn_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.lstm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.gate_output = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(GCN_DROPOUT),
                nn.Linear(FC_HIDDEN_DIM, n_commodities),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 1) — normalized closing prices
            edge_index: (2, E) — graph edges
            edge_weight: ignored (Mean+Concat uses unweighted mean)
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ========== GCN Branch (skipped in mixhop mode) ==========
        if self.fusion_mode != 'mixhop':
            temporal_outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :, :]  # (B, N, 1)
                out_t = self.spatial(x_t, edge_index, batch_size, num_nodes)  # (B, N*10)
                temporal_outputs.append(out_t)

            gcn_stack = torch.stack(temporal_outputs, dim=1)  # (B, T, N*10)
            gcn_pooled = gcn_stack.mean(dim=1)                 # (B, N*10) = (B, 2840)
            gcn_out = self.gcn_reduce(gcn_pooled)              # (B, 64)
        else:
            gcn_out = None  # computed in MixHop branch below

        # ========== MixHop Branch (replaces GCN for fusion_mode='mixhop') ==========
        if self.fusion_mode == 'mixhop':
            A = self.graph_learner()                                       # (N, N)
            x_gcn = x.mean(dim=0).squeeze(-1).T                            # (N, T)
            h1 = F.relu(self.mixhop1(x_gcn, A))                            # (N, 64)
            h2 = self.mixhop2(h1, A)                                        # (N, 64)
            h = self.gcn_norm(h2)                                           # (N, 64)
            gcn_out = h.unsqueeze(0).expand(batch_size, -1, -1)            # (B, N, 64)
            gcn_out = gcn_out[:, :self.n_commodities, :].mean(dim=1)       # (B, 64)

        # ========== LSTM Branch ==========
        x_seq = x.squeeze(-1)  # (B, T, N)
        lstm_out = self.temporal(x_seq)  # (B, 64)

        # ========== Fusion ==========
        combined = torch.cat([gcn_out, lstm_out], dim=-1)  # (B, 128)
        pred = self.output(combined)  # (B, N_commodities) — concat path

        # ========== Gated Fusion (optional, replaces concat) ==========
        if self.fusion_mode == 'gate':
            gate = torch.sigmoid(self.gate_fc(combined))          # (B, 64)
            gcn_p  = self.gcn_proj(gcn_out)                       # (B, 64)
            lstm_p = self.lstm_proj(lstm_out)                     # (B, 64)
            fused = gate * lstm_p + (1 - gate) * gcn_p            # (B, 64)
            pred = self.gate_output(fused)                        # (B, N_commodities)

        # ========== MixHop Fusion (MixHop + gated fusion) ==========
        if self.fusion_mode == 'mixhop':
            gate = torch.sigmoid(self.gate_fc(combined))          # (B, 64)
            gcn_p  = self.gcn_proj(gcn_out)                       # (B, 64)
            lstm_p = self.lstm_proj(lstm_out)                     # (B, 64)
            fused = gate * lstm_p + (1 - gate) * gcn_p            # (B, 64)
            pred = self.gate_output(fused)                        # (B, N_commodities)

        if debug:
            if self.fusion_mode == 'gate':
                gm = gate.mean().item()
                print(f"  GCN→({list(gcn_out.shape)}) || LSTM→({list(lstm_out.shape)})"
                      f" → Gate(mean={gm:.3f}) → FC → ({list(pred.shape)})")
            elif self.fusion_mode == 'mixhop':
                gm = gate.mean().item()
                nz = (self.graph_learner().detach() > 0).sum().item()
                print(f"  MixHop→({list(gcn_out.shape)}) || LSTM→({list(lstm_out.shape)})"
                      f" → Gate(mean={gm:.3f}, A_nz={nz}) → FC → ({list(pred.shape)})")
            else:
                print(f"  GCN→({list(gcn_out.shape)}) || LSTM→({list(lstm_out.shape)})"
                      f" → Concat({list(combined.shape)}) → FC → ({list(pred.shape)})")

        return pred

    def get_gate_stats(self, x: torch.Tensor,
                       edge_index: torch.Tensor = None,
                       edge_weight: torch.Tensor = None) -> dict:
        """Return gating statistics for a batch — monitors GCN vs LSTM contribution."""
        if self.fusion_mode not in ('gate', 'mixhop'):
            return {'mode': self.fusion_mode, 'note': 'not using gated fusion'}
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, num_nodes, _ = x.shape

            if self.fusion_mode == 'mixhop':
                A = self.graph_learner()
                x_gcn = x.mean(dim=0).squeeze(-1).T
                h1 = F.relu(self.mixhop1(x_gcn, A))
                h2 = self.mixhop2(h1, A)
                h = self.gcn_norm(h2)
                gcn_out = h.unsqueeze(0).expand(batch_size, -1, -1)
                gcn_out = gcn_out[:, :self.n_commodities, :].mean(dim=1)
            else:
                temporal_outputs = []
                for t in range(seq_len):
                    x_t = x[:, t, :, :]
                    out_t = self.spatial(x_t, edge_index, batch_size, num_nodes)
                    temporal_outputs.append(out_t)
                gcn_stack = torch.stack(temporal_outputs, dim=1)
                gcn_pooled = gcn_stack.mean(dim=1)
                gcn_out = self.gcn_reduce(gcn_pooled)

            x_seq = x.squeeze(-1)
            lstm_out = self.temporal(x_seq)
            combined = torch.cat([gcn_out, lstm_out], dim=-1)
            gate = torch.sigmoid(self.gate_fc(combined))

        result = {
            'gate_mean': gate.mean().item(),
            'gate_std':  gate.std().item(),
            'gate_min':  gate.min().item(),
            'gate_max':  gate.max().item(),
        }
        if self.fusion_mode == 'mixhop':
            result['mixhop_diff'] = (h1 - h2).norm().item()
        return result
