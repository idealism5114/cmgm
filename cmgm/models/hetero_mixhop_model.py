"""
HeteroMixHop — 21-dim features + per-type projection + MixHop + gated fusion.

Supports ablation variants for the paper:
  variant="full"            — complete model (all components)
  variant="edge_attn"       — EdgeAttnMixHop: content-aware edge attention
                              (structure prior A masks attention logits)
  variant="edge_attn_static" — EdgeAttnMixHop + static Pearson structure
                              (structure prior = static graph, weights dynamic)
  variant="temporal_attn"   — A+B temporal branch: per-type per-timestep
                              compression (N*F → 3×64) → LSTM(192→64) →
                              temporal attention (last + mean + attn pool)
  variant="no_type_proj"    — shared input projection (no per-type)
  variant="no_learn_graph"  — static Pearson graph (no adaptive learner)
  variant="no_mixhop"       — single-hop GCN (no MixHop multi-hop)
  variant="no_gate"         — concat fusion (no gated fusion)
  variant="gcn_only"        — LSTM branch removed
  variant="lstm_only"       — spatial branch removed
  variant="single_horizon"  — predict only the primary horizon (no multi-task)
  variant="feat7"           — 7-dim features instead of 21 (passed via feat_dim)

Architecture (full):
    21-dim features (price, returns, vol, zscore, MA ratios, RSI, BB, skew, ...)
    → Per-type Input Projection (21→64 per market)
    → AdaptiveGraphLearner → A (N×N)
    → MixHopPropagation × 2 (64→64, 64→64)
    → Per-type mean pooling → type_agg(192→64)
    → Gated fusion with LSTM(284×21→64)
    → Multi-horizon head (1d/5d/10d/20d)

Architecture (edge_attn):
    ...same up to A...
    → EdgeAttnMixHop × 2 (64→64): content-aware multi-head edge attention
      e_ij = (Q_i·K_j)/√d + log(A_ij + ε)   ← A as structure prior
      α_ij = softmax_j(e_ij),  V_agg = α @ V
      H = β·H_in + (1−β)·V_agg (MixHop-style update)
"""

import math
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
    """Per-type linear projection: feat_dim → hidden_dim."""

    def __init__(self, feat_dim: int = FEATURE_DIM,
                 hidden_dim: int = LSTM_HIDDEN_DIM):
        super().__init__()
        self.stock_proj  = nn.Linear(feat_dim, hidden_dim)
        self.bond_proj   = nn.Linear(feat_dim, hidden_dim)
        self.future_proj = nn.Linear(feat_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor,
                n_stock: int, n_bond: int) -> torch.Tensor:
        s = self.stock_proj(x_t[:, :n_stock, :])          # (B, n_stock, H)
        b = self.bond_proj(x_t[:, n_stock:n_stock+n_bond, :])
        f = self.future_proj(x_t[:, n_stock+n_bond:, :])
        return torch.cat([s, b, f], dim=1)                 # (B, N, H)


class _SingleHopGCN(nn.Module):
    """One-hop graph convolution on dense adjacency (ablation for MixHop)."""

    def __init__(self, hidden_dim: int = LSTM_HIDDEN_DIM):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_hat = A + torch.eye(A.size(0), device=A.device)
        deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_norm = A_hat / deg
        return F.relu(self.norm(self.lin(A_norm @ x)))


class EdgeAttnMixHop(nn.Module):
    """
    MixHop propagation with content-aware multi-head edge attention.

    For each hop k:
      1. Attention scores:  e_ij = (Q_i · K_j) / √d_k  +  log(A_ij + ε)
         — A (from AdaptiveGraphLearner) acts as a structure prior in logit space
      2. Normalize:         α_ij = softmax_j(e_ij)
      3. Aggregate:         V_agg = α @ V
      4. MixHop update:     H = β · H_in + (1 − β) · V_agg
      5. Selection:         out += W_k(H)

    The graph structure (who connects to whom) comes from A;
    the edge weights (how much to aggregate) are learned content-aware.
    """

    def __init__(self, in_dim: int = LSTM_HIDDEN_DIM,
                 out_dim: int = LSTM_HIDDEN_DIM,
                 K: int = 2, beta: float = 0.05, n_heads: int = 4,
                 dropout: float = 0.1, hard_mask: bool = False,
                 prior_scale: float = 1.0):
        super().__init__()
        self.K = K
        self.beta = beta
        self.n_heads = n_heads
        # hard_mask=True: attention softmax over structure neighbors only
        # (A > threshold).  Use for static sparse graphs whose tiny edge
        # weights would drown as log-priors (e.g. normalized Pearson).
        # hard_mask=False: A acts as a soft logit prior (works well when
        # A is a learned [0,1] adjacency like AdaptiveGraphLearner's).
        self.hard_mask = hard_mask
        # prior_scale: strength of the structure prior in logit space
        #   e_ij = content_score + prior_scale · log(A_ij + ε)
        self.prior_scale = prior_scale
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.head_dim = out_dim // n_heads

        # MixHop selection weights (same structure as MixHopPropagation)
        self.Ws = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(K + 1)
        ])

        # Multi-head attention projections
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N = A.size(0)
        # Clamp negatives (Pearson negative-correlation edges) to zero
        A_pos = A.clamp(min=0)
        # Soft logit prior (learned [0,1] adjacency) or hard structure mask
        log_prior = torch.log(A_pos + 1e-6)                    # (N, N)
        H = x
        H_in = x
        out = self.Ws[0](H)

        for k in range(1, self.K + 1):
            # Multi-head attention scores on current H
            Q = self.q(H).view(N, self.n_heads, self.head_dim)     # (N, h, d)
            Kt = self.k(H).view(N, self.n_heads, self.head_dim)
            V  = self.v(H).view(N, self.n_heads, self.head_dim)
            e = torch.einsum('nhd,mhd->nmh', Q, Kt) / math.sqrt(self.head_dim)  # (N, M, h)
            if self.hard_mask:
                # Attention over structure neighbors ONLY — softmax never
                # sees non-edges, so tiny static edge weights can't drown
                # the prior.
                e = e.masked_fill((A_pos <= 1e-4).unsqueeze(-1), -1e9)
            else:
                e = e + self.prior_scale * log_prior.unsqueeze(-1)   # structure prior (N, M, 1)
            alpha = F.softmax(e, dim=1)                        # over source nodes
            alpha = self.attn_drop(alpha)
            agg = torch.einsum('nmh,mhd->nhd', alpha, V)       # (N, h, d)
            agg = agg.reshape(N, -1)                           # (N, h*d)
            agg = self.out_proj(agg)                           # (N, out)

            # MixHop-style update with residual beta
            H = self.beta * H_in + (1 - self.beta) * agg
            out = out + self.Ws[k](H)

        return out


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
    Heterogeneous MixHop CMGM with ablation support.

    Forward:  (B, T, N, F) → (B, n_horizons, N_commodities)  [multi]
              (B, T, N, F) → (B, N_commodities)              [single]
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 n_stock: int = 248, n_bond: int = 12,
                 variant: str = "full", feat_dim: int = FEATURE_DIM,
                 attn_heads: int = 8, attn_dropout: float = 0.1,
                 attn_prior_scale: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.n_stock = n_stock
        self.n_bond = n_bond
        self.variant = variant
        self.feat_dim = feat_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.attn_prior_scale = attn_prior_scale

        # ── Branch switches ──
        self.use_gcn  = variant != "lstm_only"
        self.use_lstm = variant != "gcn_only"
        self.use_learn_graph = variant not in ("no_learn_graph", "edge_attn_static")
        self.use_type_proj   = variant != "no_type_proj"
        self.use_mixhop      = variant not in ("no_mixhop", "edge_attn", "edge_attn_static",
                                               "temporal_attn", "diff_input")
        self.use_edge_attn   = variant in ("edge_attn", "edge_attn_static", "temporal_attn",
                                           "diff_input")
        self.use_gate        = variant not in ("no_gate", "gcn_only", "lstm_only")

        # ── Multi-horizon output ──
        use_multi = (TARGET_TYPE == "return") and variant != "single_horizon"
        self.n_horizons = len(MULTI_HORIZONS) if use_multi else 1
        out_dim = self.n_horizons * n_commodities

        # ── Spatial branch ──
        if self.use_gcn:
            # Learnable graph (default) or static buffer (no_learn_graph)
            if self.use_learn_graph:
                self.graph_learner = AdaptiveGraphLearner(
                    num_nodes, embed_dim=10, alpha=0.5, top_k=10,
                )
            else:
                self.register_buffer('static_A', torch.zeros(num_nodes, num_nodes))

            # Per-type (default) or shared input projection
            if self.use_type_proj:
                self.type_proj = _TypeInputProjection(feat_dim, LSTM_HIDDEN_DIM)
            else:
                self.shared_proj = nn.Linear(feat_dim, LSTM_HIDDEN_DIM)

            # MixHop (default) / EdgeAttnMixHop (edge_attn) / single-hop GCN
            if self.use_mixhop:
                self.mixhop1 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
                self.mixhop2 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
            elif self.use_edge_attn:
                hard = (variant == "edge_attn_static")
                self.attn_mixhop1 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05,
                    n_heads=self.attn_heads, dropout=self.attn_dropout,
                    hard_mask=hard, prior_scale=self.attn_prior_scale)
                self.attn_mixhop2 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05,
                    n_heads=self.attn_heads, dropout=self.attn_dropout,
                    hard_mask=hard, prior_scale=self.attn_prior_scale)
            else:
                self.singlehop1 = _SingleHopGCN(LSTM_HIDDEN_DIM)
                self.singlehop2 = _SingleHopGCN(LSTM_HIDDEN_DIM)

            self.gcn_norm = nn.LayerNorm(LSTM_HIDDEN_DIM)
            self.type_pool = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)

        # ── Temporal branch ──
        if self.use_lstm:
            if variant == "temporal_attn":
                # A: per-type temporal compression (each market block → 64)
                n_future = num_nodes - n_stock - n_bond
                self.stock_compress   = nn.Linear(n_stock * feat_dim, LSTM_HIDDEN_DIM)
                self.bond_compress    = nn.Linear(n_bond * feat_dim, LSTM_HIDDEN_DIM)
                self.future_compress  = nn.Linear(n_future * feat_dim, LSTM_HIDDEN_DIM)
                # LSTM over compact market-state sequence (192 → 64)
                self.temporal = nn.LSTM(
                    input_size=LSTM_HIDDEN_DIM * 3,
                    hidden_size=LSTM_HIDDEN_DIM,
                    num_layers=LSTM_NUM_LAYERS,
                    dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                    batch_first=True,
                )
                # B: temporal attention — fuse last-state, mean-pool, attn-pool
                self.temporal_attn = nn.Linear(LSTM_HIDDEN_DIM, 1)
                self.temporal_fuse = nn.Linear(LSTM_HIDDEN_DIM * 3, LSTM_HIDDEN_DIM)
            else:
                # diff_input: concat first-order differences → 2× input size
                lstm_in = num_nodes * feat_dim
                if variant == "diff_input":
                    lstm_in *= 2
                self.temporal = nn.LSTM(
                    input_size=lstm_in,
                    hidden_size=LSTM_HIDDEN_DIM,
                    num_layers=LSTM_NUM_LAYERS,
                    dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                    batch_first=True,
                )
                # Multi-scale temporal pooling: fuse last-state with
                # mean-pool over full / 10-step / 5-step windows
                if variant == "multiscale_time":
                    self.ms_fuse = nn.Linear(LSTM_HIDDEN_DIM * 4, LSTM_HIDDEN_DIM)

        # ── Fusion ──
        if self.use_gate:
            self.gate_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)
            self.gcn_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.lstm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
        elif variant == "no_gate":
            self.concat_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)

        # ── Output head (shared across all variants) ──
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, out_dim),
        )

        # Horizon-aligned heads: each horizon has its own output head,
        # fed by its own temporal context window (only in multi-horizon mode)
        if variant == "horizon_align" and self.n_horizons > 1:
            self.ha_horizons = MULTI_HORIZONS
            self.ha_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(GCN_DROPOUT),
                    nn.Linear(FC_HIDDEN_DIM, n_commodities),
                )
                for _ in range(self.n_horizons)
            ])

    def _spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """MixHop / single-hop branch → (B, 64)."""
        N, T = x.size(2), x.size(1)

        if self.use_learn_graph:
            A = self.graph_learner()                                   # (N, N)
        else:
            A = self.static_A

        # Average over batch → per-node features → per-type projection
        x_gcn = x.mean(dim=0)                                          # (T, N, F)
        x_gcn = x_gcn.permute(1, 0, 2)                                 # (N, T, F)
        if self.use_type_proj:
            x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)  # (N, T, 64)
        else:
            x_proj = self.shared_proj(x_gcn)                           # (N, T, 64)
        x_proj = x_proj.mean(dim=1)                                    # (N, 64)

        if self.use_mixhop:
            h1 = F.relu(self.mixhop1(x_proj, A))                       # (N, 64)
            h2 = self.mixhop2(h1, A)                                   # (N, 64)
        elif self.use_edge_attn:
            h1 = F.relu(self.attn_mixhop1(x_proj, A))                  # (N, 64)
            h2 = self.attn_mixhop2(h1, A)                              # (N, 64)
        else:
            h1 = self.singlehop1(x_proj, A)                            # (N, 64)
            h2 = self.singlehop2(h1, A)                                # (N, 64)
        h = self.gcn_norm(h2)                                          # (N, 64)

        # Per-type pooling → expand to batch
        gcn_out = self.type_pool(h).unsqueeze(0).expand(x.size(0), -1) # (B, 64)
        return gcn_out

    def _temporal_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A+B temporal branch:
          per-timestep per-type compression (N*F → 3×64)
          → LSTM over (B, T, 192) → (B, T, 64)
          → fuse last-state / mean-pool / attention-pool → (B, 64)
        """
        B, T, N, n_feat = x.shape
        n_stock, n_bond = self.n_stock, self.n_bond

        # A: per-type compression per timestep
        seqs = []
        for t in range(T):
            xt = x[:, t, :, :]                                        # (B, N, F)
            s = self.stock_compress(xt[:, :n_stock].reshape(B, -1))   # (B, 64)
            b = self.bond_compress(
                xt[:, n_stock:n_stock + n_bond].reshape(B, -1))       # (B, 64)
            f = self.future_compress(
                xt[:, n_stock + n_bond:].reshape(B, -1))              # (B, 64)
            seqs.append(torch.cat([s, b, f], dim=-1))                 # (B, 192)
        x_seq = torch.stack(seqs, dim=1)                              # (B, T, 192)

        # LSTM over compact market-state sequence
        out, _ = self.temporal(x_seq)                                 # (B, T, 64)

        # B: temporal attention pooling
        h_last = out[:, -1, :]                                        # (B, 64)
        h_mean = out.mean(dim=1)                                      # (B, 64)
        scores = self.temporal_attn(out).squeeze(-1)                  # (B, T)
        alpha = F.softmax(scores, dim=1)                              # (B, T)
        h_attn = (out * alpha.unsqueeze(-1)).sum(dim=1)               # (B, 64)
        return F.relu(self.temporal_fuse(
            torch.cat([h_last, h_mean, h_attn], dim=-1)))             # (B, 64)

    def _horizon_align_predict(self, out_seq: torch.Tensor,
                               gcn_out: torch.Tensor) -> torch.Tensor:
        """
        Horizon-aligned output: each horizon h pools a context window of
        min(h, T) steps from the LSTM output (1d→1 step, 5d→5, 10d→10,
        20d→20), fuses it with the spatial representation via the shared
        gate, and predicts with its own head.

        Returns: (B, n_horizons, n_commodities)
        """
        B = out_seq.size(0)
        T_seq = out_seq.size(1)
        preds = []
        for i, h in enumerate(self.ha_horizons):
            w = min(h, T_seq)
            ctx = out_seq[:, -w:, :].mean(dim=1)                      # (B, 64)
            combined = torch.cat([gcn_out, ctx], dim=-1)              # (B, 128)
            gate = torch.sigmoid(self.gate_fc(combined))
            fused = gate * self.lstm_proj(ctx) + (1 - gate) * self.gcn_proj(gcn_out)
            preds.append(self.ha_heads[i](fused))                     # (B, Nc)
        return torch.stack(preds, dim=1)                              # (B, H, Nc)

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_weight=None,
                debug: bool = False) -> torch.Tensor:
        B, T, N, n_feat = x.shape

        # ── 1. Spatial branch (MixHop / single-hop GCN) ──
        gcn_out = self._spatial_forward(x) if self.use_gcn else None   # (B, 64)

        # ── 2. Temporal branch (LSTM) ──
        if self.use_lstm:
            if self.variant == "temporal_attn":
                lstm_out = self._temporal_attn_forward(x)             # (B, 64)
            else:
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                if self.variant == "diff_input":
                    # First-order difference stream (Δx_t = x_t − x_{t−1})
                    x_diff = torch.zeros_like(x_seq)
                    x_diff[:, 1:] = x_seq[:, 1:] - x_seq[:, :-1]
                    x_seq = torch.cat([x_seq, x_diff], dim=-1)        # (B, T, 2·N*F)
                lstm_out_seq, (h_n, _) = self.temporal(x_seq)         # (B, T, 64)
                if self.variant == "multiscale_time":
                    # Multi-scale temporal pooling:
                    # last-state + full-window + 10-step + 5-step means
                    h_last = lstm_out_seq[:, -1, :]                   # (B, 64)
                    h_all  = lstm_out_seq.mean(dim=1)                 # (B, 64)
                    h_10   = lstm_out_seq[:, -10:, :].mean(dim=1)     # (B, 64)
                    h_5    = lstm_out_seq[:, -5:, :].mean(dim=1)      # (B, 64)
                    lstm_out = F.relu(self.ms_fuse(
                        torch.cat([h_last, h_all, h_10, h_5], dim=-1)))  # (B, 64)
                elif self.variant == "horizon_align" and self.n_horizons > 1:
                    lstm_out = lstm_out_seq                           # keep full sequence
                else:
                    lstm_out = h_n[-1]                                # (B, 64)
        else:
            lstm_out = None

        # ── 3. Fusion ──
        if self.variant == "horizon_align" and self.n_horizons > 1:
            fused = None  # handled per-horizon in _horizon_align_predict
        elif self.variant == "gcn_only":
            fused = gcn_out
        elif self.variant == "lstm_only":
            fused = lstm_out
        elif self.variant == "no_gate":
            combined = torch.cat([gcn_out, lstm_out], dim=-1)          # (B, 128)
            fused = F.relu(self.concat_fc(combined))                   # (B, 64)
        else:
            combined = torch.cat([gcn_out, lstm_out], dim=-1)          # (B, 128)
            gate = torch.sigmoid(self.gate_fc(combined))
            gcn_p  = self.gcn_proj(gcn_out)
            lstm_p = self.lstm_proj(lstm_out)
            fused = gate * lstm_p + (1 - gate) * gcn_p                 # (B, 64)

        # ── 4. Output head ──
        if self.variant == "horizon_align" and self.n_horizons > 1:
            pred = self._horizon_align_predict(lstm_out, gcn_out)      # (B, H, Nc)
        else:
            pred = self.head(fused)                                    # (B, out_dim)
            if self.n_horizons > 1:
                pred = pred.view(B, self.n_horizons, self.n_commodities)   # (B, H, Nc)
            else:
                pred = pred.view(B, self.n_commodities)                # (B, Nc)

        if debug:
            print(f"  [variant={self.variant}] "
                  f"GCN→({list(gcn_out.shape) if gcn_out is not None else None}) || "
                  f"LSTM→({list(lstm_out.shape) if lstm_out is not None else None}) "
                  f"→ head → ({list(pred.shape)})")

        return pred

    def get_gate_stats(self, x, edge_index=None, edge_weight=None):
        """Return gating statistics (only meaningful for gate variants)."""
        if not self.use_gate:
            return {'mode': self.variant, 'note': 'not using gated fusion'}
        self.eval()
        with torch.no_grad():
            A = self.graph_learner() if self.use_learn_graph else self.static_A
            x_gcn = x.mean(dim=0).permute(1, 0, 2)
            if self.use_type_proj:
                x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)
            else:
                x_proj = self.shared_proj(x_gcn)
            x_proj = x_proj.mean(dim=1)
            if self.use_mixhop:
                h1 = F.relu(self.mixhop1(x_proj, A))
                h2 = self.mixhop2(h1, A)
            elif self.use_edge_attn:
                h1 = F.relu(self.attn_mixhop1(x_proj, A))
                h2 = self.attn_mixhop2(h1, A)
            else:
                h1 = self.singlehop1(x_proj, A)
                h2 = self.singlehop2(h1, A)
            h = self.gcn_norm(h2)
            B = x.size(0)
            gcn_out = self.type_pool(h).unsqueeze(0).expand(B, -1)

            if self.variant == "temporal_attn":
                lstm_out = self._temporal_attn_forward(x)
            else:
                x_seq = x.reshape(B, x.size(1), -1)
                if self.variant == "diff_input":
                    x_diff = torch.zeros_like(x_seq)
                    x_diff[:, 1:] = x_seq[:, 1:] - x_seq[:, :-1]
                    x_seq = torch.cat([x_seq, x_diff], dim=-1)
                lstm_out_seq, (h_n, _) = self.temporal(x_seq)
                if self.variant == "multiscale_time":
                    h_last = lstm_out_seq[:, -1, :]
                    h_all  = lstm_out_seq.mean(dim=1)
                    h_10   = lstm_out_seq[:, -10:, :].mean(dim=1)
                    h_5    = lstm_out_seq[:, -5:, :].mean(dim=1)
                    lstm_out = F.relu(self.ms_fuse(
                        torch.cat([h_last, h_all, h_10, h_5], dim=-1)))
                elif self.variant == "horizon_align" and self.n_horizons > 1:
                    # use primary-horizon window (5d → 5 steps) for gate stats
                    w = min(5, lstm_out_seq.size(1))
                    lstm_out = lstm_out_seq[:, -w:, :].mean(dim=1)
                else:
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
