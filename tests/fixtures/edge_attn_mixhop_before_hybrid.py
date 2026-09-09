"""Frozen pre-change reference for bitwise legacy regression tests."""
import math
import torch
from torch import nn
from torch.nn import functional as F
LSTM_HIDDEN_DIM = 64

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
                 prior_scale: float = 1.0, self_heads: int = 4,
                 cross_mask: torch.Tensor = None):
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
        # Hybrid attention: first self_heads heads are unrestricted
        # (full graph), remaining heads are restricted to pairs allowed
        # by cross_mask (directed cross-market information flow).
        self.self_heads = self_heads
        assert 0 <= self_heads <= n_heads, "self_heads must be within [0, n_heads]"
        if cross_mask is not None:
            self.register_buffer('cross_mask', cross_mask.bool())
        else:
            self.cross_mask = None
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
        # Support both single-graph (N, d) and per-sample (B, N, d) inputs.
        batched = x.dim() == 3
        B = x.size(0) if batched else None
        # Clamp negatives (Pearson negative-correlation edges) to zero
        A_pos = A.clamp(min=0)
        # Soft logit prior (learned [0,1] adjacency) or hard structure mask
        log_prior = torch.log(A_pos + 1e-6)                    # (N, N)
        H = x
        H_in = x
        out = self.Ws[0](H)

        for k in range(1, self.K + 1):
            if batched:
                # Per-sample multi-head attention scores
                Q = self.q(H).view(B, N, self.n_heads, self.head_dim)   # (B, N, h, d)
                Kt = self.k(H).view(B, N, self.n_heads, self.head_dim)
                V  = self.v(H).view(B, N, self.n_heads, self.head_dim)
                e = torch.einsum('bnhd,bmhd->bnmh', Q, Kt) / math.sqrt(self.head_dim)  # (B, N, M, h)
                if self.hard_mask:
                    e = e.masked_fill(
                        (A_pos <= 1e-4).unsqueeze(0).unsqueeze(-1), -1e9)
                else:
                    e = e + (self.prior_scale * log_prior).unsqueeze(0).unsqueeze(-1)
                if self.cross_mask is not None and self.self_heads < self.n_heads:
                    e[:, :, :, self.self_heads:] = e[:, :, :, self.self_heads:].masked_fill(
                        ~self.cross_mask.unsqueeze(0).unsqueeze(-1), -1e9)
                alpha = F.softmax(e, dim=2)                    # over source nodes
                alpha = self.attn_drop(alpha)
                agg = torch.einsum('bnmh,bmhd->bnhd', alpha, V)   # (B, N, h, d)
                agg = agg.reshape(B, N, -1)                    # (B, N, h*d)
            else:
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
                if self.cross_mask is not None and self.self_heads < self.n_heads:
                    # Cross heads (after self_heads) restricted to allowed pairs
                    e[:, :, self.self_heads:] = e[:, :, self.self_heads:].masked_fill(
                        ~self.cross_mask.unsqueeze(-1), -1e9)
                alpha = F.softmax(e, dim=1)                    # over source nodes
                alpha = self.attn_drop(alpha)
                agg = torch.einsum('nmh,mhd->nhd', alpha, V)   # (N, h, d)
                agg = agg.reshape(N, -1)                       # (N, h*d)
            agg = self.out_proj(agg)                           # (N or B, N, out)

            # MixHop-style update with residual beta
            H = self.beta * H_in + (1 - self.beta) * agg
            out = out + self.Ws[k](H)

        return out
