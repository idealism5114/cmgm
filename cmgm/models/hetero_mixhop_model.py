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
    FC_HIDDEN_DIM, FEATURE_DIM, MULTI_HORIZONS, TARGET_TYPE, SEQ_LEN,
)
from cmgm.models.model import MixHopPropagation
from cmgm.graph.adaptive_graph import AdaptiveGraphLearner
from cmgm.models.regime_dynamic import RegimeDynamicRPETransformer
from cmgm.models.switching_transformer import (
    MarketTokenTransformerBranch,
    SwitchingFilterRPEBranch,
    SwitchingTransformerBranch,
)
from cmgm.models.switching_latent_transformer import (
    SwitchingLatentTransformerBranch,
)


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


class _CausalConv1d(nn.Module):
    """Causal 1D convolution — no future leakage."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)                       # (B, C, T + pad)
        if self.pad > 0:
            out = out[:, :, :-self.pad]          # causal: drop future
        return out


class _MultiScaleTCN(nn.Module):
    """
    Multi-scale temporal convolution — replaces the LSTM branch.

      x (B, T, in_dim) → shared projection (5964→256) per timestep
      → 3 parallel causal-conv branches (kernels 3/5/7, dilations 1,2)
      → fuse(3×64→64) → (B, T, 64)

    Kernel sizes give the temporal branch an explicit multi-scale
    inductive bias (short/medium/long local patterns), mirroring the
    multi-scale pooling that proved effective in multiscale_time.
    """

    def __init__(self, in_dim: int, hidden: int = 256,
                 out_dim: int = LSTM_HIDDEN_DIM,
                 kernels: tuple = (3, 5, 7), dilations: tuple = (1, 2)):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)         # per-timestep projection
        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                _CausalConv1d(hidden, out_dim, k, dilations[0]),
                nn.ReLU(),
                _CausalConv1d(out_dim, out_dim, k, dilations[1]),
                nn.ReLU(),
            )
            self.branches.append(branch)
        self.fuse = nn.Linear(len(kernels) * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        h = F.relu(self.proj(x))                      # (B, T, hidden)
        h = h.permute(0, 2, 1)                        # (B, hidden, T)
        outs = [b(h) for b in self.branches]          # each (B, out, T)
        merged = torch.cat(outs, dim=1).permute(0, 2, 1)   # (B, T, 3·out)
        out = self.norm(self.fuse(merged))            # (B, T, out)
        return out


class _PatchTemporal(nn.Module):
    """
    PatchTST-style temporal branch (ICLR 2023).

      x (B, T, in_dim) → shared per-timestep projection (5964→256)
      → non-overlapping patches of patch_len steps (aligned with the
        primary 5d horizon: T=20 → 4 patches)
      → patch embedding (patch_len·256 → d_model)
      → small Transformer encoder over the patch tokens
      → mean-pool → (B, 64)

    Patchification is a structural multi-scale inductive bias: each token
    aggregates 5 consecutive days (≈ a week), and attention mixes the
    4 weekly states — mirroring the multi-scale idea that proved
    effective in multiscale_time, but learned.
    """

    def __init__(self, in_dim: int, t_len: int = 20, patch_len: int = 5,
                 proj_dim: int = 256, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        assert t_len % patch_len == 0, "T must be divisible by patch_len"
        self.patch_len = patch_len
        self.n_patches = t_len // patch_len

        self.proj = nn.Linear(in_dim, proj_dim)                    # per-timestep
        self.patch_embed = nn.Linear(patch_len * proj_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, LSTM_HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        B = x.size(0)
        h = F.relu(self.proj(x))                                   # (B, T, proj_dim)
        h = h.view(B, self.n_patches, self.patch_len * h.size(-1)) # (B, n_patch, pl·D)
        tok = F.relu(self.patch_embed(h))                          # (B, n_patch, d_model)
        tok = tok + self.pos_enc
        out = self.encoder(tok)                                    # (B, n_patch, d_model)
        out = self.norm(out.mean(dim=1))                           # (B, d_model)
        return self.fc(out)                                        # (B, 64)


class _MambaTemporal(nn.Module):
    """
    Mamba (SSM) temporal branch.

      x (B, T, in_dim) → projection (5964→128) → Mamba block → mean-pool
      → Linear(128→64)

    Requires the `mamba-ssm` package (Linux + CUDA only).  Import is
    deferred and optional — construction raises a clear error if missing.
    """

    def __init__(self, in_dim: int, d_model: int = 128, d_state: int = 16):
        super().__init__()
        # Priority: official CUDA impl (Linux) → mamba.py → vendored impl
        try:
            from mamba_ssm import Mamba
        except ImportError:
            try:
                from mamba import Mamba
            except ImportError:
                from cmgm.models.mamba_impl import Mamba
        self.proj = nn.Linear(in_dim, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, LSTM_HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))                       # (B, T, d_model)
        out = self.mamba(h)                            # (B, T, d_model)
        out = self.norm(out.mean(dim=1))               # (B, d_model)
        return self.fc(out)                            # (B, 64)


class _ProbSparseAttention(nn.Module):
    """
    Informer-style ProbSparse self-attention.

    Full attention with the Informer query-selection formulation:
      M(q_i, K) = max_j(q_i·k_j/√d) − mean_j(q_i·k_j/√d)
    and softmax over the full key set.  With T=20 the sampling in the
    original paper degenerates to full attention — the module keeps the
    Informer structure for paper narrative, at negligible cost.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B,H,T,T)
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class _InformerTemporal(nn.Module):
    """
    Informer-style temporal branch (encoder part).

      x (B, T, in_dim) → projection (5964→128) + positional encoding
      → 2 × [ProbSparse attention + FFN] (with LayerNorm, residual)
      → mean-pool → Linear(128→64)
    """

    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, t_len: int = 20):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, t_len, d_model) * 0.02)
        self.attns = nn.ModuleList([
            _ProbSparseAttention(d_model, n_heads) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(),
                          nn.Linear(d_model * 2, d_model)) for _ in range(n_layers)])
        self.norm1s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, LSTM_HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x)) + self.pos                  # (B, T, d_model)
        for attn, ffn, n1, n2 in zip(self.attns, self.ffns, self.norm1s, self.norm2s):
            h = n1(h + attn(h))
            h = n2(h + ffn(h))
        out = self.norm(h.mean(dim=1))                       # (B, d_model)
        return self.fc(out)                                  # (B, 64)


class _RPEAttnLayer(nn.Module):
    """Transformer encoder layer with additive relative-position bias."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(),
                                 nn.Linear(ffn_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rpe: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, d); rpe: (B, n_heads, T, T) additive bias (optional)
        B, T, d = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q(x).view(B, T, H, hd).transpose(1, 2)      # (B, H, T, hd)
        k = self.k(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v(x).view(B, T, H, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)     # (B, H, T, T)
        if rpe is not None:
            scores = scores + rpe
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        out = self.out(out)
        x = self.norm1(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class _SoftRegimeGenerator(nn.Module):
    """
    Causal soft market regime generator:
      p_t = softmax(MLP([E_t, d_t, p_{t-1}])),  K latent regimes.

      d_t = E_t − E_{t−1} (d_0 = 0); p_0 = softmax(learnable logits).
      p_t depends ONLY on time ≤ t — no future leakage.
    """

    def __init__(self, d_model: int = 128, K: int = 3):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2 + K, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )
        self.init_logits = nn.Parameter(torch.zeros(K))

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, T, d_model) → p: (B, T, K)
        B, T, d = E.shape
        d_t = torch.zeros_like(E)
        d_t[:, 1:] = E[:, 1:] - E[:, :-1]                     # dynamic slope
        p_prev = F.softmax(self.init_logits, dim=-1)          # (K,)
        ps = []
        for t in range(T):
            r = self.mlp(torch.cat([E[:, t], d_t[:, t],
                                    p_prev.unsqueeze(0).expand(B, -1)], dim=-1))
            p_t = F.softmax(r, dim=-1)                        # (B, K)
            ps.append(p_t)
            p_prev = p_t.mean(dim=0).detach()                 # carry prev state
        return torch.stack(ps, dim=1)                         # (B, T, K)


class _PrototypeRegimeGenerator(nn.Module):
    """
    Improved soft regime generator: Shared Context + K State-specific
    Prototypes (anti-collapse design).

      context_t = context_encoder([E_t, E_t − E_{t−1}])        (B, T, D_r)
      sim_tk    = context_t · prototype_k / √D_r               (B, T, K)
      trans_tk  = p_{t−1} @ transition_matrix                  (B, K)
      score_tk  = sim_tk + trans_tk
      p_t       = softmax(score_tk / temperature)              temperature = 1.0

    Causal: p_t depends only on E_t, E_{t−1}, p_{t−1}.
    """

    def __init__(self, d_model: int = 128, K: int = 3, D_r: int = 32):
        super().__init__()
        self.K = K
        self.D_r = D_r
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model * 2, D_r),
            nn.ReLU(),
        )
        self.regime_prototypes = nn.Parameter(torch.randn(K, D_r))
        self.transition_matrix = nn.Parameter(torch.zeros(K, K))
        self.temperature = 1.0

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, T, d_model) → p: (B, T, K)
        B, T, d = E.shape
        d_t = torch.zeros_like(E)
        d_t[:, 1:] = E[:, 1:] - E[:, :-1]                        # dynamic slope
        context = self.context_encoder(torch.cat([E, d_t], dim=-1))  # (B, T, D_r)
        sim = context @ self.regime_prototypes.T / (self.D_r ** 0.5)  # (B, T, K)

        p_prev = torch.full((B, self.K), 1.0 / self.K, device=E.device)
        ps = []
        for t in range(T):
            trans = p_prev @ self.transition_matrix               # (B, K)
            score = sim[:, t] + trans
            p_t = F.softmax(score / self.temperature, dim=-1)     # (B, K)
            ps.append(p_t)
            p_prev = p_t
        return torch.stack(ps, dim=1)                             # (B, T, K)

    def prototype_cosine(self) -> torch.Tensor:
        """Mean pairwise cosine similarity across prototypes (diagnostic)."""
        P = F.normalize(self.regime_prototypes, dim=-1)           # (K, D_r)
        cos = P @ P.T
        mask = ~torch.eye(self.K, dtype=torch.bool, device=cos.device)
        return cos[mask].mean()


class _TemporalTransformerBranch(nn.Module):
    """
    Temporal branch: (B, T, in_dim) → temporal transformer → (B, 64).

    Config switches (for ablation A/B/C/D/E):
      use_rpe        — standard relative position encoding (C)
      use_regime     — soft regime as input feature context (D)
      use_regime_rpe — regime-conditioned RPE (E, full model)
    """

    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256, dropout: float = 0.1,
                 t_len: int = 20, K: int = 3, use_rpe: bool = False,
                 use_regime: bool = False, use_regime_rpe: bool = False,
                 use_prototype_regime: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            _RPEAttnLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)])
        self.pool_score = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, 64)
        self.t_len = t_len
        self.K = K
        self.use_rpe = use_rpe
        self.use_regime = use_regime
        self.use_regime_rpe = use_regime_rpe
        if use_rpe or use_regime_rpe:
            self.base_rpe = nn.Parameter(
                torch.randn(2 * t_len - 1, n_heads) * 0.02)
        if use_regime or use_regime_rpe:
            if use_prototype_regime:
                self.regime_gen = _PrototypeRegimeGenerator(d_model, K)
            else:
                self.regime_gen = _SoftRegimeGenerator(d_model, K)
        if use_regime_rpe:
            self.regime_rpe = nn.Parameter(
                torch.randn(K, 2 * t_len - 1, n_heads) * 0.02)
        if use_regime and not use_regime_rpe:
            self.regime_proj = nn.Linear(K, d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, in_dim) → h_temporal (B, 64)
        B, T, _ = x_seq.shape
        E = self.proj(x_seq)                                   # (B, T, 128)

        # ── Soft regime (causal) ──
        p = None
        if self.use_regime or self.use_regime_rpe:
            p = self.regime_gen(E)                             # (B, T, K)
            self.last_regime_p = p.detach()

        # ── Regime-aware / standard RPE ──
        rpe = None
        if self.use_rpe or self.use_regime_rpe:
            idx = torch.arange(T, device=E.device)
            delta = idx.unsqueeze(0) - idx.unsqueeze(1) + (T - 1)   # (T, T)
            r_base = self.base_rpe[delta].unsqueeze(0)         # (1, T, T, H)
            if self.use_regime_rpe:
                # r_ij = base(delta) + Σ_k p_i[k]·regime_rpe[k, delta]
                r_regime = torch.einsum(
                    'btk,ktuh->btuh', p, self.regime_rpe[:, delta])  # (B, T, T, H)
                rpe = (r_base + r_regime).permute(0, 3, 1, 2)  # (B, H, T, T)
            else:
                rpe = r_base.permute(0, 3, 1, 2).expand(B, -1, -1, -1)

        # ── Regime as input feature context (D) ──
        x_in = E
        if p is not None and not self.use_regime_rpe:
            x_in = E + self.regime_proj(p)

        # ── Encoder ──
        H = x_in
        for layer in self.layers:
            H = layer(H, rpe)                                  # (B, T, 128)

        # ── Temporal attention pooling ──
        score = self.pool_score(H).squeeze(-1)                 # (B, T)
        alpha = F.softmax(score, dim=1)
        h = (H * alpha.unsqueeze(-1)).sum(dim=1)               # (B, 128)
        self.last_attn = alpha.detach()
        return self.fc(h)                                      # (B, 64)


class _ScaleAttnPool(nn.Module):
    """
    Attention pooling over a fixed window of temporal states.

    Upgrade over hard mean-pooling in multiscale_time: learns which days
    inside a scale window matter (e.g. skips post-holiday gaps, earnings
    spikes) instead of averaging all days equally.
    """

    def __init__(self, hidden_dim: int = LSTM_HIDDEN_DIM):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, w, H)
        scores = self.scorer(seq).squeeze(-1)          # (B, w)
        alpha = F.softmax(scores, dim=1)               # (B, w)
        return (seq * alpha.unsqueeze(-1)).sum(dim=1)  # (B, H)


class _TypeMeanPool(nn.Module):
    """Per-type mean pooling → concat → project.

    Supports both single-graph (N, H) → (H,) and per-sample (B, N, H) → (B, H).
    """

    def __init__(self, hidden_dim: int = LSTM_HIDDEN_DIM,
                 n_stock: int = 248, n_bond: int = 12):
        super().__init__()
        self.n_stock = n_stock
        self.n_bond = n_bond
        self.type_agg = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            stock_pool  = h[:self.n_stock, :].mean(dim=0)
            bond_pool   = h[self.n_stock:self.n_stock+self.n_bond, :].mean(dim=0)
            future_pool = h[self.n_stock+self.n_bond:, :].mean(dim=0)
            concat = torch.cat([stock_pool, bond_pool, future_pool])
            return self.type_agg(concat)
        else:
            stock_pool  = h[:, :self.n_stock, :].mean(dim=1)               # (B, H)
            bond_pool   = h[:, self.n_stock:self.n_stock+self.n_bond, :].mean(dim=1)
            future_pool = h[:, self.n_stock+self.n_bond:, :].mean(dim=1)
            concat = torch.cat([stock_pool, bond_pool, future_pool], dim=-1)  # (B, 3H)
            return self.type_agg(concat)


class HeteroMixHopCMGM(nn.Module):
    """
    Heterogeneous MixHop CMGM with ablation support.

    Forward:  (B, T, N, F) → (B, n_horizons, N_commodities)  [multi]
              (B, T, N, F) → (B, N_commodities)              [single]
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 n_stock: int = 248, n_bond: int = 12,
                 variant: str = "edge_attn", feat_dim: int = FEATURE_DIM,
                 attn_heads: int = 8, attn_dropout: float = 0.1,
                 attn_prior_scale: float = 0.5, attn_self_heads: int = 4,
                 graph_cfg: str = "full", use_embedding: bool = True,
                 relations: str = "cc"):
        super().__init__()
        supported_variants = {
            "edge_attn",
            "temporal_weighted_graph",
            "regime_dynamic_transformer",
            "regime_dynamic_semantic",
            "semantic_router",
            "loss_rebalance",
            "routing_strength",
            "context_calibrated",
            "context_balance",
            "adapter_orthogonal",
            "transformer_temporal",
            "transformer_rpe",
            "market_token_transformer",
            "market_dispersion_transformer",
            "switching_transformer",
            "switching_null_control",
            "switching_filter_rpe",
            "switching_latent_transformer",
            "switching_latent_balanced_readout",
            "switching_latent_memory",
        }
        if variant not in supported_variants:
            supported = ", ".join(sorted(supported_variants))
            raise ValueError(
                f"Unsupported variant {variant!r}; choose one of: {supported}"
            )
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.n_stock = n_stock
        self.n_bond = n_bond
        self.variant = variant
        self.feat_dim = feat_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.attn_prior_scale = attn_prior_scale
        self.attn_self_heads = attn_self_heads
        self.graph_cfg = graph_cfg
        self.use_embedding = use_embedding
        self.relations = relations
        self.graph_mode = 'normal'   # E7: 'normal' | 'zero' | 'identity'

        # ── Branch switches ──
        self.use_gcn  = variant not in ("lstm_only", "market_node_no_graph") and not (
            variant == "mkt_node" and graph_cfg == "none")
        self.use_lstm = variant != "gcn_only"
        self.use_learn_graph = variant not in ("no_learn_graph", "edge_attn_static")
        self.use_type_proj   = variant != "no_type_proj"
        self.use_mixhop      = variant not in ("no_mixhop", "edge_attn", "edge_attn_static",
                                               "temporal_attn", "diff_input", "hybrid_attn",
                                               "node_level", "comm_nodes", "batch_graph",
                                               "factor_res", "node_wise", "market_node",
                                               "market_node_no_graph", "mkt_node",
                                               "comm_residual", "comm_output_residual",
                                               "spatial_temporal_attention",
                                               "temporal_weighted_graph",
                                               "temp_weighted_comm_cond",
                                               "temporal_cross_weighted",
                                               "transformer_temporal", "transformer_rpe",
                                               "transformer_regime", "regime_rpe_transformer",
                                               "transformer_regime2", "regime_rpe_transformer2",
                                               "regime_dynamic_transformer",
                                               "regime_dynamic_semantic",
                                               "semantic_router", "loss_rebalance",
                                               "routing_strength", "context_calibrated",
                                               "context_balance", "adapter_orthogonal",
                                               "market_token_transformer",
                                               "market_dispersion_transformer",
                                               "switching_transformer",
                                               "switching_null_control",
                                               "switching_filter_rpe",
                                               "switching_latent_transformer",
                                               "switching_latent_balanced_readout",
                                               "switching_latent_memory")
        self.use_edge_attn   = variant in ("edge_attn", "edge_attn_static", "temporal_attn",
                                           "diff_input", "hybrid_attn", "node_level",
                                           "comm_nodes", "batch_graph", "factor_res",
                                           "node_wise", "market_node", "market_node_no_graph",
                                           "mkt_node", "comm_residual", "comm_output_residual",
                                           "spatial_temporal_attention",
                                           "temporal_weighted_graph",
                                           "temp_weighted_comm_cond",
                                           "temporal_cross_weighted",
                                           "transformer_temporal", "transformer_rpe",
                                           "transformer_regime", "regime_rpe_transformer",
                                           "transformer_regime2", "regime_rpe_transformer2",
                                           "regime_dynamic_transformer",
                                           "regime_dynamic_semantic",
                                           "semantic_router", "loss_rebalance",
                                           "routing_strength", "context_calibrated",
                                           "context_balance", "adapter_orthogonal",
                                           "market_token_transformer",
                                           "market_dispersion_transformer",
                                           "switching_transformer",
                                           "switching_null_control",
                                           "switching_filter_rpe",
                                           "switching_latent_transformer",
                                           "switching_latent_balanced_readout",
                                           "switching_latent_memory")
        self.use_gate        = variant not in ("no_gate", "gcn_only", "lstm_only")

        # ── Multi-horizon output ──
        use_multi = (TARGET_TYPE == "return") and variant != "single_horizon"
        self.n_horizons = len(MULTI_HORIZONS) if use_multi else 1
        out_dim = self.n_horizons * n_commodities

        # ── Per-type (default) or shared input projection ──
        # Built outside the spatial block: needed by market_node_no_graph's
        # global branch even when the GNN itself is disabled.
        if self.use_type_proj:
            self.type_proj = _TypeInputProjection(feat_dim, LSTM_HIDDEN_DIM)
        elif self.use_gcn:
            self.shared_proj = nn.Linear(feat_dim, LSTM_HIDDEN_DIM)

        # ── Spatial branch ──
        if self.use_gcn:
            # Learnable graph (default) or static buffer (no_learn_graph)
            if self.use_learn_graph:
                self.graph_learner = AdaptiveGraphLearner(
                    num_nodes, embed_dim=10, alpha=0.5, top_k=10,
                )
            else:
                self.register_buffer('static_A', torch.zeros(num_nodes, num_nodes))

            # MixHop (default) / EdgeAttnMixHop (edge_attn) / single-hop GCN
            if self.use_mixhop:
                self.mixhop1 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
                self.mixhop2 = MixHopPropagation(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05)
            elif self.use_edge_attn:
                hard = (variant == "edge_attn_static")
                cross_mask = None
                if variant == "hybrid_attn":
                    # Directed cross-market mask for the cross heads:
                    #   future query ← stock + bond   (info sinks into commodities)
                    #   stock  query ← future
                    #   bond   query ← future
                    n_fut = num_nodes - n_stock - n_bond
                    cm = torch.zeros(num_nodes, num_nodes)
                    cm[n_stock + n_bond:, :n_stock + n_bond] = 1.0      # future ← stock+bond
                    cm[:n_stock, n_stock + n_bond:] = 1.0               # stock ← future
                    cm[n_stock:n_stock + n_bond, n_stock + n_bond:] = 1.0  # bond ← future
                    cross_mask = cm
                self.attn_mixhop1 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05,
                    n_heads=self.attn_heads, dropout=self.attn_dropout,
                    hard_mask=hard, prior_scale=self.attn_prior_scale,
                    self_heads=self.attn_self_heads, cross_mask=cross_mask)
                self.attn_mixhop2 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05,
                    n_heads=self.attn_heads, dropout=self.attn_dropout,
                    hard_mask=hard, prior_scale=self.attn_prior_scale,
                    self_heads=self.attn_self_heads, cross_mask=cross_mask)
            else:
                self.singlehop1 = _SingleHopGCN(LSTM_HIDDEN_DIM)
                self.singlehop2 = _SingleHopGCN(LSTM_HIDDEN_DIM)

            self.gcn_norm = nn.LayerNorm(LSTM_HIDDEN_DIM)
            # node-level variants: keep per-node representations — no pooling
            if variant not in ("node_level", "comm_nodes", "batch_graph",
                               "factor_res", "node_wise"):
                self.type_pool = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)
            if variant in ("comm_nodes", "batch_graph"):
                # shared body + per-commodity independent heads
                body_in = (LSTM_HIDDEN_DIM * 3 if variant == "comm_nodes"
                           else LSTM_HIDDEN_DIM * 2)   # batch_graph: gcn+lstm only
                if variant == "comm_nodes":
                    # external markets (stock+bond) pooled to a global state
                    self.ext_fc = nn.Linear(LSTM_HIDDEN_DIM * 2, LSTM_HIDDEN_DIM)
                self.comm_body = nn.Sequential(
                    nn.Linear(body_in, FC_HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(GCN_DROPOUT),
                )
                self.comm_heads = nn.ModuleList([
                    nn.Linear(FC_HIDDEN_DIM, self.n_horizons)
                    for _ in range(n_commodities)
                ])
            if variant == "factor_res":
                # factor model + per-commodity direction residual
                self.type_pool = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)
                self.mean_head = nn.Linear(LSTM_HIDDEN_DIM, self.n_horizons)
                self.lam = nn.Parameter(torch.tensor(0.1))
                self.comm_body = nn.Sequential(
                    nn.Linear(LSTM_HIDDEN_DIM * 2, FC_HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(GCN_DROPOUT),
                )
                self.comm_heads = nn.ModuleList([
                    nn.Linear(FC_HIDDEN_DIM, self.n_horizons)
                    for _ in range(n_commodities)
                ])

            # ── Multi-scale graph: dual spatial branches (short/long) ──
            if variant == "multiscale_graph":
                self.register_buffer('static_A_short', torch.zeros(num_nodes, num_nodes))
                self.register_buffer('static_A_long',  torch.zeros(num_nodes, num_nodes))
                self.attn_short1 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05, hard_mask=True)
                self.attn_short2 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05, hard_mask=True)
                self.attn_long1 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05, hard_mask=True)
                self.attn_long2 = EdgeAttnMixHop(
                    LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, K=2, beta=0.05, hard_mask=True)
                self.gcn_norm_s = nn.LayerNorm(LSTM_HIDDEN_DIM)
                self.gcn_norm_l = nn.LayerNorm(LSTM_HIDDEN_DIM)
                self.type_pool_s = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)
                self.type_pool_l = _TypeMeanPool(LSTM_HIDDEN_DIM, n_stock, n_bond)
                # 3-way softmax fusion over [short, long, lstm]
                self.fuse3 = nn.Linear(LSTM_HIDDEN_DIM * 3, 3)

        # ── Temporal branch ──
        if self.use_lstm:
            if variant in ("node_wise", "market_node", "market_node_no_graph", "mkt_node"):
                # Node-wise temporal encoder: each node gets its OWN
                # temporal representation via a shared LSTM over (B*N, T, F)
                self.node_lstm = nn.LSTM(
                    input_size=feat_dim,
                    hidden_size=LSTM_HIDDEN_DIM,
                    num_layers=LSTM_NUM_LAYERS,
                    dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                    batch_first=True,
                )
                if variant == "node_wise":
                    # shared MLP over [graph_feat, temporal_feat] per commodity
                    self.node_wise_head = nn.Sequential(
                        nn.Linear(LSTM_HIDDEN_DIM * 2, FC_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(GCN_DROPOUT),
                        nn.Linear(FC_HIDDEN_DIM, self.n_horizons),
                    )
                elif variant == "mkt_node":
                    # E-series unified variant: graph_cfg / use_embedding /
                    # relations configured via kwargs
                    self.global_lstm = nn.LSTM(
                        input_size=LSTM_HIDDEN_DIM * 3,
                        hidden_size=LSTM_HIDDEN_DIM,
                        num_layers=LSTM_NUM_LAYERS,
                        dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                        batch_first=True,
                    )
                    if use_embedding:
                        self.commodity_embedding = nn.Embedding(n_commodities, 16)
                    # E2/E3: learnable residual scale / node-wise gate
                    if graph_cfg in ("res", "gate"):
                        self.graph_alpha = nn.Parameter(torch.tensor(0.1))
                    if graph_cfg == "gate":
                        self.gate_lin = nn.Linear(LSTM_HIDDEN_DIM * 2, 1)
                    # E5: relation-specific adjacency mask
                    if graph_cfg == "rel":
                        fut = n_stock + n_bond
                        M = torch.zeros(num_nodes, num_nodes)
                        if relations in ("cc", "cc_sc", "cc_bc", "cc_sc_bc", "full"):
                            M[fut:, fut:] = 1.0
                        if "sc" in relations:
                            M[fut:, :n_stock] = 1.0
                        if "bc" in relations:
                            M[fut:, n_stock:fut] = 1.0
                        if relations == "full":
                            M = torch.ones(num_nodes, num_nodes)
                        self.register_buffer('rel_mask', M)
                    # head: full (D) concats node+graph; others use graph
                    head_in = LSTM_HIDDEN_DIM * 2 + (16 if use_embedding else 0)
                    if graph_cfg == "full":
                        head_in += LSTM_HIDDEN_DIM
                    self.market_node_head = nn.Sequential(
                        nn.Linear(head_in, FC_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(GCN_DROPOUT),
                        nn.Linear(FC_HIDDEN_DIM, self.n_horizons),
                    )
                elif variant in ("market_node", "market_node_no_graph"):
                    # Global market-state branch: type-aware per-timestep
                    # pooling (stock/bond/commodity → 192) → LSTM(192→64)
                    self.global_lstm = nn.LSTM(
                        input_size=LSTM_HIDDEN_DIM * 3,
                        hidden_size=LSTM_HIDDEN_DIM,
                        num_layers=LSTM_NUM_LAYERS,
                        dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                        batch_first=True,
                    )
                    # Commodity identity embedding for the shared head
                    self.commodity_embedding = nn.Embedding(n_commodities, 16)
                    head_in = (LSTM_HIDDEN_DIM * 3 + 16 if variant == "market_node"
                               else LSTM_HIDDEN_DIM * 2 + 16)   # no_graph: no graph feat
                    self.market_node_head = nn.Sequential(
                        nn.Linear(head_in, FC_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(GCN_DROPOUT),
                        nn.Linear(FC_HIDDEN_DIM, self.n_horizons),
                    )
            elif variant == "temporal_attn":
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
            elif variant == "tcn_temporal":
                # Multi-scale TCN replaces the LSTM entirely
                self.tcn = _MultiScaleTCN(in_dim=num_nodes * feat_dim)
                # Multi-scale temporal pooling (reuse verified mechanism)
                self.ms_fuse = nn.Linear(LSTM_HIDDEN_DIM * 4, LSTM_HIDDEN_DIM)
            elif variant == "patch_temporal":
                # PatchTST-style branch (patch_len=5 aligned with 5d horizon)
                self.patch = _PatchTemporal(in_dim=num_nodes * feat_dim,
                                            t_len=SEQ_LEN, patch_len=5)
            elif variant == "mamba_temporal":
                # Mamba (SSM) branch — requires mamba-ssm (Linux CUDA)
                self.mamba = _MambaTemporal(in_dim=num_nodes * feat_dim)
            elif variant == "informer_temporal":
                # Informer-style branch (ProbSparse attention encoder)
                self.informer = _InformerTemporal(in_dim=num_nodes * feat_dim,
                                                  t_len=SEQ_LEN)
            elif variant in ("transformer_temporal", "transformer_rpe",
                             "transformer_regime", "regime_rpe_transformer",
                             "transformer_regime2", "regime_rpe_transformer2",
                             "regime_dynamic_transformer", "regime_dynamic_semantic",
                             "semantic_router", "loss_rebalance", "routing_strength",
                             "context_calibrated", "context_balance",
                             "adapter_orthogonal", "market_token_transformer",
                             "market_dispersion_transformer", "switching_transformer",
                             "switching_null_control", "switching_filter_rpe",
                             "switching_latent_transformer",
                             "switching_latent_balanced_readout",
                             "switching_latent_memory"):
                pass  # temporal branch is the transformer (no global LSTM)
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
                # Multi-scale attention pooling: attention inside each scale
                # window instead of hard mean (upgrade of multiscale_time)
                if variant == "attn_pool":
                    self.pool5  = _ScaleAttnPool()
                    self.pool10 = _ScaleAttnPool()
                    self.pool20 = _ScaleAttnPool()
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

        # comm_residual: minimal commodity-residual enhancement on top of
        # the original pooled model.  When comm_alpha → 0 the model
        # degenerates exactly to the pooled baseline.
        if variant == "comm_residual":
            self.comm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.comm_alpha = nn.Parameter(torch.tensor(0.05))
            # shared prediction head applied per commodity node
            self.comm_head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(GCN_DROPOUT),
                nn.Linear(FC_HIDDEN_DIM, self.n_horizons),
            )

        # comm_output_residual: original prediction path kept EXACTLY;
        # commodity graph representations add a small OUTPUT residual.
        # alpha = 0 → output strictly equals the original edge_attn model.
        if variant == "comm_output_residual":
            self.residual_head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_horizons),
            )
            self.residual_alpha = nn.Parameter(torch.tensor(0.01))

        # spatial_temporal_attention: per-timestep GNN + temporal attention
        # (replaces mean over T in the spatial branch).  Global branch kept
        # intact; commodity nodes never pooled.
        if variant == "spatial_temporal_attention":
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            # commodity output head — reuses the original head's shared body
            self.comm_out = nn.Linear(FC_HIDDEN_DIM, self.n_horizons)

        # temporal_weighted_graph: ONLY replaces mean(dim=1) with node-wise
        # temporal attention.  GNN structure, LSTM, fusion, head unchanged.
        if variant == "temporal_weighted_graph":
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)

        # temporal_cross_weighted: node's temporal-attention score also
        # consults its neighbors' scores via the graph adjacency —
        # score_final = MLP([own_score, neighbor_score]); softmax over T.
        if variant == "temporal_cross_weighted":
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.cross_score_fuse = nn.Linear(2, 1)

        # RegimeRPETransformer family (temporal branch only):
        #   transformer_temporal (B) / transformer_rpe (C) /
        #   transformer_regime (D) / regime_rpe_transformer (E)
        #   transformer_regime2 (R2) / regime_rpe_transformer2 (R3)
        if variant in ("transformer_temporal", "transformer_rpe",
                       "transformer_regime", "regime_rpe_transformer",
                       "transformer_regime2", "regime_rpe_transformer2"):
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)   # spatial branch
            self.temporal_transformer = _TemporalTransformerBranch(
                in_dim=num_nodes * feat_dim,
                d_model=128, n_heads=4, n_layers=2, ffn_dim=256,
                dropout=GCN_DROPOUT, t_len=SEQ_LEN, K=3,
                use_rpe=(variant in ("transformer_rpe", "regime_rpe_transformer",
                                     "regime_rpe_transformer2")),
                use_regime=(variant in ("transformer_regime", "regime_rpe_transformer",
                                        "transformer_regime2", "regime_rpe_transformer2")),
                use_regime_rpe=(variant in ("regime_rpe_transformer",
                                            "regime_rpe_transformer2")),
                use_prototype_regime=(variant in ("transformer_regime2",
                                                  "regime_rpe_transformer2")),
            )

        # S0/S0D: only temporal input construction changes.  The spatial
        # TempWeighted path, gated fusion, and prediction head remain shared.
        if variant in ("market_token_transformer", "market_dispersion_transformer"):
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.market_token_transformer = MarketTokenTransformerBranch(
                feat_dim=feat_dim,
                n_stock=n_stock,
                n_bond=n_bond,
                n_commodity=num_nodes - n_stock - n_bond,
                node_dim=32,
                d_model=128,
                n_heads=4,
                n_layers=2,
                ffn_dim=256,
                dropout=GCN_DROPOUT,
                output_dim=LSTM_HIDDEN_DIM,
                use_dispersion=(variant == "market_dispersion_transformer"),
            )

        if variant in ("switching_transformer", "switching_null_control"):
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.switching_transformer = SwitchingTransformerBranch(
                feat_dim=feat_dim,
                n_stock=n_stock,
                n_bond=n_bond,
                n_commodity=num_nodes - n_stock - n_bond,
                node_dim=32,
                d_model=128,
                n_heads=4,
                n_layers=2,
                ffn_dim=256,
                dropout=GCN_DROPOUT,
                output_dim=LSTM_HIDDEN_DIM,
                K=3,
                sticky_alpha=0.5,
                beta_max=5e-4,
                warmup_epochs=20,
                null_control=(variant == "switching_null_control"),
            )

        if variant == "switching_filter_rpe":
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.switching_filter_rpe = SwitchingFilterRPEBranch(
                feat_dim=feat_dim,
                n_stock=n_stock,
                n_bond=n_bond,
                n_commodity=num_nodes - n_stock - n_bond,
                node_dim=32,
                d_model=128,
                n_heads=4,
                n_layers=2,
                ffn_dim=256,
                dropout=GCN_DROPOUT,
                output_dim=LSTM_HIDDEN_DIM,
                max_len=SEQ_LEN,
                K=3,
                sticky_alpha=0.5,
                tau=1.0,
                beta_max=5e-4,
                warmup_epochs=20,
            )

        if variant in (
            "switching_latent_transformer",
            "switching_latent_balanced_readout",
            "switching_latent_memory",
        ):
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.switching_latent_transformer = SwitchingLatentTransformerBranch(
                feat_dim=feat_dim,
                n_stock=n_stock,
                n_bond=n_bond,
                n_commodity=num_nodes - n_stock - n_bond,
                node_dim=32,
                d_model=128,
                n_heads=4,
                n_layers=2,
                ffn_dim=256,
                dropout=GCN_DROPOUT,
                max_len=SEQ_LEN,
                K=3,
                z_dim=64,
                sticky_alpha=0.5,
                tau=1.0,
                beta_max=5e-4,
                warmup_epochs=20,
                output_dim=LSTM_HIDDEN_DIM,
                balanced_readout=(variant != "switching_latent_transformer"),
                use_latent_memory=(variant == "switching_latent_memory"),
            )

        # RegimeDynamicRPETransformer (F): modular regime → dynamics →
        # regime-aware RPE → transformer temporal branch
        # F2 (regime_dynamic_semantic): adds market-dynamics state target
        if variant in ("regime_dynamic_transformer", "regime_dynamic_semantic",
                       "semantic_router", "loss_rebalance", "routing_strength",
                       "context_calibrated", "context_balance", "adapter_orthogonal"):
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)   # spatial branch
            self.regime_dynamic = RegimeDynamicRPETransformer(
                in_dim=num_nodes * feat_dim,
                d_model=128, n_heads=4, n_layers=2, ffn_dim=256,
                dropout=GCN_DROPOUT, t_len=SEQ_LEN, K=3, D_regime=32,
                use_state_loss=(variant == "regime_dynamic_semantic"),
                use_semantic_router=(variant in (
                    "semantic_router", "loss_rebalance", "routing_strength",
                    "context_calibrated", "context_balance", "adapter_orthogonal"
                )),
                lambda_cluster=(
                    0.0005
                    if variant in (
                        "loss_rebalance", "routing_strength", "context_calibrated",
                        "context_balance", "adapter_orthogonal"
                    )
                    else 0.005
                ),
                routing_strength=(
                    0.25
                    if variant in (
                        "routing_strength", "context_calibrated", "context_balance",
                        "adapter_orthogonal"
                    )
                    else 1.0
                ),
                context_gamma={
                    "context_calibrated": 1.0,
                    "context_balance": 2.0,
                    "adapter_orthogonal": 2.0,
                }.get(variant, 5.0),
                orthogonal_dynamic=(variant == "adapter_orthogonal"),
            )

        # temp_weighted_comm_cond: commodity graph representations
        # condition the shared hidden state before the ORIGINAL per-commodity
        # output layer.  α = 0 → output strictly equals TempWeighted.
        if variant == "temp_weighted_comm_cond":
            self.temporal_score = nn.Linear(LSTM_HIDDEN_DIM, 1)
            self.comm_proj = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.alpha = nn.Parameter(torch.tensor(0.01))
            self.shuffle_comm = False    # diagnostic: permute commodity dim

        # Node-level head: shared MLP applied per commodity node —
        # input (B, n_commodities, 128) → (B, n_commodities, n_horizons)
        if variant == "node_level":
            self.node_head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM * 2, FC_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(GCN_DROPOUT),
                nn.Linear(FC_HIDDEN_DIM, self.n_horizons),
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
        self._last_h = h                                               # node reps (for comm_output_residual)

        # Per-type pooling → expand to batch
        gcn_out = self.type_pool(h).unsqueeze(0).expand(x.size(0), -1) # (B, 64)
        return gcn_out

    def _original_forward(self, x: torch.Tensor, return_h: bool = False,
                          debug: bool = False,
                          gcn_out: torch.Tensor = None) -> torch.Tensor:
        """
        THE original edge_attn forward path — single source of truth.
        `edge_attn`, `comm_output_residual` and `temporal_weighted_graph`
        all call this exact code:

          spatial (type_proj → EdgeAttnMixHop → type pooling)
          → global LSTM → gate fusion → original head → reshape

        gcn_out: optional pre-computed (B, 64) spatial representation —
                 used by variants that only replace the temporal
                 aggregation before the GNN (e.g. temporal_weighted_graph).
        return_h=True additionally returns the node representations H (N, 64)
        from the spatial branch (used by comm_output_residual).
        """
        B, T, N, n_feat = x.shape

        # ── 1. Spatial branch (single graph, unchanged) ──
        if gcn_out is None:
            gcn_out = self._spatial_forward(x)                         # (B, 64)

        # ── 2. Temporal branch (global LSTM, unchanged) ──
        x_seq = x.reshape(B, T, -1)                                    # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                             # (B, 64)

        # ── 3. Gate fusion (unchanged) ──
        combined = torch.cat([gcn_out, lstm_out], dim=-1)              # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = gate * self.lstm_proj(lstm_out) + (1 - gate) * self.gcn_proj(gcn_out)  # (B, 64)

        # ── 4. Original head + reshape (unchanged) ──
        pred = self.head(fused)                                        # (B, out_dim)
        if self.n_horizons > 1:
            pred = pred.view(B, self.n_horizons, self.n_commodities)   # (B, H, Nc)
        else:
            pred = pred.view(B, self.n_commodities)                    # (B, Nc)

        if debug:
            print(f"  [original] GCN→{list(gcn_out.shape)} LSTM→{list(lstm_out.shape)} "
                  f"→ gate → head → {list(pred.shape)}")
        if return_h:
            return pred, self._last_h
        return pred

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

    def _multiscale_graph_forward(self, x: torch.Tensor,
                                  debug: bool = False) -> torch.Tensor:
        """
        Multi-scale graph branch: dual spatial branches on short-memory
        (λ=0.9) and long-memory (λ=0.99) EWMA graphs, fused with the LSTM
        via a 3-way softmax gate.

        A_short/A_long are injected as static buffers (hard-mask mode of
        EdgeAttnMixHop — structure from EWMA, weights from content attention).
        """
        B, T, N, _ = x.shape

        # Shared per-type projection (same input prep as spatial branch)
        x_gcn = x.mean(dim=0).permute(1, 0, 2)                        # (N, T, F)
        x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)     # (N, T, 64)
        x_proj = x_proj.mean(dim=1)                                   # (N, 64)

        # ── Short-memory spatial branch ──
        h_s1 = F.relu(self.attn_short1(x_proj, self.static_A_short))  # (N, 64)
        h_s2 = self.attn_short2(h_s1, self.static_A_short)
        gcn_short = self.type_pool_s(self.gcn_norm_s(h_s2)).unsqueeze(0).expand(B, -1)  # (B, 64)

        # ── Long-memory spatial branch ──
        h_l1 = F.relu(self.attn_long1(x_proj, self.static_A_long))    # (N, 64)
        h_l2 = self.attn_long2(h_l1, self.static_A_long)
        gcn_long = self.type_pool_l(self.gcn_norm_l(h_l2)).unsqueeze(0).expand(B, -1)   # (B, 64)

        # ── Temporal branch (LSTM, unchanged) ──
        x_seq = x.reshape(B, T, -1)                                   # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                            # (B, 64)

        # ── 3-way softmax fusion ──
        combined = torch.cat([gcn_short, gcn_long, lstm_out], dim=-1)  # (B, 192)
        gate = F.softmax(self.fuse3(combined), dim=-1)                # (B, 3)
        fused = (gate[:, 0:1] * gcn_short
                 + gate[:, 1:2] * gcn_long
                 + gate[:, 2:3] * lstm_out)                           # (B, 64)

        pred = self.head(fused)
        if self.n_horizons > 1:
            pred = pred.view(B, self.n_horizons, self.n_commodities)
        else:
            pred = pred.view(B, self.n_commodities)

        if debug:
            print(f"  [multiscale_graph] short→{list(gcn_short.shape)} "
                  f"long→{list(gcn_long.shape)} lstm→{list(lstm_out.shape)} "
                  f"gate→{gate.mean(dim=0).tolist()} → head → {list(pred.shape)}")
        return pred

    def _node_level_forward(self, x: torch.Tensor,
                            debug: bool = False) -> torch.Tensor:
        """
        Node-level spatial branch: no type pooling — each commodity keeps
        its own graph representation (stocks/bonds flow in via message
        passing), fused with the global LSTM state per commodity, and
        predicted by a shared per-node head.

        Returns: (B, n_horizons, n_commodities) or (B, n_commodities).
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── Spatial: node-level graph representation ──
        x_gcn = x.mean(dim=0).permute(1, 0, 2)                        # (N, T, F)
        x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)     # (N, T, 64)
        x_proj = x_proj.mean(dim=1)                                   # (N, 64)
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(x_proj, A))                     # (N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)                                         # (N, 64)
        gcn_comm = h[fut_start:].unsqueeze(0).expand(B, -1, -1)       # (B, 24, 64)

        # ── Temporal: global LSTM state, expanded per commodity ──
        x_seq = x.reshape(B, T, -1)                                   # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                            # (B, 64)
        lstm_expand = lstm_out.unsqueeze(1).expand(B, self.n_commodities, -1)  # (B, 24, 64)

        # ── Fusion + per-node shared head ──
        fused = torch.cat([gcn_comm, lstm_expand], dim=-1)            # (B, 24, 128)
        pred = self.node_head(fused)                                  # (B, 24, H)
        if self.n_horizons > 1:
            pred = pred.permute(0, 2, 1)                              # (B, H, 24)
        else:
            pred = pred.squeeze(-1)                                   # (B, 24)

        if debug:
            print(f"  [node_level] gcn_comm→{list(gcn_comm.shape)} "
                  f"lstm→{list(lstm_expand.shape)} → head → {list(pred.shape)}")
        return pred

    def _comm_nodes_forward(self, x: torch.Tensor,
                            debug: bool = False) -> torch.Tensor:
        """
        Commodity-node spatial branch:
          - commodity nodes keep their own graph representations (24, 64)
          - stock/bond markets are pooled to a global external state (64,)
          - per-commodity fusion: [own graph feat, external state, LSTM state]
          - shared body + per-commodity independent heads (no gradient clash)
        """
        B, T, N, _ = x.shape
        n_stock, n_bond = self.n_stock, self.n_bond

        # ── Spatial: node-level graph representation ──
        x_gcn = x.mean(dim=0).permute(1, 0, 2)                        # (N, T, F)
        x_proj = self.type_proj(x_gcn, n_stock, n_bond)               # (N, T, 64)
        x_proj = x_proj.mean(dim=1)                                   # (N, 64)
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(x_proj, A))                     # (N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)                                         # (N, 64)

        # Commodity nodes kept; stock/bond pooled to external global state
        gcn_comm = h[n_stock + n_bond:].unsqueeze(0).expand(B, -1, -1)  # (B, 24, 64)
        ext = F.relu(self.ext_fc(torch.cat([
            h[:n_stock].mean(dim=0), h[n_stock:n_stock + n_bond].mean(dim=0)])))  # (64,)
        ext_expand = ext.unsqueeze(0).unsqueeze(0).expand(B, self.n_commodities, -1)

        # ── Temporal: global LSTM state, expanded per commodity ──
        x_seq = x.reshape(B, T, -1)                                   # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                            # (B, 64)
        lstm_expand = lstm_out.unsqueeze(1).expand(B, self.n_commodities, -1)

        # ── Shared body + per-commodity heads ──
        fused = torch.cat([gcn_comm, ext_expand, lstm_expand], dim=-1)  # (B, 24, 192)
        body = self.comm_body(fused)                                  # (B, 24, 64)
        preds = torch.stack([self.comm_heads[i](body[:, i])
                             for i in range(self.n_commodities)], dim=1)  # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                             # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                  # (B, 24)

        if debug:
            print(f"  [comm_nodes] gcn_comm→{list(gcn_comm.shape)} "
                  f"ext→{list(ext_expand.shape)} lstm→{list(lstm_expand.shape)} "
                  f"→ body → heads → {list(pred.shape)}")
        return pred

    def _batch_graph_forward(self, x: torch.Tensor,
                             debug: bool = False) -> torch.Tensor:
        """
        Batch-aware spatial branch:
          - graph message passing runs PER SAMPLE (no batch averaging) —
            each sample has its own node representations and attention
          - all markets keep node-level representations (no pooling);
            commodity nodes are used for prediction
          - shared body + per-commodity independent heads
        """
        B, T, N, _ = x.shape

        # ── Spatial: per-sample node representations ──
        seqs = []
        for t in range(T):
            xt = x[:, t, :, :]                                       # (B, N, F)
            seqs.append(self.type_proj(xt, self.n_stock, self.n_bond))  # (B, N, 64)
        x_proj = torch.stack(seqs, dim=1).mean(dim=1)                # (B, N, 64)

        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(x_proj, A))                    # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)                                        # (B, N, 64)
        gcn_comm = h[:, self.n_stock + self.n_bond:, :]              # (B, 24, 64)

        # ── Temporal: global LSTM state, expanded per commodity ──
        x_seq = x.reshape(B, T, -1)                                  # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                           # (B, 64)
        lstm_expand = lstm_out.unsqueeze(1).expand(B, self.n_commodities, -1)

        # ── Shared body + per-commodity heads ──
        fused = torch.cat([gcn_comm, lstm_expand], dim=-1)           # (B, 24, 128)
        body = self.comm_body(fused)                                 # (B, 24, 64)
        preds = torch.stack([self.comm_heads[i](body[:, i])
                             for i in range(self.n_commodities)], dim=1)  # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                            # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                 # (B, 24)

        if debug:
            print(f"  [batch_graph] gcn_comm→{list(gcn_comm.shape)} "
                  f"lstm→{list(lstm_expand.shape)} → body → heads → {list(pred.shape)}")
        return pred

    def _batch_spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch-aware spatial propagation (generic, per-sample representations):
          x (B, T, N, F) → per-timestep type projection → time mean
          → EdgeAttnMixHop ×2 on the shared adjacency A → (B, N, 64)

        Every batch sample keeps its OWN node representations — no
        x.mean(dim=0) anywhere on the batch axis.
        """
        B, T, N, _ = x.shape
        seqs = []
        for t in range(T):
            seqs.append(self.type_proj(x[:, t, :, :], self.n_stock, self.n_bond))
        x_proj = torch.stack(seqs, dim=1).mean(dim=1)                  # (B, N, 64)
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(x_proj, A))                     # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        return self.gcn_norm(h2)                                      # (B, N, 64)

    def _node_temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Node-wise temporal encoder:
          x (B, T, N, F) → permute → reshape (B*N, T, F)
          → shared LSTM → last hidden (B*N, D) → view → (B, N, D)

        Each node gets its OWN temporal representation (no global LSTM,
        no expand to 24 commodities).
        """
        B, T, N, F = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)           # (B*N, T, F)
        _, (h_n, _) = self.node_lstm(x_flat)
        return h_n[-1].view(B, N, -1)                                 # (B, N, 64)

    def _node_wise_forward(self, x: torch.Tensor,
                           debug: bool = False) -> torch.Tensor:
        """
        Clean node-level baseline (Model B/C):
          h_graph    = batch-aware spatial propagation  → (B, N, D)
          h_temporal = node-wise LSTM                   → (B, N, D)
          commodity subset → cat → shared MLP → predictions
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        h_graph = self._batch_spatial_forward(x)                      # (B, N, 64)
        h_temporal = self._node_temporal_forward(x)                   # (B, N, 64)

        # ── Shape assertions ──
        assert h_graph.shape[:2] == (B, N), f"graph: {h_graph.shape} != ({B},{N},_)"
        assert h_temporal.shape[:2] == (B, N), f"temporal: {h_temporal.shape} != ({B},{N},_)"

        h_graph_comm = h_graph[:, fut_start:, :]                      # (B, 24, 64)
        h_temporal_comm = h_temporal[:, fut_start:, :]                # (B, 24, 64)
        assert h_graph_comm.shape[1] == self.n_commodities, \
            f"graph comm: {h_graph_comm.shape}"
        assert h_temporal_comm.shape[1] == self.n_commodities, \
            f"temporal comm: {h_temporal_comm.shape}"

        # ── Shared MLP head ──
        z = torch.cat([h_graph_comm, h_temporal_comm], dim=-1)        # (B, 24, 128)
        preds = self.node_wise_head(z)                                # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                             # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                  # (B, 24)

        if debug:
            print(f"  [node_wise] h_graph→{list(h_graph.shape)} "
                  f"h_temporal→{list(h_temporal.shape)} "
                  f"comm→{list(z.shape)} → head → {list(pred.shape)}")
        return pred

    def _global_market_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Type-aware global market-state branch:
          per timestep: type_proj → pool stock / bond / commodity → (B, 192)
          → LSTM(192→64) → h_global (B, 64)

        Pooling is ALLOWED here — the purpose is the market-level common
        factor, not per-commodity prediction.
        """
        B, T, N, _ = x.shape
        n_stock, n_bond = self.n_stock, self.n_bond
        seqs = []
        for t in range(T):
            xt = self.type_proj(x[:, t, :, :], n_stock, n_bond)      # (B, N, 64)
            s = xt[:, :n_stock].mean(dim=1)                          # (B, 64)
            b = xt[:, n_stock:n_stock + n_bond].mean(dim=1)          # (B, 64)
            f = xt[:, n_stock + n_bond:].mean(dim=1)                 # (B, 64)
            seqs.append(torch.cat([s, b, f], dim=-1))                # (B, 192)
        g_seq = torch.stack(seqs, dim=1)                             # (B, T, 192)
        _, (h_n, _) = self.global_lstm(g_seq)
        return h_n[-1]                                               # (B, 64)

    def _market_node_forward(self, x: torch.Tensor,
                             debug: bool = False) -> torch.Tensor:
        """
        Market + Node dual representation:
          h_node    = node-wise LSTM (per-node temporal)      (B, N, 64)
          h_graph   = GNN(h_node, A) — propagation of the
                      *current window's* dynamic node states  (B, N, 64)
                      (skipped for market_node_no_graph)
          h_global  = type-aware global market factor        (B, 64)
          emb       = commodity identity embedding           (B, 24, 16)
          z = cat([h_node_comm, h_graph_comm, h_global_expand, emb]) → shared MLP
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── 1. Node temporal representation ──
        h_node = self._node_temporal_forward(x)                      # (B, N, 64)
        assert h_node.shape == (B, N, 64), f"h_node: {h_node.shape}"

        # ── 2. Graph propagation on node temporal states ──
        h_graph = None
        if self.variant == "market_node":
            A = self.graph_learner() if self.use_learn_graph else self.static_A
            h1 = F.relu(self.attn_mixhop1(h_node, A))                # (B, N, 64)
            h2 = self.attn_mixhop2(h1, A)
            h_graph = self.gcn_norm(h2)                              # (B, N, 64)
            assert h_graph.shape == (B, N, 64), f"h_graph: {h_graph.shape}"

        # ── 3. Global market factor ──
        h_global = self._global_market_forward(x)                    # (B, 64)
        assert h_global.shape == (B, 64), f"h_global: {h_global.shape}"

        # ── 4. Commodity-specific slices ──
        h_node_comm = h_node[:, fut_start:, :]                       # (B, 24, 64)
        assert h_node_comm.shape[1] == self.n_commodities
        if h_graph is not None:
            h_graph_comm = h_graph[:, fut_start:, :]                 # (B, 24, 64)
            assert h_graph_comm.shape[1] == self.n_commodities

        # ── 5. Commodity identity embedding ──
        emb = self.commodity_embedding(
            torch.arange(self.n_commodities, device=x.device))       # (24, 16)
        emb = emb.unsqueeze(0).expand(B, -1, -1)                     # (B, 24, 16)

        # ── 6. Fusion ──
        parts = [h_node_comm]
        if h_graph is not None:
            parts.append(h_graph_comm)
        parts.append(h_global.unsqueeze(1).expand(-1, self.n_commodities, -1))
        parts.append(emb)
        z = torch.cat(parts, dim=-1)                                 # (B, 24, 208)/(B, 24, 144)

        preds = self.market_node_head(z)                             # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                            # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                 # (B, 24)

        if debug:
            print(f"  [market_node] h_node={list(h_node.shape)} "
                  f"h_graph={list(h_graph.shape) if h_graph is not None else None} "
                  f"h_global={list(h_global.shape)} "
                  f"h_node_comm={list(h_node_comm.shape)} "
                  f"h_graph_comm={list(h_graph_comm.shape) if h_graph is not None else None} "
                  f"emb={list(emb.shape)} z={list(z.shape)} "
                  f"pred={list(pred.shape)}")
        return pred

    def _mkt_node_forward(self, x: torch.Tensor,
                          debug: bool = False) -> torch.Tensor:
        """
        Unified E-series forward:
          h_node (node-wise LSTM) + h_global (type-aware market factor)
          + graph branch per graph_cfg:
            none: h_eff = h_node
            full: h_eff = cat(h_node, GNN(h_node))          (D, 208)
            cc:   h_eff = GNN(h_node_comm, A_CC)            (E4, 144)
            rel:  h_eff = GNN(h_node, A·mask) commodity slice (E5)
            res:  h_eff = h_node + α·GNN(h_node)            (E2)
            gate: h_eff = h_node + σ(·)·GNN(h_node)         (E3)
          + commodity embedding (unless disabled) → shared MLP
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond
        cfg = self.graph_cfg

        h_node = self._node_temporal_forward(x)                      # (B, N, 64)
        h_global = self._global_market_forward(x)                    # (B, 64)
        h_node_comm = h_node[:, fut_start:, :]                       # (B, 24, 64)

        if cfg == "none":
            h_eff_comm = h_node_comm
        else:
            A = self.graph_learner() if self.use_learn_graph else self.static_A
            if cfg == "cc":
                A_eff = A[fut_start:, fut_start:]                    # (24, 24)
                g1 = F.relu(self.attn_mixhop1(h_node_comm, A_eff))   # (B, 24, 64)
                g2 = self.attn_mixhop2(g1, A_eff)
                h_eff_comm = self.gcn_norm(g2)
            elif cfg == "rel":
                A_eff = A * self.rel_mask
                g1 = F.relu(self.attn_mixhop1(h_node, A_eff))        # (B, N, 64)
                g2 = self.attn_mixhop2(g1, A_eff)
                h_eff_comm = self.gcn_norm(g2)[:, fut_start:, :]
            elif cfg == "res":
                g1 = F.relu(self.attn_mixhop1(h_node, A))
                g2 = self.attn_mixhop2(g1, A)
                g = self.gcn_norm(g2)
                h_eff_comm = (h_node + self.graph_alpha * g)[:, fut_start:, :]
            elif cfg == "gate":
                g1 = F.relu(self.attn_mixhop1(h_node, A))
                g2 = self.attn_mixhop2(g1, A)
                g = self.gcn_norm(g2)
                gate = torch.sigmoid(self.gate_lin(
                    torch.cat([h_node, g], dim=-1)))                 # (B, N, 1)
                self.last_gate = gate
                h_eff_comm = (h_node + gate * g)[:, fut_start:, :]
            else:  # full (D)
                g1 = F.relu(self.attn_mixhop1(h_node, A))
                g2 = self.attn_mixhop2(g1, A)
                g = self.gcn_norm(g2)
                if self.graph_mode == 'zero':                        # E7
                    h_graph = torch.zeros_like(h_node)
                elif self.graph_mode == 'identity':                  # E7
                    h_graph = h_node
                else:
                    h_graph = g
                h_eff_comm = torch.cat(
                    [h_node_comm, h_graph[:, fut_start:, :]], dim=-1)  # (B, 24, 128)

        # Commodity embedding + global expand
        parts = [h_eff_comm]
        parts.append(h_global.unsqueeze(1).expand(-1, self.n_commodities, -1))
        if self.use_embedding:
            emb = self.commodity_embedding(
                torch.arange(self.n_commodities, device=x.device))
            parts.append(emb.unsqueeze(0).expand(B, -1, -1))
        z = torch.cat(parts, dim=-1)

        preds = self.market_node_head(z)                             # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                            # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                 # (B, 24)

        if debug:
            print(f"  [mkt_node cfg={cfg}] h_node={list(h_node.shape)} "
                  f"h_global={list(h_global.shape)} h_eff_comm={list(h_eff_comm.shape)} "
                  f"z={list(z.shape)} pred={list(pred.shape)}")
        return pred

    @staticmethod
    def _pairwise_cos(h: torch.Tensor):
        """(mean, std) of pairwise cosine similarity across nodes (no self)."""
        sims = []
        for b in range(min(h.size(0), 4)):
            c = F.normalize(h[b], dim=-1)
            sim = c @ c.T
            mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            sims.append(sim[mask])
        all_sim = torch.cat(sims)
        return all_sim.mean().item(), all_sim.std().item()

    def layer_similarity(self, x: torch.Tensor) -> dict:
        """
        E6 oversmoothing diagnostic per propagation layer:
          input (node temporal) / after MixHop layer 1 / layer 2
          reported for all-node and commodity-only pairwise cosine.
        """
        with torch.no_grad():
            h_node = self._node_temporal_forward(x)                  # (B, N, 64)
            A = self.graph_learner() if self.use_learn_graph else self.static_A
            h1 = F.relu(self.attn_mixhop1(h_node, A))
            h2 = self.attn_mixhop2(h1, A)
            fut = self.n_stock + self.n_bond
            res = {}
            for name, h in [('input', h_node), ('layer1', h1), ('layer2', h2)]:
                m_all, s_all = self._pairwise_cos(h)
                m_comm, s_comm = self._pairwise_cos(h[:, fut:, :])
                res[f'{name}_all_mean'], res[f'{name}_all_std'] = m_all, s_all
                res[f'{name}_comm_mean'], res[f'{name}_comm_std'] = m_comm, s_comm
            return res

    def node_similarity(self, x: torch.Tensor) -> dict:
        """
        Oversmoothing diagnostic: mean/std of pairwise cosine similarity
        across the 24 commodity node representations (node temporal states).
        Values close to 1.0 → representations smoothed into one another.
        """
        with torch.no_grad():
            h_node = self._node_temporal_forward(x)                  # (B, N, 64)
            comm = h_node[:, self.n_stock + self.n_bond:, :]         # (B, 24, 64)
            sims = []
            for b in range(min(comm.size(0), 4)):
                c = F.normalize(comm[b], dim=-1)                     # (24, 64)
                sim = c @ c.T                                        # (24, 24)
                mask = ~torch.eye(24, dtype=torch.bool, device=c.device)
                sims.append(sim[mask])
            all_sim = torch.cat(sims)
            return {'sim_mean': all_sim.mean().item(),
                    'sim_std': all_sim.std().item()}

    def _factor_res_forward(self, x: torch.Tensor,
                            debug: bool = False) -> torch.Tensor:
        """
        Factor + residual decomposition:
          r̂_i = r̂_mean + λ·δ̂_i

          - pooling branch (factor model): type-pooled market state →
            market-mean return r̂_mean (stable, shared supervision)
          - node branch: per-commodity graph features → direction residual
            δ̂_i (tanh-bounded); λ is a learnable scale
          - self.last_r_mean stores r̂_mean so the training loop can add
            the mean-supervision auxiliary loss
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── Shared batch-aware spatial propagation ──
        seqs = []
        for t in range(T):
            seqs.append(self.type_proj(x[:, t, :, :], self.n_stock, self.n_bond))
        x_proj = torch.stack(seqs, dim=1).mean(dim=1)                  # (B, N, 64)
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(x_proj, A))                     # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)                                         # (B, N, 64)

        # ── Pooling branch: market-mean factor ──
        r_mean = self.mean_head(self.type_pool(h))                    # (B, H)
        self.last_r_mean = r_mean

        # ── Node branch: per-commodity direction residual ──
        gcn_comm = h[:, fut_start:, :]                                # (B, 24, 64)
        x_seq = x.reshape(B, T, -1)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                            # (B, 64)
        lstm_expand = lstm_out.unsqueeze(1).expand(B, self.n_commodities, -1)
        fused = torch.cat([gcn_comm, lstm_expand], dim=-1)            # (B, 24, 128)
        body = self.comm_body(fused)                                  # (B, 24, 64)
        delta = torch.stack([self.comm_heads[i](body[:, i])
                             for i in range(self.n_commodities)], dim=1)  # (B, 24, H)
        delta = torch.tanh(delta)                                     # direction, [-1, 1]

        # ── Combine: r̂_i = r̂_mean + λ·δ̂_i ──
        preds = r_mean.unsqueeze(1) + self.lam * delta                # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                             # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                  # (B, 24)

        if debug:
            print(f"  [factor_res] r_mean→{list(r_mean.shape)} "
                  f"delta→{list(delta.shape)} λ={self.lam.item():.3f} "
                  f"→ pred {list(pred.shape)}")
        return pred

    def _comm_residual_forward(self, x: torch.Tensor,
                               debug: bool = False) -> torch.Tensor:
        """
        Minimal commodity-residual enhancement (original architecture kept):

          H          = batch-aware graph propagation (B, 284, 64)
          h_global   = original type pooling → (B, 64)
          h_comm     = commodity slice (B, 24, 64), never pooled
          h_base     = original gate fusion of [global graph, global LSTM] → (B, 64)
          z          = h_base.unsqueeze(1) + α · Linear(h_comm) → (B, 24, 64)
          pred       = shared head per commodity → (B, H, 24)

        α (init 0.05) is learnable; α → 0 degenerates to the pooled model.
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── 1. Batch-aware graph propagation → node representations ──
        H = self._batch_spatial_forward(x)                             # (B, N, 64)
        # ── 2. Original global pooling (kept for baseline performance) ──
        h_global_graph = self.type_pool(H)                             # (B, 64)
        # ── 3. Commodity nodes — never pooled ──
        h_comm = H[:, fut_start:, :]                                   # (B, 24, 64)
        # ── 4. Original global LSTM ──
        x_seq = x.reshape(B, T, -1)                                    # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                             # (B, 64)
        # ── 5. h_base = original gate fusion ──
        combined = torch.cat([h_global_graph, lstm_out], dim=-1)       # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        h_base = gate * self.lstm_proj(lstm_out) + (1 - gate) * self.gcn_proj(h_global_graph)  # (B, 64)
        # ── 6. Commodity residual adapter ──
        z = h_base.unsqueeze(1) + self.comm_alpha * self.comm_proj(h_comm)  # (B, 24, 64)
        # ── 7. Shared head per commodity ──
        preds = self.comm_head(z)                                      # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                              # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                   # (B, 24)

        if debug:
            print(f"  [comm_residual] H={list(H.shape)} "
                  f"h_global={list(h_global_graph.shape)} h_comm={list(h_comm.shape)} "
                  f"α={self.comm_alpha.item():.4f} z={list(z.shape)} pred={list(pred.shape)}")
        return pred

    def _temp_weighted_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        The TempWeighted spatial branch (shared by temporal_weighted_graph
        and the RegimeRPETransformer family):

          type_proj per t → H_seq → temporal attention over T → H (B, N, 64)
          → EdgeAttnMixHop ×2 (once) → LayerNorm → type_pool → (B, 64)
        """
        B, T, N, _ = x.shape
        seqs = [self.type_proj(x[:, t, :, :], self.n_stock, self.n_bond)
                for t in range(T)]
        H_seq = torch.stack(seqs, dim=1)                               # (B, T, N, 64)
        score = self.temporal_score(H_seq)                             # (B, T, N, 1)
        alpha = F.softmax(score, dim=1)
        H = (H_seq * alpha).sum(dim=1)                                 # (B, N, 64)
        self.last_alpha = alpha.squeeze(-1).detach()
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(H, A))                           # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)
        return self.type_pool(h)                                       # (B, 64)

    def _temporal_weighted_graph_forward(self, x: torch.Tensor,
                                         debug: bool = False) -> torch.Tensor:
        """
        ONLY replaces mean(dim=1) with node-wise temporal attention.
        GNN (AdaptiveGraphLearner → MixHop/EdgeAttn ×2), global LSTM,
        gate fusion, original head — all unchanged (GNN still runs once).
        """
        gcn_out = self._temp_weighted_spatial(x)                       # (B, 64)
        pred = self._original_forward(x, debug=debug, gcn_out=gcn_out)
        if debug:
            print(f"  [temporal_weighted_graph] → {list(gcn_out.shape)} → "
                  f"{list(pred.shape)}")
        return pred

    def _temporal_cross_weighted_forward(self, x: torch.Tensor,
                                         debug: bool = False) -> torch.Tensor:
        """
        TempWeighted + neighbor-consulting temporal attention:

          score      = Linear(64→1)(H_seq)                    own score (B, T, N, 1)
          A_norm     = row-normalized adjacency (self-loops)  (N, N)
          score_neigh = A_norm @ score                        neighbor scores (B, T, N)
          score_final = Linear(2→1)([score, score_neigh])     fused (B, T, N, 1)
          α = softmax(score_final, dim=1);  H = Σ α·H_seq

        Everything after (GNN ×2 → type_pool → LSTM → gate → head) is the
        unchanged original path (shared via _original_forward).
        """
        B, T, N, _ = x.shape

        # ── Temporal-weighted aggregation with neighbor-consulting scores ──
        seqs = [self.type_proj(x[:, t, :, :], self.n_stock, self.n_bond)
                for t in range(T)]
        H_seq = torch.stack(seqs, dim=1)                               # (B, T, N, 64)
        score = self.temporal_score(H_seq)                             # (B, T, N, 1)

        A = self.graph_learner() if self.use_learn_graph else self.static_A
        A_hat = A + torch.eye(N, device=A.device)
        deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_norm = A_hat / deg                                           # (N, N)

        score_neigh = torch.einsum(
            'btn,ni->bti', score.squeeze(-1), A_norm).unsqueeze(-1)    # (B, T, N, 1)
        score_final = self.cross_score_fuse(
            torch.cat([score, score_neigh], dim=-1))                   # (B, T, N, 1)
        alpha = F.softmax(score_final, dim=1)                          # (B, T, N, 1)
        H = (H_seq * alpha).sum(dim=1)                                 # (B, N, 64)
        self.last_alpha = alpha.squeeze(-1).detach()

        # ── Existing GNN (batch mode, runs ONCE) + type pooling ──
        h1 = F.relu(self.attn_mixhop1(H, A))                           # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h = self.gcn_norm(h2)
        gcn_out = self.type_pool(h)                                    # (B, 64)

        # ── Original LSTM + gate + head (shared code) ──
        pred = self._original_forward(x, debug=debug, gcn_out=gcn_out)

        if debug:
            print(f"  [temporal_cross_weighted] H_seq={list(H_seq.shape)} "
                  f"score={list(score.shape)} score_neigh={list(score_neigh.shape)} "
                  f"α mean={alpha.mean().item():.4f} → {list(H.shape)} → {list(pred.shape)}")
        return pred

    def _temp_weighted_comm_cond_forward(self, x: torch.Tensor,
                                         debug: bool = False) -> torch.Tensor:
        """
        TempWeighted + commodity-conditioned hidden state:

          H       = temporal-weighted aggregation → GNN ×2 → (B, N, 64)
          h_comm  = H[:, 260:284] (never pooled)                  (B, 24, 64)
          fused   = original gate fusion (global graph + global LSTM) (B, 64)
          h       = head[0:3](fused)      ← original head body      (B, 64)
          h_cond  = h.unsqueeze(1) + α·comm_proj(h_comm)          (B, 24, 64)
          pred[b,h,i] = W_orig[h*24+i] @ h_cond[b,i] + b_orig[h*24+i]
                     using the ORIGINAL head[3] weights.

        α = 0 → h_cond == h (per commodity identical) → pred == TempWeighted.
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── 1. Temporal-weighted aggregation (same as TempWeighted) ──
        seqs = [self.type_proj(x[:, t, :, :], self.n_stock, self.n_bond)
                for t in range(T)]
        H_seq = torch.stack(seqs, dim=1)                           # (B, T, N, 64)
        score = self.temporal_score(H_seq)                         # (B, T, N, 1)
        alpha_t = F.softmax(score, dim=1)
        H = (H_seq * alpha_t).sum(dim=1)                           # (B, N, 64)
        self.last_alpha = alpha_t.squeeze(-1).detach()

        # ── 2. GNN ×2 (batch mode, runs once) ──
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        h1 = F.relu(self.attn_mixhop1(H, A))                       # (B, N, 64)
        h2 = self.attn_mixhop2(h1, A)
        h_spatial = self.gcn_norm(h2)                              # (B, N, 64)

        # ── 3. Original global path: pooling → LSTM → gate fusion ──
        gcn_out = self.type_pool(h_spatial)                        # (B, 64)
        x_seq = x.reshape(B, T, -1)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        lstm_out = h_n[-1]                                         # (B, 64)
        combined = torch.cat([gcn_out, lstm_out], dim=-1)          # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = gate * self.lstm_proj(lstm_out) + (1 - gate) * self.gcn_proj(gcn_out)  # (B, 64)

        # base prediction (diagnostic — the TempWeighted output)
        base_pred = self.head(fused)
        if self.n_horizons > 1:
            self.last_base_pred = base_pred.view(B, self.n_horizons, self.n_commodities).detach()
        else:
            self.last_base_pred = base_pred.view(B, self.n_commodities).detach()

        # ── 4. Original head body (first 3 modules) ──
        h = self.head[0](fused)                                    # (B, 64)
        h = self.head[1](h)
        h = self.head[2](h)                                        # (B, 64)

        # ── 5. Commodity-conditioned hidden state ──
        h_comm = h_spatial[:, fut_start:, :]                       # (B, 24, 64)
        if self.shuffle_comm:                                      # control: permute
            perm = torch.randperm(self.n_commodities, device=h_comm.device)
            h_comm = h_comm[:, perm, :]
        h_comm_proj = self.comm_proj(h_comm)                       # (B, 24, 64)
        self.last_cond_mag = (self.alpha * h_comm_proj).abs().mean().item()
        h_expand = h.unsqueeze(1).expand(B, self.n_commodities, -1)
        h_cond = h_expand + self.alpha * h_comm_proj               # (B, 24, 64)

        # ── 6. ORIGINAL per-commodity output layer (head[3]) ──
        # pred[b,h,i] = Σ_c h_cond[b,i,c] · W_orig[h*24+i, c] + bias[h*24+i]
        W = self.head[3].weight.view(self.n_horizons, self.n_commodities, -1)  # (H, 24, 64)
        b = self.head[3].bias.view(self.n_horizons, self.n_commodities)       # (H, 24)
        pred = torch.einsum('bic,hic->bhi', h_cond, W) + b.unsqueeze(0)       # (B, H, 24)
        if self.n_horizons == 1:
            pred = pred.squeeze(1)                                 # (B, 24)

        if debug:
            print(f"  [temp_weighted_comm_cond] α={self.alpha.item():.4f} "
                  f"h_spatial={list(h_spatial.shape)} h_comm={list(h_comm.shape)} "
                  f"h_cond={list(h_cond.shape)} pred={list(pred.shape)}")
        return pred

    def _transformer_temporal_forward(self, x: torch.Tensor,
                                      debug: bool = False) -> torch.Tensor:
        """
        RegimeRPETransformer family — temporal branch replaced by
        transformer (+ soft regime + regime-aware RPE), spatial branch
        kept EXACTLY as TempWeighted, original gated fusion + head kept.

          spatial: _temp_weighted_spatial(x) → h_spatial (B, 64)
          temporal: x.reshape(B,T,N*F) → _TemporalTransformerBranch → (B, 64)
          fused = gate fusion (original gate_fc/gcn_proj/lstm_proj)
          pred = original head(fused)
        """
        B, T, N, _ = x.shape

        # ── Spatial branch (TempWeighted, unchanged) ──
        h_spatial = self._temp_weighted_spatial(x)                   # (B, 64)

        # ── Temporal branch (transformer family) ──
        x_seq = x.reshape(B, T, -1)                                  # (B, T, N*F)
        h_temporal = self.temporal_transformer(x_seq)                # (B, 64)

        # ── Original gated fusion + head ──
        combined = torch.cat([h_spatial, h_temporal], dim=-1)        # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = gate * self.lstm_proj(h_temporal) + (1 - gate) * self.gcn_proj(h_spatial)
        pred = self.head(fused)                                      # (B, H*24)
        if self.n_horizons > 1:
            pred = pred.view(B, self.n_horizons, self.n_commodities)
        else:
            pred = pred.view(B, self.n_commodities)

        if debug:
            print(f"  [{self.variant}] h_spatial={list(h_spatial.shape)} "
                  f"h_temporal={list(h_temporal.shape)} → gate → {list(pred.shape)}")
        return pred

    def _market_token_predict(self, h_spatial: torch.Tensor,
                              h_temporal: torch.Tensor) -> torch.Tensor:
        """Apply the existing gated fusion and prediction head unchanged."""
        batch_size = h_spatial.shape[0]
        combined = torch.cat([h_spatial, h_temporal], dim=-1)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = (
            gate * self.lstm_proj(h_temporal)
            + (1 - gate) * self.gcn_proj(h_spatial)
        )
        pred = self.head(fused)
        if self.n_horizons > 1:
            return pred.view(batch_size, self.n_horizons, self.n_commodities)
        return pred.view(batch_size, self.n_commodities)

    def _market_token_transformer_forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
        zero_market: str = None,
        zero_component: str = None,
    ) -> torch.Tensor:
        """S0/S0D temporal replacement with the exact TempWeighted spatial path."""
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.market_token_transformer(
            x,
            zero_market=zero_market,
            zero_component=zero_component,
        )
        pred = self._market_token_predict(h_spatial, h_temporal)
        if debug:
            print(
                f"  [{self.variant}] h_spatial={list(h_spatial.shape)} "
                f"h_temporal={list(h_temporal.shape)} → gate → {list(pred.shape)}"
            )
        return pred

    def _switching_transformer_forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
        forced_probabilities: torch.Tensor = None,
    ) -> torch.Tensor:
        """S1 switching temporal branch with unchanged spatial/fusion paths."""
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_transformer(
            x, forced_probabilities=forced_probabilities
        )
        pred = self._market_token_predict(h_spatial, h_temporal)
        if debug:
            print(
                f"  [{self.variant}] h_spatial={list(h_spatial.shape)} "
                f"h_temporal={list(h_temporal.shape)} → gate → {list(pred.shape)}"
            )
        return pred

    def _switching_filter_rpe_forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
        forced_rpe_probabilities: torch.Tensor = None,
    ) -> torch.Tensor:
        """S2F temporal branch with unchanged spatial/fusion/head paths."""
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_filter_rpe(
            x, forced_rpe_probabilities=forced_rpe_probabilities
        )
        pred = self._market_token_predict(h_spatial, h_temporal)
        if debug:
            print(
                f"  [{self.variant}] h_spatial={list(h_spatial.shape)} "
                f"h_temporal={list(h_temporal.shape)} → gate → {list(pred.shape)}"
            )
        return pred

    def _switching_latent_transformer_forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
        forced_probabilities: torch.Tensor = None,
        zero_readout_component: str = None,
        zero_latent_memory: bool = False,
    ) -> torch.Tensor:
        """D0 temporal replacement with the unchanged spatial/fusion/head path."""
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_latent_transformer(
            x,
            forced_probabilities=forced_probabilities,
            zero_readout_component=zero_readout_component,
            zero_latent_memory=zero_latent_memory,
        )
        pred = self._market_token_predict(h_spatial, h_temporal)
        if debug:
            print(
                f"  [{self.variant}] h_spatial={list(h_spatial.shape)} "
                f"h_temporal={list(h_temporal.shape)} → gate → {list(pred.shape)}"
            )
        return pred

    def regime_diversity_loss(self, lambda_proto: float = 0.001) -> torch.Tensor:
        """
        Lightweight regime diversity regularization (prototype-based only):
          L_proto   = mean pairwise cosine across prototypes  (separate them)
          L_balance = Σ_k p̄_k·log(p̄_k·K) on batch-level p̄    (no monopoly)
        Returns 0 if the variant has no prototype regime generator.
        """
        tb = getattr(self, 'temporal_transformer', None)
        if tb is None:
            return torch.zeros((), device=next(self.parameters()).device)
        rg = getattr(tb, 'regime_gen', None)
        if rg is None or not hasattr(rg, 'regime_prototypes'):
            return torch.zeros((), device=next(self.parameters()).device)
        # 1. prototype separation
        l_proto = rg.prototype_cosine()
        # 2. batch-level state balance
        p = getattr(tb, 'last_regime_p', None)
        l_balance = torch.zeros((), device=l_proto.device)
        if p is not None:
            p_mean = p.mean(dim=(0, 1))                            # (K,)
            l_balance = (p_mean * torch.log(p_mean * rg.K + 1e-8)).sum()
        return lambda_proto * (l_proto + l_balance)

    def _regime_dynamic_forward(self, x: torch.Tensor,
                                debug: bool = False,
                                market_descriptor: torch.Tensor = None) -> torch.Tensor:
        """
        RegimeDynamicRPETransformer (F):
          spatial  = TempWeighted (unchanged) → h_spatial (B, 64)
          temporal = RegimeDynamicRPETransformer branch → h_temporal (B, 64)
          fused    = original gated fusion; pred = original head
        """
        B, T, N, _ = x.shape
        h_spatial = self._temp_weighted_spatial(x)                 # (B, 64)

        # ── Causal market-dynamics descriptor (F2 state target) ──
        # m_t = [abs_ret, vol, slope] from commodity ret_1d features
        # (z-scored, causal: uses only time ≤ t)
        if self.regime_dynamic.use_state_loss:
            fut_start = self.n_stock + self.n_bond
            mr = x[:, :, fut_start:, 1].mean(dim=2)                # (B, T) market ret
            abs_ret = mr.abs()
            vol = torch.zeros_like(mr)
            for t in range(1, T):
                vol[:, t] = mr[:, :t + 1].std(dim=1)
            slope = torch.zeros_like(mr)
            slope[:, 1:] = mr[:, 1:] - mr[:, :-1]
            m_t = torch.stack([abs_ret, vol, slope], dim=-1)       # (B, T, 3)
            self.regime_dynamic._m_t_for_loss = m_t                # non-detached
            self.regime_dynamic.last_m_t = m_t.detach()

        x_flat = x.reshape(B, T, -1)                               # (B, T, N*F)
        h_temporal = self.regime_dynamic(
            x_flat, market_descriptor=market_descriptor
        )                                                           # (B, 64)

        combined = torch.cat([h_spatial, h_temporal], dim=-1)      # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = gate * self.lstm_proj(h_temporal) + (1 - gate) * self.gcn_proj(h_spatial)
        pred = self.head(fused)                                    # (B, H*24)
        if self.n_horizons > 1:
            pred = pred.view(B, self.n_horizons, self.n_commodities)
        else:
            pred = pred.view(B, self.n_commodities)

        if debug:
            print(f"  [regime_dynamic] h_spatial={list(h_spatial.shape)} "
                  f"h_temporal={list(h_temporal.shape)} → {list(pred.shape)}")
        return pred

    def _comm_output_residual_forward(self, x: torch.Tensor,
                                      debug: bool = False) -> torch.Tensor:
        """
        Output residual on top of THE original edge_attn path:

          base_pred, H = self._original_forward(x, return_h=True)
              ← EXACTLY the same code as edge_attn
          h_comm       = H[commodity_idx]  (never pooled)
          residual     = Linear(64→32)→ReLU→Linear(32→H) on h_comm
          pred         = base_pred + α · residual   (α init 0.01)

        α = 0 → pred strictly equals edge_attn output.
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── 1-4. THE original path (shared code with edge_attn) ──
        base_pred, H = self._original_forward(x, return_h=True)      # (B, H, 24), (N, 64)

        # ── 5. Commodity output residual (added AFTER the original path) ──
        h_comm = H[fut_start:].unsqueeze(0).expand(B, -1, -1)        # (B, 24, 64)
        residual = self.residual_head(h_comm)                        # (B, 24, H)
        if self.n_horizons > 1:
            residual = residual.permute(0, 2, 1)                     # (B, H, 24)
        else:
            residual = residual.squeeze(-1)                          # (B, 24)

        # ── 6. Combine ──
        pred = base_pred + self.residual_alpha * residual

        # store for diagnostics
        self.last_base_pred = base_pred.detach()
        self.last_residual = residual.detach()

        if debug:
            print(f"  [comm_output_residual] α={self.residual_alpha.item():.4f} "
                  f"base={list(base_pred.shape)} residual={list(residual.shape)} "
                  f"pred={list(pred.shape)}")
        return pred

    def _spatial_temporal_attention_forward(self, x: torch.Tensor,
                                            debug: bool = False) -> torch.Tensor:
        """
        Per-timestep spatial propagation + temporal attention:

          for t in 1..T:
            x_t → type_proj → GNN(EdgeAttnMixHop on shared A) → h_t (B, N, 64)
          h_seq = stack(h_t)                     (B, T, N, 64)
          α_t   = softmax(Linear(64→1)(h_t), over T)
          h_spatial = Σ_t α_t · h_t              (B, N, 64)   ← replaces mean(dim=1)

        Global branch (kept): type_pool(h_spatial) + global LSTM → original
        fusion/head → pred_global (auxiliary supervision keeps it trained).
        Commodity branch (never pooled): h_comm = h_spatial[:, 24:] →
        original head body (shared) → comm_out → (B, H, 24).
        """
        B, T, N, _ = x.shape
        fut_start = self.n_stock + self.n_bond

        # ── 1. Per-timestep GNN propagation ──
        A = self.graph_learner() if self.use_learn_graph else self.static_A
        seqs = []
        for t in range(T):
            xt = x[:, t, :, :]                                       # (B, N, F)
            x_proj = self.type_proj(xt, self.n_stock, self.n_bond)   # (B, N, 64)
            h1 = F.relu(self.attn_mixhop1(x_proj, A))                # (B, N, 64)
            h2 = self.attn_mixhop2(h1, A)
            seqs.append(self.gcn_norm(h2))
        h_seq = torch.stack(seqs, dim=1)                             # (B, T, N, 64)

        # ── 2. Temporal attention over steps ──
        scores = self.temporal_score(h_seq).squeeze(-1)              # (B, T, N)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)               # (B, T, N, 1)
        h_spatial = (h_seq * alpha).sum(dim=1)                       # (B, N, 64)

        # ── 3. Global pooling (kept) + commodity slice (never pooled) ──
        h_global_spatial = self.type_pool(h_spatial)                 # (B, 64)
        h_comm = h_spatial[:, fut_start:, :]                         # (B, 24, 64)

        # ── 4. Global LSTM (kept) ──
        x_seq = x.reshape(B, T, -1)                                  # (B, T, N*F)
        lstm_out, (h_n, _) = self.temporal(x_seq)
        h_lstm = h_n[-1]                                             # (B, 64)

        # ── 5. Global branch: original fusion + head (auxiliary) ──
        combined = torch.cat([h_global_spatial, h_lstm], dim=-1)     # (B, 128)
        gate = torch.sigmoid(self.gate_fc(combined))
        fused = gate * self.lstm_proj(h_lstm) + (1 - gate) * self.gcn_proj(h_global_spatial)
        pred_global = self.head(fused)                               # (B, H*24)
        if self.n_horizons > 1:
            pred_global = pred_global.view(B, self.n_horizons, self.n_commodities)
        else:
            pred_global = pred_global.view(B, self.n_commodities)
        self.last_aux_pred = pred_global.detach()                    # auxiliary supervision
        self.last_pred_global = pred_global.detach()

        # ── 6. Commodity branch: original head body (shared) + comm_out ──
        body = self.head[0:3](h_comm)                                # (B, 24, 64) → (B, 24, FC)
        preds = self.comm_out(body)                                  # (B, 24, H)
        if self.n_horizons > 1:
            pred = preds.permute(0, 2, 1)                            # (B, H, 24)
        else:
            pred = preds.squeeze(-1)                                 # (B, 24)

        if debug:
            print(f"  [spatial_temporal_attention] h_seq={list(h_seq.shape)} "
                  f"h_spatial={list(h_spatial.shape)} h_global={list(h_global_spatial.shape)} "
                  f"h_comm={list(h_comm.shape)} pred={list(pred.shape)}")
        return pred

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_weight=None,
                debug: bool = False,
                market_descriptor=None) -> torch.Tensor:
        if self.variant == "multiscale_graph":
            return self._multiscale_graph_forward(x, debug)
        if self.variant == "node_level":
            return self._node_level_forward(x, debug)
        if self.variant == "comm_nodes":
            return self._comm_nodes_forward(x, debug)
        if self.variant == "batch_graph":
            return self._batch_graph_forward(x, debug)
        if self.variant == "factor_res":
            return self._factor_res_forward(x, debug)
        if self.variant == "node_wise":
            return self._node_wise_forward(x, debug)
        if self.variant in ("market_node", "market_node_no_graph"):
            return self._market_node_forward(x, debug)
        if self.variant == "mkt_node":
            return self._mkt_node_forward(x, debug)
        if self.variant == "comm_residual":
            return self._comm_residual_forward(x, debug)
        if self.variant == "comm_output_residual":
            return self._comm_output_residual_forward(x, debug)
        if self.variant == "spatial_temporal_attention":
            return self._spatial_temporal_attention_forward(x, debug)
        if self.variant == "temporal_weighted_graph":
            return self._temporal_weighted_graph_forward(x, debug)
        if self.variant == "temp_weighted_comm_cond":
            return self._temp_weighted_comm_cond_forward(x, debug)
        if self.variant == "temporal_cross_weighted":
            return self._temporal_cross_weighted_forward(x, debug)
        if self.variant in ("transformer_temporal", "transformer_rpe",
                            "transformer_regime", "regime_rpe_transformer",
                            "transformer_regime2", "regime_rpe_transformer2"):
            return self._transformer_temporal_forward(x, debug)
        if self.variant in ("market_token_transformer", "market_dispersion_transformer"):
            return self._market_token_transformer_forward(x, debug)
        if self.variant in ("switching_transformer", "switching_null_control"):
            return self._switching_transformer_forward(x, debug)
        if self.variant == "switching_filter_rpe":
            return self._switching_filter_rpe_forward(x, debug)
        if self.variant in (
            "switching_latent_transformer",
            "switching_latent_balanced_readout",
            "switching_latent_memory",
        ):
            return self._switching_latent_transformer_forward(x, debug)
        if self.variant in ("regime_dynamic_transformer", "regime_dynamic_semantic",
                            "semantic_router", "loss_rebalance", "routing_strength",
                            "context_calibrated", "context_balance",
                            "adapter_orthogonal"):
            return self._regime_dynamic_forward(
                x, market_descriptor=market_descriptor, debug=debug
            )
        if self.variant == "edge_attn":
            # single source of truth shared with comm_output_residual
            return self._original_forward(x, debug=debug)

        B, T, N, n_feat = x.shape

        # ── 1. Spatial branch (MixHop / single-hop GCN) ──
        gcn_out = self._spatial_forward(x) if self.use_gcn else None   # (B, 64)

        # ── 2. Temporal branch (LSTM or TCN) ──
        if self.use_lstm:
            if self.variant == "temporal_attn":
                lstm_out = self._temporal_attn_forward(x)             # (B, 64)
            elif self.variant == "tcn_temporal":
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                tcn_out = self.tcn(x_seq)                             # (B, T, 64)
                # Multi-scale temporal pooling (same as multiscale_time)
                h_last = tcn_out[:, -1, :]                            # (B, 64)
                h_all  = tcn_out.mean(dim=1)                          # (B, 64)
                h_10   = tcn_out[:, -10:, :].mean(dim=1)              # (B, 64)
                h_5    = tcn_out[:, -5:, :].mean(dim=1)               # (B, 64)
                lstm_out = F.relu(self.ms_fuse(
                    torch.cat([h_last, h_all, h_10, h_5], dim=-1)))   # (B, 64)
            elif self.variant == "patch_temporal":
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                lstm_out = self.patch(x_seq)                          # (B, 64)
            elif self.variant == "mamba_temporal":
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                lstm_out = self.mamba(x_seq)                          # (B, 64)
            elif self.variant == "informer_temporal":
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                lstm_out = self.informer(x_seq)                       # (B, 64)
            elif self.variant == "attn_pool":
                x_seq = x.reshape(B, T, -1)                           # (B, T, N*F)
                lstm_out_seq, (h_n, _) = self.temporal(x_seq)         # (B, T, 64)
                # Multi-scale attention pooling (windows 5/10/20 + last)
                h_last = lstm_out_seq[:, -1, :]                       # (B, 64)
                h_5  = self.pool5(lstm_out_seq[:, -5:, :])            # (B, 64)
                h_10 = self.pool10(lstm_out_seq[:, -10:, :])          # (B, 64)
                h_20 = self.pool20(lstm_out_seq[:, -20:, :])          # (B, 64)
                lstm_out = F.relu(self.ms_fuse(
                    torch.cat([h_last, h_5, h_10, h_20], dim=-1)))    # (B, 64)
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
        if self.variant in (
            "market_token_transformer",
            "market_dispersion_transformer",
            "switching_transformer",
            "switching_null_control",
            "switching_filter_rpe",
            "switching_latent_transformer",
            "switching_latent_balanced_readout",
            "switching_latent_memory",
        ):
            self.eval()
            with torch.no_grad():
                h_spatial = self._temp_weighted_spatial(x)
                temporal_branch = (
                    self.switching_latent_transformer
                    if self.variant in (
                        "switching_latent_transformer",
                        "switching_latent_balanced_readout",
                        "switching_latent_memory",
                    )
                    else (
                        self.switching_filter_rpe
                        if self.variant == "switching_filter_rpe"
                        else (
                            self.switching_transformer
                            if self.variant in (
                                "switching_transformer", "switching_null_control"
                            )
                            else self.market_token_transformer
                        )
                    )
                )
                h_temporal = temporal_branch(x)
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([h_spatial, h_temporal], dim=-1))
                )
            return {
                "mode": self.variant,
                "gate_mean": gate.mean().item(),
                "gate_std": gate.std(unbiased=False).item(),
                "gate_min": gate.min().item(),
                "gate_max": gate.max().item(),
            }
        if self.variant in ("node_level", "comm_nodes", "batch_graph",
                            "factor_res", "node_wise",
                            "market_node", "market_node_no_graph", "mkt_node",
                            "comm_residual", "comm_output_residual",
                            "spatial_temporal_attention",
                            "temporal_weighted_graph",
                            "temp_weighted_comm_cond",
                            "temporal_cross_weighted",
                            "transformer_temporal", "transformer_rpe",
                            "transformer_regime", "regime_rpe_transformer",
                            "transformer_regime2", "regime_rpe_transformer2",
                            "regime_dynamic_transformer", "regime_dynamic_semantic",
                            "semantic_router", "loss_rebalance", "routing_strength",
                            "context_calibrated", "context_balance",
                            "adapter_orthogonal"):
            return {'mode': self.variant, 'note': 'no gate — per-node fusion'}
        if self.variant == "multiscale_graph":
            self.eval()
            with torch.no_grad():
                B, T, N, _ = x.shape
                x_gcn = x.mean(dim=0).permute(1, 0, 2)
                x_proj = self.type_proj(x_gcn, self.n_stock, self.n_bond)
                x_proj = x_proj.mean(dim=1)
                h_s1 = F.relu(self.attn_short1(x_proj, self.static_A_short))
                h_s2 = self.attn_short2(h_s1, self.static_A_short)
                gcn_short = self.type_pool_s(self.gcn_norm_s(h_s2)).unsqueeze(0).expand(B, -1)
                h_l1 = F.relu(self.attn_long1(x_proj, self.static_A_long))
                h_l2 = self.attn_long2(h_l1, self.static_A_long)
                gcn_long = self.type_pool_l(self.gcn_norm_l(h_l2)).unsqueeze(0).expand(B, -1)
                x_seq = x.reshape(B, T, -1)
                lstm_out, (h_n, _) = self.temporal(x_seq)
                lstm_out = h_n[-1]
                combined = torch.cat([gcn_short, gcn_long, lstm_out], dim=-1)
                gate = F.softmax(self.fuse3(combined), dim=-1)
            return {
                'gate_short': gate[:, 0].mean().item(),
                'gate_long':  gate[:, 1].mean().item(),
                'gate_lstm':  gate[:, 2].mean().item(),
            }
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
            elif self.variant == "tcn_temporal":
                x_seq = x.reshape(B, x.size(1), -1)
                tcn_out = self.tcn(x_seq)
                h_last = tcn_out[:, -1, :]
                h_all  = tcn_out.mean(dim=1)
                h_10   = tcn_out[:, -10:, :].mean(dim=1)
                h_5    = tcn_out[:, -5:, :].mean(dim=1)
                lstm_out = F.relu(self.ms_fuse(
                    torch.cat([h_last, h_all, h_10, h_5], dim=-1)))
            elif self.variant == "patch_temporal":
                x_seq = x.reshape(B, x.size(1), -1)
                lstm_out = self.patch(x_seq)
            elif self.variant == "mamba_temporal":
                x_seq = x.reshape(B, x.size(1), -1)
                lstm_out = self.mamba(x_seq)
            elif self.variant == "informer_temporal":
                x_seq = x.reshape(B, x.size(1), -1)
                lstm_out = self.informer(x_seq)
            elif self.variant == "attn_pool":
                x_seq = x.reshape(B, x.size(1), -1)
                lstm_out_seq, (h_n, _) = self.temporal(x_seq)
                h_last = lstm_out_seq[:, -1, :]
                h_5  = self.pool5(lstm_out_seq[:, -5:, :])
                h_10 = self.pool10(lstm_out_seq[:, -10:, :])
                h_20 = self.pool20(lstm_out_seq[:, -20:, :])
                lstm_out = F.relu(self.ms_fuse(
                    torch.cat([h_last, h_5, h_10, h_20], dim=-1)))
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
