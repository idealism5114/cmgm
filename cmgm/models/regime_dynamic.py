"""
RegimeDynamicRPETransformer — modular temporal branch.

Pipeline (all shapes documented):
    x_flat (B, T, 5964)
      → proj → E (B, T, 128)
      → SoftRegimeGeneratorV2 → p (B, T, K)
      → RegimeTemporalAdapter (K independent dynamics) → z (B, T, 128)
      → RegimeAwareRPE (from p) → rpe (B, n_heads, T, T)
      → RegimeAwareTemporalAttention × n_layers → H (B, T, 128)
      → temporal attention pooling → Linear(128→64) → h_temporal (B, 64)

Causality: p_t depends only on (E_t, E_t−E_{t−1}, p_{t−1}); no future info.
Sample-specific recursion: p_prev = p_t (NO batch averaging).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRegimeGeneratorV2(nn.Module):
    """
    Soft market regime generator:
      context_t = MLP([E_t, E_t − E_{t−1}])                     (B, T, D_r)
      sim_tk    = context_t · prototype_k / √D_r                (B, T, K)
      trans_tk  = p_{t−1} @ transition_matrix                   (B, K)
      score_tk  = sim_tk + trans_tk
      p_t       = softmax(score_tk / temperature)

    Input:  E (B, T, d_model)
    Output: p (B, T, K)
    """

    def __init__(self, d_model: int = 128, K: int = 3, D_regime: int = 32):
        super().__init__()
        self.K = K
        self.D_regime = D_regime
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, D_regime),
        )
        self.regime_prototypes = nn.Parameter(torch.randn(K, D_regime) * 0.02)
        self.transition_matrix = nn.Parameter(torch.zeros(K, K))
        self.initial_regime_logits = nn.Parameter(torch.zeros(K))
        self.temperature = 1.0

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        B, T, d = E.shape
        delta = torch.zeros_like(E)
        delta[:, 1:] = E[:, 1:] - E[:, :-1]                       # dynamic slope
        context = self.context_encoder(torch.cat([E, delta], dim=-1))  # (B, T, 32)
        sim = context @ self.regime_prototypes.T / math.sqrt(self.D_regime)  # (B, T, K)

        # sample-specific recursion (per-sample p_prev, no batch averaging)
        p_prev = F.softmax(self.initial_regime_logits, dim=-1)    # (K,)
        p_prev = p_prev.unsqueeze(0).expand(B, -1)                # (B, K)
        ps = []
        for t in range(T):
            trans = p_prev @ self.transition_matrix               # (B, K)
            score = sim[:, t] + trans
            p_t = F.softmax(score / self.temperature, dim=-1)     # (B, K)
            ps.append(p_t)
            p_prev = p_t                                          # sample-specific ✓
        return torch.stack(ps, dim=1)                             # (B, T, K)


class SemanticRegimeRouter(nn.Module):
    """G router: neural context + market semantics + previous soft state."""

    def __init__(self, d_model: int = 128, K: int = 3,
                 D_regime: int = 32, gamma: float = 5.0,
                 tau_m: float = 1.0, lambda_sem: float = 1.0):
        super().__init__()
        self.K = K
        self.gamma = gamma
        self.tau_m = tau_m
        self.lambda_sem = lambda_sem
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model * 2 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, D_regime),
        )
        self.regime_prototypes = nn.Parameter(torch.randn(K, D_regime) * 0.02)
        self.transition_matrix = nn.Parameter(torch.zeros(K, K))
        self.initial_regime_logits = nn.Parameter(torch.zeros(K))

    def forward(self, E: torch.Tensor, descriptor: torch.Tensor,
                state_centers: torch.Tensor) -> torch.Tensor:
        B, T, _ = E.shape
        if descriptor.shape != (B, T, 3):
            raise ValueError(
                f"descriptor must have shape {(B, T, 3)}, got {tuple(descriptor.shape)}"
            )

        delta = torch.zeros_like(E)
        delta[:, 1:] = E[:, 1:] - E[:, :-1]
        context = self.context_encoder(torch.cat([E, delta, descriptor], dim=-1))
        context_hat = F.normalize(context, p=2, dim=-1, eps=1e-8)
        prototype_hat = F.normalize(
            self.regime_prototypes, p=2, dim=-1, eps=1e-8
        )
        context_scores = self.gamma * torch.einsum(
            'btd,kd->btk', context_hat, prototype_hat
        )
        distances = ((
            descriptor.unsqueeze(2) - state_centers.view(1, 1, self.K, 3)
        ) ** 2).sum(dim=-1)
        semantic_scores = -distances / self.tau_m

        p_prev = F.softmax(self.initial_regime_logits, dim=-1)
        p_prev = p_prev.unsqueeze(0).expand(B, -1)
        probabilities, transition_scores, final_scores = [], [], []
        for step in range(T):
            transition = p_prev @ self.transition_matrix
            score = (
                context_scores[:, step]
                + self.lambda_sem * semantic_scores[:, step]
                + transition
            )
            p_t = F.softmax(score, dim=-1)
            probabilities.append(p_t)
            transition_scores.append(transition)
            final_scores.append(score)
            p_prev = p_t

        p = torch.stack(probabilities, dim=1)
        self.last_context_score = context_scores.detach()
        self.last_semantic_score = semantic_scores.detach()
        self.last_transition_score = torch.stack(transition_scores, dim=1).detach()
        self.last_final_score = torch.stack(final_scores, dim=1).detach()
        self.last_distances = distances.detach()
        return p


class RegimeTemporalAdapter(nn.Module):
    """
    K independent regime-specific temporal dynamics adapters:
      z^(k) = MLP(E)  (k = 1..K, independent parameters)
      z     = Σ_k p_tk · z^(k)_t                                 (soft mixing)

    Input:  E (B, T, d_model), p (B, T, K)
    Output: z (B, T, d_model), zs (K, B, T, d_model)
    """

    def __init__(self, d_model: int = 128, K: int = 3):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, d_model),
            )
            for _ in range(K)
        ])

    def forward(self, E: torch.Tensor, p: torch.Tensor):
        zs = torch.stack([ad(E) for ad in self.adapters], dim=0)  # (K, B, T, d)
        z = torch.einsum('btk,kbtc->btc', p, zs)                  # (B, T, d)
        return z, zs


class RegimeAwareRPE(nn.Module):
    """
    Regime-aware relative position encoding:
      r_ij = base_rpe[Δ] + Σ_k p_i[k] · regime_rpe[k, Δ],  Δ = i − j

    Input:  p (B, T, K)
    Output: rpe (B, n_heads, T, T)
    """

    def __init__(self, t_len: int = 20, n_heads: int = 4, K: int = 3):
        super().__init__()
        self.base_rpe = nn.Parameter(torch.randn(2 * t_len - 1, n_heads) * 0.02)
        self.regime_rpe = nn.Parameter(torch.randn(K, 2 * t_len - 1, n_heads) * 0.02)

    def forward(self, p: torch.Tensor, t_len: int):
        B, T, K = p.shape
        idx = torch.arange(T, device=p.device)
        delta = idx.unsqueeze(0) - idx.unsqueeze(1) + (T - 1)     # (T, T)
        r_base = self.base_rpe[delta].unsqueeze(0)                # (1, T, T, H)
        r_regime = torch.einsum('btk,ktuh->btuh', p, self.regime_rpe[:, delta])  # (B, T, T, H)
        return (r_base + r_regime).permute(0, 3, 1, 2)            # (B, H, T, T)


class RegimeAwareTemporalAttention(nn.Module):
    """
    Lightweight self-attention block with additive regime-aware RPE:
      logits = QKᵀ/√d + rpe → softmax → V; residual + LayerNorm + FFN.

    Input:  z (B, T, d), rpe (B, n_heads, T, T) or None
    Output: (B, T, d)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 ffn_dim: int = 256, dropout: float = 0.1):
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

    def forward(self, z: torch.Tensor, rpe: torch.Tensor = None) -> torch.Tensor:
        B, T, d = z.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q(z).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        k = self.k(z).view(B, T, H, hd).transpose(1, 2)
        v = self.v(z).view(B, T, H, hd).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(hd)       # (B, H, T, T)
        if rpe is not None:
            logits = logits + rpe
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        out = self.out(out)
        z = self.norm1(z + self.drop(out))
        z = self.norm2(z + self.drop(self.ffn(z)))
        return z


class RegimeDynamicRPETransformer(nn.Module):
    """
    Full temporal branch:
      E → regime p → regime-specific adapters + soft mixing → z
      → regime-aware RPE → temporal transformer → attention pooling → (B, 64)
    """

    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256, dropout: float = 0.1,
                 t_len: int = 20, K: int = 3, D_regime: int = 32,
                 use_state_loss: bool = False,
                 use_semantic_router: bool = False,
                 lambda_cluster: float = 0.005,
                 routing_strength: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.use_semantic_router = use_semantic_router
        self.lambda_cluster = float(lambda_cluster)
        self.lambda_info_max = 0.001
        self.routing_strength = float(routing_strength)
        if not 0.0 <= self.routing_strength <= 1.0:
            raise ValueError("routing_strength must be in [0, 1]")
        self.regime_gen = (
            SemanticRegimeRouter(d_model, K, D_regime)
            if use_semantic_router
            else SoftRegimeGeneratorV2(d_model, K, D_regime)
        )
        self.adapters = RegimeTemporalAdapter(d_model, K)
        self.rpe = RegimeAwareRPE(t_len, n_heads, K)
        self.attn_layers = nn.ModuleList([
            RegimeAwareTemporalAttention(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)])
        self.pool_score = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, 64)
        self.forced_regime = None   # diagnostic: one-hot (K,) or None
        self.forced_regime_use = None  # diagnostic: direct downstream override
        self.use_state_loss = use_state_loss
        if use_state_loss or use_semantic_router:
            # latent state centers: typical market-dynamics profile per regime
            self.state_centers = nn.Parameter(torch.randn(K, 3) * 0.02)

    def forward(self, x_flat: torch.Tensor,
                market_descriptor: torch.Tensor = None) -> torch.Tensor:
        # x_flat: (B, T, in_dim) → h_temporal (B, 64)
        B, T, _ = x_flat.shape
        E = self.proj(x_flat)                                     # (B, T, 128)
        self.last_E = E.detach()

        if self.use_semantic_router:
            if market_descriptor is None:
                raise ValueError("Semantic-router variants require market_descriptor")
            p_raw = self.regime_gen(E, market_descriptor, self.state_centers)
            self._m_t_for_loss = market_descriptor
            self.last_m_t = market_descriptor.detach()
        else:
            p_raw = self.regime_gen(E)                            # (B, T, K)
        if self.forced_regime is not None:                        # diagnostic override
            p_raw = self.forced_regime.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        if self.use_semantic_router and self.routing_strength != 1.0:
            uniform_p = torch.full_like(p_raw, 1.0 / p_raw.shape[-1])
            p_use = (
                (1.0 - self.routing_strength) * uniform_p
                + self.routing_strength * p_raw
            )
        else:
            p_use = p_raw
        if self.forced_regime_use is not None:
            p_use = self.forced_regime_use.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        self.last_regime_p = p_raw.detach()  # raw recognition probability
        self.last_regime_p_use = p_use.detach()
        self._p_for_loss = p_raw                                 # non-detached (loss)

        z, zs = self.adapters(E, p_use)                           # (B, T, 128), (K, B, T, 128)
        self.last_z = z.detach()
        self.last_zs = zs.detach()
        self._zs_for_loss = zs                                    # non-detached (loss)

        rpe = self.rpe(p_use, T)                                 # (B, H, T, T)
        H = z
        for layer in self.attn_layers:
            H = layer(H, rpe)                                     # (B, T, 128)
        self.last_transformer_output = H.detach()

        score = self.pool_score(H).squeeze(-1)                    # (B, T)
        alpha = F.softmax(score, dim=1)
        h = (H * alpha.unsqueeze(-1)).sum(dim=1)                  # (B, 128)
        self.last_attn = alpha.detach()
        h_temporal = self.fc(h)                                   # (B, 64)
        self.last_h_temporal = h_temporal.detach()
        return h_temporal

    def dynamic_loss(self, lambda_dynamic: float = 0.001,
                     lambda_balance: float = 0.001,
                     lambda_state: float = 0.01) -> torch.Tensor:
        """
        F/F2: L = λd·L_dynamic_div + λb·L_balance [+ λs·L_state]
        G/H:  L = λd·L_dynamic_div + λcluster·L_cluster + λIM·L_InfoMax
          L_dynamic_div = mean pairwise cosine of adapter outputs
          L_balance     = Σ_k p̄_k·log(p̄_k·K) on batch-level p̄
          L_state       = MSE(Σ_k p_tk·μ_k, m_t)  (if use_state_loss)
        All terms use NON-detached p / zs / m (full computational graph).
        """
        p = getattr(self, '_p_for_loss', None)
        zs = getattr(self, '_zs_for_loss', None)
        if p is None or zs is None:
            return torch.zeros((), device=next(self.parameters()).device)
        K = p.shape[-1]
        zs_n = F.normalize(zs.reshape(K, -1), dim=-1)             # (K, N)
        cos = zs_n @ zs_n.T
        mask = ~torch.eye(K, dtype=torch.bool, device=cos.device)
        l_dyn = cos[mask].mean()
        if self.use_semantic_router:
            descriptor = getattr(self, '_m_t_for_loss', None)
            if descriptor is None:
                return torch.zeros((), device=p.device)
            distances = ((
                descriptor.unsqueeze(2)
                - self.state_centers.view(1, 1, K, 3)
            ) ** 2).sum(dim=-1)
            l_cluster = (p * distances).sum(dim=-1).mean()
            h_cond = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            p_mean = p.mean(dim=(0, 1))
            h_marg = -(p_mean * torch.log(p_mean + 1e-8)).sum()
            l_info_max = h_cond - h_marg
            self.last_loss_components = {
                'dynamic': l_dyn.detach(),
                'cluster': l_cluster.detach(),
                'info_max': l_info_max.detach(),
                'conditional_entropy': h_cond.detach(),
                'marginal_entropy': h_marg.detach(),
            }
            self.last_loss_weights = {
                'dynamic': float(lambda_dynamic),
                'cluster': self.lambda_cluster,
                'info_max': self.lambda_info_max,
            }
            return (
                lambda_dynamic * l_dyn
                + self.lambda_cluster * l_cluster
                + self.lambda_info_max * l_info_max
            )
        p_mean = p.mean(dim=(0, 1))                               # (K,)
        l_bal = (p_mean * torch.log(p_mean * K + 1e-8)).sum()
        loss = lambda_dynamic * l_dyn + lambda_balance * l_bal
        if self.use_state_loss:
            m_t = getattr(self, '_m_t_for_loss', None)
            if m_t is not None:
                m_hat = torch.einsum('btk,kd->btd', p, self.state_centers)  # (B, T, 3)
                self._m_hat_for_loss = m_hat
                loss = loss + lambda_state * F.mse_loss(m_hat, m_t)
        return loss
