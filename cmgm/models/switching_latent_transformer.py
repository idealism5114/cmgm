"""D0 DS3M-style deterministic switching latent-dynamics temporal branch.

Structural mapping to DS3M:
  DS3M forward GRU                 -> causal Base-RPE LongMemory Transformer
  DS3M discrete state d_t          -> differentiable soft regime p_t
  DS3M ztransition_list            -> K independent latent generators G_k
  DS3M continuous state z_t        -> deterministic micro state Z_t
  DS3M [forward hidden, z_t] input -> D0 [H_T, Z_T] temporal readout

D0 intentionally does not implement a backward network, future-target
inference, Gaussian sampling/KL, or state-specific prediction emissions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgm.models.switching_transformer import MarketAwareTemporalEncoder


class CausalBaseRPEAttention(nn.Module):
    """Causal self-attention with a shared, regime-independent Base RPE bias."""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor,
                base_relative_bias: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        expected = (batch_size, self.n_heads, time_steps, time_steps)
        if base_relative_bias.shape != expected:
            raise ValueError(
                f"base_relative_bias must have shape {expected}, "
                f"got {tuple(base_relative_bias.shape)}"
            )
        q = self.q(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        qk_logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        logits = qk_logits + base_relative_bias
        future_mask = torch.triu(
            torch.ones(
                time_steps, time_steps, dtype=torch.bool, device=x.device
            ),
            diagonal=1,
        )
        logits = logits.masked_fill(future_mask, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(
            batch_size, time_steps, self.d_model
        )
        self.last_qk_logits = qk_logits.detach()
        self.last_attention_logits = logits.detach()
        self.last_attention = attention.detach()
        return self.out(output)


class LongMemoryTransformerBlock(nn.Module):
    """One causal Base-RPE Transformer block for deterministic long memory."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalBaseRPEAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                base_relative_bias: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x, base_relative_bias)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class LongMemoryTransformer(nn.Module):
    """Two-layer causal Transformer producing H_t=f(E_<=t), without pooling."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256,
                 dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.layers = nn.ModuleList([
            LongMemoryTransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.base_rpe = nn.Parameter(
            torch.randn(self.n_heads, 2 * self.max_len - 1) * 0.02
        )

    def relative_bias(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = tokens.shape
        if time_steps > self.max_len:
            raise ValueError(
                f"time length {time_steps} exceeds max_len={self.max_len}"
            )
        table_center = self.max_len - 1
        active_start = table_center - (time_steps - 1)
        active_end = table_center + time_steps
        active_table = self.base_rpe[:, active_start:active_end]
        positions = torch.arange(time_steps, device=tokens.device)
        # Signed relative distance: query index i minus key index j.
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        delta_index = delta + (time_steps - 1)
        bias = active_table[:, delta_index].unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        self.last_base_relative_bias = bias.detach()
        self.last_relative_delta = delta.detach()
        return bias

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        base_relative_bias = self.relative_bias(tokens)
        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden, base_relative_bias)
        self.last_long_memory = hidden.detach()
        return hidden


class MarkovRegimeFilter(nn.Module):
    """One-head discriminative Markov filter used only for latent dynamics."""

    def __init__(self, d_model: int = 128, K: int = 3,
                 sticky_alpha: float = 0.5, tau: float = 1.0,
                 beta_max: float = 5e-4, warmup_epochs: int = 20,
                 eps: float = 1e-8):
        super().__init__()
        self.K = int(K)
        self.sticky_alpha = float(sticky_alpha)
        self.tau = float(tau)
        self.beta_max = float(beta_max)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)
        self.current_epoch = 1
        self.current_beta = 0.0
        self.transition_logits = nn.Parameter(torch.zeros(self.K, self.K))
        self.regime_evidence = nn.Linear(d_model, self.K)
        self._last_switch_loss = None

    def transition_matrix(self) -> torch.Tensor:
        learned = F.softmax(self.transition_logits, dim=-1)
        identity = torch.eye(
            self.K, dtype=learned.dtype, device=learned.device
        )
        return self.sticky_alpha * identity + (1.0 - self.sticky_alpha) * learned

    def set_epoch(self, epoch: int) -> float:
        self.current_epoch = int(epoch)
        if self.warmup_epochs <= 1:
            progress = 1.0
        else:
            progress = min(
                max((self.current_epoch - 1) / (self.warmup_epochs - 1), 0.0),
                1.0,
            )
        self.current_beta = self.beta_max * progress
        return self.current_beta

    def step(self, h_t: torch.Tensor, p_prev: torch.Tensor,
             transition: torch.Tensor):
        prior_t = p_prev @ transition
        evidence_t = self.regime_evidence(h_t)
        posterior_logits = (
            torch.log(prior_t + self.eps) + evidence_t / self.tau
        )
        p_t = F.softmax(posterior_logits, dim=-1)
        kl_t = (
            p_t
            * (
                torch.log(p_t + self.eps)
                - torch.log(prior_t + self.eps)
            )
        ).sum(dim=-1)
        return prior_t, evidence_t, p_t, kl_t

    def switch_loss(self) -> torch.Tensor:
        if self._last_switch_loss is None:
            return self.transition_logits.sum() * 0.0
        return self.current_beta * self._last_switch_loss


class RegimeLatentTransition(nn.Module):
    """K independent deterministic generators G_k([H_t,Z_(t-1)])."""

    def __init__(self, h_dim: int = 128, z_dim: int = 64,
                 hidden_dim: int = 128, K: int = 3):
        super().__init__()
        self.K = int(K)
        self.z_dim = int(z_dim)
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim + z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, z_dim),
            )
            for _ in range(self.K)
        ])

    def forward(self, h_t: torch.Tensor, z_prev: torch.Tensor,
                probabilities: torch.Tensor):
        transition_input = torch.cat([h_t, z_prev], dim=-1)
        candidates = torch.stack(
            [generator(transition_input) for generator in self.generators],
            dim=1,
        )
        z_t = torch.einsum("bk,bkd->bd", probabilities, candidates)
        return z_t, candidates


class SwitchingLatentTransformerBranch(nn.Module):
    """D0 long-memory plus regime-selected deterministic micro dynamics."""

    def __init__(self, feat_dim: int, n_stock: int, n_bond: int,
                 n_commodity: int, node_dim: int = 32,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256,
                 dropout: float = 0.1, max_len: int = 20,
                 K: int = 3, z_dim: int = 64,
                 sticky_alpha: float = 0.5, tau: float = 1.0,
                 beta_max: float = 5e-4, warmup_epochs: int = 20,
                 output_dim: int = 64):
        super().__init__()
        self.K = int(K)
        self.z_dim = int(z_dim)
        self.market_encoder = MarketAwareTemporalEncoder(
            feat_dim=feat_dim,
            n_stock=n_stock,
            n_bond=n_bond,
            n_commodity=n_commodity,
            node_dim=node_dim,
            d_model=d_model,
            use_dispersion=True,
        )
        self.long_memory = LongMemoryTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.regime_filter = MarkovRegimeFilter(
            d_model=d_model,
            K=self.K,
            sticky_alpha=sticky_alpha,
            tau=tau,
            beta_max=beta_max,
            warmup_epochs=warmup_epochs,
        )
        self.latent_transition = RegimeLatentTransition(
            h_dim=d_model,
            z_dim=self.z_dim,
            hidden_dim=128,
            K=self.K,
        )
        self.state_readout = nn.Linear(d_model + self.z_dim, output_dim)

    def encode_market_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.market_encoder(x)

    def latent_forward(self, long_memory: torch.Tensor,
                       forced_probabilities: torch.Tensor = None):
        batch_size, time_steps, _ = long_memory.shape
        transition = self.regime_filter.transition_matrix()
        p_prev = torch.full(
            (batch_size, self.K),
            1.0 / self.K,
            dtype=long_memory.dtype,
            device=long_memory.device,
        )
        z_prev = torch.zeros(
            batch_size,
            self.z_dim,
            dtype=long_memory.dtype,
            device=long_memory.device,
        )
        if forced_probabilities is not None:
            forced = forced_probabilities.to(
                device=long_memory.device, dtype=long_memory.dtype
            )
            if forced.dim() == 1:
                forced = forced.view(1, 1, self.K).expand(
                    batch_size, time_steps, -1
                )
            elif forced.shape != (batch_size, time_steps, self.K):
                raise ValueError(
                    "forced_probabilities must be (K,) or (B,T,K), got "
                    f"{tuple(forced.shape)}"
                )
        else:
            forced = None

        probabilities = []
        priors = []
        evidence_values = []
        latent_states = []
        candidate_values = []
        switch_kls = []
        for step in range(time_steps):
            h_t = long_memory[:, step]
            prior_t, evidence_t, p_t, kl_t = self.regime_filter.step(
                h_t, p_prev, transition
            )
            p_for_z = forced[:, step] if forced is not None else p_t
            z_t, candidates_t = self.latent_transition(
                h_t, z_prev, p_for_z
            )
            priors.append(prior_t)
            evidence_values.append(evidence_t)
            probabilities.append(p_t)
            latent_states.append(z_t)
            candidate_values.append(candidates_t)
            switch_kls.append(kl_t)
            p_prev = p_t
            z_prev = z_t

        p = torch.stack(probabilities, dim=1)
        prior = torch.stack(priors, dim=1)
        evidence = torch.stack(evidence_values, dim=1)
        z = torch.stack(latent_states, dim=1)
        candidates = torch.stack(candidate_values, dim=1)
        self.regime_filter._last_switch_loss = torch.stack(
            switch_kls, dim=1
        ).mean()
        self.regime_filter.last_switch_loss_raw = (
            self.regime_filter._last_switch_loss.detach()
        )
        self.last_regime_probabilities = p.detach()
        self.last_regime_priors = prior.detach()
        self.last_regime_evidence = evidence.detach()
        self.last_latent_states = z.detach()
        self.last_latent_candidates = candidates.detach()
        self.last_latent_probabilities = (
            forced.detach() if forced is not None else p.detach()
        )
        return p, z, candidates

    def readout(self, h_last: torch.Tensor, z_last: torch.Tensor,
                zero_component: str = None) -> torch.Tensor:
        if zero_component not in (None, "H", "Z"):
            raise ValueError("zero_component must be None, 'H', or 'Z'")
        h_effective = (
            torch.zeros_like(h_last) if zero_component == "H" else h_last
        )
        z_effective = (
            torch.zeros_like(z_last) if zero_component == "Z" else z_last
        )
        readout_input = torch.cat([h_effective, z_effective], dim=-1)
        output = self.state_readout(readout_input)
        self.last_readout_input = readout_input.detach()
        self.last_h_temporal = output.detach()
        return output

    def temporal_forward(self, tokens: torch.Tensor,
                         forced_probabilities: torch.Tensor = None,
                         zero_readout_component: str = None) -> torch.Tensor:
        long_memory = self.long_memory(tokens)
        _, latent_states, _ = self.latent_forward(
            long_memory, forced_probabilities=forced_probabilities
        )
        h_last = long_memory[:, -1]
        z_last = latent_states[:, -1]
        output = self.readout(
            h_last, z_last, zero_component=zero_readout_component
        )
        self.last_market_tokens = tokens.detach()
        self.last_long_memory = long_memory.detach()
        self.last_h_last = h_last.detach()
        self.last_z_last = z_last.detach()
        return output

    def forward(self, x: torch.Tensor,
                forced_probabilities: torch.Tensor = None,
                zero_readout_component: str = None) -> torch.Tensor:
        tokens = self.encode_market_tokens(x)
        return self.temporal_forward(
            tokens,
            forced_probabilities=forced_probabilities,
            zero_readout_component=zero_readout_component,
        )

    def set_epoch(self, epoch: int) -> float:
        return self.regime_filter.set_epoch(epoch)

    def switch_loss(self) -> torch.Tensor:
        return self.regime_filter.switch_loss()

    def transition_matrix(self) -> torch.Tensor:
        return self.regime_filter.transition_matrix()
