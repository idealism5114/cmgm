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


class LatentMemoryAttention(nn.Module):
    """Pure content attention from H_t to the unordered set Z_<t."""

    def __init__(self, query_dim: int = 128, memory_dim: int = 64,
                 n_heads: int = 4):
        super().__init__()
        if memory_dim % n_heads != 0:
            raise ValueError("memory_dim must be divisible by n_heads")
        self.memory_dim = int(memory_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.memory_dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, self.memory_dim)
        self.k_proj = nn.Linear(self.memory_dim, self.memory_dim)
        self.v_proj = nn.Linear(self.memory_dim, self.memory_dim)
        self.out_proj = nn.Linear(self.memory_dim, self.memory_dim)

    def forward(self, query: torch.Tensor, history: torch.Tensor,
                score_bias: torch.Tensor = None,
                use_content_scores: bool = True):
        if history.dim() != 3 or history.shape[1] == 0:
            raise ValueError("history must be non-empty with shape (B,L,D)")
        batch_size, history_length, _ = history.shape
        q = self.q_proj(query).view(
            batch_size, self.n_heads, self.head_dim
        )
        k = self.k_proj(history).view(
            batch_size, history_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(history).view(
            batch_size, history_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        content_logits = torch.einsum("bhd,bhld->bhl", q, k) * self.scale
        logits = content_logits if use_content_scores else torch.zeros_like(
            content_logits
        )
        if score_bias is not None:
            if score_bias.shape != logits.shape:
                raise ValueError(
                    "score_bias must match (B,H,L), got "
                    f"{tuple(score_bias.shape)} vs {tuple(logits.shape)}"
                )
            logits = logits + score_bias
        attention = F.softmax(logits, dim=-1)
        memory = torch.einsum("bhl,bhld->bhd", attention, v).reshape(
            batch_size, self.memory_dim
        )
        self.last_query = q.detach()
        self.last_keys = k.detach()
        self.last_values = v.detach()
        self.last_content_logits = content_logits.detach()
        self.last_score_bias = (
            torch.zeros_like(content_logits)
            if score_bias is None else score_bias.detach()
        )
        self.last_logits = logits.detach()
        return self.out_proj(memory), attention


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
                 output_dim: int = 64,
                 balanced_readout: bool = False,
                 use_latent_memory: bool = False,
                 zero_init_memory_projection: bool = True,
                 use_regime_relative_memory: bool = False,
                 use_dynamic_slope: bool = False):
        super().__init__()
        self.K = int(K)
        self.z_dim = int(z_dim)
        self.balanced_readout = bool(balanced_readout)
        self.use_latent_memory = bool(use_latent_memory)
        self.zero_init_memory_projection = bool(
            zero_init_memory_projection
        )
        self.use_regime_relative_memory = bool(
            use_regime_relative_memory
        )
        self.use_dynamic_slope = bool(use_dynamic_slope)
        if self.use_latent_memory and not self.balanced_readout:
            raise ValueError("latent memory requires the D0B balanced readout")
        if self.use_regime_relative_memory and not self.use_latent_memory:
            raise ValueError("regime-relative memory requires latent memory")
        if self.use_dynamic_slope and not self.balanced_readout:
            raise ValueError("dynamic slope requires the D0B balanced readout")
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
        if self.balanced_readout:
            # D0B changes only the final H_T/Z_T readout.  The latent
            # trajectories above remain byte-for-byte shared with D0.
            self.long_memory_readout = nn.Linear(d_model, output_dim)
            self.micro_state_readout = nn.Linear(self.z_dim, output_dim)
            self.long_memory_norm = nn.LayerNorm(output_dim)
            self.micro_state_norm = nn.LayerNorm(output_dim)
            self.state_readout = nn.Linear(2 * output_dim, output_dim)
        else:
            self.state_readout = nn.Linear(d_model + self.z_dim, output_dim)

        # Construct D1A-only modules strictly after every D0B shared module so
        # identical seeds preserve the entire D0B initialization sequence.
        if self.use_latent_memory:
            self.latent_memory_attention = LatentMemoryAttention(
                query_dim=d_model, memory_dim=self.z_dim, n_heads=4
            )
            self.memory_projections = nn.ModuleList([
                nn.Linear(self.z_dim, 128, bias=False)
                for _ in range(self.K)
            ])
            if self.zero_init_memory_projection:
                for projection in self.memory_projections:
                    nn.init.zeros_(projection.weight)
        if self.use_regime_relative_memory:
            self.regime_relative_lag_bias = nn.Parameter(torch.zeros(
                self.K,
                self.latent_memory_attention.n_heads,
                max_len - 1,
            ))
        # D0C-only modules are created last so every D0B/shared parameter keeps
        # exactly the same seeded initialization. These projections are active
        # from the first optimization step via PyTorch's default initialization.
        if self.use_dynamic_slope:
            self.slope_projections = nn.ModuleList([
                nn.Linear(d_model, 128, bias=False)
                for _ in range(self.K)
            ])

    def centered_regime_relative_lag_bias(self) -> torch.Tensor:
        if not self.use_regime_relative_memory:
            raise RuntimeError("regime-relative latent memory is disabled")
        table = self.regime_relative_lag_bias
        return table - table.mean(dim=0, keepdim=True)

    def regime_memory_bias(self, probabilities: torch.Tensor,
                           history_length: int,
                           lag_indices: torch.Tensor = None) -> torch.Tensor:
        """Return B_reg with lag=t-j and lag 1 stored at table index 0."""
        if not self.use_regime_relative_memory:
            return probabilities.new_zeros(
                probabilities.shape[0], 0, history_length
            )
        if lag_indices is None:
            lags = torch.arange(
                history_length, 0, -1, device=probabilities.device
            )
            lag_indices = lags - 1
        else:
            lag_indices = lag_indices.to(probabilities.device)
        if lag_indices.numel() != history_length:
            raise ValueError("lag_indices length must equal history_length")
        if lag_indices.min() < 0 or lag_indices.max() >= self.regime_relative_lag_bias.shape[-1]:
            raise ValueError("lag index is outside the configured causal table")
        centered = self.centered_regime_relative_lag_bias()
        selected = centered.index_select(-1, lag_indices)
        return torch.einsum("bk,khl->bhl", probabilities, selected)

    def encode_market_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.market_encoder(x)

    def latent_forward(self, long_memory: torch.Tensor,
                       forced_probabilities: torch.Tensor = None,
                       zero_latent_memory: bool = False,
                       zero_regime_memory_bias: bool = False,
                       forced_rpe_probabilities: torch.Tensor = None,
                       slope_scale: float = 1.0):
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
        if forced_rpe_probabilities is not None:
            forced_rpe = forced_rpe_probabilities.to(
                device=long_memory.device, dtype=long_memory.dtype
            )
            if forced_rpe.dim() == 1:
                forced_rpe = forced_rpe.view(1, 1, self.K).expand(
                    batch_size, time_steps, -1
                )
            elif forced_rpe.shape != (batch_size, time_steps, self.K):
                raise ValueError(
                    "forced_rpe_probabilities must be (K,) or (B,T,K), got "
                    f"{tuple(forced_rpe.shape)}"
                )
        else:
            forced_rpe = None

        probabilities = []
        priors = []
        evidence_values = []
        latent_states = []
        candidate_values = []
        switch_kls = []
        z_history = []
        memory_values = []
        memory_attentions = []
        base_preactivations = []
        memory_injections = []
        memory_scores = []
        memory_query_norms = []
        memory_key_norms = []
        memory_value_norms = []
        memory_content_scores = []
        memory_regime_biases = []
        long_memory_slopes = torch.zeros_like(long_memory)
        long_memory_slopes[:, 1:] = (
            long_memory[:, 1:] - long_memory[:, :-1]
        )
        slope_contributions = []
        slope_base_preactivations = []
        for step in range(time_steps):
            h_t = long_memory[:, step]
            prior_t, evidence_t, p_t, kl_t = self.regime_filter.step(
                h_t, p_prev, transition
            )
            p_for_z = forced[:, step] if forced is not None else p_t
            if self.use_latent_memory:
                if step == 0 or zero_latent_memory:
                    memory_t = torch.zeros_like(z_prev)
                    attention_t = None
                else:
                    # z_history is intentionally non-detached and contains
                    # exactly Z_0,...,Z_(t-1), never the current or future Z.
                    history_t = torch.stack(z_history, dim=1)
                    score_bias_t = None
                    if self.use_regime_relative_memory:
                        rpe_p_t = (
                            forced_rpe[:, step]
                            if forced_rpe is not None else p_for_z
                        )
                        if zero_regime_memory_bias:
                            score_bias_t = long_memory.new_zeros(
                                batch_size,
                                self.latent_memory_attention.n_heads,
                                step,
                            )
                        else:
                            score_bias_t = self.regime_memory_bias(
                                rpe_p_t, step
                            )
                    memory_t, attention_t = self.latent_memory_attention(
                        h_t, history_t, score_bias=score_bias_t
                    )
                transition_input = torch.cat([h_t, z_prev], dim=-1)
                state_candidates = []
                state_base = []
                state_injections = []
                for generator, memory_projection in zip(
                    self.latent_transition.generators,
                    self.memory_projections,
                ):
                    base_t = generator[0](transition_input)
                    injection_t = memory_projection(memory_t)
                    hidden_t = generator[1](base_t + injection_t)
                    state_candidates.append(generator[2](hidden_t))
                    state_base.append(base_t)
                    state_injections.append(injection_t)
                candidates_t = torch.stack(state_candidates, dim=1)
                z_t = torch.einsum("bk,bkd->bd", p_for_z, candidates_t)
                memory_values.append(memory_t)
                memory_attentions.append(attention_t)
                if attention_t is None:
                    empty_norm = long_memory.new_zeros(
                        batch_size, self.latent_memory_attention.n_heads
                    )
                    memory_scores.append(None)
                    memory_query_norms.append(empty_norm)
                    memory_key_norms.append(empty_norm)
                    memory_value_norms.append(empty_norm)
                else:
                    memory_scores.append(
                        self.latent_memory_attention.last_logits
                    )
                    memory_content_scores.append(
                        self.latent_memory_attention.last_content_logits
                    )
                    memory_regime_biases.append(
                        self.latent_memory_attention.last_score_bias
                    )
                    memory_query_norms.append(
                        self.latent_memory_attention.last_query.norm(dim=-1)
                    )
                    memory_key_norms.append(
                        self.latent_memory_attention.last_keys.norm(dim=-1).mean(dim=-1)
                    )
                    memory_value_norms.append(
                        self.latent_memory_attention.last_values.norm(dim=-1).mean(dim=-1)
                    )
                base_preactivations.append(torch.stack(state_base, dim=1))
                memory_injections.append(
                    torch.stack(state_injections, dim=1)
                )
                if attention_t is None:
                    memory_content_scores.append(None)
                    memory_regime_biases.append(None)
            else:
                if self.use_dynamic_slope:
                    transition_input = torch.cat([h_t, z_prev], dim=-1)
                    delta_h_t = long_memory_slopes[:, step] * slope_scale
                    state_candidates = []
                    state_base = []
                    state_slopes = []
                    for generator, slope_projection in zip(
                        self.latent_transition.generators,
                        self.slope_projections,
                    ):
                        base_t = generator[0](transition_input)
                        slope_t = slope_projection(delta_h_t)
                        hidden_t = generator[1](base_t + slope_t)
                        state_candidates.append(generator[2](hidden_t))
                        state_base.append(base_t)
                        state_slopes.append(slope_t)
                    candidates_t = torch.stack(state_candidates, dim=1)
                    z_t = torch.einsum(
                        "bk,bkd->bd", p_for_z, candidates_t
                    )
                    slope_base_preactivations.append(
                        torch.stack(state_base, dim=1)
                    )
                    slope_contributions.append(
                        torch.stack(state_slopes, dim=1)
                    )
                else:
                    z_t, candidates_t = self.latent_transition(
                        h_t, z_prev, p_for_z
                    )
            priors.append(prior_t)
            evidence_values.append(evidence_t)
            probabilities.append(p_t)
            latent_states.append(z_t)
            candidate_values.append(candidates_t)
            switch_kls.append(kl_t)
            if self.use_latent_memory:
                z_history.append(z_t)
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
        if self.use_latent_memory:
            padded_attention = long_memory.new_zeros(
                batch_size, time_steps, self.latent_memory_attention.n_heads,
                time_steps,
            )
            padded_scores = long_memory.new_zeros(
                batch_size, time_steps, self.latent_memory_attention.n_heads,
                time_steps,
            )
            padded_content_scores = torch.zeros_like(padded_scores)
            padded_regime_biases = torch.zeros_like(padded_scores)
            for step, attention_t in enumerate(memory_attentions):
                if attention_t is not None:
                    padded_attention[:, step, :, :step] = attention_t.detach()
                    padded_scores[:, step, :, :step] = memory_scores[step]
                    padded_content_scores[:, step, :, :step] = (
                        memory_content_scores[step]
                    )
                    padded_regime_biases[:, step, :, :step] = (
                        memory_regime_biases[step]
                    )
            self.last_latent_memories = torch.stack(
                memory_values, dim=1
            ).detach()
            self.last_latent_memory_attention = padded_attention.detach()
            self.last_latent_memory_scores = padded_scores.detach()
            self.last_latent_memory_content_scores = (
                padded_content_scores.detach()
            )
            self.last_regime_memory_biases = padded_regime_biases.detach()
            self.last_latent_memory_query_norms = torch.stack(
                memory_query_norms, dim=1
            ).detach()
            self.last_latent_memory_key_norms = torch.stack(
                memory_key_norms, dim=1
            ).detach()
            self.last_latent_memory_value_norms = torch.stack(
                memory_value_norms, dim=1
            ).detach()
            self.last_generator_base_preactivations = torch.stack(
                base_preactivations, dim=1
            ).detach()
            self.last_generator_memory_injections = torch.stack(
                memory_injections, dim=1
            ).detach()
        if self.use_dynamic_slope:
            slope_tensor = torch.stack(slope_contributions, dim=1)
            self.last_long_memory_slopes = long_memory_slopes.detach()
            self.last_slope_contributions = slope_tensor.detach()
            self.last_generator_base_preactivations = torch.stack(
                slope_base_preactivations, dim=1
            ).detach()
            self.last_regime_weighted_slope = torch.einsum(
                "btk,btkd->btd", self.last_latent_probabilities, slope_tensor
            ).detach()
        return p, z, candidates

    def readout(self, h_last: torch.Tensor, z_last: torch.Tensor,
                zero_component: str = None) -> torch.Tensor:
        if zero_component not in (None, "H", "Z"):
            raise ValueError("zero_component must be None, 'H', or 'Z'")
        if self.balanced_readout:
            h_long = self.long_memory_norm(
                self.long_memory_readout(h_last)
            )
            h_micro = self.micro_state_norm(
                self.micro_state_readout(z_last)
            )
            # D0B null interventions are deliberately post-normalization:
            # zeroing raw inputs would leave projection/LN bias effects.
            h_effective = (
                torch.zeros_like(h_long) if zero_component == "H" else h_long
            )
            z_effective = (
                torch.zeros_like(h_micro) if zero_component == "Z" else h_micro
            )
            readout_input = torch.cat([h_effective, z_effective], dim=-1)
            self.last_h_long = h_long.detach()
            self.last_h_micro = h_micro.detach()
            self.last_h_long_effective = h_effective.detach()
            self.last_h_micro_effective = z_effective.detach()
            self.last_balanced_concat = readout_input.detach()
        else:
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
                         zero_readout_component: str = None,
                         zero_latent_memory: bool = False,
                         zero_regime_memory_bias: bool = False,
                         forced_rpe_probabilities: torch.Tensor = None,
                         slope_scale: float = 1.0) -> torch.Tensor:
        long_memory = self.long_memory(tokens)
        _, latent_states, _ = self.latent_forward(
            long_memory,
            forced_probabilities=forced_probabilities,
            zero_latent_memory=zero_latent_memory,
            zero_regime_memory_bias=zero_regime_memory_bias,
            forced_rpe_probabilities=forced_rpe_probabilities,
            slope_scale=slope_scale,
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
                zero_readout_component: str = None,
                zero_latent_memory: bool = False,
                zero_regime_memory_bias: bool = False,
                forced_rpe_probabilities: torch.Tensor = None,
                slope_scale: float = 1.0) -> torch.Tensor:
        tokens = self.encode_market_tokens(x)
        return self.temporal_forward(
            tokens,
            forced_probabilities=forced_probabilities,
            zero_readout_component=zero_readout_component,
            zero_latent_memory=zero_latent_memory,
            zero_regime_memory_bias=zero_regime_memory_bias,
            forced_rpe_probabilities=forced_rpe_probabilities,
            slope_scale=slope_scale,
        )

    def set_epoch(self, epoch: int) -> float:
        return self.regime_filter.set_epoch(epoch)

    def switch_loss(self) -> torch.Tensor:
        return self.regime_filter.switch_loss()

    def transition_matrix(self) -> torch.Tensor:
        return self.regime_filter.transition_matrix()
