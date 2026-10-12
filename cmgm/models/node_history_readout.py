"""Commodity-preserving readout over spatial nodes and temporal history."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CONCAT_VARIANT = "switching_latent_balanced_concat_fusion"
NODE_MEAN_VARIANT = "switching_latent_balanced_node_history_mean"
NODE_ATTENTION_VARIANT = "switching_latent_balanced_node_history_attention"
NODE_HISTORY_VARIANTS = (NODE_MEAN_VARIANT, NODE_ATTENTION_VARIANT)
ALL_VARIANTS = (CONCAT_VARIANT, *NODE_HISTORY_VARIANTS)


class NodeHistoryReadout(nn.Module):
    """Shared node-conditioned historical readout.

    The common V/O/fusion modules are deliberately constructed before optional
    Q/K modules so mean and attention variants start with identical shared
    readout parameters under the same seed.
    """

    def __init__(self, *, attention: bool, hidden_dim: int = 64,
                 memory_dim: int = 128, readout_dim: int = 32):
        super().__init__()
        self.attention = bool(attention)
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.readout_dim = readout_dim
        self.W_v = nn.Linear(memory_dim, readout_dim, bias=False)
        self.W_o = nn.Linear(readout_dim, hidden_dim, bias=False)
        self.fusion_projection = nn.Linear(hidden_dim * 4, hidden_dim)
        if self.attention:
            self.W_q = nn.Linear(hidden_dim, readout_dim, bias=False)
            self.W_k = nn.Linear(memory_dim, readout_dim, bias=False)

    def forward(self, s_nodes: torch.Tensor, history: torch.Tensor,
                s_global: torch.Tensor, t_global: torch.Tensor, *,
                force_uniform_attention: bool = False,
                return_attention: bool = False):
        if s_nodes.ndim != 3 or s_nodes.shape[-1] != self.hidden_dim:
            raise ValueError("s_nodes must have shape (B,Nc,hidden_dim)")
        if history.ndim != 3 or history.shape[0] != s_nodes.shape[0] or history.shape[-1] != self.memory_dim:
            raise ValueError("history must have shape (B,T,memory_dim)")
        if history.shape[1] < 1:
            raise ValueError("history must contain at least one observed timestep")
        expected_global = (s_nodes.shape[0], self.hidden_dim)
        if s_global.shape != expected_global or t_global.shape != expected_global:
            raise ValueError("global branch states must both have shape (B,hidden_dim)")
        if force_uniform_attention and not self.attention:
            raise ValueError("force_uniform_attention applies only to the attention variant")

        values = self.W_v(history)  # (B,T,32)
        if self.attention:
            if force_uniform_attention:
                attention = values.new_full(
                    (values.shape[0], s_nodes.shape[1], values.shape[1]),
                    1.0 / values.shape[1],
                )
            else:
                query = self.W_q(s_nodes)
                keys = self.W_k(history)
                scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.readout_dim)
                attention = scores.softmax(dim=-1)
            u_nodes = self.W_o(torch.matmul(attention, values))
        else:
            attention = None
            u_mean = self.W_o(values.mean(dim=1))
            u_nodes = u_mean.unsqueeze(1).expand(-1, s_nodes.shape[1], -1)

        n_commodities = s_nodes.shape[1]
        s_broadcast = s_global.unsqueeze(1).expand(-1, n_commodities, -1)
        t_broadcast = t_global.unsqueeze(1).expand(-1, n_commodities, -1)
        features = torch.cat([s_nodes, u_nodes, s_broadcast, t_broadcast], dim=-1)
        f_nodes = F.gelu(self.fusion_projection(features))
        if return_attention:
            return f_nodes, {
                "attention": attention,
                "s_nodes": s_nodes,
                "u_nodes": u_nodes,
                "s_global": s_global,
                "t_global": t_global,
            }
        return f_nodes


def apply_original_head_per_node(head: nn.Sequential, f_nodes: torch.Tensor,
                                 n_horizons: int, n_commodities: int) -> torch.Tensor:
    """Apply head body per commodity and select its own flattened output rows."""
    if f_nodes.ndim != 3 or f_nodes.shape[1:] != (n_commodities, 64):
        raise ValueError("f_nodes must have shape (B,Nc,64)")
    if not isinstance(head[3], nn.Linear):
        raise TypeError("The original prediction head's final module must be Linear")
    if head[3].out_features != n_horizons * n_commodities:
        raise ValueError("Prediction head output rows do not match (horizons, commodities)")
    z = head[2](head[1](head[0](f_nodes)))
    weight = head[3].weight.view(n_horizons, n_commodities, -1)
    bias = head[3].bias.view(n_horizons, n_commodities)
    return torch.einsum("bcd,hcd->bhc", z, weight) + bias.unsqueeze(0)


def attention_diagnostics(attention: torch.Tensor) -> dict:
    """Summarize attention behavior (descriptive, not causal attribution)."""
    if attention.ndim != 3:
        raise ValueError("attention must have shape (B,Nc,T)")
    steps = attention.shape[-1]
    eps = torch.finfo(attention.dtype).eps
    entropy = -(attention * attention.clamp_min(eps).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(steps) if steps > 1 else torch.zeros_like(entropy)
    uniform = 1.0 / steps
    # Mean absolute L1-coordinate distance over all unordered commodity pairs.
    if attention.shape[1] > 1:
        pair_diff = (attention[:, :, None, :] - attention[:, None, :, :]).abs().mean(dim=-1)
        pair_mask = torch.triu(torch.ones(attention.shape[1], attention.shape[1],
                                          dtype=torch.bool, device=attention.device), diagonal=1)
        commodity_difference = pair_diff[:, pair_mask].mean()
    else:
        commodity_difference = attention.new_zeros(())
    return {
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "max_time_weight_mean": float(attention.max(dim=-1).values.mean()),
        "mean_abs_deviation_from_uniform": float((attention - uniform).abs().mean()),
        "mean_pairwise_commodity_attention_abs_difference": float(commodity_difference),
        "time_steps": int(steps),
        "commodity_count": int(attention.shape[1]),
    }
