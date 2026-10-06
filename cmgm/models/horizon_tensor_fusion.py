"""Low-rank, horizon-conditioned bilinear fusion for the D0B global states."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


VARIANT = "switching_latent_balanced_horizon_tensor_fusion"
DISPLAY_NAME = "D0B-HorizonTensorFusion"
HORIZONS = (1, 5, 10, 20)
HIDDEN_DIM = 64
RANK = 8
KERNELS = 2


class HorizonTensorFusion(nn.Module):
    """Fuse two (B,64) branch states into four horizon-specific (B,64) states."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, rank: int = RANK,
                 horizons: tuple[int, ...] = HORIZONS):
        super().__init__()
        if hidden_dim != HIDDEN_DIM or rank != RANK or tuple(horizons) != HORIZONS:
            raise ValueError("HorizonTensorFusion is fixed to d=64, r=8, horizons=(1,5,10,20)")
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.horizons = tuple(horizons)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.u_s = nn.Linear(hidden_dim, rank, bias=False)
        self.u_t = nn.Linear(hidden_dim, rank, bias=False)
        self.G0 = nn.Parameter(torch.empty(hidden_dim, rank, rank))
        self.G1 = nn.Parameter(torch.empty(hidden_dim, rank, rank))
        self.G2 = nn.Parameter(torch.empty(hidden_dim, rank, rank))
        self.E = nn.Parameter(torch.empty(len(horizons), KERNELS))
        self.A_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.A_t = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_tensor_parameters()
        # Eval-only frozen interventions are explicit call arguments instead.

    def reset_tensor_parameters(self) -> None:
        for kernel in (self.G0, self.G1, self.G2):
            nn.init.xavier_uniform_(kernel.reshape(self.hidden_dim, -1))
        nn.init.normal_(self.E, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

    def effective_kernels(self, zero_embedding: bool = False) -> torch.Tensor:
        embedding = torch.zeros_like(self.E) if zero_embedding else self.E
        return (self.G0.unsqueeze(0)
                + embedding[:, 0, None, None, None] * self.G1.unsqueeze(0)
                + embedding[:, 1, None, None, None] * self.G2.unsqueeze(0))

    def forward(self, s_raw: torch.Tensor, t_raw: torch.Tensor, *,
                disable_interaction: bool = False,
                zero_embedding: bool = False,
                return_components: bool = False):
        if s_raw.ndim != 2 or s_raw.shape[-1] != self.hidden_dim or t_raw.shape != s_raw.shape:
            raise ValueError(f"Expected matching branch states (B,{self.hidden_dim})")
        s, t = self.norm_s(s_raw), self.norm_t(t_raw)
        u, v = self.u_s(s), self.u_t(t)
        # Keep the full r x r outer product; never construct a 64 x 64 tensor.
        outer = torch.einsum("ba,bc->bac", u, v)
        kernels = self.effective_kernels(zero_embedding=zero_embedding)
        interaction = torch.einsum("bac,hdac->bhd", outer, kernels)
        if disable_interaction:
            interaction = torch.zeros_like(interaction)
        linear = self.A_s(s) + self.A_t(t)
        fused = F.gelu(linear[:, None, :] + interaction + self.b[None, None, :])
        if fused.shape != (s.shape[0], len(self.horizons), self.hidden_dim):
            raise AssertionError(f"Unexpected fused shape {tuple(fused.shape)}")
        if return_components:
            return fused, {
                "s": s, "t": t, "u": u, "v": v, "outer": outer,
                "interaction": interaction, "linear": linear[:, None, :].expand_as(fused),
                "effective_kernels": kernels,
            }
        return fused

    @torch.no_grad()
    def diagnostics(self, s_raw: torch.Tensor, t_raw: torch.Tensor) -> dict:
        """Read-only norms and cycle/kernel differences; does not alter RNG or weights."""
        fused, parts = self(s_raw, t_raw, return_components=True)
        inter_norm = parts["interaction"].norm(dim=-1).mean(dim=0)
        linear_norm = parts["linear"].norm(dim=-1).mean(dim=0)
        kernels = parts["effective_kernels"]
        return {
            "interaction_norm_by_horizon": inter_norm.cpu().tolist(),
            "linear_norm_by_horizon": linear_norm.cpu().tolist(),
            "interaction_to_linear_norm_ratio":
                (inter_norm / linear_norm.clamp_min(1e-12)).cpu().tolist(),
            "embedding_norm": float(self.E.norm()),
            "kernel_pairwise_frobenius": {
                f"{i}_{j}": float((kernels[i] - kernels[j]).norm())
                for i in range(len(self.horizons)) for j in range(i + 1, len(self.horizons))
            },
            "kernel_component_norms": {"G0": float(self.G0.norm()),
                                       "G1": float(self.G1.norm()),
                                       "G2": float(self.G2.norm())},
            "fused_shape": list(fused.shape),
        }


def predict_by_horizon(head: nn.Sequential, fused: torch.Tensor,
                       n_commodities: int) -> torch.Tensor:
    """Use the existing shared head body and only each horizon's output rows."""
    if fused.ndim != 3 or fused.shape[1:] != (len(HORIZONS), HIDDEN_DIM):
        raise ValueError("Expected horizon-specific fused states (B,4,64)")
    final = head[3]
    expected_out = len(HORIZONS) * n_commodities
    if final.out_features != expected_out or final.in_features != HIDDEN_DIM:
        raise ValueError("Shared prediction head does not match four horizon-major outputs")
    body = head[:3](fused)
    weight = final.weight.view(len(HORIZONS), n_commodities, HIDDEN_DIM)
    bias = final.bias.view(len(HORIZONS), n_commodities)
    return torch.einsum("bhd,hnd->bhn", body, weight) + bias.unsqueeze(0)

