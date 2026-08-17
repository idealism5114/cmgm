"""
Minimal pure-PyTorch Mamba (SSM) implementation — vendored into the project
so no GitHub / CUDA-build dependency is required.

Follows the Mamba block structure (Gu & Dao, 2023):
    in_proj → causal conv1d → SiLU → selective scan → gated by SiLU(z) → out_proj

The selective scan is implemented as a sequential loop — perfectly fine for
our short sequences (T=20) and runs on any device (CPU/GPU, macOS/Linux).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelectiveScan(nn.Module):
    """
    Sequential selective scan:
        h_t = exp(dt·A) ⊙ h_{t−1} + dt·B_t ⊙ x_t
        y_t = C_t · h_t
    """

    def __init__(self, d_inner: int, d_state: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        # A parameterized as −exp(A_log) → always stable (negative real part)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.dt_bias = nn.Parameter(torch.randn(d_inner))

    def forward(self, x, dt, B, C):
        # x: (B, T, d_inner), dt: (B, T, d_inner), B, C: (B, T, d_state)
        Bsz, T, D = x.shape
        A = -torch.exp(self.A_log)                                   # (D, ds)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, D, ds)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)                       # (B, T, 1, ds)
        h = torch.zeros(Bsz, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)      # (B, D, ds)
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))            # (B, D)
        return torch.stack(ys, dim=1)                                # (B, T, D)


class Mamba(nn.Module):
    """Single Mamba block — API-compatible with mamba_ssm/mamba.py for
    construction: Mamba(d_model, d_state, d_conv, expand)."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, **kwargs):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                groups=d_inner, padding=d_conv - 1)
        # dt_rank = d_state for simplicity
        self.x_proj = nn.Linear(d_inner, d_state * 3)
        self.dt_proj = nn.Linear(d_state, d_inner)
        self.ssm = _SelectiveScan(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)

        self._init_params()

    def _init_params(self):
        # dt init: softplus(dt_bias) ≈ 0.001
        with torch.no_grad():
            self.ssm.dt_bias.fill_(math.log(0.001) - 1.0)
            # A ∈ (−16, −1)
            self.ssm.A_log.uniform_(-math.log(16.0), -math.log(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) → (B, T, d_model)
        B, T, _ = x.shape
        xz = self.in_proj(x)                                         # (B, T, 2·d_inner)
        x, z = xz.chunk(2, dim=-1)                                   # (B, T, d_inner) each
        x = self.conv1d(x.transpose(1, 2))                           # (B, d_inner, T+pad)
        x = x.transpose(1, 2)[:, :T, :]                              # causal
        x = F.silu(x)

        dt, Bc, C = self.x_proj(x).split([self.d_state] * 3, dim=-1)
        dt = F.softplus(self.dt_proj(dt) + self.ssm.dt_bias)         # (B, T, d_inner)

        y = self.ssm(x, dt, Bc, C)                                   # (B, T, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)                                      # (B, T, d_model)
