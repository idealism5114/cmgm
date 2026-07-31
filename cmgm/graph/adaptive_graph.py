"""
AdaptiveGraphLearner — Learnable graph structure (gradient-friendly version).

Core formula (improved for gradient flow):
    M1 = E1 @ Θ1                       # No tanh — avoids saturation
    M2 = E2 @ Θ2
    A  = ReLU(tanh(α · (M1 @ M2ᵀ - M2 @ M1ᵀ)))   # Single nonlinearity
    A  = A · σ(10 · (A - threshold))             # Soft top-k via sigmoid

Fixes applied (vs original MTGNN):
  1. α is a learnable parameter (init 0.5), not a fixed scalar at 3
  2. Removed tanh on M1/M2 — one tanh is enough
  3. Hard top-k replaced with differentiable sigmoid gating
  4. top-k increased to 10 for better gradient signal (284 nodes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    Learnable directed graph adjacency matrix with improved gradient flow.

    Args:
        num_nodes: Number of nodes in the graph (e.g. 284)
        embed_dim: Node embedding dimension (default: 10)
        alpha:     Initial saturation rate (default: 0.5, learnable)
        top_k:     Number of neighbors to keep per node (default: 10)
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10,
                 alpha: float = 0.5, top_k: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k

        # Learnable α (initialized small to avoid tanh saturation)
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Learnable node embeddings
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))

        # Learnable transformations
        self.Θ1 = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.Θ2 = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier uniform initialization for all parameters."""
        for p in [self.E1, self.E2]:
            nn.init.xavier_uniform_(p)
        for p in [self.Θ1, self.Θ2]:
            nn.init.xavier_uniform_(p)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            A: (num_nodes, num_nodes) — non-negative, soft top-k adjacency
        """
        # Projected embeddings (no tanh — keeps gradient flowing)
        M1 = self.E1 @ self.Θ1                              # (N, d)
        M2 = self.E2 @ self.Θ2                              # (N, d)

        # Anti-symmetric similarity → adjacency
        # Single tanh after anti-symmetric difference (not before)
        S = M1 @ M2.T - M2 @ M1.T                           # (N, N)
        A = F.relu(torch.tanh(self.alpha * S))               # (N, N)

        # Soft top-k: differentiable sigmoid gating
        # Instead of hard zeroing non-top-k positions, use a smooth gate
        if self.top_k is not None and self.top_k < self.num_nodes:
            # threshold = k-th largest value per row, shape (N, 1)
            threshold = torch.topk(A, self.top_k, dim=1).values[:, -1:]
            # Sigmoid gate: ~1 for values above threshold, ~0 below
            # gradient flows through both A and threshold
            gate = torch.sigmoid((A - threshold) * 10.0)
            A = A * gate

        return A
