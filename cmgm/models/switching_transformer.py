"""Modular temporal components for the new switching-transformer route.

S0 contains market-aware input tokenization and an ordinary Transformer only.
Switching-state inference and relative-position bias are intentionally absent.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketAwareTemporalEncoder(nn.Module):
    """Encode raw nodes into one hierarchical market-aware token per day."""

    MARKET_NAMES = ("stock", "bond", "commodity")

    def __init__(self, feat_dim: int, n_stock: int, n_bond: int,
                 n_commodity: int, node_dim: int = 32,
                 d_model: int = 128, use_dispersion: bool = False):
        super().__init__()
        market_sizes = {
            "stock": int(n_stock),
            "bond": int(n_bond),
            "commodity": int(n_commodity),
        }
        if any(size <= 0 for size in market_sizes.values()):
            raise ValueError(f"all market sizes must be positive, got {market_sizes}")
        self.feat_dim = int(feat_dim)
        self.node_dim = int(node_dim)
        self.d_model = int(d_model)
        self.use_dispersion = bool(use_dispersion)
        self.market_sizes = market_sizes
        self.stock_encoder = self._make_node_encoder()
        self.bond_encoder = self._make_node_encoder()
        self.commodity_encoder = self._make_node_encoder()
        self.stock_attention_score = nn.Linear(self.node_dim, 1)
        self.bond_attention_score = nn.Linear(self.node_dim, 1)
        self.commodity_attention_score = nn.Linear(self.node_dim, 1)
        market_width = self.node_dim * (2 if self.use_dispersion else 1)
        self.daily_projection = nn.Sequential(
            nn.Linear(market_width * len(self.MARKET_NAMES), self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def _make_node_encoder(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.feat_dim, self.node_dim),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )

    def _market_modules(self):
        return {
            "stock": (self.stock_encoder, self.stock_attention_score),
            "bond": (self.bond_encoder, self.bond_attention_score),
            "commodity": (
                self.commodity_encoder,
                self.commodity_attention_score,
            ),
        }

    def _split_markets(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"x must have shape (B,T,N,F), got {tuple(x.shape)}")
        expected_nodes = sum(self.market_sizes.values())
        if x.shape[2] != expected_nodes or x.shape[3] != self.feat_dim:
            raise ValueError(
                f"expected node/feature shape ({expected_nodes},{self.feat_dim}), "
                f"got {tuple(x.shape[2:])}"
            )
        stock_end = self.market_sizes["stock"]
        bond_end = stock_end + self.market_sizes["bond"]
        return {
            "stock": x[:, :, :stock_end, :],
            "bond": x[:, :, stock_end:bond_end, :],
            "commodity": x[:, :, bond_end:, :],
        }

    def forward(self, x: torch.Tensor, zero_market: str = None,
                zero_component: str = None) -> torch.Tensor:
        if zero_market is not None and zero_market not in self.MARKET_NAMES:
            raise ValueError(f"zero_market must be one of {self.MARKET_NAMES}")
        valid_components = {
            f"{name}_{component}"
            for name in self.MARKET_NAMES
            for component in ("level", "dispersion")
        } | {"all_dispersion"}
        if zero_component is not None and zero_component not in valid_components:
            raise ValueError(
                f"zero_component must be one of {sorted(valid_components)}"
            )
        if zero_component is not None and not self.use_dispersion:
            raise ValueError("zero_component diagnostics require use_dispersion=True")
        market_inputs = self._split_markets(x)
        node_encodings = {}
        attentions = {}
        market_tokens = {}
        market_dispersions = {}
        market_representations = {}
        for name, (node_encoder, attention_score) in self._market_modules().items():
            encoded = node_encoder(market_inputs[name])            # (B,T,N_m,32)
            logits = attention_score(encoded)                      # (B,T,N_m,1)
            attention = F.softmax(logits, dim=2)                   # node dimension
            token = (attention * encoded).sum(dim=2)               # level: (B,T,32)
            dispersion = (
                encoded.std(dim=2, unbiased=False)                 # (B,T,32)
                if self.use_dispersion
                else None
            )
            if zero_market == name:
                token = torch.zeros_like(token)
                if dispersion is not None:
                    dispersion = torch.zeros_like(dispersion)
            if zero_component == f"{name}_level":
                token = torch.zeros_like(token)
            if (
                zero_component == f"{name}_dispersion"
                or zero_component == "all_dispersion"
            ):
                dispersion = torch.zeros_like(dispersion)
            node_encodings[name] = encoded
            attentions[name] = attention.squeeze(-1)
            market_tokens[name] = token
            if dispersion is not None:
                market_dispersions[name] = dispersion
            market_representations[name] = (
                torch.cat([token, dispersion], dim=-1)
                if self.use_dispersion
                else token
            )

        concatenated = torch.cat(
            [market_representations[name] for name in self.MARKET_NAMES], dim=-1
        )                                                   # S0: (B,T,96); S0D: (B,T,192)
        daily_tokens = self.daily_projection(concatenated)          # (B,T,128)

        self.last_node_encodings = {
            name: value.detach() for name, value in node_encodings.items()
        }
        self.last_market_attentions = {
            name: value.detach() for name, value in attentions.items()
        }
        self.last_market_tokens = {
            name: value.detach() for name, value in market_tokens.items()
        }
        self.last_market_dispersions = {
            name: value.detach() for name, value in market_dispersions.items()
        }
        self.last_market_concat = concatenated.detach()
        self.last_daily_tokens = daily_tokens.detach()
        return daily_tokens


class TemporalSelfAttention(nn.Module):
    """Ordinary multi-head self-attention with no positional or regime bias."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        q = self.q(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v(x).view(
            batch_size, time_steps, self.n_heads, self.head_dim
        ).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention = F.softmax(logits, dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(
            batch_size, time_steps, self.d_model
        )
        return self.out(output)


class TemporalTransformerBlock(nn.Module):
    """Pre-bias-free ordinary temporal Transformer block."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = TemporalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class BaseTemporalTransformer(nn.Module):
    """Ordinary Transformer plus the existing temporal attention pooling."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256,
                 dropout: float = 0.1, output_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pool_score = nn.Linear(d_model, 1)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden)
        scores = self.pool_score(hidden).squeeze(-1)               # (B,T)
        attention = F.softmax(scores, dim=1)                       # time dimension
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)     # (B,128)
        output = self.output_projection(pooled)                    # (B,64)
        self.last_transformer_output = hidden.detach()
        self.last_temporal_attention = attention.detach()
        self.last_temporal_output = output.detach()
        return output


class MarketTokenTransformerBranch(nn.Module):
    """S0 temporal branch: market-aware tokens followed by a plain Transformer."""

    def __init__(self, feat_dim: int, n_stock: int, n_bond: int,
                 n_commodity: int, node_dim: int = 32,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 256,
                 dropout: float = 0.1, output_dim: int = 64,
                 use_dispersion: bool = False):
        super().__init__()
        self.market_encoder = MarketAwareTemporalEncoder(
            feat_dim=feat_dim,
            n_stock=n_stock,
            n_bond=n_bond,
            n_commodity=n_commodity,
            node_dim=node_dim,
            d_model=d_model,
            use_dispersion=use_dispersion,
        )
        self.transformer = BaseTemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            output_dim=output_dim,
        )

    def encode_market_tokens(self, x: torch.Tensor,
                             zero_market: str = None,
                             zero_component: str = None) -> torch.Tensor:
        return self.market_encoder(
            x,
            zero_market=zero_market,
            zero_component=zero_component,
        )

    def temporal_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer(tokens)

    def forward(self, x: torch.Tensor, zero_market: str = None,
                zero_component: str = None) -> torch.Tensor:
        tokens = self.encode_market_tokens(
            x,
            zero_market=zero_market,
            zero_component=zero_component,
        )
        return self.temporal_forward(tokens)
