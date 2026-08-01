"""
Graph-based baselines: GCN-only, GCN+GAT hybrid.

GCN-only: removes the LSTM from CMGM — pools over time after GCN.
GCN+GAT: replaces one GCN layer with GAT (graph attention).

These share the same graph structure as CMGM but differ in the
temporal aggregation and/or spatial convolution mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from typing import Dict
import time

from ..config import (
    GCN_INPUT_DIM, GCN_HIDDEN_DIM, GCN_OUTPUT_DIM,
    GCN_DROPOUT, FC_HIDDEN_DIM,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
)
from ..training.train import make_loss


# =============================================================================
# GCN baseline — no LSTM, mean pooling over time
# =============================================================================

class GCNOnlySpatial(nn.Module):
    """Two-layer GCN with configurable input dim + LayerNorm for stability."""

    def __init__(self, in_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, GCN_HIDDEN_DIM)
        self.input_norm = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.conv1 = GCNConv(GCN_HIDDEN_DIM, GCN_HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.conv2 = GCNConv(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.norm2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.input_norm(self.input_proj(x)))
        x = self.dropout(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)
        return x


class GCNOnlyModel(nn.Module):
    """GCN-only baseline. Configurable in_dim for multi-feature."""

    def __init__(self, num_nodes: int, n_commodities: int,
                 in_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.gcn = GCNOnlySpatial(in_dim=in_dim)

        self.fc = nn.Sequential(
            nn.Linear(num_nodes * GCN_OUTPUT_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, n_commodities),
        )

    def _batch_edge_index(self, edge_index, num_nodes, batch_size, device):
        offsets = torch.arange(batch_size, device=device) * num_nodes
        batched = edge_index.repeat(1, batch_size)
        offset_repeat = offsets.repeat_interleave(edge_index.shape[1])
        batched = batched + offset_repeat.unsqueeze(0)
        return batched

    def forward(self, x, edge_index, edge_weight, debug=False):
        batch_size, seq_len, num_nodes, _ = x.shape

        temporal_embs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (B, N, 1)
            x_flat = x_t.reshape(batch_size * num_nodes, -1)  # (B*N, 1)

            batched_ei = self._batch_edge_index(
                edge_index, num_nodes, batch_size, x_flat.device
            )
            batched_ew = edge_weight.repeat(batch_size).to(x_flat.device)

            out = self.gcn(x_flat, batched_ei, batched_ew)  # (B*N, 64)
            out = out.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)  # (B, N*64)
            temporal_embs.append(out)

        # Stack: (B, T, N*64)
        stacked = torch.stack(temporal_embs, dim=1)
        # Mean pool over time: (B, N*64)
        pooled = stacked.mean(dim=1)

        pred = self.fc(pooled)  # (B, N_commodities)
        return pred


# =============================================================================
# GCN+GAT hybrid — replaces second GCN layer with GAT
# =============================================================================

class GCNGATSpatial(nn.Module):
    """Hybrid spatial module: InputProj → GCNConv → GATConv, with LayerNorm."""

    def __init__(self, in_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, GCN_HIDDEN_DIM)
        self.input_norm = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gcn = GCNConv(GCN_HIDDEN_DIM, GCN_HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gat = GATConv(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM, heads=1)
        self.norm2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.input_norm(self.input_proj(x)))
        x = self.dropout(x)
        # GCN layer
        x = self.gcn(x, edge_index, edge_weight)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        # GAT layer (edge_weight not used by GAT in the same way)
        x = self.gat(x, edge_index)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)
        return x


class GCNGATModel(nn.Module):
    """GCN+GAT hybrid. Configurable in_dim."""

    def __init__(self, num_nodes: int, n_commodities: int,
                 in_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.spatial = GCNGATSpatial(in_dim=in_dim)

        self.fc = nn.Sequential(
            nn.Linear(num_nodes * GCN_OUTPUT_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, n_commodities),
        )

    def _batch_edge_index(self, edge_index, num_nodes, batch_size, device):
        offsets = torch.arange(batch_size, device=device) * num_nodes
        batched = edge_index.repeat(1, batch_size)
        offset_repeat = offsets.repeat_interleave(edge_index.shape[1])
        batched = batched + offset_repeat.unsqueeze(0)
        return batched

    def forward(self, x, edge_index, edge_weight, debug=False):
        batch_size, seq_len, num_nodes, _ = x.shape

        temporal_embs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            x_flat = x_t.reshape(batch_size * num_nodes, -1)

            batched_ei = self._batch_edge_index(
                edge_index, num_nodes, batch_size, x_flat.device
            )
            batched_ew = edge_weight.repeat(batch_size).to(x_flat.device)

            out = self.spatial(x_flat, batched_ei, batched_ew)
            out = out.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)
            temporal_embs.append(out)

        stacked = torch.stack(temporal_embs, dim=1)
        pooled = stacked.mean(dim=1)

        pred = self.fc(pooled)
        return pred


# =============================================================================
# Shared training function
# =============================================================================

def train_graph_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    model_name: str,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
) -> Dict:
    """Training loop for graph-based baselines."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = make_loss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"  Device: {device}, Epochs: {num_epochs}, Patience: {patience}")

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            ei = edge_index.to(device)
            ew = edge_weight.to(device)

            optimizer.zero_grad()
            pred = model(X_batch, ei, ew)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                ei = edge_index.to(device)
                ew = edge_weight.to(device)
                pred = model(X_batch, ei, ew)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [Early stopping at epoch {epoch}]")
                break

    train_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    print(f"  Best val loss: {best_val_loss:.6f} | Time: {train_time:.1f}s")
    return {
        'model': model,
        'history': history,
        'train_time': train_time,
        'best_val_loss': best_val_loss,
    }


def train_gcn_only(
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    n_commodities: int,
    device: torch.device,
    in_dim: int = 1,
    **kwargs,
) -> Dict:
    """Train GCN-only baseline."""
    print(f"\n{'=' * 50}")
    print("GCN-Only Baseline (no LSTM)")
    print(f"{'=' * 50}")
    model = GCNOnlyModel(num_nodes, n_commodities, in_dim=in_dim)
    return train_graph_baseline(
        model, train_loader, val_loader, edge_index, edge_weight,
        'GCN-Only', device, **kwargs,
    )


def train_gcn_gat(
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    n_commodities: int,
    device: torch.device,
    in_dim: int = 1,
    **kwargs,
) -> Dict:
    """Train GCN+GAT hybrid baseline."""
    print(f"\n{'=' * 50}")
    print("GCN+GAT Hybrid Baseline")
    print(f"{'=' * 50}")
    model = GCNGATModel(num_nodes, n_commodities, in_dim=in_dim)
    return train_graph_baseline(
        model, train_loader, val_loader, edge_index, edge_weight,
        'GCN+GAT', device, **kwargs,
    )
