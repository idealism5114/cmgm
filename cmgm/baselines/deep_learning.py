"""
Deep learning baselines: LSTM, BiLSTM.

These models process the raw price sequence without graph preprocessing.
Input: (B, T, N) — all assets as features, no graph structure.
Output: (B, N_commodities) — next-day commodity price predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import time

from ..config import (
    SEQ_LEN, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, FC_HIDDEN_DIM, GCN_DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
)
from ..training.train import make_loss


class LSTMModel(nn.Module):
    """
    LSTM baseline — no graph preprocessing.

    Architecture:
      Input:  (B, T, N)
      → LSTM(N → LSTM_HIDDEN) → final hidden
      → FC(LSTM_HIDDEN → FC_HIDDEN → N_commodities)

    Tensor shapes:
      Input:       (B, T, N)
      LSTM:        (B, T, N) → hidden (B, LSTM_HIDDEN)
      FC:          (B, LSTM_HIDDEN) → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 feat_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.feat_dim = feat_dim

        lstm_input = num_nodes * feat_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, n_commodities),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) — input features

        Returns:
            pred: (B, N_commodities)
        """
        # Flatten node×feature dims: (B, T, N, F) → (B, T, N*F)
        if x.dim() == 4:
            B, T = x.shape[0], x.shape[1]
            x = x.reshape(B, T, -1)

        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


class BiLSTMModel(nn.Module):
    """
    BiLSTM baseline — bidirectional LSTM, no graph preprocessing.

    Architecture:
      Input:  (B, T, N)
      → BiLSTM(N → LSTM_HIDDEN, bidirectional) → concat final states
      → FC(2*LSTM_HIDDEN → FC_HIDDEN → N_commodities)

    Tensor shapes:
      Input:       (B, T, N)
      BiLSTM:      (B, T, N) → hidden (num_layers*2, B, LSTM_HIDDEN)
      Concat:      (B, 2*LSTM_HIDDEN)
      FC:          (B, 2*LSTM_HIDDEN) → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 feat_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.feat_dim = feat_dim

        lstm_input = num_nodes * feat_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional → 2 * LSTM_HIDDEN (concat forward + backward)
        self.fc = nn.Sequential(
            nn.Linear(2 * LSTM_HIDDEN_DIM, FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GCN_DROPOUT),
            nn.Linear(FC_HIDDEN_DIM, n_commodities),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) — input features

        Returns:
            pred: (B, N_commodities)
        """
        if x.dim() == 4:
            B, T = x.shape[0], x.shape[1]
            x = x.reshape(B, T, -1)  # (B, T, N*F)

        # BiLSTM
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, LSTM_HIDDEN)

        # Concat forward and backward from the last layer
        # For bidirectional, h_n[-2] = forward, h_n[-1] = backward
        h_forward = h_n[-2]   # (B, LSTM_HIDDEN)
        h_backward = h_n[-1]  # (B, LSTM_HIDDEN)
        h_concat = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2*LSTM_HIDDEN)

        out = self.fc(h_concat)
        return out


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
) -> Dict:
    """
    Generic training loop for deep learning baselines.

    Returns:
        dict with 'model', 'history', 'train_time'
    """
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
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:3d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
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

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    print(f"  Best val loss: {best_val_loss:.6f} | Time: {train_time:.1f}s")

    return {
        'model': model,
        'history': history,
        'train_time': train_time,
        'best_val_loss': best_val_loss,
    }


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_nodes: int,
    n_commodities: int,
    device: torch.device,
    feat_dim: int = 1,
    **kwargs,
) -> Dict:
    """Train LSTM baseline."""
    print(f"\n{'=' * 50}")
    print("LSTM Baseline")
    print(f"{'=' * 50}")
    model = LSTMModel(num_nodes, n_commodities, feat_dim=feat_dim)
    return train_dl_model(model, train_loader, val_loader, 'LSTM', device, **kwargs)


def train_bilstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_nodes: int,
    n_commodities: int,
    device: torch.device,
    feat_dim: int = 1,
    **kwargs,
) -> Dict:
    """Train BiLSTM baseline."""
    print(f"\n{'=' * 50}")
    print("BiLSTM Baseline")
    print(f"{'=' * 50}")
    model = BiLSTMModel(num_nodes, n_commodities, feat_dim=feat_dim)
    return train_dl_model(model, train_loader, val_loader, 'BiLSTM', device, **kwargs)
