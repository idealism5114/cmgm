"""
Training with Cosine Annealing Warm Restarts + Gradient Clipping.

Differences from cmgm.train:
  1. Scheduler: CosineAnnealingWarmRestarts (not ReduceLROnPlateau)
  2. Gradient clipping: torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

from cmgm.config import (
    LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE,
    GCN_DROPOUT, LSTM_DROPOUT,
)
from cmgm.training.train import make_loss
from cmgm.training.train import validate_epoch


def _train_epoch_clipped(
    model, loader, edge_index, edge_weight,
    optimizer, criterion, device, max_norm=1.0, debug=False,
):
    """
    Train one epoch with gradient clipping.

    Identical to cmgm.train.train_epoch() + clip_grad_norm_.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 4:
            X_batch, y_batch, batch_ei, batch_ew = batch
            cur_ei = batch_ei.to(device)
            cur_ew = batch_ew.to(device)
        else:
            X_batch, y_batch = batch
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch, cur_ei, cur_ew, debug=(debug and batch_idx == 0))
        loss = criterion(pred, y_batch)
        loss.backward()

        # ── Gradient clipping ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _get_cosine_scheduler(optimizer, T_0=30, T_mult=2, eta_min=1e-6):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    checkpoint_path: Optional[str] = None,
    # ── Cosine annealing parameters ──
    T_0: int = 30,
    T_mult: int = 2,
    eta_min: float = 1e-6,
    # ── Gradient clipping ──
    max_norm: float = 1.0,
    # ── Warmup ──
    warmup_epochs: int = 10,
) -> Dict:
    """
    Full training loop: LR Warmup + CosineAnnealingWarmRestarts + Gradient Clip.

    Differences from cmgm.train.train():
      - LR warmup: linear 0 → lr over {warmup_epochs} epochs
      - Scheduler: CosineAnnealingWarmRestarts (not ReduceLROnPlateau)
      - Gradient clipping: clip_grad_norm_(..., max_norm)
    """
    print(f"\n{'=' * 60}")
    print(f"CMGM Training (Warmup + Cosine + Clip)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs} (early stopping patience={patience})")
    print(f"LR: {lr}, Weight decay: {weight_decay}")
    print(f"Warmup: {warmup_epochs} epochs (linear 0→{lr})")
    print(f"CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult})")
    print(f"Gradient clipping: max_norm={max_norm}")
    print(f"Dropout: GCN={GCN_DROPOUT}, LSTM={LSTM_DROPOUT}")
    print(f"{'=' * 60}")

    t0 = time.time()

    model = model.to(device)

    # Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # MSE loss
    criterion = make_loss()

    # Cosine Annealing scheduler (created AFTER warmup to keep T_0 aligned)
    scheduler = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'lr_history': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ── LR Warmup: linear 0 → lr over warmup_epochs ──
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor
        elif epoch == warmup_epochs + 1:
            # Initialize cosine scheduler at the target LR
            scheduler = _get_cosine_scheduler(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
            )

        train_loss = _train_epoch_clipped(
            model, train_loader, edge_index, edge_weight,
            optimizer, criterion, device,
            max_norm=max_norm, debug=(epoch == 1),
        )

        val_loss = validate_epoch(
            model, val_loader, edge_index, edge_weight,
            criterion, device
        )

        # ── Step scheduler (only after warmup) ──
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr_history'].append(current_lr)

        print(f"  Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Best: {best_val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. "
                      f"Stopping at epoch {epoch}.")
                print(f"[Early Stopping] Best validation loss: {best_val_loss:.6f} "
                      f"at epoch {history['best_epoch']}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Checkpoint] Restored best model from epoch {history['best_epoch']}")

    if checkpoint_path and best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'history': history,
            'best_val_loss': best_val_loss,
            'best_epoch': history['best_epoch'],
        }, checkpoint_path)
        print(f"[Checkpoint] Saved to {checkpoint_path}")

    print(f"{'=' * 60}\n")
    history['train_time'] = time.time() - t0
    return history
