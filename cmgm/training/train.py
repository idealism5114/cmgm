"""
Training pipeline for CMGM (Section 3.4).

Implements:
  - MSE / Huber loss optimization
  - Early stopping with patience
  - Learning rate scheduling
  - Training/validation loss tracking
  - Model checkpointing (best validation loss)
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

from cmgm.config import (
    LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE,
    LOSS_TYPE, HUBER_DELTA,
)


def make_loss() -> nn.Module:
    """Return the configured loss function (MSE or Huber)."""
    if LOSS_TYPE == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA)
    return nn.MSELoss()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    debug: bool = False,
) -> float:
    """
    Train the CMGM model for one epoch.

    Section 3.4: "The model is trained using the Adam optimizer with
    Mean Squared Error (MSE) as the loss function."

    Args:
        model: CMGM model
        loader: Training DataLoader
        edge_index: Graph edges, shape (2, E)
        edge_weight: Edge weights, shape (E,)
        optimizer: Adam optimizer
        criterion: MSE loss
        device: torch device
        debug: print debug info

    Returns:
        avg_loss: Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Support both static graphs (X, y) and dynamic graphs (X, y, ei, ew)
        market_descriptor = None
        if len(batch) == 4:
            X_batch, y_batch, batch_ei, batch_ew = batch
            cur_ei = batch_ei.to(device)
            cur_ew = batch_ew.to(device)
        elif len(batch) == 3:
            X_batch, y_batch, market_descriptor = batch
            market_descriptor = market_descriptor.to(device)
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)
        else:
            X_batch, y_batch = batch
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)

        # X_batch: (B, T, N, 1) — normalized closing prices
        # y_batch: (B, N_commodities) — next-day normalized commodity prices
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        # Models with internal graph_learner (e.g. AdaptiveCMGM) don't
        # need edge_index/edge_weight passed from outside.
        if hasattr(model, 'graph_learner'):
            if market_descriptor is None:
                pred = model(X_batch, debug=(debug and batch_idx == 0))
            else:
                pred = model(
                    X_batch,
                    market_descriptor=market_descriptor,
                    debug=(debug and batch_idx == 0),
                )
        else:
            pred = model(X_batch, cur_ei, cur_ew, debug=(debug and batch_idx == 0))
        # pred: (B, N_commodities)

        # Compute loss — supports multi-horizon (B,H,Nc) or single (B,Nc)
        if pred.dim() == 3:  # multi-horizon: (B, H, Nc)
            loss = sum(criterion(pred[:, h, :], y_batch[:, h, :])
                       for h in range(pred.size(1)))
        else:
            loss = criterion(pred, y_batch)

        # Auxiliary loss: factor_res — supervise the market-mean branch
        # (r̂_mean stored in model.last_r_mean during forward)
        aux = getattr(model, 'last_r_mean', None)
        if aux is not None:
            if y_batch.dim() == 3:
                y_mean = y_batch.mean(dim=-1)                    # (B, H)
                loss = loss + criterion(aux, y_mean)
        # Auxiliary loss: spatial_temporal_attention — keep the global
        # branch trained (pred_global stored in model.last_aux_pred)
        aux_pred = getattr(model, 'last_aux_pred', None)
        if aux_pred is not None:
            if y_batch.dim() == 3:
                loss = loss + criterion(aux_pred, y_batch)

        # Regime diversity regularization (prototype-based regime generator)
        if hasattr(model, 'regime_diversity_loss'):
            loss = loss + model.regime_diversity_loss()
        # RegimeDynamic (F): adapter divergence + balance regularization
        rd = getattr(model, 'regime_dynamic', None)
        if rd is not None:
            loss = loss + rd.dynamic_loss()
        switching_branch = getattr(model, 'switching_transformer', None)
        if (
            switching_branch is not None
            and not switching_branch.null_control
        ):
            loss = loss + switching_branch.switch_loss()

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the CMGM model on validation set.

    Args:
        model: CMGM model
        loader: Validation DataLoader
        edge_index: Graph edges, shape (2, E)
        edge_weight: Edge weights, shape (E,)
        criterion: MSE loss
        device: torch device

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        market_descriptor = None
        if len(batch) == 4:
            X_batch, y_batch, batch_ei, batch_ew = batch
            cur_ei = batch_ei.to(device)
            cur_ew = batch_ew.to(device)
        elif len(batch) == 3:
            X_batch, y_batch, market_descriptor = batch
            market_descriptor = market_descriptor.to(device)
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)
        else:
            X_batch, y_batch = batch
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if hasattr(model, 'graph_learner'):
            if market_descriptor is None:
                pred = model(X_batch, debug=False)
            else:
                pred = model(
                    X_batch, market_descriptor=market_descriptor, debug=False
                )
        else:
            pred = model(X_batch, cur_ei, cur_ew, debug=False)
        if pred.dim() == 3:
            loss = sum(criterion(pred[:, h, :], y_batch[:, h, :])
                       for h in range(pred.size(1)))
        else:
            loss = criterion(pred, y_batch)
        # Auxiliary loss: factor_res — supervise the market-mean branch
        aux = getattr(model, 'last_r_mean', None)
        if aux is not None and y_batch.dim() == 3:
            loss = loss + criterion(aux, y_batch.mean(dim=-1))
        # Auxiliary loss: spatial_temporal_attention — keep the global
        # branch trained (pred_global stored in model.last_aux_pred)
        aux_pred = getattr(model, 'last_aux_pred', None)
        if aux_pred is not None and y_batch.dim() == 3:
            loss = loss + criterion(aux_pred, y_batch)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


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
) -> Dict:
    """
    Full training loop with early stopping.

    Section 3.4: Training Protocol:
      - Optimizer: Adam
      - Loss: MSE
      - Early stopping with patience of 20 epochs
      - Learning rate: 0.001

    Args:
        model: CMGM model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        edge_index: Graph edge indices, shape (2, E)
        edge_weight: Graph edge weights, shape (E,)
        device: torch device
        num_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        checkpoint_path: Path to save best model checkpoint

    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'best_epoch'
    """
    print(f"\n{'=' * 60}")
    print(f"CMGM Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs} (early stopping patience={patience})")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Move model to device
    model = model.to(device)

    # Section 3.4: Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Section 3.4: Loss (MSE or Huber)
    criterion = make_loss()

    # Learning rate scheduler: reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2,
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'lr_history': [],
        'switch_beta': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        switching_branch = getattr(model, 'switching_transformer', None)
        current_switch_beta = None
        if switching_branch is not None:
            current_switch_beta = switching_branch.set_epoch(epoch)

        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, edge_index, edge_weight,
            optimizer, criterion, device, debug=(epoch == 1)
        )

        # Validate
        val_loss = validate_epoch(
            model, val_loader, edge_index, edge_weight,
            criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr_history'].append(current_lr)
        history['switch_beta'].append(current_switch_beta)

        # Print progress
        print(f"  Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Best: {best_val_loss:.6f}")

        # Gate stats (for models with gated fusion, e.g. CMGM(fusion_mode='gate'))
        if hasattr(model, 'get_gate_stats') and epoch % 10 == 0:
            try:
                sample_batch = next(iter(val_loader))
                if len(sample_batch) == 4:
                    x_sample = sample_batch[0].to(device)
                else:
                    x_sample = sample_batch[0].to(device)
                stats = model.get_gate_stats(x_sample, edge_index.to(device), edge_weight.to(device))
                if 'gate_mean' in stats:
                    direction = 'LSTM' if stats['gate_mean'] > 0.5 else 'GCN'
                    print(f"  Gate   | mean={stats['gate_mean']:.3f} "
                          f"std={stats['gate_std']:.3f} "
                          f"[{stats['gate_min']:.3f}, {stats['gate_max']:.3f}]"
                          f" → 偏向{direction}")
            except Exception:
                pass  # silently ignore if gate stats not available

        # Early stopping: check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
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

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        switching_branch = getattr(model, 'switching_transformer', None)
        if switching_branch is not None:
            switching_branch.set_epoch(history['best_epoch'])
        print(f"\n[Checkpoint] Restored best model from epoch {history['best_epoch']}")

    # Save checkpoint if path provided
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
