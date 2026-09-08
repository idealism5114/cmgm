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
import json
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
    MULTI_HORIZONS, TARGET_HORIZON,
)

from cmgm.training.target_scale_huber import (
    VARIANT as TARGET_SCALE_VARIANT, OBJECTIVE as TARGET_SCALE_OBJECTIVE,
    TargetScaleHuber, horizon_terms as _target_scale_terms,
    detached_terms as _target_scale_detached_terms,
)

FIVE_DAY_ONLY_VARIANT = 'switching_latent_balanced_readout_5d_only'
FIVE_DAY_OBJECTIVE_MULTIPLIER = 4.0
GROUPED_VARIANT = 'switching_latent_balanced_readout_5_10_20'
GROUPED_HORIZONS = (5, 10, 20)
GROUPED_MULTIPLIER = 4.0 / 3.0
GROUPED_OBJECTIVE = '4/3*(5d+10d+20d)'
NO_SWITCH_KL_VARIANT = 'switching_latent_balanced_readout_no_switch_kl'


def _effective_switch_loss(model, branch):
    """Training contribution only; retain the filter's original KL and schedule."""
    if getattr(model, 'disable_switch_kl', False):
        # Independent zero: no KL autograd path, even when raw KL is nonzero.
        return branch.regime_filter.transition_logits.new_zeros(())
    return branch.switch_loss()


@torch.no_grad()
def _raw_regime_statistics(branch):
    p, prior = branch.last_regime_probabilities, branch.last_regime_priors
    return {
        'raw_KL': branch.regime_filter._last_switch_loss.detach().item(),
        'posterior_entropy': -(p * p.clamp_min(1e-8).log()).sum(-1).mean().item(),
        'prior_entropy': -(prior * prior.clamp_min(1e-8).log()).sum(-1).mean().item(),
        'posterior_prior_L1': (p-prior).abs().sum(-1).mean().item(),
    }


def _is_five_day_only(model):
    return getattr(model, 'variant', None) == FIVE_DAY_ONLY_VARIANT


def _target_scale_huber_prediction_loss(prediction, target, horizons, target_scales,
                                         reference_horizon=5, reference_delta=.02):
    if reference_horizon != 5 or reference_delta != .02:
        raise ValueError('TargetScaleHuber permits only the 5d/.02 anchor')
    calibration = (target_scales if isinstance(target_scales, TargetScaleHuber)
                   else TargetScaleHuber(tuple(target_scales), tuple(horizons), reference_horizon, reference_delta))
    _, weighted = _target_scale_terms(prediction, target, calibration, horizons)
    return sum(weighted.values())


def _prediction_loss(model, prediction, target, criterion):
    """Shared train/validation objectives; default variants retain sum_h L_h."""
    if getattr(model, 'variant', None) == TARGET_SCALE_VARIANT:
        calibration = getattr(model, 'target_scale_huber', None)
        if not isinstance(calibration, TargetScaleHuber):
            raise ValueError('TargetScaleHuber requires frozen TRAIN scales before loss computation')
        return _target_scale_huber_prediction_loss(prediction, target, MULTI_HORIZONS, calibration)
    if getattr(model, 'variant', None) == GROUPED_VARIANT:
        if (len(MULTI_HORIZONS) != 4 or set(MULTI_HORIZONS) != {1, 5, 10, 20}
                or prediction.dim() != 3 or prediction.size(1) != 4
                or prediction.shape != target.shape):
            raise ValueError('Grouped diagnostic requires matching 1/5/10/20 outputs')
        indices = [MULTI_HORIZONS.index(h) for h in GROUPED_HORIZONS]
        return GROUPED_MULTIPLIER * sum(
            criterion(prediction[:, i, :], target[:, i, :]) for i in indices)
    if _is_five_day_only(model):
        if (TARGET_HORIZON != 5 or len(MULTI_HORIZONS) != 4
                or prediction.dim() != 3 or prediction.size(1) != 4
                or prediction.shape != target.shape):
            raise ValueError('5d-only diagnostic requires four matching horizons and TARGET_HORIZON=5')
        idx = MULTI_HORIZONS.index(TARGET_HORIZON)
        return FIVE_DAY_OBJECTIVE_MULTIPLIER * criterion(
            prediction[:, idx, :], target[:, idx, :])
    if prediction.dim() == 3:
        return sum(criterion(prediction[:, h, :], target[:, h, :])
                   for h in range(prediction.size(1)))
    return criterion(prediction, target)


@torch.no_grad()
def _horizon_loss_values(prediction, target, criterion):
    """Detached descriptive losses; never part of backward or model selection."""
    return {str(h): criterion(prediction[:, i, :], target[:, i, :]).item()
            for i, h in enumerate(MULTI_HORIZONS)}


def make_loss() -> nn.Module:
    """Return the configured loss function (MSE or Huber)."""
    if LOSS_TYPE == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA)
    return nn.MSELoss()


def _switching_branch(model: nn.Module):
    """Return the active S1/S1C/S2F/D0 branch without changing interfaces."""
    branch = getattr(model, 'switching_latent_transformer', None)
    if branch is not None:
        return branch
    branch = getattr(model, 'switching_filter_rpe', None)
    if branch is not None:
        return branch
    return getattr(model, 'switching_transformer', None)


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
    five_only = _is_five_day_only(model)
    objective_sums = dict(raw_L5=0., scaled_prediction_loss=0., switch_loss=0., total_loss=0.)
    grouped = getattr(model, 'variant', None) == GROUPED_VARIANT
    no_switch = getattr(model, 'variant', None) == NO_SWITCH_KL_VARIANT
    target_scale = getattr(model, 'variant', None) == TARGET_SCALE_VARIANT
    if target_scale:
        objective_sums = {f'{kind}_huber_{h}': 0. for kind in ('raw','weighted') for h in MULTI_HORIZONS}
        objective_sums.update(prediction_loss=0., switch_loss=0., total_loss=0.)
    if no_switch:
        objective_sums = dict(prediction_loss=0., raw_KL=0., posterior_entropy=0.,
                              prior_entropy=0., posterior_prior_L1=0., weighted_switch_loss=0.,
                              total_loss=0., beta_effective=0.)
    if grouped:
        objective_sums = dict(raw_L1=0., raw_L5=0., raw_L10=0., raw_L20=0.,
                              group_raw=0., group_scaled=0., switch_loss=0., total_loss=0.)

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
        loss = _prediction_loss(model, pred, y_batch, criterion)
        if target_scale:
            target_scale_values = _target_scale_detached_terms(pred, y_batch, model.target_scale_huber, MULTI_HORIZONS)
            target_scale_values['prediction_loss'] = loss.detach().item()
        if no_switch:
            prediction_value = loss.detach().item()
        if five_only:
            scaled_prediction_value = loss.detach().item()
        if grouped:
            scaled_prediction_value = loss.detach().item()
            raw_horizon_values = _horizon_loss_values(pred, y_batch, criterion)

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
        switching_branch = _switching_branch(model)
        if (
            switching_branch is not None
            and not getattr(switching_branch, 'null_control', False)
        ):
            switch_loss = _effective_switch_loss(model, switching_branch)
            loss = loss + switch_loss

        if five_only:
            objective_sums['raw_L5'] += scaled_prediction_value / FIVE_DAY_OBJECTIVE_MULTIPLIER
            objective_sums['scaled_prediction_loss'] += scaled_prediction_value
            objective_sums['switch_loss'] += switch_loss.detach().item()
            objective_sums['total_loss'] += loss.detach().item()
        if grouped:
            for h, value in raw_horizon_values.items():
                objective_sums[f'raw_L{h}'] += value
            objective_sums['group_raw'] += sum(raw_horizon_values[str(h)] for h in GROUPED_HORIZONS)
            objective_sums['group_scaled'] += scaled_prediction_value
            objective_sums['switch_loss'] += switch_loss.detach().item()
            objective_sums['total_loss'] += loss.detach().item()
        if no_switch:
            objective_sums['prediction_loss'] += prediction_value
            for key, value in _raw_regime_statistics(switching_branch).items():
                objective_sums[key] += value
            objective_sums['weighted_switch_loss'] += switch_loss.item()
            objective_sums['total_loss'] += loss.detach().item()

        if target_scale:
            target_scale_values.update(switch_loss=switch_loss.detach().item(), total_loss=loss.detach().item())
            for key,value in target_scale_values.items():
                objective_sums[key] += value

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if five_only or grouped or no_switch or target_scale:
        model._last_train_objective = {k: v / max(num_batches, 1) for k, v in objective_sums.items()}
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
    five_only = _is_five_day_only(model)
    horizon_sums = {str(h): 0. for h in MULTI_HORIZONS}
    grouped = getattr(model, 'variant', None) == GROUPED_VARIANT
    target_scale = getattr(model, 'variant', None) == TARGET_SCALE_VARIANT
    target_sums = {f'{kind}_huber_{h}': 0. for kind in ('raw','weighted') for h in MULTI_HORIZONS}
    primary_abs, primary_squared, primary_count = 0., 0., 0

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
        loss = _prediction_loss(model, pred, y_batch, criterion)
        if target_scale:
            values = _target_scale_detached_terms(pred, y_batch, model.target_scale_huber, MULTI_HORIZONS)
            for key,value in values.items():
                target_sums[key] += value
            idx = MULTI_HORIZONS.index(5)
            residual = pred[:,idx].double() - y_batch[:,idx].double()
            primary_abs += residual.abs().sum().item()
            primary_squared += residual.square().sum().item()
            primary_count += residual.numel()
        if five_only or grouped:
            for h, value in _horizon_loss_values(pred, y_batch, criterion).items():
                horizon_sums[h] += value
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

    if five_only:
        horizon_means = {h: value / max(num_batches, 1) for h, value in horizon_sums.items()}
        model._last_val_objective = {
            'raw_L5': horizon_means['5'],
            'scaled_prediction_loss': total_loss / max(num_batches, 1),
            'aux_multi_horizon_loss': sum(horizon_means.values()),
            'per_horizon': horizon_means,
        }
    if grouped:
        horizon_means = {h: value / max(num_batches, 1) for h, value in horizon_sums.items()}
        model._last_val_objective = {
            **{f'raw_L{h}': value for h, value in horizon_means.items()},
            'group_raw': sum(horizon_means[str(h)] for h in GROUPED_HORIZONS),
            'group_scaled': total_loss / max(num_batches, 1),
            'aux_multi_horizon_loss': sum(horizon_means.values()),
            'per_horizon': horizon_means,
        }
    if target_scale:
        model._last_val_objective = {k:v/max(num_batches,1) for k,v in target_sums.items()}
        model._last_val_objective.update(prediction_loss=total_loss/max(num_batches,1),
                                         MAE_5d=primary_abs/primary_count, MSE_5d=primary_squared/primary_count)
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
    checkpoint_metadata: Optional[Dict] = None,
    epoch_diagnostic=None,
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
        checkpoint_metadata: Optional provenance fields stored with checkpoint

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

    target_scale = getattr(model, 'variant', None) == TARGET_SCALE_VARIANT
    if target_scale:
        if not isinstance(getattr(model, 'target_scale_huber', None), TargetScaleHuber):
            raise ValueError('Frozen TRAIN scales are required before optimizer construction')
        if getattr(model, 'disable_switch_kl', False):
            raise ValueError('TargetScaleHuber must retain switching KL')

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
        'epoch_diagnostics': {},
    }
    five_only = _is_five_day_only(model)
    grouped = getattr(model, 'variant', None) == GROUPED_VARIANT
    no_switch = getattr(model, 'variant', None) == NO_SWITCH_KL_VARIANT
    if no_switch:
        if not getattr(model, 'disable_switch_kl', False):
            raise ValueError('NoSwitchKL requires disable_switch_kl=True')
        history['switch_kl_enabled'] = False
        history['objective_history'] = []
    if five_only:
        history['objective'] = '4x_5d_only'
        history['objective_history'] = []
    if grouped:
        history['objective'] = GROUPED_OBJECTIVE
        history['objective_history'] = []
    if target_scale:
        history['objective'] = TARGET_SCALE_OBJECTIVE
        history['objective_history'] = []
        history['scale_metadata'] = model.target_scale_huber.metadata()
        history['switch_kl_enabled'] = True
    persistence_branch = _switching_branch(model)
    persistence_filter = (
        getattr(persistence_branch, 'regime_filter', None)
        if persistence_branch is not None else None
    )
    learnable_persistence = getattr(
        persistence_filter, 'learnable_sticky_alpha', False
    )
    if learnable_persistence:
        history['alpha_history'] = []
        history['sticky_logit_history'] = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        switching_branch = _switching_branch(model)
        current_switch_beta = None
        if switching_branch is not None:
            current_switch_beta = switching_branch.set_epoch(epoch)
            if no_switch:
                current_switch_beta = 0.0

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
        if five_only:
            row = {'epoch': epoch, 'train': dict(model._last_train_objective),
                   'val': dict(model._last_val_objective), 'lr': current_lr,
                   'switch_beta': current_switch_beta}
            history['objective_history'].append(row)
            print(f"  [D0B-5dOnly epoch {epoch}] "
                  f"raw_L5={row['train']['raw_L5']:.9g} "
                  f"scaled_prediction_loss={row['train']['scaled_prediction_loss']:.9g} "
                  f"switch_loss={row['train']['switch_loss']:.9g} "
                  f"total_loss={row['train']['total_loss']:.9g} "
                  f"val_scaled_5d_loss={val_loss:.9g} "
                  f"aux_multi_horizon_val_loss={row['val']['aux_multi_horizon_loss']:.9g}")
        if learnable_persistence:
            alpha_value = persistence_filter.sticky_alpha_value().detach().item()
            logit_value = persistence_filter.sticky_logit.detach().item()
            history['alpha_history'].append(alpha_value)
            history['sticky_logit_history'].append(logit_value)
            print(f"  [D0E epoch {epoch}] sticky_logit={logit_value:.9g} "
                  f"alpha={alpha_value:.9g}")
        if grouped:
            row = {'epoch': epoch, 'train': dict(model._last_train_objective),
                   'val': dict(model._last_val_objective), 'lr': current_lr,
                   'switch_beta': current_switch_beta}
            history['objective_history'].append(row)
            values = ' '.join(f'{k}={v:.9g}' for k, v in row['train'].items())
            print(f"  [D0B grouped epoch {epoch}] {values} "
                  f"val_grouped_objective={val_loss:.9g} raw_val_L5={row['val']['raw_L5']:.9g}")
        if no_switch:
            row = {'epoch': epoch, 'train': dict(model._last_train_objective),
                   'val_prediction_loss': val_loss, 'lr': current_lr,
                   'beta_effective': 0., 'reference_schedule_beta': persistence_filter.current_beta}
            history['objective_history'].append(row)
            values = ' '.join(f'{k}={v:.9g}' for k,v in row['train'].items())
            print(f"  [D0B-NoSwitchKL epoch {epoch}] {values} val_prediction_loss={val_loss:.9g}")
        if target_scale:
            row = {'epoch':epoch, 'train':dict(model._last_train_objective),
                   'val':dict(model._last_val_objective), 'val_prediction_loss':val_loss,
                   'lr':current_lr, 'switch_beta':current_switch_beta}
            history['objective_history'].append(row)
            print('[D0B-TargetScaleHuber epoch] ' + json.dumps(row), flush=True)
        diagnostic_epochs = (1, 5, 10, 20) if no_switch else (1, 5, 10)
        if epoch_diagnostic is not None and epoch in diagnostic_epochs:
            history['epoch_diagnostics'][f'epoch{epoch}'] = (
                epoch_diagnostic(model, f'epoch{epoch}')
            )

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

    if five_only or grouped or no_switch or target_scale:
        history['final_epoch'] = len(history['objective_history'])
    if no_switch and epoch_diagnostic is not None:
        history['epoch_diagnostics']['final'] = epoch_diagnostic(model, f"final(epoch {history['final_epoch']})")
    if grouped or target_scale:
        # Persist the elapsed training-loop time before checkpoint diagnostics.
        history['training_elapsed_seconds'] = time.time() - t0
    if learnable_persistence and history['alpha_history']:
        history['final_epoch'] = len(history['alpha_history'])
        history['final_epoch_alpha'] = history['alpha_history'][-1]
        history['final_epoch_sticky_logit'] = history['sticky_logit_history'][-1]
        print(f"  [D0E final epoch {history['final_epoch']}] "
              f"alpha={history['final_epoch_alpha']:.9g} "
              f"sticky_logit={history['final_epoch_sticky_logit']:.9g}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        switching_branch = _switching_branch(model)
        if switching_branch is not None:
            switching_branch.set_epoch(history['best_epoch'])
        if epoch_diagnostic is not None:
            history['epoch_diagnostics']['best'] = epoch_diagnostic(
                model, f"best(epoch {history['best_epoch']})"
            )
        print(f"\n[Checkpoint] Restored best model from epoch {history['best_epoch']}")
        if learnable_persistence:
            history['best_alpha'] = persistence_filter.sticky_alpha_value().detach().item()
            history['best_sticky_logit'] = persistence_filter.sticky_logit.detach().item()
            print(f"  [D0E best epoch {history['best_epoch']}] "
                  f"alpha={history['best_alpha']:.9g} "
                  f"sticky_logit={history['best_sticky_logit']:.9g}")

    # Save checkpoint if path provided
    if checkpoint_path and best_model_state is not None:
        checkpoint = {
            'model_state_dict': best_model_state,
            'history': history,
            'best_val_loss': best_val_loss,
            'best_epoch': history['best_epoch'],
        }
        if checkpoint_metadata:
            checkpoint['metadata'] = dict(checkpoint_metadata)
        if five_only:
            checkpoint.setdefault('metadata', {}).update({
                'variant': model.variant,
                'objective': '4x_5d_only',
                'objective_description': 'scale-matched 5d-only diagnostic objective',
                'objective_multiplier': FIVE_DAY_OBJECTIVE_MULTIPLIER,
                'best_epoch': history['best_epoch'],
                'best_val_scaled_5d_loss': best_val_loss,
                'raw_val_5d_loss': best_val_loss / FIVE_DAY_OBJECTIVE_MULTIPLIER,
                'parameter_count': sum(p.numel() for p in model.parameters()),
            })
        if learnable_persistence:
            checkpoint.setdefault('metadata', {}).update({
                'variant': model.variant,
                'best_epoch': history['best_epoch'],
                'best_val_loss': best_val_loss,
                'final_learned_alpha': history['best_alpha'],
                'sticky_logit': history['best_sticky_logit'],
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'last_training_epoch': history['final_epoch'],
                'last_training_epoch_alpha': history['final_epoch_alpha'],
                'alpha_semantics': 'final_learned_alpha belongs to the restored best checkpoint',
            })
        if grouped:
            best_row = history['objective_history'][history['best_epoch'] - 1]
            checkpoint.setdefault('metadata', {}).update({
                'variant': model.variant, 'objective': GROUPED_OBJECTIVE,
                'objective_description': 'scale-matched grouped-objective diagnostic',
                'objective_multiplier': GROUPED_MULTIPLIER,
                'best_epoch': history['best_epoch'],
                'best_grouped_val_loss': best_val_loss,
                'raw_val_5d_loss': best_row['val']['raw_L5'],
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'training_elapsed_seconds': history['training_elapsed_seconds'],
            })
        if target_scale:
            best_row = history['objective_history'][history['best_epoch']-1]
            checkpoint.setdefault('metadata', {}).update({
                **model.target_scale_huber.metadata(), 'variant':model.variant,
                'best_epoch':history['best_epoch'],
                'parameter_count':sum(p.numel() for p in model.parameters()),
                'best_val_scale_aware_prediction_loss':best_val_loss,
                'best_val_5d_MAE':best_row['val']['MAE_5d'],
                'best_val_5d_MSE':best_row['val']['MSE_5d'],
                'training_elapsed_seconds':history['training_elapsed_seconds'],
                'switch_kl_enabled':True,
                'metric_standard':'pooled MAE/MSE/RMSE; RMSE=sqrt(MSE); unmasked sign Hit',
            })
        if no_switch:
            checkpoint.setdefault('metadata', {}).update({
                'variant': model.variant, 'best_epoch': history['best_epoch'],
                'best_val_loss': best_val_loss,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'switch_kl_enabled': False, 'beta_effective': 0.,
                'objective': 'sum_1d_5d_10d_20d',
                'reference_beta_max': persistence_filter.beta_max,
                'reference_warmup_epochs': persistence_filter.warmup_epochs,
            })
        torch.save(checkpoint, checkpoint_path)
        print(f"[Checkpoint] Saved to {checkpoint_path}")

    print(f"{'=' * 60}\n")
    history['train_time'] = time.time() - t0
    return history
