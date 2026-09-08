"""Frozen TRAIN-target-std calibration; no model parameters or scale fitting on errors."""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

VARIANT = 'switching_latent_balanced_readout_target_scale_huber'
OBJECTIVE = 'train_target_std_5d_anchor_gradient_cap_preserving_huber'
HORIZONS = (1, 5, 10, 20)
REFERENCE_STD = (.0166215601, .0361432539, .0500428743, .0703851644)


@dataclass(frozen=True)
class TargetScaleHuber:
    target_scales: tuple
    horizons: tuple = HORIZONS
    reference_horizon: int = 5
    reference_delta: float = .02

    def __post_init__(self):
        # Store only immutable Python constants, never optimizer parameters/buffers.
        object.__setattr__(self, 'target_scales', tuple(float(v) for v in self.target_scales))
        object.__setattr__(self, 'horizons', tuple(self.horizons))
        if set(self.horizons) != set(HORIZONS) or len(self.horizons) != 4:
            raise ValueError('TargetScaleHuber requires exactly 1/5/10/20d')
        if self.reference_horizon != 5 or self.reference_delta != .02:
            raise ValueError('Only 5d anchor and delta_ref=.02 are authorized')
        if len(self.target_scales) != 4 or not all(math.isfinite(s) and s > 0 for s in self.target_scales):
            raise ValueError('All TRAIN target scales must be finite and positive')

    @property
    def deltas(self):
        anchor = self.target_scales[self.horizons.index(5)]
        return tuple(.02 if h == 5 else .02 * (s / anchor)
                     for h, s in zip(self.horizons, self.target_scales))

    @property
    def weights(self):
        return tuple(1. if h == 5 else .02 / d for h, d in zip(self.horizons, self.deltas))

    def metadata(self):
        return {'objective': OBJECTIVE, 'target_scale_source': 'TRAIN target std',
                'ddof': 0, 'unbiased': False, 'dtype': 'float64',
                'aggregation': 'all TRAIN windows × commodities; includes tail; frozen before training',
                'reference_horizon': 5, 'reference_delta': .02, 'horizons': list(self.horizons),
                **{f'target_scale_{h}': s for h, s in zip(self.horizons, self.target_scales)},
                **{f'delta_{h}': d for h, d in zip(self.horizons, self.deltas)},
                **{f'weight_{h}': w for h, w in zip(self.horizons, self.weights)},
                'caps': {str(h): w*d for h,w,d in zip(self.horizons,self.weights,self.deltas)}}


def estimate_training_scales(train_dataset, expected_samples=1396):
    """The only fitting entry: dataset targets, all windows (no dropped tail).

    Caller must pass data['loaders']['train'].dataset; no VAL/TEST/residual inputs.
    Fail closed if the current target construction differs from the prior diagnostic.
    """
    if len(train_dataset) != expected_samples:
        raise ValueError(f'TRAIN window count changed: {len(train_dataset)} != {expected_samples}; stop')
    horizons = tuple(train_dataset.horizons)
    targets = np.stack([np.asarray(train_dataset[i][1], dtype=np.float64) for i in range(len(train_dataset))])
    if targets.shape != (expected_samples, 4, 24) or not np.isfinite(targets).all():
        raise ValueError(f'Unexpected TRAIN target population {targets.shape}; stop')
    scales = tuple(float(targets[:, horizons.index(h)].std(ddof=0)) for h in HORIZONS)
    if not np.allclose(scales, REFERENCE_STD, rtol=1e-6, atol=1e-10):
        raise ValueError(f'TRAIN target std differs from diagnostic: actual={scales}, reference={REFERENCE_STD}; stop before training')
    frozen = TargetScaleHuber(scales)
    return frozen, {'PASS': True, 'samples': len(targets), 'shape': list(targets.shape),
                    'reference_std': list(REFERENCE_STD),
                    'max_abs_reference_diff': float(np.max(np.abs(np.array(scales)-REFERENCE_STD))),
                    'rtol': 1e-6, 'atol': 1e-10, **frozen.metadata()}


def restore_scale_metadata(metadata):
    if metadata.get('objective') != OBJECTIVE or metadata.get('target_scale_source') != 'TRAIN target std':
        raise ValueError('Checkpoint lacks frozen TRAIN target scale metadata')
    c = TargetScaleHuber(tuple(metadata[f'target_scale_{h}'] for h in HORIZONS))
    if metadata.get('ddof') != 0 or metadata.get('reference_horizon') != 5 or metadata.get('reference_delta') != .02:
        raise ValueError('Checkpoint scale definition/anchor does not match this experiment')
    for key,value in c.metadata().items():
        if key.startswith(('delta_', 'weight_')) and not math.isclose(metadata[key],value,rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError(f'Checkpoint has inconsistent {key}')
    return c


def horizon_terms(prediction, target, calibration, horizons=HORIZONS):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 4:
        raise ValueError('TargetScaleHuber requires matching [B,4,Nc] predictions and targets')
    if set(horizons) != set(HORIZONS) or len(horizons) != 4:
        raise ValueError('Unexpected horizon mapping')
    raw, weighted = {}, {}
    for h,d,w in zip(calibration.horizons,calibration.deltas,calibration.weights):
        idx = list(horizons).index(h)
        value = F.huber_loss(prediction[:,idx], target[:,idx], reduction='mean', delta=d)
        raw[str(h)] = value
        weighted[str(h)] = w * value
    return raw, weighted


def detached_terms(prediction, target, calibration, horizons=HORIZONS):
    with torch.no_grad():
        raw, weighted = horizon_terms(prediction, target, calibration, horizons)
        return {**{f'raw_huber_{h}':v.item() for h,v in raw.items()},
                **{f'weighted_huber_{h}':v.item() for h,v in weighted.items()}}
