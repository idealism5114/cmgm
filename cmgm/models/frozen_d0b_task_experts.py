"""Independent prediction residuals on a frozen, independently trained D0B.

No routing or representation mixture. The original D0B forward is untouched.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE_VARIANT = 'switching_latent_balanced_readout'
HORIZONS = (1, 5, 10, 20)
ROLES = ('G', 'S', 'L')
CRITERIA = ('global', 'short', 'long')
EPSILON = .1
MAIN_CHECKPOINTS = {'G': 'global', 'S': 'short', 'L': 'long'}


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        v = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(v.dtype).encode())
        digest.update(str(tuple(v.shape)).encode())
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()


class FrozenD0BFeatures(nn.Module):
    """Extract actual projections and actual head input with one base forward.

    Hooks collect this call's tensors, never a diagnostic cache. no_grad (rather
    than inference tensors) lets a trainable residual Linear save its input.
    """
    def __init__(self, base):
        super().__init__()
        if base.variant != BASE_VARIANT or base.n_horizons != len(HORIZONS):
            raise ValueError('Base must be independent four-horizon D0B, not a jointly trained B model')
        self.base = base.requires_grad_(False).eval()
        self.s_dim = base.gcn_proj.out_features
        self.t_dim = base.lstm_proj.out_features
        self.fused_dim = base.head[0].in_features
        self.input_dim = self.s_dim + self.t_dim + self.fused_dim
        self.n_commodities = base.n_commodities
        if base.head[-1].out_features != len(HORIZONS) * self.n_commodities:
            raise ValueError('D0B output rows must be horizon-major')
        self.train(False)

    def train(self, mode=True):
        # Called recursively by any outer .train(): the entire base stays eval.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        self.base.eval()
        captured = {}
        def output(name):
            def hook(module, args, value):
                captured.setdefault(name, []).append(value)
            return hook
        def input_hook(module, args):
            captured.setdefault('z_base', []).append(args[0])
        handles = [self.base.gcn_proj.register_forward_hook(output('s')),
                   self.base.lstm_proj.register_forward_hook(output('t')),
                   self.base.head.register_forward_pre_hook(input_hook)]
        try:
            y0 = self.base(x)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {'s', 't', 'z_base'} or any(len(v) != 1 for v in captured.values()):
            raise ValueError('Expected one original D0B projection/fusion/head path')
        s, t, z = (captured[key][0] for key in ('s', 't', 'z_base'))
        if (s.shape != (len(x), self.s_dim) or t.shape != (len(x), self.t_dim)
                or z.shape != (len(x), self.fused_dim)
                or y0.shape != (len(x), 4, self.n_commodities)):
            raise ValueError('Frozen D0B actual representation/output dimensions changed')
        u = torch.cat((s, t, z), dim=-1).detach()
        if not torch.isfinite(u).all() or not torch.isfinite(y0).all():
            raise FloatingPointError('Nonfinite frozen features or base prediction')
        return dict(u=u, y_base=y0.detach(), s=s.detach(), t=t.detach(), z_base=z.detach())


class ResidualPredictionExpert(nn.Module):
    def __init__(self, input_dim, n_commodities, horizons=HORIZONS):
        super().__init__()
        if tuple(horizons) != HORIZONS:
            raise ValueError('Fixed horizon order is [1,5,10,20]')
        self.input_dim, self.n_commodities = input_dim, n_commodities
        self.network = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 4 * n_commodities))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, u):
        if u.ndim != 2 or u.shape[-1] != self.input_dim:
            raise ValueError('Expected (batch,input_dim) frozen features')
        return self.network(u).reshape(len(u), 4, self.n_commodities)

    def predict(self, u, y_base):
        residual = self(u)
        if residual.shape != y_base.shape:
            raise ValueError('Expert/base horizon or commodity order mismatch')
        return y_base + residual


class FrozenD0BExpert(nn.Module):
    """Optional online adapter; formal runs may cache its frozen inputs."""
    def __init__(self, features, expert):
        super().__init__()
        self.features, self.expert = features, expert
        if features.input_dim != expert.input_dim or features.n_commodities != expert.n_commodities:
            raise ValueError('Adapter expert dimensions differ from base')

    def forward(self, x):
        frozen = self.features(x)
        return self.expert.predict(frozen['u'], frozen['y_base'])


def matched_experts(input_dim, n_commodities, seed=42):
    """Construct one CPU state and explicitly copy it; preserve caller RNG."""
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        template = ResidualPredictionExpert(input_dim, n_commodities)
        state = {k: v.clone() for k, v in template.state_dict().items()}
        experts = {role: copy.deepcopy(template) for role in ROLES}
        for expert in experts.values():
            expert.load_state_dict(state, strict=True)
    hashes = {role: tensor_state_hash(e.state_dict()) for role, e in experts.items()}
    ids = [{id(p) for p in experts[role].parameters()} for role in ROLES]
    differences = {role: max(float((v - experts[role].state_dict()[k]).abs().max()) for k, v in state.items()) for role in ROLES}
    independent = all(not ids[a] & ids[b] for a, b in ((0,1),(0,2),(1,2)))
    audit = dict(seed=seed, initial_state_sha256=tensor_state_hash(state), role_hashes=hashes,
        max_abs_diff=differences, independent_parameter_objects=independent,
        copied_keys=list(state), trainable_parameters=sum(p.numel() for p in template.parameters()),
        input_dim=input_dim, n_commodities=n_commodities,
        PASS=len(set(hashes.values())) == 1 and independent and max(differences.values()) == 0)
    if not audit['PASS']:
        raise ValueError('Independent expert initialization mismatch')
    return experts, audit


def horizon_weights(criterion, epsilon=EPSILON):
    if epsilon <= 0:
        raise ValueError('Cross-group outputs must receive positive supervision')
    if criterion == 'global':
        return (1., 1., 1., 1.)
    high, low = 2 / (1 + epsilon), 2 * epsilon / (1 + epsilon)
    if criterion == 'short':
        return (high, high, low, low)
    if criterion == 'long':
        return (low, low, high, high)
    raise ValueError(criterion)


def horizon_losses(prediction, target, delta):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 4:
        raise ValueError('Expected matching (batch,4,commodities)')
    return torch.stack([F.huber_loss(prediction[:, i], target[:, i], delta=delta, reduction='mean') for i in range(4)])


def weighted_objective(losses, criterion, epsilon=EPSILON):
    # Sequential sum matches the existing prediction-loss reduction order.
    return sum(w * loss for w, loss in zip(horizon_weights(criterion, epsilon), losses))


def combine_strategies(base, predictions):
    """Label-free row selection; no fitted mixture or oracle."""
    if set(predictions) != {'G-global', 'G-short', 'G-long', 'S-short', 'L-long'}:
        raise ValueError('Need five prespecified expert/checkpoint outputs')
    if any(p.shape != base.shape for p in predictions.values()):
        raise ValueError('Checkpoint predictions have different ordering or dimensions')
    concatenate = torch.cat if torch.is_tensor(base) else np.concatenate
    joined = lambda short, long: concatenate((short[:, :2], long[:, 2:]), 1)
    return dict(D0B=base, G_global=predictions['G-global'], S_short=predictions['S-short'],
        L_long=predictions['L-long'], generic_group_control=joined(predictions['G-short'],predictions['G-long']),
        supervised_task_assignment=joined(predictions['S-short'],predictions['L-long']))
