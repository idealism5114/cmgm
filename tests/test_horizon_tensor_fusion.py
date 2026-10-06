import torch
from torch import nn

from cmgm.config import FEATURE_DIM, SEQ_LEN
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.horizon_tensor_fusion import (
    HORIZONS, HorizonTensorFusion, predict_by_horizon,
)
from cmgm.training.train import _prediction_loss, _switching_branch, make_loss, train_epoch


def _models(seed=812):
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2)
    torch.manual_seed(seed)
    original = HeteroMixHopCMGM(**kwargs, variant="switching_latent_balanced_readout")
    torch.manual_seed(seed)
    tensor = HeteroMixHopCMGM(
        **kwargs, variant="switching_latent_balanced_horizon_tensor_fusion"
    )
    return original, tensor


def test_tensor_interaction_matches_explicit_reference_and_parameter_count():
    torch.manual_seed(4)
    module = HorizonTensorFusion().eval()
    s, t = torch.randn(3, 64), torch.randn(3, 64)
    fused, parts = module(s, t, return_components=True)
    kernels = module.effective_kernels()
    # Independent flatten/matmul reference checks both tensor axes and order.
    reference = torch.stack([
        parts["outer"].flatten(1) @ kernels[h].flatten(1).T
        for h in range(4)
    ], dim=1)
    torch.testing.assert_close(parts["interaction"], reference, rtol=1e-5, atol=1e-6)
    assert fused.shape == (3, 4, 64)
    assert parts["outer"].shape == (3, 8, 8)
    assert sum(p.numel() for p in module.parameters()) == 21_832


def test_horizon_rows_are_selected_from_original_shared_head():
    torch.manual_seed(5)
    head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 96)).eval()
    fused = torch.randn(2, 4, 64)
    actual = predict_by_horizon(head, fused, 24)
    body = head[:3](fused)
    weight = head[3].weight.view(4, 24, 64)
    bias = head[3].bias.view(4, 24)
    reference = torch.stack([
        body[:, h] @ weight[h].T + bias[h] for h in range(4)
    ], dim=1)
    torch.testing.assert_close(actual, reference)


def test_cycle_conditioning_and_zero_compressed_input():
    torch.manual_seed(6)
    module = HorizonTensorFusion().eval()
    s, t = torch.randn(2, 64), torch.randn(2, 64)
    with torch.no_grad():
        module.E.zero_()
    torch.testing.assert_close(module.effective_kernels(), module.G0[None].expand(4, -1, -1, -1))
    base, parts = module(s, t, return_components=True)
    assert torch.count_nonzero(parts["interaction"]) > 0
    rng_before = torch.get_rng_state().clone()
    diagnostic = module.diagnostics(s, t)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert len(diagnostic["interaction_norm_by_horizon"]) == 4
    without_interaction, no_interaction_parts = module(
        s, t, disable_interaction=True, return_components=True
    )
    assert torch.count_nonzero(no_interaction_parts["interaction"]) == 0
    expected_linear = torch.nn.functional.gelu(
        no_interaction_parts["linear"] + module.b[None, None, :]
    )
    torch.testing.assert_close(without_interaction, expected_linear)
    _, zero_embedding_parts = module(s, t, zero_embedding=True, return_components=True)
    torch.testing.assert_close(
        zero_embedding_parts["effective_kernels"],
        module.G0[None].expand(4, -1, -1, -1),
    )

    original_e = module.E.detach().clone()
    with torch.no_grad():
        module.E[2, 0] = .7
    changed = module(s, t)
    assert torch.equal(base[:, [0, 1, 3]], changed[:, [0, 1, 3]])
    assert not torch.equal(base[:, 2], changed[:, 2])
    head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 96)).eval()
    base_pred = predict_by_horizon(head, base, 24)
    changed_pred = predict_by_horizon(head, changed, 24)
    assert torch.equal(base_pred[:, [0, 1, 3]], changed_pred[:, [0, 1, 3]])
    assert not torch.equal(base_pred[:, 2], changed_pred[:, 2])
    with torch.no_grad():
        module.E.copy_(original_e)

    saved = module.u_s.weight.detach().clone()
    with torch.no_grad():
        module.u_s.weight.zero_()
    _, zero_parts = module(s, t, return_components=True)
    assert torch.count_nonzero(zero_parts["u"]) == 0
    assert torch.count_nonzero(zero_parts["interaction"]) == 0
    with torch.no_grad():
        module.u_s.weight.copy_(saved)


def test_shared_initialization_and_full_model_gradient_path_checkpoint(tmp_path):
    original, model = _models()
    old_params = dict(original.named_parameters())
    new_params = dict(model.named_parameters())
    shared = [name for name in old_params if not name.startswith("gate_fc.")]
    assert all(name in new_params for name in shared)
    assert max((old_params[n] - new_params[n]).abs().max().item() for n in shared) == 0
    assert "gate_fc" not in model._modules
    assert sum(p.numel() for p in model.horizon_tensor_fusion.parameters()) == 21_832
    assert sum(p.numel() for p in new_params.values()) - sum(p.numel() for p in old_params.values()) == 13_576

    x = torch.randn(2, SEQ_LEN, 30, FEATURE_DIM)
    y = torch.randn(2, 4, 24)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        permuted = model(x.flip(0)).flip(0)
    assert pred.shape == (2, 4, 24) and torch.isfinite(pred).all()
    torch.testing.assert_close(pred, permuted)

    model.train()
    model.zero_grad(set_to_none=True)
    pred = model(x)
    _prediction_loss(model, pred, y, make_loss()).backward()
    for name, parameter in model.horizon_tensor_fusion.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for prefix in ("type_proj", "graph_learner", "switching_latent_transformer",
                   "gcn_proj", "lstm_proj", "head"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), prefix

    branch = _switching_branch(model)
    assert branch is model.switching_latent_transformer
    beta1 = branch.set_epoch(1)
    beta20 = branch.set_epoch(20)
    assert beta1 == 0 and beta20 == 5e-4

    class ZeroCriterion(nn.Module):
        def forward(self, prediction, target):
            return prediction.sum() * 0

    class NoUpdateOptimizer:
        def zero_grad(self):
            model.zero_grad(set_to_none=True)
        def step(self):
            pass

    original_switch_loss = branch.switch_loss
    branch.switch_loss = lambda: next(model.parameters()).sum() * 0 + .125
    loader = [(x.detach(), y)]
    try:
        observed = train_epoch(model, loader, torch.empty(2, 0, dtype=torch.long),
                               torch.empty(0), NoUpdateOptimizer(), ZeroCriterion(),
                               torch.device("cpu"))
    finally:
        branch.switch_loss = original_switch_loss
    assert abs(observed - .125) < 1e-6

    model.eval()
    with torch.no_grad():
        expected = model(x)
    checkpoint = tmp_path / "tensor.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = HeteroMixHopCMGM(
        30, 24, n_stock=4, n_bond=2,
        variant="switching_latent_balanced_horizon_tensor_fusion",
    )
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(x)
    torch.testing.assert_close(actual, expected)


def test_readout_variant_keeps_original_gate_path():
    original, _ = _models(seed=99)
    x = torch.randn(2, SEQ_LEN, 30, FEATURE_DIM)
    original.eval()
    with torch.no_grad():
        spatial = original._temp_weighted_spatial(x)
        temporal = original.switching_latent_transformer(x)
        combined = torch.cat([spatial, temporal], dim=-1)
        gate = torch.sigmoid(original.gate_fc(combined))
        fused = gate * original.lstm_proj(temporal) + (1 - gate) * original.gcn_proj(spatial)
        expected = original.head(fused).view(2, 4, 24)
        actual = original(x)
    torch.testing.assert_close(actual, expected)
