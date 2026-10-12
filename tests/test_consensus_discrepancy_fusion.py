import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.config import FEATURE_DIM, SEQ_LEN
from cmgm.models.consensus_discrepancy_fusion import (
    ConsensusDiscrepancyFusion,
    mean_fusion,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import (
    _effective_switch_loss, _prediction_loss, _switching_branch, make_loss, train_epoch,
)


VAR_BASE = "switching_latent_balanced_readout"
VAR_MEAN = "switching_latent_balanced_mean_fusion"
VAR_RESIDUAL = "switching_latent_balanced_consensus_discrepancy_fusion"


def test_residual_formula_parameter_count_and_zero_discrepancy_even_after_training_weights():
    torch.manual_seed(71)
    module = ConsensusDiscrepancyFusion().eval()
    assert sum(p.numel() for p in module.parameters()) == 3_088
    s, t = torch.randn(5, 64), torch.randn(5, 64)
    fused, parts = module(s, t, return_components=True)
    c = 0.5 * (s + t)
    d = s - t
    context = module.A_c(c) + module.b
    r = torch.tanh(context + module.A_d(d)) - torch.tanh(context)
    expected_delta = module.B(r)
    torch.testing.assert_close(parts["context"], context)
    torch.testing.assert_close(parts["r"], r)
    torch.testing.assert_close(parts["delta"], expected_delta)
    torch.testing.assert_close(fused, c + expected_delta)

    with torch.no_grad():
        module.B.weight.normal_()
    equal = torch.randn(5, 64)
    equal_fused, equal_parts = module(equal, equal, return_components=True)
    assert torch.count_nonzero(equal_parts["d"]) == 0
    assert torch.count_nonzero(equal_parts["delta"]) == 0
    torch.testing.assert_close(equal_fused, equal)


def test_initial_mean_equivalence_and_gradient_staging():
    torch.manual_seed(72)
    module = ConsensusDiscrepancyFusion()
    s, t = torch.randn(4, 64), torch.randn(4, 64)
    torch.testing.assert_close(module(s, t), mean_fusion(s, t), rtol=0, atol=0)
    assert torch.count_nonzero(module.B.weight) == 0

    target = torch.randn(4, 64)
    loss = (module(s, t) - target).square().mean()
    loss.backward()
    assert module.B.weight.grad is not None
    assert torch.isfinite(module.B.weight.grad).all()
    assert module.B.weight.grad.abs().sum() > 0

    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    optimizer.step()
    module.zero_grad(set_to_none=True)
    second = (module(s, t) - target).square().mean()
    second.backward()
    for name in ("A_c.weight", "A_d.weight", "b"):
        grad = dict(module.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def _build_model(variant, seed=81, n_nodes=30):
    torch.manual_seed(seed)
    return HeteroMixHopCMGM(
        num_nodes=n_nodes, n_commodities=24, n_stock=n_nodes - 26, n_bond=2,
        variant=variant,
    )


def test_shared_initialization_parameter_deltas_and_variant_surface():
    # N=284 provides the requested project-scale count audit without running a forward.
    baseline = _build_model(VAR_BASE, n_nodes=284)
    mean = _build_model(VAR_MEAN, n_nodes=284)
    residual = _build_model(VAR_RESIDUAL, n_nodes=284)
    base_params = dict(baseline.named_parameters())
    for model in (mean, residual):
        params = dict(model.named_parameters())
        shared = [n for n in base_params if n != "gate_fc.weight" and n != "gate_fc.bias"]
        assert all(n in params for n in shared)
        assert all(torch.equal(base_params[n], params[n]) for n in shared)
        assert "gate_fc" not in model._modules
    assert sum(p.numel() for p in residual.consensus_discrepancy_fusion.parameters()) == 3_088
    n_base = sum(p.numel() for p in baseline.parameters())
    n_mean = sum(p.numel() for p in mean.parameters())
    n_resid = sum(p.numel() for p in residual.parameters())
    assert n_base == 520_549
    assert n_mean == 512_293
    assert n_resid == 515_381
    assert n_base - n_mean == 8_256
    assert n_resid - n_mean == 3_088
    assert not any(n.startswith("consensus_discrepancy_fusion.") for n, _ in mean.named_parameters())


def test_full_model_forward_kl_base_regression_and_strict_checkpoint(tmp_path):
    torch.manual_seed(93)
    models = {v: _build_model(v, seed=93) for v in (VAR_BASE, VAR_MEAN, VAR_RESIDUAL)}
    x = torch.randn(2, SEQ_LEN, 30, FEATURE_DIM)
    y = torch.randn(2, 4, 24)

    for model in models.values():
        model.eval()
        with torch.no_grad():
            pred = model(x)
            permuted = model(x.flip(0)).flip(0)
        assert pred.shape == (2, 4, 24)
        assert torch.isfinite(pred).all()
        torch.testing.assert_close(pred, permuted)
        branch = _switching_branch(model)
        assert branch is model.switching_latent_transformer
        assert branch.set_epoch(3) >= 0
        switch_loss = _effective_switch_loss(model, branch)
        assert switch_loss.ndim == 0 and torch.isfinite(switch_loss)
        objective = _prediction_loss(model, pred, y, make_loss()) + switch_loss
        assert torch.isfinite(objective)

    # Recompute the baseline through the unchanged original sigmoid-gate equation.
    baseline = models[VAR_BASE]
    with torch.no_grad():
        hs = baseline._temp_weighted_spatial(x)
        ht = baseline.switching_latent_transformer(x)
        gate = torch.sigmoid(baseline.gate_fc(torch.cat([hs, ht], dim=-1)))
        reference = baseline.head(
            gate * baseline.lstm_proj(ht) + (1 - gate) * baseline.gcn_proj(hs)
        ).view(2, 4, 24)
        torch.testing.assert_close(baseline(x), reference)

    # At initialization the full model must exactly match the parameter-free mean model.
    torch.testing.assert_close(models[VAR_RESIDUAL](x), models[VAR_MEAN](x), rtol=0, atol=0)

    residual = models[VAR_RESIDUAL]
    residual.train()
    residual.zero_grad(set_to_none=True)
    pred = residual(x)
    (_prediction_loss(residual, pred, y, make_loss())
     + _effective_switch_loss(residual, _switching_branch(residual))).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in residual.consensus_discrepancy_fusion.parameters())

    # Exercise the existing train_epoch path once: native switching KL is added
    # through _switching_branch for these new variants, with no MoE auxiliary loss.
    residual.switching_latent_transformer.set_epoch(1)
    optimizer = torch.optim.Adam(residual.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch_loss = train_epoch(
        residual, DataLoader(TensorDataset(x, y), batch_size=2),
        torch.empty(2, 0, dtype=torch.long), torch.empty(0), optimizer,
        make_loss(), torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(epoch_loss))

    residual.eval()
    with torch.no_grad():
        expected = residual(x)
    checkpoint = tmp_path / "consensus_discrepancy.pt"
    torch.save(residual.state_dict(), checkpoint)
    restored = _build_model(VAR_RESIDUAL, seed=93)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(x)
    torch.testing.assert_close(actual, expected)

    base_copy = _build_model(VAR_BASE, seed=93)
    base_copy.load_state_dict(models[VAR_BASE].state_dict(), strict=True)
    base_copy.eval()
    with torch.no_grad():
        torch.testing.assert_close(base_copy(x), models[VAR_BASE](x))

    mean_copy = _build_model(VAR_MEAN, seed=93)
    mean_copy.load_state_dict(models[VAR_MEAN].state_dict(), strict=True)
    mean_copy.eval()
    with torch.no_grad():
        torch.testing.assert_close(mean_copy(x), models[VAR_MEAN](x))

    with torch.no_grad():
        residual_hs = restored._temp_weighted_spatial(x)
        residual_ht = restored.switching_latent_transformer(x)
    restored.train()
    restored.head.eval()  # Mixed child modes must also survive the diagnostic.
    modes_before = [m.training for m in restored.modules()]
    diagnostic = restored.consensus_discrepancy_diagnostic(x, disable_correction=True)
    assert [m.training for m in restored.modules()] == modes_before
    assert diagnostic["disable_correction"] is True
    assert diagnostic["applied_delta_l2_mean_per_sample"] == 0
    torch.testing.assert_close(diagnostic["fused"],
                               0.5 * (restored.gcn_proj(residual_hs)
                                      + restored.lstm_proj(residual_ht)))

    # The already supported tensor-fusion variant remains constructible and strictly reloadable.
    tensor = _build_model("switching_latent_balanced_horizon_tensor_fusion", seed=94)
    tensor_copy = _build_model("switching_latent_balanced_horizon_tensor_fusion", seed=95)
    tensor_copy.load_state_dict(tensor.state_dict(), strict=True)
