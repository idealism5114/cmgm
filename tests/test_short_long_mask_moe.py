"""Synthetic-only checks for the short/long task-mask MoE."""
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_mask_moe import (
    EXPERT_NAMES, FIXED_VARIANT, GLOBAL_VARIANT, LEARNED_VARIANT,
    MASK_MOE_VARIANTS,
)
from cmgm.models.task_conditioned_moe import BASE_VARIANT
from cmgm.scripts.d0b_short_long_mask_moe import matched_initialization, parameters
from cmgm.scripts.d0b_task_conditioned_moe import initialization_audit
from cmgm.training.train import _prediction_loss, _switching_branch, train


KWARGS = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
              commodity_indices=(6, 30))


def make(variant):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(variant=variant, **KWARGS)


def test_structure_initialization_and_d0b_regression():
    base = make(BASE_VARIANT).eval()
    models = {name: make(name).eval() for name in MASK_MOE_VARIANTS}
    x = torch.randn(2, 20, 30, 21)
    assert all(initialization_audit(KWARGS, name, 42)["PASS"] for name in MASK_MOE_VARIANTS)
    assert matched_initialization(KWARGS, 42)["PASS"]
    with torch.no_grad():
        original = base(x)
        assert all(model(x).shape == (2, 4, 24) for model in models.values())
        torch.testing.assert_close(models[FIXED_VARIANT](x),
                                   models[LEARNED_VARIANT](x), rtol=0, atol=0)
    torch.testing.assert_close(base(x), original, rtol=0, atol=0)
    for model in models.values():
        assert not hasattr(model, "gate_fc")
        assert not any(name.startswith("gate_fc.") for name in model.state_dict())
        assert _switching_branch(model) is model.switching_latent_transformer
    global_model, fixed, learned = (models[name] for name in MASK_MOE_VARIANTS)
    assert not any("logits" in name or "theta" in name
                   for name, _ in fixed.short_long_mask_moe.named_parameters())
    assert parameters(global_model)["experts"] == parameters(fixed)["experts"] == parameters(learned)["experts"]
    assert parameters(global_model)["routing_logits"] == 4
    assert parameters(fixed)["routing_logits"] == 0
    assert parameters(learned)["routing_logits"] == 4
    assert set(parameters(learned)["experts"]) == set(EXPERT_NAMES)
    assert parameters(learned)["trainable"] == parameters(learned)["total"]


def test_masked_forward_and_horizon_rows():
    model = make(LEARNED_VARIANT).eval()
    module = model.short_long_mask_moe
    s, t = torch.randn(2, 64), torch.randn(2, 64)
    memory, latent = torch.randn(2, 20, 128), torch.randn(2, 20, 64)
    with torch.no_grad():
        states = module(s, t, memory, latent)
        assert states.shape == (2, 4, 64)
        torch.testing.assert_close(states[:, 0], states[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(states[:, 2], states[:, 3], rtol=0, atol=0)
        prediction = module.predict(model.head, states, 24)
        body = model.head[:3](states)
        weight = model.head[3].weight.view(4, 24, 64)
        bias = model.head[3].bias.view(4, 24)
        manual = torch.stack([body[:, h] @ weight[h].T + bias[h] for h in range(4)], dim=1)
        torch.testing.assert_close(prediction, manual, atol=1e-7, rtol=0)
        assert torch.isfinite(prediction).all()
    route = module.routing_weights()
    torch.testing.assert_close(route["short"], torch.full((2,), .5))
    torch.testing.assert_close(route["long"], torch.full((2,), .5))
    module.theta_short.data = torch.tensor((2.0, -1.0))
    with torch.no_grad():
        changed = module(s, t, memory, latent)
    torch.testing.assert_close(changed[:, 2:], states[:, 2:], atol=0, rtol=0)
    assert not torch.allclose(changed[:, :2], states[:, :2])


def test_group_gradients_and_live_temporal_single_pass():
    model = make(LEARNED_VARIANT).train()
    module = model.short_long_mask_moe
    captured = {}
    original = module.forward

    def spy(s, t, memory, latent):
        memory.retain_grad()
        latent.retain_grad()
        captured.update(memory=memory, latent=latent)
        return original(s, t, memory, latent)

    module.forward = spy
    calls = []
    hook = model.switching_latent_transformer.register_forward_hook(lambda *args: calls.append(1))
    with patch.object(model, "_temp_weighted_spatial", wraps=model._temp_weighted_spatial) as spatial:
        prediction = model(torch.randn(2, 20, 30, 21))
        assert spatial.call_count == 1
    hook.remove()
    assert len(calls) == 1
    assert captured["memory"].requires_grad and captured["latent"].requires_grad
    assert prediction.shape == (2, 4, 24) and torch.isfinite(prediction).all()
    params = [next(module.shared_expert.parameters()),
              next(module.short_memory_expert.parameters()),
              next(module.short_shock_expert.parameters()),
              next(module.long_memory_expert.parameters()),
              next(module.long_structure_expert.parameters()),
              module.theta_short, module.theta_long]
    short = torch.autograd.grad(prediction[:, :2].square().mean(), params,
                                retain_graph=True, allow_unused=True)
    long = torch.autograd.grad(prediction[:, 2:].square().mean(), params,
                               retain_graph=True, allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in short[:3] + (short[5],))
    assert all(g is None or torch.count_nonzero(g) == 0 for g in (short[3], short[4], short[6]))
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in (long[0], long[3], long[4], long[6]))
    assert all(g is None or torch.count_nonzero(g) == 0 for g in (long[1], long[2], long[5]))
    model.switching_latent_transformer.set_epoch(3)
    objective = _prediction_loss(model, prediction, torch.randn_like(prediction)*.01,
                                 nn.HuberLoss(delta=.02))
    kl = model.switching_latent_transformer.switch_loss()
    total = objective + kl
    assert torch.isfinite(total) and torch.isfinite(kl)
    total.backward()
    assert captured["memory"].grad.abs().sum() > 0
    assert captured["latent"].grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())
    assert all(parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_global_all_experts_and_frozen_val_interventions():
    global_model = make(GLOBAL_VARIANT).eval()
    x = torch.randn(2, 20, 30, 21)
    with torch.no_grad():
        original = global_model(x)
    global_model.short_long_mask_moe.global_logits.data[0] = 2.
    with torch.no_grad():
        changed = global_model(x)
    assert all(not torch.allclose(original[:, i], changed[:, i]) for i in range(4))
    model = make(LEARNED_VARIANT).eval()
    fusion = model.short_long_mask_moe
    fusion.theta_short.data = torch.tensor((2., -1.))
    fusion.theta_long.data = torch.tensor((-1., 2.))
    with torch.no_grad():
        original = model(x)
        with fusion.intervention("swap_short_long"):
            swapped = model(x)
        torch.testing.assert_close(model(x), original, rtol=0, atol=0)
        assert not torch.allclose(swapped, original)
        with fusion.intervention("remove_short_memory"):
            no_short = model(x)
        torch.testing.assert_close(no_short[:, 2:], original[:, 2:], rtol=0, atol=0)
        with fusion.intervention("remove_long_structure"):
            no_long = model(x)
        torch.testing.assert_close(no_long[:, :2], original[:, :2], rtol=0, atol=0)


def test_checkpoint_and_existing_trainer(tmp_path):
    model = make(LEARNED_VARIANT)
    x = torch.randn(2, 20, 30, 21)
    y = torch.randn(2, 4, 24) * .01
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    checkpoint = tmp_path / "best.pt"
    history = train(model, loader, loader, torch.empty(2, 0, dtype=torch.long),
                    torch.empty(0), torch.device("cpu"), num_epochs=1, lr=1e-4,
                    weight_decay=1e-5, patience=10, checkpoint_path=str(checkpoint),
                    checkpoint_metadata={"variant": LEARNED_VARIANT})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["best_epoch"] == history["best_epoch"] == 1
    restored = make(LEARNED_VARIANT).eval()
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
