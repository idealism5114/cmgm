import math

import torch
from torch import nn

from cmgm.models.node_history_readout import (
    NODE_ATTENTION_VARIANT,
    NODE_MEAN_VARIANT,
    NodeHistoryReadout,
    apply_original_head_per_node,
    attention_diagnostics,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_node_history_readout import (
    BASE_VARIANT,
    CONCAT_VARIANT,
    MEAN_FUSION_VARIANT,
    _audit_seed,
)
from cmgm import config


def test_attention_formula_and_gradient_connectivity():
    torch.manual_seed(3)
    module = NodeHistoryReadout(attention=True).eval()
    s_nodes = torch.randn(2, 3, 64, requires_grad=True)
    history = torch.randn(2, 5, 128, requires_grad=True)
    s_global, t_global = torch.randn(2, 64), torch.randn(2, 64)
    actual, aux = module(s_nodes, history, s_global, t_global, return_attention=True)
    q = module.W_q(s_nodes)
    k = module.W_k(history)
    v = module.W_v(history)
    scores = q @ k.transpose(-1, -2) / math.sqrt(32)
    expected_attention = scores.softmax(-1)
    expected_u = module.W_o(expected_attention @ v)
    features = torch.cat([
        s_nodes, expected_u,
        s_global[:, None, :].expand(-1, 3, -1),
        t_global[:, None, :].expand(-1, 3, -1),
    ], dim=-1)
    expected = torch.nn.functional.gelu(module.fusion_projection(features))
    torch.testing.assert_close(aux["attention"], expected_attention)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(aux["attention"].sum(-1), torch.ones(2, 3))
    actual.square().mean().backward()
    for layer in (module.W_q, module.W_k, module.W_v, module.W_o, module.fusion_projection):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert s_nodes.grad is not None and s_nodes.grad.abs().sum() > 0
    assert history.grad is not None and history.grad.abs().sum() > 0


def test_uniform_attention_equals_mean_and_t1_diagnostic():
    torch.manual_seed(4)
    mean = NodeHistoryReadout(attention=False).eval()
    attn = NodeHistoryReadout(attention=True).eval()
    attn.W_v.load_state_dict(mean.W_v.state_dict())
    attn.W_o.load_state_dict(mean.W_o.state_dict())
    attn.fusion_projection.load_state_dict(mean.fusion_projection.state_dict())
    s_nodes, history = torch.randn(2, 4, 64), torch.randn(2, 7, 128)
    sg, tg = torch.randn(2, 64), torch.randn(2, 64)
    expected = mean(s_nodes, history, sg, tg)
    actual, aux = attn(s_nodes, history, sg, tg, force_uniform_attention=True,
                       return_attention=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(aux["attention"], torch.full((2, 4, 7), 1 / 7))
    assert attention_diagnostics(torch.ones(2, 4, 1))["normalized_entropy_mean"] == 0.0


def test_original_head_rows_are_selected_by_horizon_and_commodity():
    head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0), nn.Linear(64, 3 * 4))
    head.eval()
    with torch.no_grad():
        head[0].weight.copy_(torch.eye(64)); head[0].bias.zero_()
        head[3].weight.zero_()
        head[3].bias.copy_(torch.arange(12, dtype=torch.float))
    shared = torch.randn(2, 64)
    node_representations = shared[:, None, :].expand(-1, 4, -1)
    node = apply_original_head_per_node(head, node_representations, 3, 4)
    direct = head(shared).view(2, 3, 4)
    torch.testing.assert_close(node, direct, rtol=0, atol=0)
    assert node[0, 0, 2].item() == 2 and node[0, 2, 1].item() == 9


def test_full_model_variant_dispatch_initialization_order_and_checkpoint_roundtrip(tmp_path):
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
                  commodity_indices=(6, 30))
    torch.manual_seed(9)
    base = HeteroMixHopCMGM(**kwargs, variant=BASE_VARIANT).eval()
    variants = (MEAN_FUSION_VARIANT, CONCAT_VARIANT, NODE_MEAN_VARIANT, NODE_ATTENTION_VARIANT)
    models = {}
    for variant in variants:
        assert _audit_seed(variant, kwargs, 9)["PASS"]
        torch.manual_seed(9)
        models[variant] = HeteroMixHopCMGM(**kwargs, variant=variant).eval()
    mean_readout = models[NODE_MEAN_VARIANT].node_history_readout
    attn_readout = models[NODE_ATTENTION_VARIANT].node_history_readout
    for name in ("W_v", "W_o", "fusion_projection"):
        left, right = getattr(mean_readout, name), getattr(attn_readout, name)
        assert torch.equal(left.weight, right.weight)
        if left.bias is not None:
            assert torch.equal(left.bias, right.bias)
    assert not hasattr(mean_readout, "W_q") and not hasattr(mean_readout, "W_k")
    assert sum(p.numel() for p in attn_readout.parameters()) - sum(
        p.numel() for p in mean_readout.parameters()) == 6144

    x = torch.randn(2, config.SEQ_LEN, 30, config.FEATURE_DIM)
    with torch.no_grad():
        base_prediction = base(x)
        for model in models.values():
            prediction = model(x)
            assert prediction.shape == base_prediction.shape == (2, 4, 24)
            assert torch.isfinite(prediction).all()
    # No registered gate remains in any added fusion variant.
    for variant, model in models.items():
        assert "gate_fc" not in model._modules
    actual_kwargs = dict(num_nodes=284, n_commodities=24, n_stock=248, n_bond=12,
                         commodity_indices=(260, 284))
    expected_parameters = {
        BASE_VARIANT: 520549,
        MEAN_FUSION_VARIANT: 512293,
        CONCAT_VARIANT: 520549,
        NODE_MEAN_VARIANT: 534885,
        NODE_ATTENTION_VARIANT: 541029,
    }
    for variant, expected_count in expected_parameters.items():
        torch.manual_seed(5)
        actual_config_model = HeteroMixHopCMGM(**actual_kwargs, variant=variant)
        assert sum(p.numel() for p in actual_config_model.parameters()) == expected_count
    path = tmp_path / "node-attention.pt"
    torch.save(models[NODE_ATTENTION_VARIANT].state_dict(), path)
    restored = HeteroMixHopCMGM(**kwargs, variant=NODE_ATTENTION_VARIANT).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), models[NODE_ATTENTION_VARIANT](x))
    base_restored = HeteroMixHopCMGM(**kwargs, variant=BASE_VARIANT).eval()
    base_restored.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(base_restored(x), base_prediction)

    # One temporal market encoding only; optional history is the live memory
    # from the same call and the legacy no-flag path remains a Tensor.
    attention_model = models[NODE_ATTENTION_VARIANT]
    calls = []
    handle = attention_model.switching_latent_transformer.market_encoder.register_forward_hook(
        lambda *_: calls.append(1)
    )
    with torch.no_grad():
        legacy_result = attention_model.switching_latent_transformer(x)
        memory_result, history = attention_model.switching_latent_transformer(x, return_memory=True)
    handle.remove()
    assert torch.is_tensor(legacy_result) and legacy_result.shape == (2, 64)
    assert history.shape == (2, config.SEQ_LEN, 128)
    torch.testing.assert_close(memory_result, legacy_result)
    assert len(calls) == 2  # one per explicit call; model node-forward itself calls once

    # Full node path keeps gradients into both original encoders and all active
    # readout layers; native switch KL remains available and differentiable.
    attention_model.train()
    attention_model.zero_grad(set_to_none=True)
    attention_model.switching_latent_transformer.set_epoch(1)
    model_encoder_calls = []
    model_hook = attention_model.switching_latent_transformer.market_encoder.register_forward_hook(
        lambda *_: model_encoder_calls.append(1)
    )
    pred = attention_model(x)
    model_hook.remove()
    assert len(model_encoder_calls) == 1
    loss = pred.square().mean() + attention_model.switching_latent_transformer.switch_loss()
    assert torch.isfinite(loss)
    loss.backward()
    for layer in (attention_model.gcn_proj, attention_model.lstm_proj,
                  attention_model.head[0], attention_model.head[3],
                  attention_model.node_history_readout.W_q,
                  attention_model.node_history_readout.W_k,
                  attention_model.node_history_readout.W_v,
                  attention_model.node_history_readout.W_o,
                  attention_model.node_history_readout.fusion_projection):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert attention_model.type_proj.stock_proj.weight.grad is not None
    assert attention_model.switching_latent_transformer.market_encoder.stock_encoder[0].weight.grad is not None
    unused = [name for name, parameter in attention_model.named_parameters()
              if parameter.requires_grad and parameter.grad is None]
    assert not unused, f"unexpected inactive trainable parameters: {unused}"


def test_explicit_dataset_commodity_interval_is_required():
    try:
        HeteroMixHopCMGM(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
                         variant=NODE_MEAN_VARIANT)
    except ValueError as exc:
        assert "commodity node interval explicitly" in str(exc)
    else:
        raise AssertionError("node readout must not guess target node order")
