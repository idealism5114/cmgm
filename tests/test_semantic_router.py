import numpy as np
import torch

from cmgm.data.data_loader import (
    MarketSequenceDataset,
    build_market_descriptor_timeline,
)
from cmgm.models.regime_dynamic import (
    RegimeDynamicRPETransformer,
    SemanticRegimeRouter,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM


def _synthetic_prices():
    stock = np.linspace(100, 108, 9)[:, None]
    bond = np.linspace(50, 51, 9)[:, None]
    commodity_a = np.array([10, 11, 10, 12, 11, 13, 12, 14, 13])[:, None]
    commodity_b = np.array([20, 19, 21, 20, 22, 21, 23, 22, 24])[:, None]
    return np.concatenate([stock, bond, commodity_a, commodity_b], axis=1)


def test_descriptor_is_causal_and_uses_train_statistics_only():
    prices = _synthetic_prices()
    indices = {"stock": (0, 1), "bond": (1, 2), "commodity": (2, 4)}
    normalized, raw, stats = build_market_descriptor_timeline(
        prices, indices, train_end=6, lookback=5
    )

    changed_future = prices.copy()
    changed_future[6:, 2:] *= 100
    normalized_changed, raw_changed, stats_changed = build_market_descriptor_timeline(
        changed_future, indices, train_end=6, lookback=5
    )

    np.testing.assert_allclose(raw[:6], raw_changed[:6])
    np.testing.assert_allclose(normalized[:6], normalized_changed[:6])
    np.testing.assert_allclose(stats["mean"], stats_changed["mean"])
    np.testing.assert_allclose(stats["std"], stats_changed["std"])
    assert raw[1, 1] == 0.0  # only one observed return is available


def test_dataset_slices_the_precomputed_descriptor_timeline():
    prices = _synthetic_prices().astype(np.float32)
    indices = {"stock": (0, 1), "bond": (1, 2), "commodity": (2, 4)}
    descriptor, _, _ = build_market_descriptor_timeline(
        prices, indices, train_end=6, lookback=5
    )
    dataset = MarketSequenceDataset(
        prices,
        indices,
        seq_len=3,
        feature_matrix=prices[..., None],
        raw_prices=prices,
        target_type="return",
        horizons=[1],
        market_descriptors=descriptor,
    )
    _, _, descriptor_window = dataset[2]
    np.testing.assert_allclose(descriptor_window.numpy(), descriptor[2:5])


def test_semantic_router_is_causal():
    torch.manual_seed(1)
    router = SemanticRegimeRouter(d_model=8, K=3, D_regime=4)
    encoded = torch.randn(2, 20, 8)
    descriptor = torch.randn(2, 20, 3)
    centers = torch.randn(3, 3)
    p_original = router(encoded, descriptor, centers)

    encoded_changed = encoded.clone()
    descriptor_changed = descriptor.clone()
    encoded_changed[:, 10:] = torch.randn_like(encoded_changed[:, 10:])
    descriptor_changed[:, 10:] = torch.randn_like(descriptor_changed[:, 10:])
    p_changed = router(encoded_changed, descriptor_changed, centers)
    torch.testing.assert_close(p_original[:, :10], p_changed[:, :10])


def test_g_auxiliary_loss_reaches_all_router_parameters():
    torch.manual_seed(2)
    model = RegimeDynamicRPETransformer(
        in_dim=6,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        t_len=20,
        K=3,
        D_regime=4,
        use_semantic_router=True,
    )
    output = model(torch.randn(3, 20, 6), torch.randn(3, 20, 3))
    loss = output.square().mean() + model.dynamic_loss()
    loss.backward()

    router = model.regime_gen
    parameters = [
        router.regime_prototypes,
        router.context_encoder[0].weight,
        router.transition_matrix,
        model.state_centers,
    ]
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_h_differs_from_g_only_by_cluster_coefficient():
    common = dict(
        in_dim=6,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        t_len=20,
        K=3,
        D_regime=4,
        use_semantic_router=True,
    )
    torch.manual_seed(3)
    model_g = RegimeDynamicRPETransformer(**common, lambda_cluster=0.005)
    torch.manual_seed(3)
    model_h = RegimeDynamicRPETransformer(**common, lambda_cluster=0.0005)

    assert model_g.state_dict().keys() == model_h.state_dict().keys()
    for name, tensor_g in model_g.state_dict().items():
        torch.testing.assert_close(tensor_g, model_h.state_dict()[name])
    assert model_g.lambda_cluster == 0.005
    assert model_h.lambda_cluster == 0.0005
    assert model_g.lambda_info_max == model_h.lambda_info_max == 0.001

    x = torch.randn(3, 20, 6)
    descriptor = torch.randn(3, 20, 3)
    model_g.eval()
    model_h.eval()
    prediction_g = model_g(x, descriptor)
    prediction_h = model_h(x, descriptor)
    torch.testing.assert_close(prediction_g, prediction_h)
    loss_g = model_g.dynamic_loss()
    loss_h = model_h.dynamic_loss()
    torch.testing.assert_close(
        loss_g - loss_h,
        torch.tensor(0.0045) * model_g.last_loss_components["cluster"],
    )
    assert model_g.last_loss_weights == {
        "dynamic": 0.001,
        "cluster": 0.005,
        "info_max": 0.001,
    }
    assert model_h.last_loss_weights == {
        "dynamic": 0.001,
        "cluster": 0.0005,
        "info_max": 0.001,
    }


def test_h_variant_wires_the_same_architecture_as_g():
    common = dict(
        num_nodes=4,
        n_commodities=2,
        n_stock=1,
        n_bond=1,
        feat_dim=21,
    )
    torch.manual_seed(4)
    model_g = HeteroMixHopCMGM(variant="semantic_router", **common)
    torch.manual_seed(4)
    model_h = HeteroMixHopCMGM(variant="loss_rebalance", **common)

    assert model_g.state_dict().keys() == model_h.state_dict().keys()
    for name, tensor_g in model_g.state_dict().items():
        torch.testing.assert_close(tensor_g, model_h.state_dict()[name])
    assert model_g.regime_dynamic.lambda_cluster == 0.005
    assert model_h.regime_dynamic.lambda_cluster == 0.0005


def test_i_preserves_h_raw_routing_and_loss_but_smooths_downstream_use():
    common = dict(
        in_dim=6,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        t_len=20,
        K=3,
        D_regime=4,
        use_semantic_router=True,
        lambda_cluster=0.0005,
    )
    torch.manual_seed(5)
    model_h = RegimeDynamicRPETransformer(**common, routing_strength=1.0)
    torch.manual_seed(5)
    model_i = RegimeDynamicRPETransformer(**common, routing_strength=0.25)
    model_h.eval()
    model_i.eval()

    assert model_h.state_dict().keys() == model_i.state_dict().keys()
    for name, tensor_h in model_h.state_dict().items():
        torch.testing.assert_close(tensor_h, model_i.state_dict()[name])

    x = torch.randn(3, 20, 6)
    descriptor = torch.randn(3, 20, 3)
    model_h(x, descriptor)
    model_i(x, descriptor)
    torch.testing.assert_close(model_h.last_regime_p, model_i.last_regime_p)
    expected_use = 0.75 * torch.full_like(model_i.last_regime_p, 1.0 / 3.0)
    expected_use = expected_use + 0.25 * model_i.last_regime_p
    torch.testing.assert_close(model_i.last_regime_p_use, expected_use)
    torch.testing.assert_close(
        model_i.last_regime_p_use.sum(dim=-1),
        torch.ones_like(model_i.last_regime_p_use[..., 0]),
    )

    loss_h = model_h.dynamic_loss()
    loss_i = model_i.dynamic_loss()
    torch.testing.assert_close(loss_h, loss_i)
    for component in ("cluster", "info_max", "dynamic"):
        torch.testing.assert_close(
            model_h.last_loss_components[component],
            model_i.last_loss_components[component],
        )


def test_i_prediction_path_remains_differentiable_to_router():
    torch.manual_seed(6)
    model = RegimeDynamicRPETransformer(
        in_dim=6,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        t_len=20,
        K=3,
        D_regime=4,
        use_semantic_router=True,
        lambda_cluster=0.0005,
        routing_strength=0.25,
    )
    output = model(torch.randn(3, 20, 6), torch.randn(3, 20, 3))
    output.square().mean().backward()
    router = model.regime_gen
    parameters = (
        router.regime_prototypes,
        router.context_encoder[0].weight,
        router.transition_matrix,
        model.state_centers,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(parameter.grad.norm().item() > 0.0 for parameter in parameters)


def test_i_variant_wires_only_routing_strength_differently_from_h():
    common = dict(
        num_nodes=4,
        n_commodities=2,
        n_stock=1,
        n_bond=1,
        feat_dim=21,
    )
    torch.manual_seed(7)
    model_h = HeteroMixHopCMGM(variant="loss_rebalance", **common)
    torch.manual_seed(7)
    model_i = HeteroMixHopCMGM(variant="routing_strength", **common)

    assert model_h.state_dict().keys() == model_i.state_dict().keys()
    for name, tensor_h in model_h.state_dict().items():
        torch.testing.assert_close(tensor_h, model_i.state_dict()[name])
    assert model_h.regime_dynamic.lambda_cluster == 0.0005
    assert model_i.regime_dynamic.lambda_cluster == 0.0005
    assert model_h.regime_dynamic.lambda_info_max == 0.001
    assert model_i.regime_dynamic.lambda_info_max == 0.001
    assert model_h.regime_dynamic.routing_strength == 1.0
    assert model_i.regime_dynamic.routing_strength == 0.25


def test_i_forced_raw_and_downstream_overrides_are_distinct():
    torch.manual_seed(8)
    model = RegimeDynamicRPETransformer(
        in_dim=6,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_dim=16,
        t_len=20,
        K=3,
        D_regime=4,
        use_semantic_router=True,
        lambda_cluster=0.0005,
        routing_strength=0.25,
    )
    model.eval()
    x = torch.randn(2, 20, 6)
    descriptor = torch.randn(2, 20, 3)
    one_hot = torch.tensor([1.0, 0.0, 0.0])

    model.forced_regime = one_hot
    model(x, descriptor)
    torch.testing.assert_close(
        model.last_regime_p_use[0, 0], torch.tensor([0.5, 0.25, 0.25])
    )

    model.forced_regime = None
    model.forced_regime_use = one_hot
    model(x, descriptor)
    torch.testing.assert_close(model.last_regime_p_use[0, 0], one_hot)
