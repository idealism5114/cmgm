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


def test_j_differs_from_i_only_by_fixed_context_gamma():
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
        routing_strength=0.25,
    )
    torch.manual_seed(9)
    model_i = RegimeDynamicRPETransformer(**common, context_gamma=5.0)
    torch.manual_seed(9)
    model_j = RegimeDynamicRPETransformer(**common, context_gamma=1.0)
    model_i.eval()
    model_j.eval()

    assert model_i.state_dict().keys() == model_j.state_dict().keys()
    for name, tensor_i in model_i.state_dict().items():
        torch.testing.assert_close(tensor_i, model_j.state_dict()[name])
    assert model_i.routing_strength == model_j.routing_strength == 0.25
    assert model_i.lambda_cluster == model_j.lambda_cluster == 0.0005
    assert model_i.lambda_info_max == model_j.lambda_info_max == 0.001
    assert model_i.context_gamma == model_i.regime_gen.gamma == 5.0
    assert model_j.context_gamma == model_j.regime_gen.gamma == 1.0

    x = torch.randn(3, 20, 6)
    descriptor = torch.randn(3, 20, 3)
    model_i(x, descriptor)
    model_j(x, descriptor)
    torch.testing.assert_close(
        model_j.regime_gen.last_context_score,
        model_i.regime_gen.last_context_score / 5.0,
    )
    torch.testing.assert_close(
        model_j.regime_gen.last_semantic_score,
        model_i.regime_gen.last_semantic_score,
    )
    expected_use = 0.75 * torch.full_like(model_j.last_regime_p, 1.0 / 3.0)
    expected_use = expected_use + 0.25 * model_j.last_regime_p
    torch.testing.assert_close(model_j.last_regime_p_use, expected_use)


def test_j_variant_wires_the_same_parameters_and_i_configuration():
    common = dict(
        num_nodes=4,
        n_commodities=2,
        n_stock=1,
        n_bond=1,
        feat_dim=21,
    )
    torch.manual_seed(10)
    model_i = HeteroMixHopCMGM(variant="routing_strength", **common)
    torch.manual_seed(10)
    model_j = HeteroMixHopCMGM(variant="context_calibrated", **common)

    assert model_i.state_dict().keys() == model_j.state_dict().keys()
    for name, tensor_i in model_i.state_dict().items():
        torch.testing.assert_close(tensor_i, model_j.state_dict()[name])
    assert sum(p.numel() for p in model_i.parameters()) == sum(
        p.numel() for p in model_j.parameters()
    )
    rd_i = model_i.regime_dynamic
    rd_j = model_j.regime_dynamic
    assert rd_i.routing_strength == rd_j.routing_strength == 0.25
    assert rd_i.lambda_cluster == rd_j.lambda_cluster == 0.0005
    assert rd_i.lambda_info_max == rd_j.lambda_info_max == 0.001
    assert rd_i.regime_gen.tau_m == rd_j.regime_gen.tau_m == 1.0
    assert rd_i.regime_gen.lambda_sem == rd_j.regime_gen.lambda_sem == 1.0
    assert rd_i.regime_gen.gamma == 5.0
    assert rd_j.regime_gen.gamma == 1.0


def test_k_differs_from_j_only_by_fixed_context_gamma():
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
        routing_strength=0.25,
    )
    torch.manual_seed(11)
    model_j = RegimeDynamicRPETransformer(**common, context_gamma=1.0)
    torch.manual_seed(11)
    model_k = RegimeDynamicRPETransformer(**common, context_gamma=2.0)
    model_j.eval()
    model_k.eval()

    assert model_j.state_dict().keys() == model_k.state_dict().keys()
    for name, tensor_j in model_j.state_dict().items():
        torch.testing.assert_close(tensor_j, model_k.state_dict()[name])
    assert model_j.routing_strength == model_k.routing_strength == 0.25
    assert model_j.lambda_cluster == model_k.lambda_cluster == 0.0005
    assert model_j.lambda_info_max == model_k.lambda_info_max == 0.001
    assert model_j.context_gamma == model_j.regime_gen.gamma == 1.0
    assert model_k.context_gamma == model_k.regime_gen.gamma == 2.0

    x = torch.randn(3, 20, 6)
    descriptor = torch.randn(3, 20, 3)
    model_j(x, descriptor)
    model_k(x, descriptor)
    torch.testing.assert_close(
        model_k.regime_gen.last_context_score,
        2.0 * model_j.regime_gen.last_context_score,
    )
    torch.testing.assert_close(
        model_k.regime_gen.last_semantic_score,
        model_j.regime_gen.last_semantic_score,
    )
    expected_use = 0.75 * torch.full_like(model_k.last_regime_p, 1.0 / 3.0)
    expected_use = expected_use + 0.25 * model_k.last_regime_p
    torch.testing.assert_close(model_k.last_regime_p_use, expected_use)


def test_k_variant_matches_j_parameters_and_all_non_gamma_configuration():
    common = dict(
        num_nodes=4,
        n_commodities=2,
        n_stock=1,
        n_bond=1,
        feat_dim=21,
    )
    torch.manual_seed(12)
    model_j = HeteroMixHopCMGM(variant="context_calibrated", **common)
    torch.manual_seed(12)
    model_k = HeteroMixHopCMGM(variant="context_balance", **common)

    assert model_j.state_dict().keys() == model_k.state_dict().keys()
    for name, tensor_j in model_j.state_dict().items():
        torch.testing.assert_close(tensor_j, model_k.state_dict()[name])
    assert sum(p.numel() for p in model_j.parameters()) == sum(
        p.numel() for p in model_k.parameters()
    )
    rd_j = model_j.regime_dynamic
    rd_k = model_k.regime_dynamic
    assert rd_j.routing_strength == rd_k.routing_strength == 0.25
    assert rd_j.lambda_cluster == rd_k.lambda_cluster == 0.0005
    assert rd_j.lambda_info_max == rd_k.lambda_info_max == 0.001
    assert rd_j.regime_gen.tau_m == rd_k.regime_gen.tau_m == 1.0
    assert rd_j.regime_gen.lambda_sem == rd_k.regime_gen.lambda_sem == 1.0
    assert rd_j.regime_gen.gamma == 1.0
    assert rd_k.regime_gen.gamma == 2.0


def test_l_differs_from_k_only_by_cosine_squared_diversity_objective():
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
        routing_strength=0.25,
        context_gamma=2.0,
    )
    torch.manual_seed(13)
    model_k = RegimeDynamicRPETransformer(**common, orthogonal_dynamic=False)
    torch.manual_seed(13)
    model_l = RegimeDynamicRPETransformer(**common, orthogonal_dynamic=True)
    model_k.eval()
    model_l.eval()

    assert model_k.state_dict().keys() == model_l.state_dict().keys()
    for name, tensor_k in model_k.state_dict().items():
        torch.testing.assert_close(tensor_k, model_l.state_dict()[name])
    assert not model_k.orthogonal_dynamic
    assert model_l.orthogonal_dynamic

    x = torch.randn(3, 20, 6)
    descriptor = torch.randn(3, 20, 3)
    prediction_k = model_k(x, descriptor)
    prediction_l = model_l(x, descriptor)
    torch.testing.assert_close(prediction_k, prediction_l)
    torch.testing.assert_close(model_k.last_regime_p, model_l.last_regime_p)
    torch.testing.assert_close(model_k.last_regime_p_use, model_l.last_regime_p_use)

    loss_k = model_k.dynamic_loss()
    loss_l = model_l.dynamic_loss()
    components_k = model_k.last_loss_components
    components_l = model_l.last_loss_components
    for component in ("cluster", "info_max", "pairwise_cosine", "pairwise_cosine_sq"):
        torch.testing.assert_close(components_k[component], components_l[component])
    torch.testing.assert_close(
        components_k["dynamic"], components_k["pairwise_cosine"]
    )
    torch.testing.assert_close(
        components_l["dynamic"], components_l["pairwise_cosine_sq"]
    )
    torch.testing.assert_close(
        loss_l - loss_k,
        torch.tensor(0.001)
        * (components_l["pairwise_cosine_sq"] - components_k["pairwise_cosine"]),
    )


def test_l_variant_matches_k_parameters_and_all_non_loss_configuration():
    common = dict(
        num_nodes=4,
        n_commodities=2,
        n_stock=1,
        n_bond=1,
        feat_dim=21,
    )
    torch.manual_seed(14)
    model_k = HeteroMixHopCMGM(variant="context_balance", **common)
    torch.manual_seed(14)
    model_l = HeteroMixHopCMGM(variant="adapter_orthogonal", **common)

    assert model_k.state_dict().keys() == model_l.state_dict().keys()
    for name, tensor_k in model_k.state_dict().items():
        torch.testing.assert_close(tensor_k, model_l.state_dict()[name])
    assert sum(p.numel() for p in model_k.parameters()) == sum(
        p.numel() for p in model_l.parameters()
    )
    rd_k = model_k.regime_dynamic
    rd_l = model_l.regime_dynamic
    assert rd_k.routing_strength == rd_l.routing_strength == 0.25
    assert rd_k.lambda_cluster == rd_l.lambda_cluster == 0.0005
    assert rd_k.lambda_info_max == rd_l.lambda_info_max == 0.001
    assert rd_k.regime_gen.gamma == rd_l.regime_gen.gamma == 2.0
    assert rd_k.regime_gen.tau_m == rd_l.regime_gen.tau_m == 1.0
    assert rd_k.regime_gen.lambda_sem == rd_l.regime_gen.lambda_sem == 1.0
    assert not rd_k.orthogonal_dynamic
    assert rd_l.orthogonal_dynamic
