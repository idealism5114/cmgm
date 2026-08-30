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
