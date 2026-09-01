import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.switching_transformer import (
    BaseTemporalTransformer,
    CenteredRegimeRelativePositionBias,
    MarkovFilteringRegimeInference,
    MarketAwareTemporalEncoder,
    MarketTokenTransformerBranch,
    RegimeEvidenceTransformer,
    SwitchingFilterRPEBranch,
    SwitchingRegimeInference,
    SwitchingTransformerBranch,
)
from cmgm.training.train import make_loss, train, train_epoch


def _branch():
    return MarketTokenTransformerBranch(
        feat_dim=5,
        n_stock=4,
        n_bond=3,
        n_commodity=2,
        node_dim=32,
        d_model=128,
        n_heads=4,
        n_layers=2,
        ffn_dim=256,
        dropout=0.1,
        output_dim=64,
    )


def test_market_encoder_shapes_and_attention_axis():
    torch.manual_seed(10)
    encoder = MarketAwareTemporalEncoder(5, 4, 3, 2)
    tokens = encoder(torch.randn(2, 7, 9, 5))

    assert tokens.shape == (2, 7, 128)
    expected = {"stock": 4, "bond": 3, "commodity": 2}
    for name, n_nodes in expected.items():
        assert encoder.last_node_encodings[name].shape == (2, 7, n_nodes, 32)
        assert encoder.last_market_tokens[name].shape == (2, 7, 32)
        attention = encoder.last_market_attentions[name]
        assert attention.shape == (2, 7, n_nodes)
        torch.testing.assert_close(
            attention.sum(dim=2), torch.ones(2, 7), atol=1e-6, rtol=1e-6
        )


def test_market_token_branch_is_permutation_invariant_within_each_market():
    torch.manual_seed(11)
    branch = _branch().eval()
    x = torch.randn(3, 8, 9, 5)
    reference_tokens = branch.encode_market_tokens(x)
    reference_output = branch.temporal_forward(reference_tokens)

    for start, end in ((0, 4), (4, 7), (7, 9)):
        permuted = x.clone()
        permutation = torch.arange(end - start - 1, -1, -1)
        permuted[:, :, start:end] = x[:, :, start:end].index_select(2, permutation)
        tokens = branch.encode_market_tokens(permuted)
        output = branch.temporal_forward(tokens)
        torch.testing.assert_close(tokens, reference_tokens, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(output, reference_output, atol=1e-6, rtol=1e-6)


def test_market_token_branch_is_batch_independent():
    torch.manual_seed(12)
    branch = _branch().eval()
    x = torch.randn(4, 6, 9, 5)
    tokens_batch = branch.encode_market_tokens(x)
    output_batch = branch.temporal_forward(tokens_batch)
    tokens_single = branch.encode_market_tokens(x[:1])
    output_single = branch.temporal_forward(tokens_single)

    torch.testing.assert_close(tokens_batch[:1], tokens_single)
    torch.testing.assert_close(output_batch[:1], output_single, atol=1e-6, rtol=1e-6)


def test_s0_is_plain_transformer_without_regime_or_rpe_components():
    branch = _branch()
    names = {name.lower() for name, _ in branch.named_modules()}
    assert not any("regime" in name for name in names)
    assert not any("rpe" in name for name in names)
    assert isinstance(branch.transformer, BaseTemporalTransformer)
    assert len(branch.transformer.layers) == 2
    assert branch.transformer.layers[0].attention.n_heads == 4
    assert branch.transformer.layers[0].attention.d_model == 128


def test_s0_full_model_keeps_spatial_fusion_head_and_output_interface():
    torch.manual_seed(13)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="market_token_transformer",
    ).eval()
    x = torch.randn(2, 7, 6, 5)

    with torch.no_grad():
        prediction = model(x)
        h_spatial = model._temp_weighted_spatial(x)
        h_temporal = model.market_token_transformer(x)
        combined = torch.cat([h_spatial, h_temporal], dim=-1)
        gate = torch.sigmoid(model.gate_fc(combined))
        fused = (
            gate * model.lstm_proj(h_temporal)
            + (1 - gate) * model.gcn_proj(h_spatial)
        )
        expected = model.head(fused).view(
            x.shape[0], model.n_horizons, model.n_commodities
        )

    assert h_spatial.shape == (2, 64)
    assert h_temporal.shape == (2, 64)
    assert prediction.shape == expected.shape
    torch.testing.assert_close(prediction, expected)
    assert not hasattr(model, "regime_dynamic")


def test_market_encoder_has_fewer_parameters_than_real_flat_projection():
    encoder = MarketAwareTemporalEncoder(
        feat_dim=21, n_stock=248, n_bond=12, n_commodity=24
    )
    market_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    flat_projection_parameters = 284 * 21 * 128 + 128
    assert market_parameters < flat_projection_parameters


def test_s0d_hidden_dispersion_matches_population_std_and_shapes():
    torch.manual_seed(14)
    encoder = MarketAwareTemporalEncoder(
        feat_dim=5,
        n_stock=4,
        n_bond=3,
        n_commodity=2,
        use_dispersion=True,
    ).eval()
    tokens = encoder(torch.randn(2, 7, 9, 5))

    assert encoder.last_market_concat.shape == (2, 7, 192)
    assert tokens.shape == (2, 7, 128)
    for name in encoder.MARKET_NAMES:
        encoded = encoder.last_node_encodings[name]
        expected = encoded.std(dim=2, unbiased=False)
        dispersion = encoder.last_market_dispersions[name]
        assert dispersion.shape == (2, 7, 32)
        assert encoder.last_market_tokens[name].shape == (2, 7, 32)
        torch.testing.assert_close(dispersion, expected)


def test_s0_and_s0d_differ_only_in_daily_projection_input_width():
    s0 = MarketTokenTransformerBranch(5, 4, 3, 2, use_dispersion=False)
    s0d = MarketTokenTransformerBranch(5, 4, 3, 2, use_dispersion=True)
    shapes_s0 = {name: tuple(value.shape) for name, value in s0.state_dict().items()}
    shapes_s0d = {name: tuple(value.shape) for name, value in s0d.state_dict().items()}

    assert shapes_s0.keys() == shapes_s0d.keys()
    differing_shapes = {
        name for name in shapes_s0 if shapes_s0[name] != shapes_s0d[name]
    }
    assert differing_shapes == {"market_encoder.daily_projection.0.weight"}
    assert shapes_s0["market_encoder.daily_projection.0.weight"] == (128, 96)
    assert shapes_s0d["market_encoder.daily_projection.0.weight"] == (128, 192)
    assert (
        sum(parameter.numel() for parameter in s0d.parameters())
        - sum(parameter.numel() for parameter in s0.parameters())
        == 96 * 128
    )


def test_s0d_zero_component_and_all_dispersion_diagnostics_are_inference_only():
    torch.manual_seed(15)
    branch = MarketTokenTransformerBranch(
        5, 4, 3, 2, use_dispersion=True
    ).eval()
    x = torch.randn(2, 7, 9, 5)
    reference = branch.encode_market_tokens(x)
    zero_stock_level = branch.encode_market_tokens(
        x, zero_component="stock_level"
    )
    zero_stock_dispersion = branch.encode_market_tokens(
        x, zero_component="stock_dispersion"
    )
    zero_all_dispersion = branch.encode_market_tokens(
        x, zero_component="all_dispersion"
    )

    assert (reference - zero_stock_level).abs().max() > 0
    assert (reference - zero_stock_dispersion).abs().max() > 0
    assert (reference - zero_all_dispersion).abs().max() > 0


def test_s0d_is_permutation_invariant_and_batch_independent():
    torch.manual_seed(16)
    branch = MarketTokenTransformerBranch(
        5, 4, 3, 2, use_dispersion=True
    ).eval()
    x = torch.randn(3, 8, 9, 5)
    reference_tokens = branch.encode_market_tokens(x)
    reference_output = branch.temporal_forward(reference_tokens)

    for start, end in ((0, 4), (4, 7), (7, 9)):
        permuted = x.clone()
        permutation = torch.arange(end - start - 1, -1, -1)
        permuted[:, :, start:end] = x[:, :, start:end].index_select(2, permutation)
        tokens = branch.encode_market_tokens(permuted)
        output = branch.temporal_forward(tokens)
        torch.testing.assert_close(tokens, reference_tokens, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(output, reference_output, atol=1e-6, rtol=1e-6)

    single_tokens = branch.encode_market_tokens(x[:1])
    single_output = branch.temporal_forward(single_tokens)
    torch.testing.assert_close(
        reference_tokens[:1], single_tokens, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        reference_output[:1], single_output, atol=1e-6, rtol=1e-6
    )


def test_s0d_full_model_forward_and_dispersion_path_gradient():
    torch.manual_seed(17)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="market_dispersion_transformer",
    )
    prediction = model(torch.randn(2, 7, 6, 5))
    prediction.square().mean().backward()

    assert prediction.shape == (2, model.n_horizons, 2)
    assert model.market_token_transformer.market_encoder.use_dispersion
    projection_grad = (
        model.market_token_transformer.market_encoder.daily_projection[0].weight.grad
    )
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert projection_grad[:, 32:64].abs().sum() > 0
    assert not hasattr(model, "regime_dynamic")


def test_s1_shared_initialization_matches_s0d_common_modules():
    common = dict(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
    )
    torch.manual_seed(18)
    s0d = HeteroMixHopCMGM(variant="market_dispersion_transformer", **common)
    torch.manual_seed(18)
    s1 = HeteroMixHopCMGM(variant="switching_transformer", **common)

    s0d_encoder = s0d.market_token_transformer.market_encoder.state_dict()
    s1_encoder = s1.switching_transformer.market_encoder.state_dict()
    assert s0d_encoder.keys() == s1_encoder.keys()
    for name in s0d_encoder:
        torch.testing.assert_close(s0d_encoder[name], s1_encoder[name])

    s0d_forecast = s0d.market_token_transformer.transformer.state_dict()
    s1_forecast = s1.switching_transformer.transformer.state_dict()
    assert s0d_forecast.keys() == s1_forecast.keys()
    for name in s0d_forecast:
        torch.testing.assert_close(s0d_forecast[name], s1_forecast[name])


def test_s1c_is_parameter_identical_to_s1_and_preserves_regime_rng_path():
    common = dict(
        feat_dim=5,
        n_stock=4,
        n_bond=3,
        n_commodity=2,
        d_model=16,
        n_heads=4,
        n_layers=2,
        ffn_dim=32,
        output_dim=8,
        dropout=0.2,
    )
    torch.manual_seed(1801)
    s1 = SwitchingTransformerBranch(**common)
    torch.manual_seed(1801)
    s1c = SwitchingTransformerBranch(**common, null_control=True)

    assert sum(p.numel() for p in s1.parameters()) == sum(
        p.numel() for p in s1c.parameters()
    )
    assert s1.state_dict().keys() == s1c.state_dict().keys()
    for name, value in s1.state_dict().items():
        torch.testing.assert_close(
            value, s1c.state_dict()[name], rtol=0.0, atol=0.0
        )

    x = torch.randn(2, 7, 9, 5)
    s1.train()
    s1c.train()
    torch.manual_seed(1802)
    s1(x)
    torch.manual_seed(1802)
    s1c(x)
    torch.testing.assert_close(
        s1.last_regime_evidence, s1c.last_regime_evidence,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        s1.last_regime_probabilities, s1c.last_regime_probabilities,
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        s1.last_regime_intervention_real,
        s1c.last_regime_intervention_real,
        rtol=0.0, atol=0.0,
    )


def test_s1c_null_intervention_loss_and_prediction_gradients_are_exactly_cut():
    torch.manual_seed(1803)
    branch = SwitchingTransformerBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, ffn_dim=32,
        output_dim=8, dropout=0.0, null_control=True,
    )
    branch.set_epoch(20)
    output = branch(torch.randn(2, 7, 9, 5))
    return_loss = output.square().mean()
    switch_loss = branch.switch_loss()
    total_loss = return_loss + switch_loss

    assert branch.scheduled_beta == 5e-4
    assert branch.effective_beta == 0.0
    assert branch.regime_inference._last_switch_loss.requires_grad
    assert switch_loss.item() == 0.0
    assert total_loss.item() == return_loss.item()
    assert branch.last_regime_intervention_real.norm() > 0
    assert branch.last_regime_intervention_effective.abs().max() == 0
    torch.testing.assert_close(
        branch.last_conditioned_tokens,
        branch.last_market_tokens,
        rtol=0.0,
        atol=0.0,
    )

    total_loss.backward()
    regime_parameters = [
        *branch.regime_evidence.parameters(),
        *branch.regime_inference.posterior_heads.parameters(),
        branch.regime_inference.transition_logits,
        branch.regime_embeddings,
    ]
    assert all(parameter.grad is None for parameter in regime_parameters)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in branch.market_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in branch.transformer.parameters()
    )

    initial_regime = [parameter.detach().clone() for parameter in regime_parameters]
    optimizer = torch.optim.Adam(
        branch.parameters(), lr=1e-3, weight_decay=0.1
    )
    optimizer.step()
    for initial, parameter in zip(initial_regime, regime_parameters):
        torch.testing.assert_close(
            initial, parameter.detach(), rtol=0.0, atol=0.0
        )


def test_s1c_is_invariant_to_forced_null_states():
    torch.manual_seed(1804)
    branch = SwitchingTransformerBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, ffn_dim=32,
        output_dim=8, dropout=0.0, null_control=True,
    ).eval()
    x = torch.randn(2, 7, 9, 5)
    state_a = torch.tensor([1.0, 0.0, 0.0])
    state_b = torch.tensor([0.0, 0.0, 1.0])
    output_a = branch(x, forced_probabilities=state_a)
    real_a = branch.last_regime_intervention_real.clone()
    output_b = branch(x, forced_probabilities=state_b)
    real_b = branch.last_regime_intervention_real.clone()

    assert (real_a - real_b).abs().max() > 0
    torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)
    assert branch.last_regime_intervention_effective.abs().max() == 0


def test_s1c_full_model_variant_is_registered_without_extra_parameters():
    common = dict(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
    )
    torch.manual_seed(1805)
    s1 = HeteroMixHopCMGM(variant="switching_transformer", **common)
    torch.manual_seed(1805)
    s1c = HeteroMixHopCMGM(variant="switching_null_control", **common)
    assert sum(p.numel() for p in s1.parameters()) == sum(
        p.numel() for p in s1c.parameters()
    )
    assert s1.state_dict().keys() == s1c.state_dict().keys()
    assert s1c.switching_transformer.null_control
    prediction = s1c(torch.randn(2, 7, 6, 5))
    assert prediction.shape == (2, s1c.n_horizons, 2)


def test_s1c_train_epoch_keeps_all_switching_parameters_frozen():
    torch.manual_seed(1806)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="switching_null_control",
    )
    branch = model.switching_transformer
    branch.set_epoch(20)
    regime_parameters = [
        *branch.regime_evidence.parameters(),
        *branch.regime_inference.posterior_heads.parameters(),
        branch.regime_inference.transition_logits,
        branch.regime_embeddings,
    ]
    initial = [parameter.detach().clone() for parameter in regime_parameters]
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 7, 6, 5),
            torch.randn(2, model.n_horizons, 2),
        ),
        batch_size=2,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0.1
    )
    train_epoch(
        model,
        loader,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0),
        optimizer,
        make_loss(),
        torch.device("cpu"),
    )

    assert branch.regime_inference._last_switch_loss is not None
    assert all(parameter.grad is None for parameter in regime_parameters)
    for before, after in zip(initial, regime_parameters):
        torch.testing.assert_close(
            before, after.detach(), rtol=0.0, atol=0.0
        )


def test_s2f_markov_filter_matches_log_prior_plus_evidence_formula():
    torch.manual_seed(1901)
    filtering = MarkovFilteringRegimeInference(
        d_model=8, K=3, sticky_alpha=0.5, tau=1.0
    )
    evidence = torch.randn(2, 6, 8)
    probabilities = filtering(evidence)
    transition = filtering.transition_matrix()

    assert not hasattr(filtering, "posterior_heads")
    assert probabilities.min() >= 0
    assert (probabilities.sum(dim=-1) - 1).abs().max() < 1e-6
    torch.testing.assert_close(
        transition.diagonal(), torch.full((3,), 2.0 / 3.0),
        atol=1e-6, rtol=1e-6,
    )
    previous = torch.full((2, 3), 1.0 / 3.0)
    expected_probabilities = []
    expected_priors = []
    for step in range(evidence.shape[1]):
        prior = previous @ transition
        observation = filtering.observation_head(evidence[:, step])
        posterior = torch.softmax(torch.log(prior + filtering.eps) + observation, -1)
        expected_priors.append(prior)
        expected_probabilities.append(posterior)
        previous = posterior
    torch.testing.assert_close(
        filtering.last_priors, torch.stack(expected_priors, dim=1)
    )
    torch.testing.assert_close(
        probabilities, torch.stack(expected_probabilities, dim=1)
    )


def test_s2f_centered_regime_rpe_uniform_sanity_and_signed_delta():
    torch.manual_seed(1902)
    rpe = CenteredRegimeRelativePositionBias(max_len=7, n_heads=2, K=3)
    uniform = torch.full((2, 7, 3), 1.0 / 3.0)
    base_bias, regime_bias = rpe.components(uniform)

    assert rpe.base_rpe.shape == (2, 13)
    assert rpe.regime_rpe.shape == (3, 2, 13)
    assert base_bias.shape == (2, 2, 7, 7)
    assert regime_bias.shape == (2, 2, 7, 7)
    assert regime_bias.abs().max() < 1e-6
    expected_delta = torch.arange(7).unsqueeze(1) - torch.arange(7).unsqueeze(0)
    torch.testing.assert_close(rpe.last_relative_delta, expected_delta)
    centered = rpe.centered_regime_rpe()
    assert centered.mean(dim=0).abs().max() < 1e-7


def test_s2f_prediction_path_reaches_filter_transition_and_regime_rpe():
    torch.manual_seed(1903)
    branch = SwitchingFilterRPEBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, n_layers=2,
        ffn_dim=32, output_dim=8, dropout=0.0, max_len=7,
    )
    output = branch(torch.randn(3, 7, 9, 5))
    return_loss = output.square().mean()
    groups = {
        "evidence": list(branch.regime_evidence.parameters()),
        "observation": list(branch.regime_inference.observation_head.parameters()),
        "transition": [branch.regime_inference.transition_logits],
        "base_rpe": [branch.transformer.relative_position.base_rpe],
        "regime_rpe": [branch.transformer.relative_position.regime_rpe],
        "forecast": list(branch.transformer.layers.parameters()),
    }
    gradients = torch.autograd.grad(
        return_loss,
        [parameter for values in groups.values() for parameter in values],
        allow_unused=True,
    )
    offset = 0
    for values in groups.values():
        group_gradients = gradients[offset:offset + len(values)]
        offset += len(values)
        assert any(
            gradient is not None and gradient.abs().sum() > 0
            for gradient in group_gradients
        )
    assert not hasattr(branch, "regime_embeddings")
    assert not hasattr(branch.regime_inference, "posterior_heads")


def test_s2f_forced_states_change_only_rpe_conditioned_forecast():
    torch.manual_seed(1904)
    branch = SwitchingFilterRPEBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, n_layers=2,
        ffn_dim=32, output_dim=8, dropout=0.0, max_len=7,
    ).eval()
    x = torch.randn(2, 7, 9, 5)
    state0 = torch.tensor([1.0, 0.0, 0.0])
    state2 = torch.tensor([0.0, 0.0, 1.0])
    output0 = branch(x, forced_rpe_probabilities=state0)
    p0 = branch.last_regime_probabilities.clone()
    bias0 = branch.transformer.relative_position.last_regime_bias.clone()
    output2 = branch(x, forced_rpe_probabilities=state2)
    p2 = branch.last_regime_probabilities.clone()
    bias2 = branch.transformer.relative_position.last_regime_bias.clone()

    torch.testing.assert_close(p0, p2, rtol=0.0, atol=0.0)
    assert (bias0 - bias2).abs().max() > 0
    assert (output0 - output2).abs().max() > 0


def test_s2f_causality_batch_independence_and_market_permutation():
    torch.manual_seed(1905)
    branch = SwitchingFilterRPEBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, n_layers=2,
        ffn_dim=32, output_dim=8, dropout=0.0, max_len=12,
    ).eval()
    x = torch.randn(3, 12, 9, 5)
    tokens = branch.encode_market_tokens(x)
    output = branch.temporal_forward(tokens)
    probabilities = branch.last_regime_probabilities.clone()
    priors = branch.regime_inference.last_priors.clone()

    perturbed = x.clone()
    perturbed[:, 7:] = torch.randn_like(perturbed[:, 7:])
    branch(perturbed)
    torch.testing.assert_close(
        probabilities[:, :7], branch.last_regime_probabilities[:, :7],
        atol=1e-6, rtol=1e-6,
    )
    torch.testing.assert_close(
        priors[:, :7], branch.regime_inference.last_priors[:, :7],
        atol=1e-6, rtol=1e-6,
    )

    single_tokens = branch.encode_market_tokens(x[:1])
    single_output = branch.temporal_forward(single_tokens)
    single_p = branch.last_regime_probabilities.clone()
    single_prior = branch.regime_inference.last_priors.clone()
    torch.testing.assert_close(tokens[:1], single_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output[:1], single_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probabilities[:1], single_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(priors[:1], single_prior, atol=1e-6, rtol=1e-6)

    for start, end in ((0, 4), (4, 7), (7, 9)):
        permuted = x.clone()
        order = torch.arange(end - start - 1, -1, -1)
        permuted[:, :, start:end] = x[:, :, start:end].index_select(2, order)
        permuted_tokens = branch.encode_market_tokens(permuted)
        permuted_output = branch.temporal_forward(permuted_tokens)
        torch.testing.assert_close(permuted_tokens, tokens, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            branch.last_regime_probabilities, probabilities, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(permuted_output, output, atol=1e-6, rtol=1e-6)


def test_s2f_full_model_registration_and_switch_loss_training_path():
    torch.manual_seed(1906)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="switching_filter_rpe",
    )
    branch = model.switching_filter_rpe
    branch.set_epoch(20)
    prediction = model(torch.randn(2, 7, 6, 5))
    total_loss = prediction.square().mean() + branch.switch_loss()
    total_loss.backward()

    assert prediction.shape == (2, model.n_horizons, 2)
    assert branch.regime_inference._last_switch_loss.item() >= 0
    assert branch.regime_inference.transition_logits.grad is not None
    assert branch.regime_inference.transition_logits.grad.abs().sum() > 0


def test_s2f_reuses_s0d_encoder_and_has_no_s1_conditioning_mechanisms():
    common = dict(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
    )
    torch.manual_seed(1907)
    s0d = HeteroMixHopCMGM(
        variant="market_dispersion_transformer", **common
    )
    torch.manual_seed(1907)
    s2f = HeteroMixHopCMGM(variant="switching_filter_rpe", **common)
    s0d_encoder = s0d.market_token_transformer.market_encoder.state_dict()
    s2f_encoder = s2f.switching_filter_rpe.market_encoder.state_dict()
    assert s0d_encoder.keys() == s2f_encoder.keys()
    for name in s0d_encoder:
        torch.testing.assert_close(
            s0d_encoder[name], s2f_encoder[name], rtol=0.0, atol=0.0
        )

    branch = s2f.switching_filter_rpe
    x = torch.randn(2, 7, 6, 5)
    branch(x)
    torch.testing.assert_close(
        branch.last_market_tokens,
        branch.transformer.last_input_tokens,
        rtol=0.0,
        atol=0.0,
    )
    assert not hasattr(branch, "regime_embeddings")
    assert not hasattr(branch.regime_inference, "posterior_heads")


def test_regime_evidence_transformer_is_causal():
    torch.manual_seed(19)
    evidence = RegimeEvidenceTransformer(
        d_model=16, n_heads=4, ffn_dim=32, dropout=0.0
    ).eval()
    tokens = torch.randn(2, 12, 16)
    perturbed = tokens.clone()
    perturbed[:, 7:] = torch.randn_like(perturbed[:, 7:])
    original_output = evidence(tokens)
    perturbed_output = evidence(perturbed)
    torch.testing.assert_close(
        original_output[:, :7], perturbed_output[:, :7], atol=1e-6, rtol=1e-6
    )


def test_switching_transition_probability_and_recursion_are_valid():
    torch.manual_seed(20)
    switching = SwitchingRegimeInference(d_model=8, K=3)
    p = switching(torch.randn(2, 6, 8))
    transition = switching.transition_matrix()

    torch.testing.assert_close(
        transition.sum(dim=-1), torch.ones(3), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        transition.diagonal(), torch.full((3,), 2.0 / 3.0), atol=1e-6, rtol=1e-6
    )
    off_diagonal = transition[~torch.eye(3, dtype=torch.bool)]
    torch.testing.assert_close(
        off_diagonal, torch.full((6,), 1.0 / 6.0), atol=1e-6, rtol=1e-6
    )
    assert p.min() >= 0
    assert (p.sum(dim=-1) - 1).abs().max() < 1e-6
    expected = torch.einsum(
        "bti,btik->btk",
        switching.last_previous_probabilities,
        switching.last_posterior_heads,
    )
    torch.testing.assert_close(p, expected)
    expected_prior = torch.einsum(
        "bti,ik->btk",
        switching.last_previous_probabilities,
        transition.detach(),
    )
    torch.testing.assert_close(switching.last_priors, expected_prior)


def test_switching_beta_warmup_and_non_detached_kl_gradients():
    torch.manual_seed(21)
    branch = SwitchingTransformerBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, ffn_dim=32,
        output_dim=8, dropout=0.0,
    )
    assert branch.set_epoch(1) == 0.0
    assert abs(branch.set_epoch(20) - 5e-4) < 1e-12
    branch(torch.randn(2, 6, 9, 5))
    loss = branch.switch_loss()
    loss.backward()

    assert loss.item() >= 0
    assert branch.regime_inference.transition_logits.grad is not None
    assert branch.regime_inference.transition_logits.grad.abs().sum() > 0
    for head in branch.regime_inference.posterior_heads:
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0
    evidence_grad = branch.regime_evidence.block.attention.q.weight.grad
    assert evidence_grad is not None
    assert evidence_grad.abs().sum() > 0


def test_uniform_probability_has_zero_centered_regime_intervention():
    torch.manual_seed(22)
    branch = SwitchingTransformerBranch(5, 4, 3, 2, d_model=16, n_heads=4)
    uniform = torch.full((2, 7, 3), 1.0 / 3.0)
    tokens = torch.randn(2, 7, 16)
    conditioned = branch.condition_tokens(tokens, uniform)
    torch.testing.assert_close(conditioned, tokens, atol=1e-6, rtol=1e-6)
    assert branch.last_regime_intervention.abs().max() < 1e-6


def test_s1_causality_batch_independence_and_market_permutation():
    torch.manual_seed(23)
    branch = SwitchingTransformerBranch(
        5, 4, 3, 2, d_model=16, n_heads=4, ffn_dim=32,
        output_dim=8, dropout=0.0,
    ).eval()
    x = torch.randn(3, 12, 9, 5)
    tokens = branch.encode_market_tokens(x)
    output = branch.temporal_forward(tokens)
    probabilities = branch.last_regime_probabilities.clone()

    perturbed = x.clone()
    perturbed[:, 7:] = torch.randn_like(perturbed[:, 7:])
    perturbed_tokens = branch.encode_market_tokens(perturbed)
    branch.temporal_forward(perturbed_tokens)
    torch.testing.assert_close(
        probabilities[:, :7], branch.last_regime_probabilities[:, :7],
        atol=1e-6, rtol=1e-6,
    )

    single_tokens = branch.encode_market_tokens(x[:1])
    single_output = branch.temporal_forward(single_tokens)
    single_p = branch.last_regime_probabilities.clone()
    torch.testing.assert_close(tokens[:1], single_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output[:1], single_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probabilities[:1], single_p, atol=1e-6, rtol=1e-6)

    for start, end in ((0, 4), (4, 7), (7, 9)):
        permuted = x.clone()
        order = torch.arange(end - start - 1, -1, -1)
        permuted[:, :, start:end] = x[:, :, start:end].index_select(2, order)
        permuted_tokens = branch.encode_market_tokens(permuted)
        permuted_output = branch.temporal_forward(permuted_tokens)
        torch.testing.assert_close(permuted_tokens, tokens, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            branch.last_regime_probabilities, probabilities, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(permuted_output, output, atol=1e-6, rtol=1e-6)


def test_s1_full_model_uses_only_new_switching_branch_and_prediction_connects():
    torch.manual_seed(24)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="switching_transformer",
    )
    prediction = model(torch.randn(2, 7, 6, 5))
    prediction.square().mean().backward()

    assert prediction.shape == (2, model.n_horizons, 2)
    assert not hasattr(model, "regime_dynamic")
    assert not hasattr(model, "market_token_transformer")
    branch = model.switching_transformer
    module_names = {name.lower() for name, _ in branch.named_modules()}
    assert not any("rpe" in name for name in module_names)
    assert not any("prototype" in name for name in module_names)
    assert not hasattr(branch, "state_centers")
    assert branch.regime_embeddings.grad is not None
    assert branch.regime_embeddings.grad.abs().sum() > 0
    for head in branch.regime_inference.posterior_heads:
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0
    evidence_grad = branch.regime_evidence.block.attention.q.weight.grad
    assert evidence_grad is not None
    assert evidence_grad.abs().sum() > 0


def test_training_restores_best_epoch_switch_beta():
    torch.manual_seed(25)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="switching_transformer",
    )
    dataset = TensorDataset(
        torch.randn(4, 7, 6, 5),
        torch.randn(4, model.n_horizons, 2),
    )
    loader = DataLoader(dataset, batch_size=2)
    history = train(
        model,
        loader,
        loader,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0),
        torch.device("cpu"),
        num_epochs=2,
        patience=3,
    )

    inference = model.switching_transformer.regime_inference
    expected_beta = inference.beta_max * min(
        max((history["best_epoch"] - 1) / (inference.warmup_epochs - 1), 0.0),
        1.0,
    )
    assert inference.current_epoch == history["best_epoch"]
    assert abs(inference.current_beta - expected_beta) < 1e-12
