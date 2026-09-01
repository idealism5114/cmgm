import torch

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.switching_transformer import (
    BaseTemporalTransformer,
    MarketAwareTemporalEncoder,
    MarketTokenTransformerBranch,
)


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
