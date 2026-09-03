import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.switching_latent_transformer import (
    LongMemoryTransformer,
    MarkovRegimeFilter,
    RegimeLatentTransition,
    SwitchingLatentTransformerBranch,
)
from cmgm.training.train import make_loss, train_epoch


def _d0_branch(dropout=0.0, max_len=12, balanced_readout=False):
    return SwitchingLatentTransformerBranch(
        feat_dim=5,
        n_stock=4,
        n_bond=3,
        n_commodity=2,
        node_dim=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        ffn_dim=32,
        dropout=dropout,
        max_len=max_len,
        K=3,
        z_dim=8,
        output_dim=8,
        balanced_readout=balanced_readout,
    )


def test_d0_long_memory_is_causal_and_uses_signed_base_rpe():
    torch.manual_seed(2001)
    memory = LongMemoryTransformer(
        d_model=16, n_heads=4, n_layers=2, ffn_dim=32,
        dropout=0.0, max_len=12,
    ).eval()
    tokens = torch.randn(2, 12, 16)
    perturbed = tokens.clone()
    perturbed[:, 7:] = torch.randn_like(perturbed[:, 7:])
    original = memory(tokens)
    changed = memory(perturbed)

    assert original.shape == (2, 12, 16)
    assert memory.base_rpe.shape == (4, 23)
    expected_delta = torch.arange(12).unsqueeze(1) - torch.arange(12).unsqueeze(0)
    torch.testing.assert_close(memory.last_relative_delta, expected_delta)
    torch.testing.assert_close(
        original[:, :7], changed[:, :7], atol=1e-6, rtol=1e-6
    )
    future_attention = memory.layers[0].attention.last_attention[
        :, :, torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    ]
    assert future_attention.abs().max() == 0


def test_d0_markov_filter_step_matches_formula():
    torch.manual_seed(2002)
    filtering = MarkovRegimeFilter(d_model=8, K=3)
    h_t = torch.randn(3, 8)
    p_prev = torch.softmax(torch.randn(3, 3), dim=-1)
    transition = filtering.transition_matrix()
    prior, evidence, posterior, kl = filtering.step(
        h_t, p_prev, transition
    )

    expected_prior = p_prev @ transition
    expected_evidence = filtering.regime_evidence(h_t)
    expected_posterior = torch.softmax(
        torch.log(expected_prior + filtering.eps) + expected_evidence,
        dim=-1,
    )
    torch.testing.assert_close(prior, expected_prior)
    torch.testing.assert_close(evidence, expected_evidence)
    torch.testing.assert_close(posterior, expected_posterior)
    assert posterior.min() >= 0
    assert (posterior.sum(dim=-1) - 1).abs().max() < 1e-6
    assert kl.min() >= -1e-6


def test_d0_branch_recursion_shapes_and_only_base_rpe():
    torch.manual_seed(2003)
    branch = _d0_branch(max_len=7)
    output = branch(torch.randn(2, 7, 9, 5))

    assert output.shape == (2, 8)
    assert branch.last_market_tokens.shape == (2, 7, 16)
    assert branch.last_long_memory.shape == (2, 7, 16)
    assert branch.last_regime_probabilities.shape == (2, 7, 3)
    assert branch.last_regime_priors.shape == (2, 7, 3)
    assert branch.last_latent_states.shape == (2, 7, 8)
    assert branch.last_latent_candidates.shape == (2, 7, 3, 8)
    assert branch.last_readout_input.shape == (2, 24)
    torch.testing.assert_close(
        output,
        branch.state_readout(torch.cat([branch.last_h_last, branch.last_z_last], dim=-1)),
    )
    assert branch.latent_transition.generators[0][0].in_features == 24
    assert not branch.balanced_readout
    assert not hasattr(branch, "long_memory_readout")
    assert not hasattr(branch, "micro_state_readout")
    assert not hasattr(branch.long_memory, "regime_rpe")
    assert not hasattr(branch, "regime_embeddings")
    assert not hasattr(branch, "pool_score")


def test_d0_prediction_gradients_reach_regime_and_all_generators():
    torch.manual_seed(2004)
    branch = _d0_branch(max_len=7)
    prediction = branch(torch.randn(3, 7, 9, 5))
    loss = prediction.square().mean()
    groups = {
        "market": list(branch.market_encoder.parameters()),
        "memory": list(branch.long_memory.layers.parameters()),
        "base_rpe": [branch.long_memory.base_rpe],
        "regime": list(branch.regime_filter.regime_evidence.parameters()),
        "transition": [branch.regime_filter.transition_logits],
        "G0": list(branch.latent_transition.generators[0].parameters()),
        "G1": list(branch.latent_transition.generators[1].parameters()),
        "G2": list(branch.latent_transition.generators[2].parameters()),
        "readout": list(branch.state_readout.parameters()),
    }
    flat = [parameter for values in groups.values() for parameter in values]
    gradients = torch.autograd.grad(loss, flat, allow_unused=True)
    offset = 0
    for values in groups.values():
        selected = gradients[offset:offset + len(values)]
        offset += len(values)
        assert any(
            gradient is not None and gradient.abs().sum() > 0
            for gradient in selected
        )


def test_d0_forced_regime_recomputes_full_latent_trajectory():
    torch.manual_seed(2005)
    branch = _d0_branch(max_len=7).eval()
    x = torch.randn(2, 7, 9, 5)
    real = branch(x)
    real_z = branch.last_latent_states.clone()
    state0 = branch(x, forced_probabilities=torch.tensor([1.0, 0.0, 0.0]))
    state0_z = branch.last_latent_states.clone()
    state2 = branch(x, forced_probabilities=torch.tensor([0.0, 0.0, 1.0]))
    state2_z = branch.last_latent_states.clone()

    assert (real_z[:, -1] - state0_z[:, -1]).abs().max() > 0
    assert (state0_z[:, -1] - state2_z[:, -1]).abs().max() > 0
    assert (real - state0).abs().max() > 0
    assert (state0 - state2).abs().max() > 0
    assert (state0_z[:, 1] - state2_z[:, 1]).abs().max() > 0


def test_d0_zero_h_and_zero_z_are_readout_only_interventions():
    torch.manual_seed(2006)
    branch = _d0_branch(max_len=7).eval()
    x = torch.randn(2, 7, 9, 5)
    real = branch(x)
    real_z = branch.last_latent_states.clone()
    zero_z = branch(x, zero_readout_component="Z")
    zero_z_trajectory = branch.last_latent_states.clone()
    zero_h = branch(x, zero_readout_component="H")
    zero_h_trajectory = branch.last_latent_states.clone()

    torch.testing.assert_close(real_z, zero_z_trajectory, rtol=0.0, atol=0.0)
    torch.testing.assert_close(real_z, zero_h_trajectory, rtol=0.0, atol=0.0)
    assert (real - zero_z).abs().max() > 0
    assert (real - zero_h).abs().max() > 0


def test_d0_full_causality_batch_independence_and_market_permutation():
    torch.manual_seed(2007)
    branch = _d0_branch(max_len=12).eval()
    x = torch.randn(3, 12, 9, 5)
    output = branch(x)
    E = branch.last_market_tokens.clone()
    H = branch.last_long_memory.clone()
    p = branch.last_regime_probabilities.clone()
    Z = branch.last_latent_states.clone()

    perturbed = x.clone()
    perturbed[:, 7:] = torch.randn_like(perturbed[:, 7:])
    branch(perturbed)
    torch.testing.assert_close(H[:, :7], branch.last_long_memory[:, :7], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p[:, :7], branch.last_regime_probabilities[:, :7], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(Z[:, :7], branch.last_latent_states[:, :7], atol=1e-6, rtol=1e-6)

    single_output = branch(x[:1])
    torch.testing.assert_close(E[:1], branch.last_market_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(H[:1], branch.last_long_memory, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p[:1], branch.last_regime_probabilities, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(Z[:1], branch.last_latent_states, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output[:1], single_output, atol=1e-6, rtol=1e-6)

    for start, end in ((0, 4), (4, 7), (7, 9)):
        permuted = x.clone()
        order = torch.arange(end - start - 1, -1, -1)
        permuted[:, :, start:end] = x[:, :, start:end].index_select(2, order)
        permuted_output = branch(permuted)
        torch.testing.assert_close(E, branch.last_market_tokens, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(H, branch.last_long_memory, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(p, branch.last_regime_probabilities, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(Z, branch.last_latent_states, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(output, permuted_output, atol=1e-6, rtol=1e-6)


def test_d0_generator_label_permutation_is_invariant():
    torch.manual_seed(2008)
    transition = RegimeLatentTransition(h_dim=8, z_dim=4, hidden_dim=8, K=3)
    H = torch.randn(2, 6, 8)
    p = torch.softmax(torch.randn(2, 6, 3), dim=-1)

    def trajectory(module, probabilities):
        z_prev = torch.zeros(2, 4)
        values = []
        for step in range(6):
            z_prev, _ = module(H[:, step], z_prev, probabilities[:, step])
            values.append(z_prev)
        return torch.stack(values, dim=1)

    reference = trajectory(transition, p)
    generator0 = transition.generators[0]
    generator1 = transition.generators[1]
    transition.generators[0] = generator1
    transition.generators[1] = generator0
    permuted_p = p[:, :, [1, 0, 2]]
    changed = trajectory(transition, permuted_p)
    torch.testing.assert_close(reference, changed, atol=1e-6, rtol=1e-6)


def test_d0_full_model_registration_and_training_hook():
    torch.manual_seed(2009)
    model = HeteroMixHopCMGM(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
        variant="switching_latent_transformer",
    )
    branch = model.switching_latent_transformer
    branch.set_epoch(20)
    before = branch.regime_filter.transition_logits.detach().clone()
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 7, 6, 5),
            torch.randn(2, model.n_horizons, 2),
        ),
        batch_size=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_epoch(
        model,
        loader,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0),
        optimizer,
        make_loss(),
        torch.device("cpu"),
    )

    assert branch.regime_filter._last_switch_loss is not None
    assert (
        branch.regime_filter.transition_logits.detach() - before
    ).abs().sum() > 0


def test_d0_reuses_s0d_market_encoder_initialization():
    common = dict(
        num_nodes=6,
        n_commodities=2,
        n_stock=2,
        n_bond=2,
        feat_dim=5,
    )
    torch.manual_seed(2010)
    s0d = HeteroMixHopCMGM(
        variant="market_dispersion_transformer", **common
    )
    torch.manual_seed(2010)
    d0 = HeteroMixHopCMGM(
        variant="switching_latent_transformer", **common
    )
    left = s0d.market_token_transformer.market_encoder.state_dict()
    right = d0.switching_latent_transformer.market_encoder.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0.0, atol=0.0)


def test_d0b_balances_only_the_final_readout_and_preserves_shared_initialization():
    torch.manual_seed(2011)
    d0 = _d0_branch(max_len=7)
    torch.manual_seed(2011)
    d0b = _d0_branch(max_len=7, balanced_readout=True)

    for module_name in (
        "market_encoder", "long_memory", "regime_filter", "latent_transition"
    ):
        d0_state = getattr(d0, module_name).state_dict()
        d0b_state = getattr(d0b, module_name).state_dict()
        assert d0_state.keys() == d0b_state.keys()
        for name in d0_state:
            torch.testing.assert_close(
                d0_state[name], d0b_state[name], rtol=0.0, atol=0.0
            )

    assert not d0.balanced_readout
    assert d0b.balanced_readout
    assert d0.state_readout.in_features == 24
    assert d0b.long_memory_readout.in_features == 16
    assert d0b.micro_state_readout.in_features == 8
    assert d0b.state_readout.in_features == 16


def test_d0b_post_projection_layer_norm_balances_scales_without_changing_latents():
    torch.manual_seed(2012)
    branch = _d0_branch(max_len=7, balanced_readout=True).eval()
    x = torch.randn(3, 7, 9, 5)
    output = branch(x)

    assert output.shape == (3, 8)
    assert branch.last_h_long.shape == (3, 8)
    assert branch.last_h_micro.shape == (3, 8)
    assert branch.last_balanced_concat.shape == (3, 16)
    long_norm = branch.last_h_long.norm(dim=-1).mean()
    micro_norm = branch.last_h_micro.norm(dim=-1).mean()
    assert 0.8 < (micro_norm / long_norm).item() < 1.2

    latent_reference = branch.last_latent_states.clone()
    branch(x, zero_readout_component="Z")
    torch.testing.assert_close(
        latent_reference, branch.last_latent_states, rtol=0.0, atol=0.0
    )
    assert branch.last_h_micro.abs().sum() > 0
    assert branch.last_h_micro_effective.abs().sum() == 0
    branch(x, zero_readout_component="H")
    torch.testing.assert_close(
        latent_reference, branch.last_latent_states, rtol=0.0, atol=0.0
    )
    assert branch.last_h_long.abs().sum() > 0
    assert branch.last_h_long_effective.abs().sum() == 0


def test_d0b_prediction_gradients_reach_both_balanced_readout_paths():
    torch.manual_seed(2013)
    branch = _d0_branch(max_len=7, balanced_readout=True)
    loss = branch(torch.randn(3, 7, 9, 5)).square().mean()
    groups = (
        branch.long_memory_readout,
        branch.micro_state_readout,
        branch.long_memory_norm,
        branch.micro_state_norm,
        branch.state_readout,
    )
    parameters = [parameter for group in groups for parameter in group.parameters()]
    gradients = torch.autograd.grad(loss, parameters)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_d0b_full_model_registration_and_parameter_delta():
    common = dict(
        num_nodes=6, n_commodities=2, n_stock=2, n_bond=2, feat_dim=5
    )
    torch.manual_seed(2014)
    d0 = HeteroMixHopCMGM(variant="switching_latent_transformer", **common)
    torch.manual_seed(2014)
    d0b = HeteroMixHopCMGM(
        variant="switching_latent_balanced_readout", **common
    )
    assert d0b.switching_latent_transformer.balanced_readout
    assert d0b(torch.randn(2, 7, 6, 5)).shape == (2, d0b.n_horizons, 2)
    expected_delta = (
        sum(
            parameter.numel()
            for name in (
                "long_memory_readout", "micro_state_readout",
                "long_memory_norm", "micro_state_norm", "state_readout",
            )
            for parameter in getattr(d0b.switching_latent_transformer, name).parameters()
        )
        - sum(
            parameter.numel()
            for parameter in d0.switching_latent_transformer.state_readout.parameters()
        )
    )
    assert (
        sum(parameter.numel() for parameter in d0b.parameters())
        - sum(parameter.numel() for parameter in d0.parameters())
    ) == expected_delta
