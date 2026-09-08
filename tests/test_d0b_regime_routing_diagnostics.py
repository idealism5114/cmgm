"""Check the intervention boundary against the actual unmodified D0B model."""
import pytest
import torch

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_regime_routing_diagnostics import (
    complete_readout, fixed_permutation, latent_forward_with_routing_intervention,
    normal_reference, routing_distribution, gradient_diagnostics, metrics,
)


@pytest.fixture
def example():
    torch.manual_seed(142)
    model = HeteroMixHopCMGM(6, 2, n_stock=2, n_bond=2, feat_dim=5,
                            variant="switching_latent_balanced_readout").eval()
    x = torch.randn(3, 7, 6, 5)
    with torch.no_grad():
        native, spatial = normal_reference(model, x)
    return model, x, native, spatial


def test_default_and_alpha_half_reproduce_real_d0b_and_preserve_state(example):
    model, x, native, spatial = example
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    branch = model.switching_latent_transformer
    with torch.no_grad():
        for options in ({}, {"sticky_alpha_override": 0.5}):
            trace = latent_forward_with_routing_intervention(branch, native["H"], **options)
            result = complete_readout(model, spatial, trace)
            for key in ("p", "q", "prior", "evidence", "Z", "candidates", "h_long", "h_micro", "h_temporal", "prediction"):
                torch.testing.assert_close(result[key], native[key], rtol=0, atol=1e-7)
        for alpha in (0., .25, .75, 1.):
            trace = latent_forward_with_routing_intervention(branch, native["H"], sticky_alpha_override=alpha)
            torch.testing.assert_close(trace["q"], trace["p"], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            assert torch.equal(value, snapshot[key])
        torch.testing.assert_close(model(x), native["prediction"], rtol=0, atol=0)


@pytest.mark.parametrize("mode,temperature", [("native", .5), ("native", .75), ("native", 2.),
    ("hard", 1.), ("uniform", 1.), ("lag", 1.), ("shuffle_unrestricted", 1.), ("state1", 1.)])
def test_routing_never_changes_posterior_and_replays_own_previous_z(example, mode, temperature):
    model, x, native, spatial = example
    branch = model.switching_latent_transformer
    permutation = fixed_permutation(len(x), 42, x.device)
    with torch.no_grad():
        result = latent_forward_with_routing_intervention(branch, native["H"], mode=mode,
            routing_temperature=temperature, permutation=permutation)
        for key in ("p", "prior", "evidence"):
            torch.testing.assert_close(result[key], native[key], rtol=0, atol=0)
        # Independent direct G evaluations verify recurrence, not just final mixture.
        for t in (0, 1, 6):
            previous = torch.zeros_like(result["Z"][:, 0]) if t == 0 else result["Z"][:, t - 1]
            joined = torch.cat([native["H"][:, t], previous], -1)
            candidates = torch.stack([g(joined) for g in branch.latent_transition.generators], 1)
            torch.testing.assert_close(candidates, result["candidates"][:, t], rtol=0, atol=0)
            expected = torch.einsum("bk,bkd->bd", result["q"][:, t], candidates)
            torch.testing.assert_close(expected, result["Z"][:, t], rtol=0, atol=0)
        if mode == "lag":
            torch.testing.assert_close(result["q"][:, 0], torch.full_like(result["q"][:, 0], 1/3))
            torch.testing.assert_close(result["q"][:, 1:], native["p"][:, :-1])
        if mode == "state1":
            forced = native["p"].new_tensor([0., 1., 0.])
            _, z, candidates = branch.latent_forward(native["H"], forced_probabilities=forced)
            torch.testing.assert_close(result["Z"], z, rtol=0, atol=0)


def test_alpha_identity_is_not_a_frozen_regime_and_future_does_not_change_prefix(example):
    model, x, native, _ = example
    branch = model.switching_latent_transformer
    with torch.no_grad():
        result = latent_forward_with_routing_intervention(branch, native["H"], sticky_alpha_override=1.)
        torch.testing.assert_close(result["A"], torch.eye(3))
        torch.testing.assert_close(result["prior"][:, 1:], result["p"][:, :-1])
        assert (result["p"][:, 1:] - result["p"][:, :-1]).abs().max() > 1e-5
        changed = native["H"].clone()
        changed[:, 4:] += 10
        later = latent_forward_with_routing_intervention(branch, changed, sticky_alpha_override=1.)
        for key in ("p", "q", "Z"):
            torch.testing.assert_close(result[key][:, :4], later[key][:, :4], rtol=0, atol=0)


def test_shuffle_uses_whole_donor_trajectory_and_causal_mode_masks_future_donors():
    p = torch.tensor([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    permutation = torch.tensor([2, 0, 1])
    q = routing_distribution(p, mode="shuffle_unrestricted", permutation=permutation)
    torch.testing.assert_close(q, p[permutation])
    causal = routing_distribution(p, mode="shuffle", permutation=permutation, sample_starts=torch.arange(3))
    torch.testing.assert_close(causal[0], p[0])
    torch.testing.assert_close(causal[1:], p[permutation][1:])


def test_prediction_only_gradients_do_not_update_or_leave_parameter_grads(example):
    model, x, native, _ = example
    state = {k: v.clone() for k, v in model.state_dict().items()}
    y = torch.randn_like(native["prediction"]) * .03
    result = gradient_diagnostics(model, x, y)
    assert set(result["norms"]) == {"1", "5", "10", "20"}
    for values in result["norms"].values():
        assert values["regime evidence"] > 0
        assert values["transition logits"] > 0
        assert all(values[f"G{k}"] > 0 for k in range(3))
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_point_metrics_match_existing_evaluation_convention():
    import numpy as np
    from cmgm.training.evaluate import compute_metrics
    rng = np.random.default_rng(91)
    target = rng.normal(size=(13, 3)).astype(np.float32)
    pred = rng.normal(size=(13, 3)).astype(np.float32)
    actual, original = metrics(pred, target), compute_metrics(pred, target)
    for key in ("MAE", "RMSE"):
        np.testing.assert_equal(actual[key], original[key])
    np.testing.assert_equal(actual["Hit"], original["Hit_Ratio"])
