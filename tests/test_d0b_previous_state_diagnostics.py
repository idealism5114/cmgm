import torch

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_previous_state_diagnostics import (
    _fixed_derangement,
    replay_latent_trajectory,
    run_previous_state_diagnostics,
)


def _model():
    return HeteroMixHopCMGM(
        num_nodes=9,
        n_commodities=2,
        n_stock=4,
        n_bond=3,
        feat_dim=5,
        variant="switching_latent_balanced_readout",
    ).eval()


def test_d0b_replay_interventions_are_recursive_and_leave_model_unchanged():
    torch.manual_seed(4101)
    model = _model()
    x = torch.randn(4, 20, 9, 5)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = run_previous_state_diagnostics(model, x, seed=73)

    normal = result["normal"]
    zero = result["interventions"]["zero_previous"]
    assert result["H_sanity_max_diff"] == 0.0
    assert result["p_sanity_max_diff"] == 0.0
    torch.testing.assert_close(normal.z[:, 0], zero.z[:, 0], rtol=0.0, atol=0.0)
    assert (normal.z[:, 2:] - zero.z[:, 2:]).abs().max() > 0
    assert result["shuffle_unchanged_fraction"] == 0.0
    assert result["comparison"]["zero_micro"]["prediction"][0] > 0
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0.0, atol=0.0)


def test_lagged_replay_never_uses_a_future_state_and_shuffle_is_deranged():
    torch.manual_seed(4102)
    model = _model()
    branch = model.switching_latent_transformer
    x = torch.randn(5, 20, 9, 5)
    with torch.no_grad():
        branch(x)
        H = branch.last_long_memory
        p = branch.last_regime_probabilities
        normal_z = branch.last_latent_states
        lagged_z, _ = replay_latent_trajectory(branch, H, p, mode="lagged")

    # At t=0 both receive zero; at t=1 lagged is explicitly defined as Z0,
    # which is exactly the normal recurrence input.
    torch.testing.assert_close(lagged_z[:, :2], normal_z[:, :2], atol=1e-7, rtol=0.0)
    assert (lagged_z[:, 2:] - normal_z[:, 2:]).abs().max() > 0
    order = _fixed_derangement(5, 99, x.device)
    assert not torch.any(order == torch.arange(5))
