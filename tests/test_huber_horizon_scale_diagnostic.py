"""Residual population/Huber boundary and inference-integrity regressions."""
from types import SimpleNamespace
import numpy as np
import torch

from cmgm.scripts.d0b_huber_horizon_scale_diagnostic import (
    residual_statistics, split_statistics, robust_scale, network_gradients, VARIANT,
)
from cmgm.training.metric_standard import population_metrics
from cmgm.training.evaluate import compute_metrics
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM


def test_population_rmse_and_zero_target_hit():
    p = np.array([[0., 4.], [0., 0.]])
    y = np.zeros_like(p)
    m = population_metrics(p, y)
    assert m == {'MAE':1., 'MSE':4., 'RMSE':2., 'Hit':.75}
    report = compute_metrics(p, y)
    assert report['MSE'] == 4 and report['RMSE'] == 2
    assert report['RMSE_mean_asset_legacy'] != report['RMSE']
    assert report['Hit_Ratio'] == .75


def test_huber_regions_and_analytic_gradients_match_torch_at_boundary():
    p = torch.tensor([-.04,-.02,-.01,0,.01,.02,.04],dtype=torch.float64,requires_grad=True)
    y = torch.zeros_like(p)
    loss = torch.nn.functional.huber_loss(p,y,delta=.02,reduction='sum')
    g, = torch.autograd.grad(loss,p)
    r = residual_statistics(p.detach().numpy(),y.detach().numpy())
    assert np.isclose(r['huber_loss'],loss.item()/len(p))
    assert np.isclose(r['gradient']['mean_abs'],g.abs().mean().item())
    assert np.isclose(r['gradient']['saturated_fraction'],4/7)
    assert np.isclose(r['linear_fraction'],2/7)
    assert np.isclose(r['boundary_fraction'],2/7)
    assert r['sanity']['saturation_error_corrected_for_boundary'] < 1e-15


def test_population_losses_not_unweighted_batch_means_and_dynamic_horizons():
    p = np.zeros((65,4,2)); p[-1] = .1
    y = np.zeros_like(p)
    r = split_statistics(p,y,['a','b'],SimpleNamespace(variant=VARIANT))
    expected = .02*(.1-.01)/65
    assert np.isclose(r['horizons']['5']['huber_loss'],expected)
    assert np.isclose(r['sum_huber'],4*expected)
    assert r['training_helper_error'] < 1e-15
    assert sum(r['horizons'][h]['huber_share'] for h in r['horizons']) == 1


def test_train_robust_scale_uses_centered_mad():
    s=robust_scale(np.array([10.,11.,12.]))
    assert s['MAD']==1 and s['robust_std']==1.4826


def test_network_probe_covers_parameters_and_has_no_grad_or_state_mutation():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(9,3,n_stock=3,n_bond=3,feat_dim=21,variant=VARIANT).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    report=network_gradients(model,(torch.randn(2,20,9,21),torch.randn(2,4,3)*.02),'cpu')
    assert set(report['norms'])=={'1','5','10','20'}
    assert {'spatial branch','fusion/gate','prediction head'} <= set(report['norms']['5'])
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(not m.training for m in model.modules())


def test_future_ablation_reporting_keeps_all_horizons_and_tail():
    from cmgm.scripts.main_ablation import evaluate_horizon_metrics
    from torch.utils.data import DataLoader, TensorDataset
    class Fixed(torch.nn.Module):
        def forward(self,x,market_descriptor=None):
            return x
    p=torch.zeros(65,4,2);p[-1]=torch.tensor([1.,2.,3.,4.])[:,None]
    y=torch.zeros_like(p)
    result=evaluate_horizon_metrics(Fixed(),DataLoader(TensorDataset(p,y),batch_size=64),'cpu')
    assert set(result)=={'1','5','10','20'}
    for i,h in enumerate(('1','5','10','20')):
        assert np.isclose(result[h]['MSE'],(i+1)**2/65)
        assert np.isclose(result[h]['RMSE']**2,result[h]['MSE'])
        assert set(result[h])=={'MAE','MSE','RMSE','Hit'}
