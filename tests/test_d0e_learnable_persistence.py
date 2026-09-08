"""Synthetic logic tests only; no market-data experiment or real training."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.switching_latent_transformer import MarkovRegimeFilter
from cmgm.scripts.d0e_diagnostics import (
    BASE_VARIANT, VARIANT, LOGIT_KEY, comparison_report, fixed_sanity,
    gradient_probe, shared_initialization_check, transition_diagnostics,
)


def make(variant=VARIANT):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(6, 2, n_stock=2, n_bond=2, feat_dim=5, variant=variant)


def batch():
    generator = torch.Generator().manual_seed(174)
    return torch.randn(3, 7, 6, 5, generator=generator), torch.randn(3, 4, 2, generator=generator) * .03


def test_only_one_parameter_and_real_dimension_initialization_equality():
    torch.manual_seed(42)
    model = HeteroMixHopCMGM(284, 24, n_stock=248, n_bond=12, variant=VARIANT)
    x = torch.randn(2, 20, 284, 21)
    rng = torch.get_rng_state().clone()
    result = shared_initialization_check(model, x)
    assert result["PASS"]
    assert result["D0B_params"] == 520549
    assert result["D0E_params"] == 520550
    assert all(v == 0 for v in result["forward_max_diffs"].values())
    assert torch.equal(rng, torch.get_rng_state())


def test_scalar_is_only_active_alpha_and_legacy_state_dict_is_unchanged():
    old = MarkovRegimeFilter(d_model=8, sticky_alpha=.7)
    assert isinstance(old.sticky_alpha_value(), float)
    assert not hasattr(old, "sticky_logit")
    with torch.no_grad():
        old.transition_logits.copy_(torch.randn(3, 3))
    expected = .7 * torch.eye(3) + .3 * old.transition_logits.softmax(-1)
    # Preserve the literal old expression and floating-point operation order.
    expected = old.sticky_alpha * torch.eye(3) + (1. - old.sticky_alpha) * old.transition_logits.softmax(-1)
    torch.testing.assert_close(old.transition_matrix(), expected, rtol=0, atol=0)
    new = MarkovRegimeFilter(d_model=8, sticky_alpha=.9, learnable_sticky_alpha=True)
    assert new.sticky_logit.shape == torch.Size([])
    assert new.sticky_logit.item() == 0
    assert new.sticky_alpha_value().item() == .5
    before = new.transition_matrix().detach().clone()
    new.sticky_alpha = .1  # The legacy float must not participate in D0E.
    torch.testing.assert_close(before, new.transition_matrix(), rtol=0, atol=0)


def test_alpha_grad_matches_finite_difference_and_transition_decomposition():
    filtering = MarkovRegimeFilter(d_model=4, learnable_sticky_alpha=True).double()
    with torch.no_grad():
        filtering.sticky_logit.fill_(.3)
        filtering.transition_logits.copy_(torch.randn(3,3,dtype=torch.float64) * .1)
    H = torch.randn(2,5,4,dtype=torch.float64)
    def objective():
        p = torch.full((2,3),1/3,dtype=torch.float64)
        for t in range(5):
            _,_,p,_ = filtering.step(H[:,t],p,filtering.transition_matrix())
        return (p * torch.tensor([.1,.7,-.2],dtype=torch.float64)).sum()
    analytic = torch.autograd.grad(objective(),filtering.sticky_logit)[0].item()
    step = 1e-5
    with torch.no_grad():
        filtering.sticky_logit.fill_(.3 + step); positive = objective().item()
        filtering.sticky_logit.fill_(.3 - step); negative = objective().item()
    assert abs(analytic) > 1e-7
    assert analytic == pytest.approx((positive-negative)/(2*step),rel=1e-5,abs=1e-8)
    model = make()
    with torch.no_grad():
        model.switching_latent_transformer.regime_filter.sticky_logit.fill_(.7)
        model.switching_latent_transformer.regime_filter.transition_logits.add_(torch.randn(3,3)*.1)
    result = transition_diagnostics(model, np.zeros((3,3)).tolist())
    assert result["decomposition_max_error"] < 1e-7


def test_gradient_probe_preserves_rng_grads_modes_and_reaches_alpha():
    model = make().train()
    model.switching_latent_transformer.set_epoch(20)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = gradient_probe(model,batch(),"unit-test",per_horizon=True)
    assert model.training
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in model.state_dict().items():
        assert torch.equal(value,state[key])
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    assert result["gradients"]["sticky_logit"]["prediction_only_norm"] > 0
    assert result["gradients"]["transition logits"]["prediction_only_norm"] > 0
    assert result["total_loss"] > result["prediction_loss"]
    for h in ("1","5","10","20"):
        v = result["per_horizon"][h]
        assert v["dL_d_alpha"] == pytest.approx(v["dL_d_sticky_logit"]/.25)


def test_causality_batch_and_market_invariance_with_nondefault_learned_alpha():
    model = make()
    with torch.no_grad():
        model.switching_latent_transformer.regime_filter.sticky_logit.fill_(1.2)
    result = fixed_sanity(model,batch()[0])
    assert result["PASS"]
    assert all(v == 0 for v in result["causality"].values())


def test_variant_registration_checkpoint_path_and_protocol_defaults(tmp_path, monkeypatch):
    import sys
    from cmgm.scripts.main_ablation import VARIANTS, D_SERIES_VARIANTS, select_variants, _checkpoint_path_for_variant, parse_args
    assert VARIANT in D_SERIES_VARIANTS
    assert ("D0E-LearnablePersistence",VARIANT) in VARIANTS
    assert select_variants("D0E-LearnablePersistence") == [("D0E-LearnablePersistence",VARIANT)]
    assert _checkpoint_path_for_variant(VARIANT,tmp_path).name == VARIANT+"_best.pt"
    monkeypatch.setattr(sys,"argv",["main_ablation","--variants","D0E-LearnablePersistence"])
    args=parse_args()
    assert (args.epochs,args.patience,args.seed,args.batch_size,args.seq_len)==(200,10,42,64,20)
    assert (config.LEARNING_RATE,config.WEIGHT_DECAY,config.HUBER_DELTA)==(1e-4,1e-5,.02)
    b=make().switching_latent_transformer
    assert b.balanced_readout and b.regime_filter.learnable_sticky_alpha
    assert not any((b.use_latent_memory,b.use_dynamic_slope,b.use_balanced_transition_input,b.use_regime_relative_memory))


def test_history_best_restore_metadata_and_single_optimizer_group_without_training(tmp_path,monkeypatch):
    """Mock epoch computations; neither optimizer.step nor data training runs."""
    training=importlib.import_module("cmgm.training.train")
    model=make(); filtering=model.switching_latent_transformer.regime_filter
    x,y=batch(); loader=DataLoader(TensorDataset(x,y),batch_size=3)
    epoch=[0]; group_checks=[]
    def fake_train(model,loader,ei,ew,optimizer,criterion,device,debug=False):
        epoch[0]+=1
        with torch.no_grad(): filtering.sticky_logit.fill_(epoch[0]/10)
        group_checks.append(len(optimizer.param_groups)==1 and optimizer.param_groups[0]["weight_decay"]==1e-5
                            and any(p is filtering.sticky_logit for p in optimizer.param_groups[0]["params"]))
        return 1/epoch[0]
    losses=iter([7.,6.,5.,4.,3.,2.,1.,2.,3.,4.])
    monkeypatch.setattr(training,"train_epoch",fake_train)
    monkeypatch.setattr(training,"validate_epoch",lambda *a,**kw:next(losses))
    def forbidden_step(*a,**kw): raise AssertionError("unit test must not optimize")
    monkeypatch.setattr(torch.optim.Adam,"step",forbidden_step)
    path=tmp_path/'d0e.pt'
    callback_stages=[]
    def callback(model,stage):
        callback_stages.append(stage)
        return {"alpha":float(filtering.sticky_alpha_value().detach())}
    history=training.train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.zeros(0),torch.device('cpu'),
                           num_epochs=12,patience=3,checkpoint_path=str(path),checkpoint_metadata={'git_sha':'synthetic','seed':42},epoch_diagnostic=callback)
    assert len(history['alpha_history'])==10 and history['best_epoch']==7 and history['final_epoch']==10
    assert callback_stages==['epoch1','epoch5','epoch10','best(epoch 7)']
    assert history['best_sticky_logit']==pytest.approx(.7)
    assert history['final_epoch_sticky_logit']==pytest.approx(1.)
    assert filtering.current_epoch==7
    assert all(group_checks)
    saved=torch.load(path,weights_only=True)
    assert saved['metadata']['sticky_logit']==history['best_sticky_logit']
    assert saved['metadata']['final_learned_alpha']==history['best_alpha']
    assert saved['metadata']['seed']==42 and saved['metadata']['git_sha']=='synthetic'
    assert saved['metadata']['parameter_count']==sum(p.numel() for p in model.parameters())
    restored=make();restored.load_state_dict(saved['model_state_dict'],strict=True)
    torch.testing.assert_close(restored.state_dict()[LOGIT_KEY],model.state_dict()[LOGIT_KEY],rtol=0,atol=0)


def test_synthetic_checkpoint_report_smoke_has_all_required_tables(tmp_path):
    model=make();baseline=make(BASE_VARIANT)
    x,y=batch();loaders={s:DataLoader(TensorDataset(x,y),batch_size=3,shuffle=False) for s in ('train','val','test')}
    baseline_file=tmp_path/'baseline.pt'
    torch.save({'model_state_dict':baseline.state_dict(),'best_epoch':1},baseline_file)
    payload={'best_epoch':1,'history':{},'metadata':{'variant':VARIANT,'initial_transition_logits':np.zeros((3,3)).tolist()}}
    report=comparison_report(model,payload,{'loaders':loaders},baseline_file,tmp_path/'report')
    assert report['models']['D0E']['params']-report['models']['D0B']['params']==1
    assert report['models']['D0E']['sanity']['PASS']
    text=(tmp_path/'report/REPORT.md').read_text()
    for title in ('表 1','表 2','表 3','表 4','表 5','表 6','六个问题'):
        assert title in text
    for split in ('TRAIN','VAL','TEST'):
        assert 'candidate_specialization' in report['models']['D0E']['splits'][split]
    assert 'sticky_logit' in report['models']['D0E']['gradients']['gradients']


def test_d0e_entrypoint_and_legacy_diagnostics_integrate_without_running_training(tmp_path,monkeypatch):
    """Exercise the full new orchestration; replace train() with a serializer."""
    from cmgm.scripts.main_ablation import run_variant
    training=importlib.import_module('cmgm.training.train')
    # Reproduce visible GPUs with an unusable driver: CPU setup must never
    # request CUDA RNG state (fork_rng's default would initialize both GPUs).
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    def forbidden_cuda_rng(*args, **kwargs):
        raise AssertionError('CPU diagnostic setup must not initialize CUDA RNG')
    monkeypatch.setattr(torch.cuda, 'get_rng_state', forbidden_cuda_rng)
    torch.manual_seed(11)
    x=torch.randn(3,20,6,21);y=torch.randn(3,4,2)*.03
    loaders={s:DataLoader(TensorDataset(x,y),batch_size=3) for s in ('train','val','test')}
    data={'n_nodes':6,'n_commodities':2,'market_indices':{'stock':(0,2),'bond':(2,4),'commodity':(4,6)},
          'loaders':loaders,'norm_stats':{'mean':np.ones(6),'std':np.ones(6)},'raw_prices_test':np.ones((50,6))}
    torch.manual_seed(42)
    baseline=HeteroMixHopCMGM(6,2,n_stock=2,n_bond=2,feat_dim=21,variant=BASE_VARIANT)
    baseline_path=tmp_path/'baseline.pt'
    torch.save({'model_state_dict':baseline.state_dict(),'best_epoch':1},baseline_path)
    calls=[]
    def fake_training(model,*a,**kw):
        calls.append(model.variant)
        assert kw['num_epochs']==200 and kw['patience']==10
        assert 'lr' not in kw and 'weight_decay' not in kw
        history={'best_epoch':1,'final_epoch':1,'alpha_history':[.5],'sticky_logit_history':[0.],
                 'train_loss':[1.],'val_loss':[1.],'lr_history':[1e-4],'switch_beta':[0.],'epoch_diagnostics':{}}
        torch.save({'model_state_dict':model.state_dict(),'history':history,'best_epoch':1,
                    'best_val_loss':1.,'metadata':kw['checkpoint_metadata']},kw['checkpoint_path'])
        return history
    monkeypatch.setattr(training,'train',fake_training)
    args=SimpleNamespace(seed=42,seq_len=20,epochs=200,patience=10,batch_size=64,
                         d0b_checkpoint=baseline_path,checkpoint_dir=tmp_path/'checkpoints',d0e_report_dir=tmp_path/'reports')
    result=run_variant('D0E-LearnablePersistence',VARIANT,args,torch.device('cpu'),data)
    assert calls==[VARIANT]
    assert result['alpha']==.5
    assert result['variant']=='D0E-LearnablePersistence'
    assert (args.checkpoint_dir/(VARIANT+'_best.pt')).is_file()
    assert result['diagnostics']['transition_drift']==0
    assert 'REPORT.md' in result['report_path']
