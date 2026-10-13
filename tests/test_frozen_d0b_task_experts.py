"""Only synthetic arrays and temporary checkpoints; no real data or TEST."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.frozen_d0b_task_experts import (
    BASE_VARIANT, FrozenD0BFeatures, FrozenD0BExpert, ResidualPredictionExpert,
    matched_experts, tensor_state_hash, horizon_weights, horizon_losses,
    weighted_objective, combine_strategies,
)
from cmgm.scripts.d0b_supervised_task_experts import evaluate_expert, fit_expert, load_selector
from cmgm.training.train import _prediction_loss


@pytest.fixture
def base():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return HeteroMixHopCMGM(30,24,n_stock=4,n_bond=2,commodity_indices=(6,30),variant=BASE_VARIANT).eval()


def test_original_forward_hook_features_zero_residual_and_frozen_eval(base):
    features=FrozenD0BFeatures(base)
    experts,audit=matched_experts(features.input_dim,24)
    wrapper=FrozenD0BExpert(features,experts['S']).train()
    assert wrapper.training and wrapper.expert.training
    assert not features.training and not any(m.training for m in base.modules())
    x=torch.randn(3,20,30,21)
    digest=tensor_state_hash(base.state_dict())
    with torch.no_grad():original=base(x)
    with patch.object(base,'_temp_weighted_spatial',wraps=base._temp_weighted_spatial) as spatial, \
            patch.object(base.switching_latent_transformer,'forward',wraps=base.switching_latent_transformer.forward) as temporal:
        frozen=features(x)
        assert spatial.call_count==temporal.call_count==1
    assert frozen['u'].shape==(3,192) and frozen['y_base'].shape==(3,4,24)
    torch.testing.assert_close(frozen['y_base'],original,atol=0,rtol=0)
    torch.testing.assert_close(frozen['u'],torch.cat((frozen['s'],frozen['t'],frozen['z_base']),-1),atol=0,rtol=0)
    assert all(not value.requires_grad and not torch.is_inference(value) for value in frozen.values())
    assert audit['PASS'] and audit['max_abs_diff']=={'G':0.,'S':0.,'L':0.}
    for expert in experts.values():
        assert torch.equal(expert.predict(frozen['u'],frozen['y_base']),original)
    assert torch.equal(wrapper(x),original)
    assert tensor_state_hash(base.state_dict())==digest
    assert not any(isinstance(m,(nn.Dropout,nn.LayerNorm)) for m in experts['G'].modules())


def test_explicit_init_copy_independent_parameters_and_preserved_rng():
    torch.manual_seed(7);before=torch.get_rng_state().clone()
    experts,audit=matched_experts(192,24)
    assert torch.equal(before,torch.get_rng_state())
    assert audit['trainable_parameters']==14528
    assert len(set(audit['role_hashes'].values()))==1
    with torch.no_grad():experts['S'].network[0].weight.add_(1)
    assert not torch.equal(experts['G'].network[0].weight,experts['S'].network[0].weight)
    assert torch.equal(experts['G'].network[0].weight,experts['L'].network[0].weight)


def test_output_horizon_major_order_and_input_dimensions():
    expert=ResidualPredictionExpert(192,24)
    with torch.no_grad():expert.network[-1].bias.copy_(torch.arange(96))
    pred=expert(torch.randn(2,192))
    assert pred.shape==(2,4,24)
    for h in range(4):torch.testing.assert_close(pred[0,h],torch.arange(h*24,(h+1)*24,dtype=pred.dtype))
    with pytest.raises(ValueError):expert(torch.randn(2,128))


@pytest.mark.parametrize('criterion',('global','short','long'))
def test_loss_formula_weight_sum_epsilon_one_and_all_horizons(criterion):
    p=torch.linspace(-.09,.1,96).reshape(2,4,12).requires_grad_()
    y=torch.zeros_like(p);terms=horizon_losses(p,y,.02)
    w=horizon_weights(criterion)
    assert sum(w)==pytest.approx(4) and min(w)>0
    original=_prediction_loss(SimpleNamespace(variant=BASE_VARIANT),p,y,nn.HuberLoss(delta=.02))
    assert torch.equal(weighted_objective(terms,criterion,epsilon=1),original)
    ls=(terms[0]+terms[1])/2;ll=(terms[2]+terms[3])/2
    expected={'global':terms.sum(),'short':4/1.1*(ls+.1*ll),'long':4/1.1*(.1*ls+ll)}[criterion]
    torch.testing.assert_close(weighted_objective(terms,criterion),expected)
    weighted_objective(terms,criterion).backward()
    assert torch.isfinite(p.grad).all() and (p.grad.abs().sum((0,2))>0).all()


def test_output_then_internal_gradients_optimizer_scope_and_roundtrip(base,tmp_path):
    features=FrozenD0BFeatures(base);experts,_=matched_experts(192,24)
    model=FrozenD0BExpert(features,experts['S']).train()
    optimizer=torch.optim.Adam(model.expert.parameters(),lr=.001,weight_decay=1e-5)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    initial={k:v.clone() for k,v in model.expert.state_dict().items()}
    base_before=tensor_state_hash(base.state_dict())
    x=torch.randn(3,20,30,21);y=torch.randn(3,4,24)*.03
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred=model(x);loss=weighted_objective(horizon_losses(pred,y,.02),'short')
        assert torch.isfinite(pred).all() and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.expert.parameters())
        if step==0:
            assert model.expert.network[-1].weight.grad.abs().sum()>0
            assert model.expert.network[0].weight.grad.abs().sum()==0
            assert model.expert.network[2].weight.grad.abs().sum()==0
        else:
            assert model.expert.network[0].weight.grad.abs().sum()>0
            assert model.expert.network[2].weight.grad.abs().sum()>0
        optimizer.step()
        assert not any(m.training for m in base.modules())
    assert not torch.equal(initial['network.4.weight'],model.expert.network[-1].weight)
    assert tensor_state_hash(base.state_dict())==base_before and all(p.grad is None for p in base.parameters())
    path=tmp_path/'expert.pt';torch.save(model.expert.state_dict(),path)
    restored=ResidualPredictionExpert(192,24);restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    model.eval();frozen=features(x)
    torch.testing.assert_close(restored.predict(frozen['u'],frozen['y_base']),model(x),atol=0,rtol=0)


def test_six_label_free_strategies_choose_correct_rows():
    base=np.zeros((3,4,2));predictions={k:np.full_like(base,value) for k,value in
        {'G-global':1,'G-short':2,'G-long':3,'S-short':4,'L-long':5}.items()}
    result=combine_strategies(base,predictions)
    assert len(result)==6
    np.testing.assert_array_equal(result['generic_group_control'][:,:2],predictions['G-short'][:,:2])
    np.testing.assert_array_equal(result['generic_group_control'][:,2:],predictions['G-long'][:,2:])
    np.testing.assert_array_equal(result['supervised_task_assignment'][:,:2],predictions['S-short'][:,:2])
    np.testing.assert_array_equal(result['supervised_task_assignment'][:,2:],predictions['L-long'][:,2:])
    torch_result=combine_strategies(torch.from_numpy(base),{k:torch.from_numpy(v) for k,v in predictions.items()})
    for k,v in result.items():np.testing.assert_array_equal(v,torch_result[k].numpy())


def test_reject_jointly_trained_base_and_restore_original_checkpoint_interface(base,tmp_path):
    ck=tmp_path/'base.pt';torch.save(base.state_dict(),ck)
    before=ck.read_bytes();state=tensor_state_hash(base.state_dict())
    restored=HeteroMixHopCMGM(30,24,n_stock=4,n_bond=2,commodity_indices=(6,30),variant=BASE_VARIANT).eval()
    restored.load_state_dict(torch.load(ck,weights_only=True),strict=True)
    x=torch.randn(2,20,30,21)
    with torch.no_grad():torch.testing.assert_close(base(x),restored(x),atol=0,rtol=0)
    FrozenD0BFeatures(base)
    assert tensor_state_hash(base.state_dict())==state and ck.read_bytes()==before
    base.variant='switching_latent_balanced_residual_experts_task_dynamic'
    with pytest.raises(ValueError,match='independent'):FrozenD0BFeatures(base)


def test_three_selectors_global_monitor_full_eval_and_strict_checkpoint(tmp_path):
    torch.manual_seed(2);expert=ResidualPredictionExpert(192,24)
    u=torch.randn(7,192);y0=torch.zeros(7,4,24);y=torch.randn(7,4,24)*.03
    dataset=TensorDataset(u,y0,y)
    train=DataLoader(dataset,batch_size=3,shuffle=False,drop_last=True)
    val=DataLoader(dataset,batch_size=3,shuffle=False,drop_last=False)
    protocol=dict(lr=1e-4,weight_decay=1e-5,max_epochs=3,early_stopping_patience=10)
    config=dict(input_dim=192,n_commodities=24,base_audit={'checkpoint_sha256':'synthetic'},data_audit={'synthetic':True},delta=.02)
    metadata=dict(base_checkpoint_sha256='synthetic',data_audit=config['data_audit'],epsilon=.1,delta=.02,horizons=[1,5,10,20])
    history=fit_expert(expert,train,val,'L',.02,protocol,tmp_path,metadata)
    assert len(history['epochs'])==3
    for row in history['epochs']:
        assert row['train_origins_used']==6 and row['val']['origins']==7
        assert len(row['gradient_diagnostics']['mean_output_weight_grad_L2_by_horizon'])==4
        assert all(g>0 for g in row['gradient_diagnostics']['mean_output_weight_grad_L2_by_horizon'])
    for criterion in ('global','short','long'):
        model,payload=load_selector(tmp_path/f'{criterion}_best.pt',config,'L',criterion,history)
        assert payload['epoch']==1+int(np.argmin([r['val']['selector_batch_mean'][criterion] for r in history['epochs']]))
        score,pred=evaluate_expert(model,val,.02,'cpu')
        assert pred.shape==(7,4,24) and np.isfinite(pred).all()
        assert score['selector_batch_mean'][criterion]==payload['selection_loss_batch_mean']
        # Uneven batches must retain the historical batch-mean, separately named.
        assert score['horizon_Huber_batch_mean']!=score['horizon_Huber_sample_mean_float32_batch_terms']


def test_synthetic_train_export_and_summary_interfaces(base,tmp_path,monkeypatch):
    """Exercise all nine checkpoint/export interfaces without real input data."""
    import cmgm.scripts.d0b_supervised_task_experts as runner
    from cmgm.scripts.d0b_short_long_expert_diagnostics import write_json,sha256
    study=tmp_path/'synthetic_study';study.mkdir()
    features=FrozenD0BFeatures(base)
    frozen=features(torch.randn(7,20,30,21))
    target=np.random.default_rng(4).normal(0,.03,size=(7,4,24)).astype(np.float32)
    packed={f'{split}_{k}':v for split in ('train','val') for k,v in dict(
        u=frozen['u'].numpy(),y_base=frozen['y_base'].numpy(),target=target,
        origin_dates=np.datetime64('2020-01-01')+np.arange(7).astype('timedelta64[D]'),
        commodity_order=np.array([str(i) for i in range(24)])).items()}
    np.savez_compressed(study/'frozen_features.npz',**packed)
    experts,audit=matched_experts(192,24);torch.save(experts['G'].state_dict(),study/'expert_initial_state.pt')
    write_json(study/'initialization_audit.json',audit)
    fixture=tmp_path/'frozen_base.pt';torch.save(base.state_dict(),fixture)
    config=dict(base_variant=BASE_VARIANT,seed=42,epsilon=.1,delta=.02,input_dim=192,n_commodities=24,
        frozen_parameters=sum(p.numel() for p in base.parameters()),trainable_per_expert=14528,
        original_environment={'cudnn_deterministic':True,'cudnn_benchmark':False},
        original_protocol=dict(batch_size=3,train_drop_last=True,lr=1e-4,weight_decay=1e-5,max_epochs=2,early_stopping_patience=10),
        base_audit={'checkpoint_sha256':sha256(fixture),'protected_files_sha256':{str(fixture):sha256(fixture)}},
        data_audit={'commodity_order':[str(i) for i in range(24)],'splits':{'train':{'origins':7},'val':{'origins':7}}},source=runner.source_record())
    write_json(study/'config.json',config)
    monkeypatch.setattr(runner,'prepare',lambda args:study)
    monkeypatch.setattr(runner,'verify_study',lambda path:config)
    monkeypatch.setattr(runner,'CHECKPOINT_ROOT',tmp_path/'checkpoints')
    args=SimpleNamespace(study=study,device='cpu',threads=2,output=tmp_path/'summary')
    for role in ('G','S','L'):
        args.expert=role;runner.train_role(args)
        result,history=runner.validate_run(study,role,config)
        assert result['training_complete'] and not result['test_evaluated']
        assert result['main_checkpoint']=={'G':'global','S':'short','L':'long'}[role]
        assert all(Path.exists() for Path in (tmp_path/'checkpoints'/study.name/role).glob('*_best.pt'))
    runner.summarize(args)
    summary=__import__('json').loads((args.output/'metrics.json').read_text())
    assert len(summary['val'])==6
    assert (args.output/'expert_capability.csv').exists()
    assert (args.output/'checkpoint_index.json').exists()
    assert len(__import__('json').loads((args.output/'checkpoint_index.json').read_text()))==9
