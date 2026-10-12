"""Synthetic-only four-task MoE checks; never load market data or TEST."""
from unittest.mock import patch
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.four_task_moe import (
    FILM_VARIANT, FOUR_TASK_VARIANTS, GLOBAL_VARIANT, PER_HORIZON_VARIANT,
    predict_with_task_film,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.horizon_tensor_fusion import predict_by_horizon
from cmgm.models.task_conditioned_moe import BASE_VARIANT, GLOBAL_VARIANT as THREE_GLOBAL
from cmgm.training.train import _prediction_loss, _switching_branch, train
from cmgm.scripts.d0b_four_task_moe import (
    VARIANTS, compare_results, evaluate, parameters, route_statistics,
)
from cmgm.scripts.d0b_task_conditioned_moe import sha256, write_json

KWARGS = dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,commodity_indices=(6,30))


def make(variant):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(variant=variant,**KWARGS)


def test_live_sequences_and_original_d0b_behavior():
    base=make(BASE_VARIANT).eval()
    candidate=make(PER_HORIZON_VARIANT).eval()
    x=torch.randn(2,20,30,21)
    with torch.no_grad():
        original=base(x)
        ordinary=base.switching_latent_transformer(x)
        live,memory,latent=base.switching_latent_transformer(x,return_live_sequences=True)
    torch.testing.assert_close(live,ordinary,rtol=0,atol=0)
    assert memory.shape==(2,20,128) and latent.shape==(2,20,64)
    torch.testing.assert_close(base(x),original,rtol=0,atol=0)
    shared=[k for k in base.state_dict() if not k.startswith('gate_fc.')]
    assert all(torch.equal(base.state_dict()[k],candidate.state_dict()[k]) for k in shared)
    assert not hasattr(candidate,'gate_fc')
    assert _switching_branch(candidate) is candidate.switching_latent_transformer
    base_clone=make(BASE_VARIANT).eval()
    base_clone.load_state_dict(base.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(base_clone(x),original,rtol=0,atol=0)


def test_experts_router_head_and_film_equivalence():
    models={v:make(v).eval() for v in FOUR_TASK_VARIANTS}
    x=torch.randn(2,20,30,21)
    with torch.no_grad():
        outputs={v:m(x) for v,m in models.items()}
    assert all(y.shape==(2,4,24) and torch.isfinite(y).all() for y in outputs.values())
    torch.testing.assert_close(outputs[GLOBAL_VARIANT],outputs[PER_HORIZON_VARIANT],atol=3e-7,rtol=0)
    torch.testing.assert_close(outputs[PER_HORIZON_VARIANT],outputs[FILM_VARIANT],atol=3e-7,rtol=0)
    for variant,model in models.items():
        route=model.four_task_moe_fusion.routing_weights()
        assert route.shape==(4,4)
        torch.testing.assert_close(route.sum(-1),torch.ones(4),atol=1e-7,rtol=0)
        if variant!=FILM_VARIANT:
            assert not hasattr(model.four_task_moe_fusion,'film_scale_delta')
    reference=models[GLOBAL_VARIANT].four_task_moe_fusion.state_dict()
    for variant in (PER_HORIZON_VARIANT,FILM_VARIANT):
        current=models[variant].four_task_moe_fusion.state_dict()
        assert all(torch.equal(value,current[key]) for key,value in reference.items()
                   if key.startswith(('recent_expert.','long_expert.',
                                      'spatial_expert.','interaction_expert.')))
    module=models[PER_HORIZON_VARIANT].four_task_moe_fusion
    expected=(8256,16448,4160,3072)
    actual=tuple(sum(p.numel() for p in expert.parameters()) for expert in
                 (module.recent_expert,module.long_expert,module.spatial_expert,module.interaction_expert))
    assert actual==expected
    shared=torch.randn(2,64)
    fused=shared[:,None].expand(-1,4,-1)
    with torch.no_grad():
        torch.testing.assert_close(predict_by_horizon(models[GLOBAL_VARIANT].head,fused,24),
                                   models[GLOBAL_VARIANT].head(shared).view(2,4,24),atol=3e-7,rtol=0)
        torch.testing.assert_close(predict_with_task_film(
            models[FILM_VARIANT].head,fused,
            models[FILM_VARIANT].four_task_moe_fusion.film_scale_delta,
            models[FILM_VARIANT].four_task_moe_fusion.film_shift,24),
            predict_by_horizon(models[FILM_VARIANT].head,fused,24),atol=3e-7,rtol=0)
    # Change just one task's route; no other horizon's head rows can change.
    model=models[PER_HORIZON_VARIANT]
    model.four_task_moe_fusion.logits.data[1]=torch.tensor((1.,-1.,0.,0.))
    with torch.no_grad():
        changed=model(x)
    for index in (0,2,3):
        torch.testing.assert_close(changed[:,index],outputs[PER_HORIZON_VARIANT][:,index],atol=1e-7,rtol=0)
    assert not torch.allclose(changed[:,1],outputs[PER_HORIZON_VARIANT][:,1])


def test_live_gradient_single_pass_and_native_switch_kl():
    model=make(PER_HORIZON_VARIANT).train()
    x=torch.randn(2,20,30,21)
    captured={}
    original=model.four_task_moe_fusion.forward
    def spy(s,t,memory,latent):
        memory.retain_grad();latent.retain_grad()
        captured.update(memory=memory,latent=latent)
        return original(s,t,memory,latent)
    model.four_task_moe_fusion.forward=spy
    calls=[]
    hook=model.switching_latent_transformer.register_forward_hook(lambda *unused:calls.append(1))
    with patch.object(model,'_temp_weighted_spatial',wraps=model._temp_weighted_spatial) as spatial:
        prediction=model(x)
        assert spatial.call_count==1
    hook.remove()
    assert len(calls)==1 and prediction.shape==(2,4,24)
    assert captured['memory'].requires_grad and captured['latent'].requires_grad
    target=torch.randn_like(prediction)*.01
    model.switching_latent_transformer.set_epoch(3)
    pred_loss=_prediction_loss(model,prediction,target,nn.HuberLoss(delta=.02))
    native_kl=model.switching_latent_transformer.switch_loss()
    total=pred_loss+native_kl
    assert torch.isfinite(total) and torch.isfinite(native_kl)
    total.backward()
    for key in ('memory','latent'):
        grad=captured[key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,key
    fusion=model.four_task_moe_fusion
    assert fusion.logits.grad is not None and fusion.logits.grad.abs().sum(dim=1).min()>0
    for expert in (fusion.recent_expert,fusion.long_expert,fusion.spatial_expert,fusion.interaction_expert):
        grads=[p.grad for p in expert.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    for layer in (model.gcn_proj,model.lstm_proj,model.head[0],model.head[3]):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0


def test_swap_and_checkpoint_training_entry(tmp_path):
    model=make(PER_HORIZON_VARIANT).eval()
    fusion=model.four_task_moe_fusion
    fusion.logits.data[0]=torch.tensor((1.,0.,0.,0.))
    fusion.logits.data[2]=torch.tensor((0.,1.,0.,0.))
    original=fusion.routing_weights().clone()
    with fusion.swap_tasks(0,2):
        swapped=fusion.routing_weights()
        torch.testing.assert_close(swapped[0],original[2])
        torch.testing.assert_close(swapped[2],original[0])
    torch.testing.assert_close(fusion.routing_weights(),original)
    model=make(PER_HORIZON_VARIANT)
    x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.01
    loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=False)
    checkpoint=tmp_path/'best.pt'
    history=train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),
                  torch.device('cpu'),num_epochs=1,lr=1e-4,weight_decay=1e-5,
                  patience=10,checkpoint_path=str(checkpoint),
                  checkpoint_metadata={'variant':PER_HORIZON_VARIANT})
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['best_epoch']==history['best_epoch']==1
    restored=make(PER_HORIZON_VARIANT).eval()
    restored.load_state_dict(payload['model_state_dict'],strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)


def test_film_gradients_and_parameter_delta():
    film=make(FILM_VARIANT)
    plain=make(PER_HORIZON_VARIANT)
    assert parameters(film)['total']-parameters(plain)['total']==512
    prediction=film(torch.randn(2,20,30,21))
    prediction.square().mean().backward()
    for name in ('film_scale_delta','film_shift'):
        grad=getattr(film.four_task_moe_fusion,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_route_statistics_and_synthetic_val_evaluator():
    global_model=make(GLOBAL_VARIANT).eval()
    stats=route_statistics(global_model)
    assert stats['weights_by_horizon']==[[.25]*4]*4
    assert all(abs(value)<1e-12 for value in stats['pairwise_js_nats'].values())
    task=make(PER_HORIZON_VARIANT).eval()
    task.four_task_moe_fusion.logits.data[0,0]=2.
    task.four_task_moe_fusion.logits.data[1,1]=2.
    stats=route_statistics(task)
    assert stats['pairwise_js_nats']['1d_vs_5d']>0
    x=torch.randn(3,20,30,21);y=torch.randn(3,4,24)*.01
    loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=False,drop_last=False)
    metrics=evaluate(task,loader,torch.device('cpu'))
    assert set(metrics)=={'1','5','10','20'}
    assert all(set(values)=={'MAE','MSE','RMSE','Hit'} for values in metrics.values())


def test_seed42_val_gate_uses_only_main_and_matched_global(tmp_path):
    paths=[]
    values={variant:.023 for variant in VARIANTS}
    values[BASE_VARIANT]=.022
    values[GLOBAL_VARIANT]=.0218
    values[PER_HORIZON_VARIANT]=.0216
    for variant in VARIANTS:
        directory=tmp_path/variant
        directory.mkdir()
        checkpoint=directory/'best.pt'
        checkpoint.write_bytes(variant.encode())
        config=dict(protocol={'loss':'same'},data_audit={'train':'same','val':'same'},
                    git_commit='same',source_sha256={'source':'same'},seed=42)
        write_json(directory/'config.json',config)
        result=dict(status='COMPLETE',variant=variant,seed=42,test_evaluated=False,
                    checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                    metrics={'val':{str(h):{'MAE':values[variant]} for h in (1,5,10,20)}},
                    best_val_prediction_huber_batch_mean=.01,best_epoch=3,
                    parameters={'total':100},train_seconds=2.)
        path=directory/'result.json'
        write_json(path,result)
        paths.append(path)
    output=tmp_path/'comparison.json'
    compare_results(paths,output)
    assert json.loads(output.read_text())['pass'] is True
