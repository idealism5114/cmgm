"""Synthetic checks only: no dataset, fitting, or TEST access."""
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.horizon_tensor_fusion import predict_by_horizon
from cmgm.models.task_conditioned_moe import (
    ALL_VARIANTS, BASE_VARIANT, CONTENT_VARIANT, GLOBAL_VARIANT,
    MOE_VARIANTS, PER_HORIZON_VARIANT, SHORT_LONG_VARIANT,
)
from cmgm.training.train import _prediction_loss, _switching_branch, train
from cmgm.scripts.d0b_task_conditioned_moe import compare_results, sha256, write_json

KWARGS = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
              commodity_indices=(6, 30))


def make(variant):
    torch.manual_seed(42)
    return HeteroMixHopCMGM(variant=variant, **KWARGS)


def test_shared_initialization_and_original_path():
    base = make(BASE_VARIANT).eval()
    x = torch.randn(2, 20, 30, 21)
    with torch.no_grad():
        expected = base(x)
    for variant in MOE_VARIANTS:
        candidate = make(variant).eval()
        original = base.state_dict()
        current = candidate.state_dict()
        shared = [k for k in original if not k.startswith('gate_fc.')]
        assert all(torch.equal(original[k], current[k]) for k in shared)
        assert not hasattr(candidate, 'gate_fc')
        assert _switching_branch(candidate) is candidate.switching_latent_transformer
        with torch.no_grad():
            output = candidate(x)
        assert output.shape == (2, 4, 24) and torch.isfinite(output).all()
    clone = make(BASE_VARIANT).eval()
    clone.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(clone(x), expected, rtol=0, atol=0)


def test_matched_moe_constructor_rng_state():
    states=[]
    for variant in MOE_VARIANTS:
        make(variant)
        states.append(torch.random.get_rng_state())
    assert all(torch.equal(states[0], state) for state in states[1:])


def test_routing_and_head_rows_and_gradients():
    global_model, grouped = make(GLOBAL_VARIANT).eval(), make(SHORT_LONG_VARIANT).eval()
    grouped.task_moe_fusion.logits.data.copy_(global_model.task_moe_fusion.logits.data.expand(2, -1))
    x = torch.randn(2, 20, 30, 21)
    with torch.no_grad():
        torch.testing.assert_close(global_model(x), grouped(x), rtol=0, atol=1e-7)
        same = torch.randn(2, 64)
        horizon_states = same[:, None].expand(-1, 4, -1)
        torch.testing.assert_close(
            predict_by_horizon(grouped.head, horizon_states, 24),
            grouped.head(same).view(2, 4, 24), atol=3e-7, rtol=0,
        )
        weight = grouped.head[3].weight.view(4, 24, 64)
        body = grouped.head[:3](horizon_states)
        for h in range(4):
            reference = torch.nn.functional.linear(body[:, h], weight[h],
                                                    grouped.head[3].bias.view(4, 24)[h])
            torch.testing.assert_close(predict_by_horizon(grouped.head, horizon_states, 24)[:, h],
                                       reference, atol=3e-7, rtol=0)
        route = grouped.task_moe_fusion.routing_weights(torch.randn(2,64),torch.randn(2,64))
        torch.testing.assert_close(route.sum(-1), torch.ones(2,4), atol=1e-7, rtol=0)
        torch.testing.assert_close(route, torch.full((2,4,3),1/3), atol=1e-7, rtol=0)

    grouped.train()
    prediction = grouped(x)
    target = torch.randn_like(prediction) * 0.02
    grouped.switching_latent_transformer.set_epoch(3)
    pred_loss = _prediction_loss(grouped, prediction, target, nn.HuberLoss(delta=.02))
    manual = sum(nn.HuberLoss(delta=.02)(prediction[:, i], target[:, i]) for i in range(4))
    torch.testing.assert_close(pred_loss, manual)
    kl = grouped.switching_latent_transformer.switch_loss()
    loss = pred_loss + kl
    assert torch.isfinite(loss) and torch.isfinite(kl)
    loss.backward()
    named = dict(grouped.named_parameters())
    for name in ('task_moe_fusion.logits',
                 'task_moe_fusion.temporal_expert.mlp.0.weight',
                 'task_moe_fusion.spatial_expert.mlp.0.weight',
                 'task_moe_fusion.interaction_expert.mlp.0.weight',
                 'gcn_proj.weight', 'lstm_proj.weight', 'head.3.weight'):
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert named['task_moe_fusion.logits'].grad.abs().sum(dim=1).min() > 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'best.pt'
        torch.save(grouped.state_dict(), path)
        restored = make(SHORT_LONG_VARIANT).eval()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        grouped.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(x), grouped(x), rtol=0, atol=0)


def test_content_and_horizon_routes_and_eval_interventions():
    hs, ht = torch.randn(3,64), torch.randn(3,64)
    s, t = torch.randn(3,64), torch.randn(3,64)
    for variant in (CONTENT_VARIANT, PER_HORIZON_VARIANT, SHORT_LONG_VARIANT):
        fusion = make(variant).task_moe_fusion
        fused = fusion(s,t,hs,ht)
        assert fused.shape == (3,4,64)
        assert torch.isfinite(fused).all()
        if variant == CONTENT_VARIANT:
            fusion.content_router[-1].weight.data.normal_(0,.01)
            fusion.content_router[-1].bias.data.normal_(0,.01)
            pi = fusion.routing_weights(hs,ht)
            torch.testing.assert_close(pi[:,0], pi[:,3])
        if variant == SHORT_LONG_VARIANT:
            fusion.eval()
            with fusion.intervention('swap_short_long'):
                swapped = fusion.routing_weights(hs,ht)
            ordinary = fusion.routing_weights(hs,ht)
            torch.testing.assert_close(swapped[:,0], ordinary[:,2])
            with fusion.intervention('spatial'):
                forced = fusion.routing_weights(hs,ht)
            torch.testing.assert_close(forced[...,1], torch.ones(3,4))


def test_val_only_comparison_gate(tmp_path):
    paths=[]
    values={BASE_VARIANT:.023, GLOBAL_VARIANT:.022,
            CONTENT_VARIANT:.0225, SHORT_LONG_VARIANT:.021,
            PER_HORIZON_VARIANT:.0221}
    for variant in ALL_VARIANTS:
        directory=tmp_path/variant
        directory.mkdir()
        checkpoint=directory/'best.pt'
        checkpoint.write_bytes(variant.encode())
        config_record=dict(variant=variant,protocol={'objective':'VAL Huber'},
                           data_audit={'train':'same','val':'same'},git_commit='same',
                           source_sha256={'model':'same'})
        write_json(directory/'config.json',config_record)
        metrics={str(h):{'MAE':values[variant] if h==5 else .03}
                 for h in (1,5,10,20)}
        result=dict(status='COMPLETE',variant=variant,seed=42,
                    test_evaluated=False,checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint),
                    metrics={'val':metrics},best_val_prediction_huber_batch_mean=.01,
                    best_epoch=3,parameters_total=100)
        path=directory/'result.json'
        write_json(path,result)
        paths.append(path)
    output=tmp_path/'comparison.json'
    compare_results(paths,output)
    import json
    assert json.loads(output.read_text())['pass'] is True


def test_existing_trainer_one_synthetic_epoch(tmp_path):
    model=make(SHORT_LONG_VARIANT)
    x=torch.randn(2,20,30,21)
    y=torch.randn(2,4,24)*.01
    loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=False,drop_last=False)
    checkpoint=tmp_path/'best.pt'
    history=train(model,loader,loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),
                  torch.device('cpu'),num_epochs=1,lr=1e-4,weight_decay=1e-5,
                  patience=10,checkpoint_path=str(checkpoint),
                  checkpoint_metadata={'variant':SHORT_LONG_VARIANT})
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert history['best_epoch']==payload['best_epoch']==1
    assert torch.isfinite(torch.tensor(payload['best_val_loss']))
    restored=make(SHORT_LONG_VARIANT).eval()
    restored.load_state_dict(payload['model_state_dict'],strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
