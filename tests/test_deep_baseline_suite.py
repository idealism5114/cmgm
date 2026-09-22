import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from cmgm.models.comparison_baselines import ORDER,TRAINABLE,MODELS,MODEL_CONFIGS,make_model
from cmgm.scripts.baseline_protocol import sanity,prediction_loss
MI=dict(stock=(0,4),bond=(4,6),commodity=(6,30))


def test_registry_exact_and_no_excluded_models():
    assert ORDER==TRAINABLE==('RNN','GRU','LSTM','VanillaTransformer','GraphWaveNet','MTGNN','MSGNet','CrossGNN')
    assert set(MODELS)==set(ORDER)==set(MODEL_CONFIGS)
    for name in ('Linear','ZeroReturn','Ridge','iTransformer','TCN'):
        with pytest.raises(ValueError):make_model(name,MI)


@pytest.mark.parametrize('name',ORDER)
def test_all_models_sanity_finite_gradients_and_roundtrip(name,tmp_path):
    torch.manual_seed(42);model=make_model(name,MI);x=torch.randn(3,20,30,21);target=torch.randn(3,4,24)*.02
    check=sanity(model,x);assert check['PASS'],check
    model.train();pred=model(x);assert pred.shape==(3,4,24)
    loss=prediction_loss(pred,target);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    model.eval();expected=model(x)
    path=tmp_path/(name+'.pt');torch.save(model.state_dict(),path)
    restored=make_model(name,MI).eval();restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(expected,restored(x),atol=0,rtol=0)
    # Every sample must remain independent when unrelated batch companions change.
    changed=x.clone();changed[1:]=100*torch.randn_like(changed[1:])
    torch.testing.assert_close(expected[0],model(changed)[0],atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('name,attr',[('RNN','rnn'),('GRU','gru'),('LSTM','lstm')])
def test_recurrent_family_final_hidden_and_slots(name,attr):
    m=make_model(name,MI).eval();x=torch.randn(2,20,30,21);seq=m.adapter.sequence(x)
    assert seq.shape==(2,20,588)
    module=getattr(m,attr);assert module.hidden_size==128 and module.num_layers==2 and module.dropout==.1 and not module.bidirectional
    _,hidden=module(seq);h=hidden[0] if name=='LSTM' else hidden
    torch.testing.assert_close(m(x),m.head(h[-1]).reshape(2,4,24))
    with torch.no_grad():m.head.weight.zero_();m.head.bias.copy_(torch.arange(96))
    assert torch.equal(m(x)[0],torch.arange(96).reshape(4,24))


@pytest.mark.parametrize('name',('GraphWaveNet','MTGNN','MSGNet','CrossGNN'))
def test_graph_commodity_row_and_horizon_order(name):
    m=make_model(name,MI).eval();x=torch.randn(2,20,30,21)
    if name=='MTGNN':expected=m.head(m.temporal_states(x)[:,-1,4:]).permute(0,2,1)
    elif name=='GraphWaveNet':expected=m.core(m.adapter(x).permute(0,3,2,1))[:,:,4:,-1]
    else:expected=m.core(m.scalar_input(x))[:,:,4:]
    torch.testing.assert_close(m(x),expected,atol=0,rtol=0)
    if name=='GraphWaveNet':
        with torch.no_grad():m.core.end[-1].weight.zero_();m.core.end[-1].bias.copy_(torch.arange(4))
    elif name=='MTGNN':
        with torch.no_grad():m.head.weight.zero_();m.head.bias.copy_(torch.arange(4))
    else:
        with torch.no_grad():m.core.horizon_head.weight.zero_();m.core.horizon_head.bias.copy_(torch.arange(4))
    torch.testing.assert_close(m(x),torch.arange(4).float()[None,:,None].expand(2,4,24),atol=0,rtol=0)


def test_msgnet_scales_and_crossgnn_graphs_device_independent():
    for name in ('MSGNet','CrossGNN'):
        m=make_model(name,MI).double().eval();x=torch.randn(2,20,30,21,dtype=torch.float64);m(x)
        assert m.scalar_input(x).shape==(2,20,28)
        assert m.node_value_projection.in_features==21 and m.node_value_projection.out_features==1
        assert m.mechanism_sanity()['PASS']
        frequencies=m.core.last['frequencies'];assert (frequencies>0).all() and (frequencies<=10).all()
        if name=='MSGNet':
            assert frequencies.shape==(2,3)
            weights=m.core.last['scale_weights'];torch.testing.assert_close(weights.sum(-1),torch.ones(2,dtype=torch.float64))
            assert len({g.nodevec1.data_ptr() for g in m.core.graphs})==3
        else:
            assert frequencies.shape==(2,4) and m.core.last['periods'].shape==(2,5)
            a=m.core.last['time_adjacency'];assert a.shape==(2,40,40)
            torch.testing.assert_close(a.sum(1),torch.ones(2,40,dtype=torch.float64))
            assert m.core.last['variable_adjacency'].shape==(28,28) and not m.core.anti_ood
            assert all(n in dict(m.core.named_parameters()) for n in ('timevec1','timevec2','nodevec1','nodevec2'))
    from cmgm.models.baselines.msgnet import sample_fft
    freq,amp=sample_fft(torch.zeros(2,20,28),3)
    assert torch.equal(freq,torch.tensor([[1,2,3],[1,2,3]])) and (amp==0).all()
    with pytest.raises(ValueError):sample_fft(torch.randn(2,4,28),3)


def test_causal_gwn_and_transformer_prefix():
    for name in ('GraphWaveNet','VanillaTransformer'):
        m=make_model(name,MI).eval();x=torch.randn(2,20,30,21);z=x.clone();z[:,10:]*=100
        torch.testing.assert_close(m.temporal_states(x)[:,:10],m.temporal_states(z)[:,:10],atol=0,rtol=0)
        if name=='GraphWaveNet':
            a=m.core.adjacency();assert a.shape==(28,28)
            torch.testing.assert_close(a.sum(-1),torch.ones(28))
            assert all(not isinstance(mod,nn.BatchNorm2d) for mod in m.modules())
            assert [l.left_padding for l in m.core.layers]==[1,2]*4


def test_completed_reuse_requires_full_v3_identity_and_hash(tmp_path):
    from cmgm.scripts.baseline_comparison import completed_training,PROTOCOL
    from cmgm.scripts.formal_v2_protocol import sha
    audit=dict(split_fingerprint={'train':'fixture'});sources={'fixture':'sha'};path=tmp_path/'rnn.pt'
    cp=dict(training_complete=True,metadata=dict(model='RNN',configuration=MODEL_CONFIGS['RNN'],protocol=PROTOCOL,data_fingerprint=audit['split_fingerprint'],source_hashes=sources))
    torch.save(cp,path)
    assert completed_training(path,'RNN',audit,sources,sha(path))['training_complete']
    for wrong in (None,'wrong'):
        with pytest.raises(ValueError):completed_training(path,'RNN',audit,sources,wrong)
    with pytest.raises(ValueError):completed_training(path,'RNN',audit,{},sha(path))


def test_report_nine_rows_honest_ranking_and_all_metrics(tmp_path):
    from cmgm.scripts.baseline_comparison import OURS,PROTOCOL
    from cmgm.scripts.baseline_report import write_report
    r=dict(status='SYNTHETIC',protocol=PROTOCOL,models={},model_status={},sanity={})
    for i,name in enumerate((*ORDER,OURS)):
        value=.02+i*.001
        metrics={s:{str(h):dict(MAE=value,MSE=value**2,RMSE=value,Hit=.6) for h in (1,5,10,20)} for s in ('train','val','test')}
        r['models'][name]=dict(metrics=metrics,parameters=dict(trainable=1,nontrainable=0,buffer_elements=0),per_commodity=[dict(commodity=str(c),MAE=value,MSE=value**2) for c in range(24)])
        r['model_status'][name]='COMPLETE';r['sanity'][name]=dict(PASS=True)
    write_report(r,tmp_path)
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==18 and 'Q8.' in text and 'NO; exceptions:' in text
    assert len(r['tables']['overall'])==9 and r['tables']['overall'][0]['DeltaMAE_vs_Ours']<0
    assert len(r['tables']['all_horizons'])==108
    assert len((tmp_path/'per_commodity_5d.csv').read_text().splitlines())==217


def test_ours_provenance_requires_completed_data_and_sanity(tmp_path):
    from cmgm.scripts.baseline_comparison import audit_ours,BASE
    from cmgm.scripts.formal_v2_protocol import sha
    audit=dict(mapping=[dict(commodity='test',full_node=6,target_output=0)],split_fingerprint={'train':'same'},PASS=True)
    data={**audit,'mapping':[dict(commodity='test',node_index=6,target_index=0,output_index=0)]}
    cfg=dict(variant=BASE);md=dict(variant=BASE,data=data,config=cfg,source_hashes={'historical':'source'})
    path=tmp_path/'ours.pt';torch.save(dict(metadata=md,best_epoch=7),path)
    r=dict(status='COMPLETE',trained=True,config=cfg,data=data,source_hashes=md['source_hashes'],checkpoint_sha256=sha(path),sanity=dict(PASS=True),best_sanity=dict(PASS=True),best_checkpoint_metadata=dict(best_epoch=7))
    report=tmp_path/'results.json';report.write_text(json.dumps(r))
    assert all(audit_ours(path,report,audit)[2]['checks'].values())
    r['best_sanity']['PASS']=False;report.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='OURS provenance mismatch'):audit_ours(path,report,audit)


def test_preflight_never_fits_and_resume_verifies_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset,DataLoader
    import cmgm.scripts.baseline_comparison as suite
    from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
    from cmgm.scripts.formal_v2_protocol import sha
    from cmgm.scripts.baseline_protocol import evaluate,loaders
    x,y=torch.randn(4,20,30,21),torch.randn(4,4,24)*.02
    data=dict(n_nodes=30,market_indices=MI,feature_names=[str(i) for i in range(30)],loaders={s:DataLoader(TensorDataset(x,y),batch_size=4) for s in ('train','val','test')})
    audit=dict(mapping=[dict(commodity=str(i),full_node=6+i,target_output=i) for i in range(24)],split_fingerprint={s:{'sha256':'fixture','origins':4,'timeline':44} for s in ('train','val','test')},PASS=True)
    candidate_data={**audit,'mapping':[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]}
    m=HeteroMixHopCMGM(30,24,n_stock=4,n_bond=2,variant=suite.BASE)
    metrics=evaluate(m,loaders(data,full=True),torch.device('cpu'),data['feature_names'][6:])['metrics']
    path=tmp_path/'ours.pt';cfg=dict(variant=suite.BASE)
    md=dict(variant=suite.BASE,data=candidate_data,config=cfg,source_hashes={'native':'fixture'})
    torch.save(dict(model_state_dict=m.state_dict(),metadata=md,best_epoch=1),path)
    report=tmp_path/'ours.json';report.write_text(json.dumps(dict(status='COMPLETE',trained=True,config=cfg,data=candidate_data,source_hashes=md['source_hashes'],checkpoint_sha256=sha(path),sanity={'PASS':True},best_sanity={'PASS':True},best_checkpoint_metadata={'best_epoch':1},evaluation=dict(metrics=metrics))))
    monkeypatch.setattr(suite,'data_audit',lambda d:audit)
    monkeypatch.setattr(suite,'train_one',lambda *a,**k:pytest.fail('Preflight created a fit'))
    args=SimpleNamespace(run=False,sanity_only=True,output_dir=tmp_path/'suite',checkpoint_dir=tmp_path/'baselines',ours_checkpoint=path,ours_results=report,resume=None)
    r=suite.run_suite(args,torch.device('cpu'),data)
    assert r['status']=='PREPARED' and not r['training_executed'] and r['reference_check']['PASS']
    assert all(r['sanity'][n]['PASS'] for n in ORDER) and not args.checkpoint_dir.exists()
    args.resume='latest';before=sha(path);r2=suite.run_suite(args,torch.device('cpu'),data)
    assert r2['status']=='PREPARED' and sha(path)==before
    monkeypatch.setattr(suite,'sources',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='Resume protocol/data/config/source mismatch'):suite.run_suite(args,torch.device('cpu'),data)
