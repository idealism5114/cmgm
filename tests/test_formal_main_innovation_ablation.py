import ast
import inspect
from pathlib import Path
import pytest
import torch
from cmgm.models.formal_d0b_main_ablation import NAMES, NEW, MainInnovationAblation
from cmgm.scripts.formal_main_ablation_audit import init_audit,sanity

DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))

@pytest.mark.parametrize('name',NAMES)
def test_candidate_shared_initialization_and_structural_checks(name):
    x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.02
    m,init=init_audit(name,DATA,torch.device('cpu'),x)
    assert init['PASS'] and init['shared_parameter_initial_max_abs_diff']==0
    before={k:v.clone() for k,v in m.state_dict().items()}
    check=sanity(m,x,y)
    assert check['PASS'],check
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())


def test_shared_transition_is_not_uniform_routing_and_retains_kl():
    m=MainInnovationAblation('w/o Regime-Specific Transitions',DATA).eval()
    b=m.switching_latent_transformer;b.set_epoch(20)
    m(torch.randn(2,20,30,21))
    assert (b.last_regime_probabilities-1/3).abs().max()>1e-4
    c=b.last_latent_candidates
    assert torch.equal(c[:,:,0],c[:,:,1]) and torch.equal(c[:,:,0],c[:,:,2])
    assert m.auxiliary_loss().requires_grad and m.auxiliary_loss().item()>0


def test_new_trainer_is_exact_previous_training_protocol():
    from cmgm.scripts.formal_ablation_study import fit as old
    from cmgm.scripts.formal_main_innovation_ablation import fit as new
    assert ast.dump(ast.parse(inspect.getsource(new)))==ast.dump(ast.parse(inspect.getsource(old)))


def test_reuse_never_enters_training(tmp_path):
    from cmgm.scripts.formal_main_innovation_ablation import run_one
    with pytest.raises(AssertionError,match='never retrained'):
        run_one(NAMES[0],dict(results={},reuse={NAMES[0]:{'reused':True}}),tmp_path,None,None,None)


def test_report_fixed_order_and_negative_evidence(tmp_path):
    import csv
    from cmgm.scripts.formal_main_ablation_report import report
    metric=dict(MAE=.1,MSE=.02,RMSE=.02**.5,Hit=.5)
    def entry(mae):return dict(metrics={'test':{'5':{**metric,'MAE':mae}}},per_commodity=[],mechanism={},runtime={},sanity={})
    r=dict(results={NAMES[-1]:entry(.1),NAMES[0]:entry(.09)},reuse={n:dict(old_checkpoint_found=False,definition_exact_match=False,reused=False) for n in NAMES},
           supplementary={},supplementary_reuse={},sanity={},initialization={},data={'mapping':[]},protocol={},status='PARTIAL')
    report(r,tmp_path)
    rows=list(csv.DictReader((tmp_path/'main_architecture_ablation.csv').open()))
    assert [v['Variant'] for v in rows]==list(NAMES)
    assert float(rows[0]['DeltaMAE'])==pytest.approx(-.01)
    evidence=list(csv.DictReader((tmp_path/'contribution_evidence.csv').open()))
    assert evidence[0]['Supported']=='NO'
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==13 and 'Q11.' in text
    for forbidden in ('w/o Spatial Branch','w/o Temporal Branch','w/o Adaptive Fusion'):
        assert forbidden not in text and forbidden not in NAMES
    assert len(NAMES)==11 and len(NEW)==9
    assert not (tmp_path/'branch_ablation.csv').exists()


def test_fusion_variants_remove_only_intended_modules():
    full=MainInnovationAblation(NAMES[-1],DATA)
    simple=MainInnovationAblation('w/o MoE Fusion',DATA)
    static=MainInnovationAblation('w/o Candidate-Aware Routing',DATA)
    assert not hasattr(simple,'candidate_moe_fusion')
    assert type(simple.simple_fusion) is torch.nn.Linear
    assert sum(p.numel() for p in simple.simple_fusion.parameters())==8256
    assert not hasattr(static,'candidate_moe_fusion') and not hasattr(static.global_mixture_fusion,'router')
    assert static.global_mixture_fusion.global_mixture_logits.shape==(2,)
    assert torch.equal(static.global_mixture_fusion.global_mixture_logits.softmax(0),torch.tensor([.5,.5]))
    for name in NAMES[:8]:
        m=MainInnovationAblation(name,DATA)
        assert type(m.candidate_moe_fusion) is type(full.candidate_moe_fusion)
        assert m.variant=='switching_latent_balanced_candidate_2expert_moe'


def test_full_cannot_be_retrained(tmp_path):
    from cmgm.scripts.formal_main_innovation_ablation import run_one
    with pytest.raises(AssertionError,match='never train'):
        run_one(NAMES[-1],dict(results={},reuse={}),tmp_path,None,None,None)


def test_readonly_reference_audit_completion_and_protocol(tmp_path):
    import json
    from cmgm.scripts.formal_main_innovation_ablation import audit_reuse
    from cmgm.scripts.d0b_candidate_2expert_moe import PROTOCOL
    from cmgm.scripts.formal_v2_protocol import sha
    path=tmp_path/'full.pt';source=tmp_path/'results.json';data={'fixture':1}
    history=dict(val_loss=[.1]+[.2]*10,best_epoch=1)
    hashes={s:sha(Path(s)) for s in ('cmgm/models/candidate_moe_fusion.py','cmgm/models/switching_latent_transformer.py')}
    md=dict(variant=PROTOCOL['variant'],seed=42,config=PROTOCOL,data=data,source_hashes=hashes)
    cp=dict(history=history,best_epoch=1,best_val_loss=.1,metadata=md)
    torch.save(cp,path)
    r=dict(status='COMPLETE',trained=True,history=history,checkpoint_sha256=sha(path),config=PROTOCOL,data=data,
        source_hashes=hashes,initialization={'PASS':True},sanity={'PASS':True},best_sanity={'PASS':True},best_checkpoint_metadata={'best_epoch':1})
    source.write_text(json.dumps(r))
    row=audit_reuse(NAMES[-1],source,path,data)
    assert row['reused'] and row['training_complete']
    r['trained']=False;source.write_text(json.dumps(r))
    assert not audit_reuse(NAMES[-1],source,path,data)['reused']
    r['trained']=True;r['checkpoint_sha256']='corrupted';source.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='hash mismatch'):audit_reuse(NAMES[-1],source,path,data)


def test_fresh_reference_mismatch_stops_without_publishing_metrics(tmp_path,monkeypatch):
    import json
    from torch.utils.data import DataLoader,TensorDataset
    import cmgm.scripts.formal_main_innovation_ablation as runner
    m=MainInnovationAblation(NAMES[-1],DATA)
    path=tmp_path/'full.pt';source=tmp_path/'reference.json'
    torch.save({'model_state_dict':m.state_dict()},path);source.write_text('{}')
    expected={'test':{'5':{'MAE':.02}}}
    row=dict(reused=True,path=str(path),sha256=runner.sha(path),source_report=str(source),
             source_report_sha256=runner.sha(source),expected_metrics=expected)
    r=dict(reuse={NAMES[-1]:row},results={})
    data={**DATA,'loaders':{'train':DataLoader(TensorDataset(torch.randn(2,20,30,21),torch.zeros(2,4,24)))}}
    monkeypatch.setattr(runner,'loaders',lambda data,full:data['loaders'])
    monkeypatch.setattr(runner,'sanity',lambda *a:{'PASS':True})
    monkeypatch.setattr(runner,'evaluate',lambda *a:{'metrics':{'test':{'5':{'MAE':.025}}}})
    monkeypatch.setattr(runner,'save',lambda *a:None)
    with pytest.raises(ValueError,match='does not reproduce'):
        runner.evaluate_reused(r,tmp_path,data,torch.device('cpu'))
    assert not r['results'] and row['fresh_reproduction']['PASS'] is False
    assert runner.sha(path)==row['sha256']


def test_standard_gcn_is_single_hop_with_directed_learned_adjacency():
    from cmgm.models.formal_d0b_main_ablation import StandardGraphPropagation
    layer=StandardGraphPropagation(2,2).double()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2,dtype=torch.float64));layer.linear.bias.zero_()
    A=torch.tensor([[0.,2.,0.],[0.,0.,3.],[1.,0.,0.]],dtype=torch.float64,requires_grad=True)
    x=torch.arange(12,dtype=torch.float64).reshape(2,3,2).requires_grad_()
    aug=A+torch.eye(3,dtype=torch.float64);D=torch.diag(aug.sum(1).rsqrt())
    expected=torch.stack([D@aug@D@sample for sample in x])
    torch.testing.assert_close(layer(x,A),expected,rtol=0,atol=1e-14)
    grads=torch.autograd.grad(layer(x,A).square().sum(),(x,A,layer.linear.weight))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    assert not hasattr(layer,'K') and not hasattr(layer,'beta')
    m=MainInnovationAblation('w/o EdgeAttnMixHop',DATA)
    assert isinstance(m.standard_gcn1,StandardGraphPropagation) and isinstance(m.standard_gcn2,StandardGraphPropagation)
    assert not any('ordinary_mixhop' in name for name,_ in m.named_modules())
    assert sum(p.numel() for p in m.standard_gcn1.parameters())==4160


def test_gcn_revision_retains_ten_rows_and_excludes_only_old_edge_result(tmp_path):
    import copy,json
    import cmgm.scripts.formal_main_innovation_ablation as runner
    from cmgm.models.formal_d0b_main_ablation import DEFINITIONS,REUSE
    old_protocol={k:v for k,v in runner.PROTOCOL.items() if k!='edge_propagation_control'}
    old=dict(status='COMPLETE',protocol=old_protocol,data={'fixture':1},source_hashes={'cmgm/models/formal_d0b_main_ablation.py':'before','cmgm/training/train.py':'same'},results={})
    r=dict(protocol=runner.PROTOCOL,data=old['data'],source_hashes={**old['source_hashes'],'cmgm/models/formal_d0b_main_ablation.py':'after'},results={},reuse={})
    for i,name in enumerate(NAMES):
        path=tmp_path/f'{i}.pt'
        definition=DEFINITIONS[name] if name!='w/o EdgeAttnMixHop' else 'Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05); Candidate MoE retained'
        torch.save(dict(training_complete=True,best_epoch=4,metadata=dict(name=name,definition=definition,seed=42,protocol=old_protocol,data=old['data'],source_hashes=old['source_hashes'])),path)
        old['results'][name]=dict(path=str(path),sha256=runner.sha(path),sanity={'PASS':True},metrics={'test':{'5':{'MAE':.01+i*.001}}})
        if name in REUSE:r['reuse'][name]=dict(reused=True,sha256=runner.sha(path))
    source=tmp_path/'results.json';source.write_text(json.dumps(old));before=source.read_bytes()
    clean=copy.deepcopy(r)
    runner.reuse_unchanged_for_gcn(r,source)
    assert len(r['results'])==10 and 'w/o EdgeAttnMixHop' not in r['results']
    assert source.read_bytes()==before
    for name,entry in r['results'].items():assert entry['metrics']==old['results'][name]['metrics']
    clean['source_hashes']['cmgm/training/train.py']='changed'
    with pytest.raises(ValueError,match='Unrelated source'):runner.reuse_unchanged_for_gcn(clean,source)


from cmgm.models.formal_d0b_main_ablation import GLOBAL,GLOBAL_NAMES,GLOBAL_FULL


@pytest.mark.parametrize('name',GLOBAL_NAMES)
def test_global_temporal_native_initialization_rng_wiring_and_sanity(name):
    from cmgm.scripts.formal_main_ablation_audit import native
    from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
    x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.02
    seed_all(42);reference=native(DATA,GLOBAL);rng=torch.get_rng_state().clone()
    seed_all(42);m=MainInnovationAblation(name,DATA,mode='global-temporal')
    assert torch.equal(rng,torch.get_rng_state())  # No transient Candidate router construction.
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in reference.state_dict().items())
    assert m.variant==GLOBAL and not hasattr(m,'candidate_moe_fusion')
    assert not any('router' in k for k,_ in m.global_mixture_fusion.named_parameters())
    m,initial=init_audit(name,DATA,torch.device('cpu'),x,mode='global-temporal')
    assert initial['PASS'] and initial['mismatch_count']==0 and initial['shared_parameter_initial_max_abs_diff']==0
    assert initial['initial_alpha_max_error']==0
    before={k:v.clone() for k,v in m.state_dict().items()}
    check=sanity(m,x,y)
    assert check['PASS'],check
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())
    m.eval();p=m(x);logits=m.global_mixture_fusion.global_mixture_logits
    grad=torch.autograd.grad(prediction_loss(p,y),logits)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    if name==GLOBAL_NAMES[0]:
        assert check['uniform_prior_error']==0 and check['KL_zero'] and check['three_generators']
    elif name==GLOBAL_NAMES[1]:
        assert check['revised_structure']['candidate12_error']==0 and check['revised_structure']['KL_retained']
    elif name==GLOBAL_NAMES[2]:
        assert check['zero_micro']==0 and check['revised_structure']['recurrence_retained'] and check['revised_structure']['KL_retained']
    else:
        reference.eval()
        with torch.no_grad():torch.testing.assert_close(p,reference(x),rtol=0,atol=0)


def test_global_targeted_mode_never_trains_full_or_extra_controls(tmp_path):
    from cmgm.scripts.formal_main_innovation_ablation import run_one,GLOBAL_PROTOCOL,suite_names
    r=dict(results={},reuse={},protocol=GLOBAL_PROTOCOL)
    assert suite_names(r)==GLOBAL_NAMES and len(GLOBAL_NAMES[:-1])==3
    with pytest.raises(AssertionError,match='never train'):run_one(GLOBAL_FULL,r,tmp_path,None,None,None)
    with pytest.raises(ValueError,match='outside selected'):run_one(NAMES[0],r,tmp_path,None,None,None)
    with pytest.raises(ValueError):MainInnovationAblation(NAMES[0],DATA,mode='global-temporal')


@pytest.mark.parametrize('deltas,case',[( (.01,.02), 'Case A'),((.01,-.02),'Case B'),((-.01,-.02),'Case C'),((.00001,.00001),'Case C')])
def test_global_report_only_four_rows_and_unfavorable_results(tmp_path,deltas,case):
    import csv
    from cmgm.scripts.formal_main_ablation_report import report
    from cmgm.scripts.formal_main_innovation_ablation import GLOBAL_PROTOCOL
    def entry(mae):return dict(metrics={'test':{'5':dict(MAE=mae,MSE=.02,RMSE=.02**.5,Hit=.5)}},
        mechanism={'global_mixture':dict(alpha_T=.49,alpha_ST=.51)},sanity={},per_commodity=[],runtime={})
    r=dict(protocol=GLOBAL_PROTOCOL,results={n:entry(.1+d) for n,d in zip(GLOBAL_NAMES,(*deltas,.005,0))},
        reuse={n:dict(reused=n==GLOBAL_FULL) for n in GLOBAL_NAMES},sanity={},initialization={},source_hashes={},
        data={'mapping':[]},status='COMPLETE')
    report(r,tmp_path)
    rows=list(csv.DictReader((tmp_path/'main_architecture_ablation.csv').open()))
    assert [row['Variant'] for row in rows]==list(GLOBAL_NAMES)
    assert float(rows[0]['DeltaMAE'])==pytest.approx(deltas[0])
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==16 and 'Q11.' in text and 'post-hoc mechanism study' in text
    assert case in text
    if case=='Case C':assert 'Removing the second expert-level router does not restore' in text
    assert not (tmp_path/'spatial_innovation_ablation.csv').exists()


from cmgm.models.formal_d0b_main_ablation import TS_NAMES,TS_FULL,TS_VARIANT


@pytest.mark.parametrize('name',TS_NAMES)
def test_ts_full_and_controls_shared_init_branch_wiring_and_unchanged_state(name):
    from cmgm.scripts.formal_main_ablation_audit import native
    from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
    x=torch.randn(2,20,30,21);y=torch.randn(2,4,24)*.02
    seed_all(42);reference=native(DATA,TS_VARIANT);rng=torch.get_rng_state().clone()
    seed_all(42);model=MainInnovationAblation(name,DATA,mode='ts-global-expert')
    if name!='w/o EdgeAttnMixHop':assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
    m,initial=init_audit(name,DATA,torch.device('cpu'),x,mode='ts-global-expert')
    assert initial['PASS'] and initial['mismatch_count']==0 and initial['shared_parameter_initial_max_abs_diff']==0
    assert initial['tst_shared_initialization']['PASS'] and initial['initial_alpha_max_error']==0
    before={k:v.clone() for k,v in m.state_dict().items()}
    check=sanity(m,x,y)
    assert check['PASS'],check
    assert check['revised_structure']['expert_parameters']==dict(Temporal=8320,Spatial=8320)
    assert check['revised_structure']['branch_specific_expert_wiring']
    assert check['revised_structure']['no_interaction_expert']
    assert check['revised_structure']['expert_symmetry']
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())
    if name=='w/o EdgeAttnMixHop':
        assert check['revised_structure']['ordinary_mixhop_exact']
        assert check['revised_structure']['same_A_both_blocks']
        assert check['revised_structure']['edge_qkv_inactive']
        assert check['revised_structure']['no_gcn_replacement']
    m.eval();p=m(x);parameters=list(m.global_mixture_fusion.parameters())
    grads=torch.autograd.grad(prediction_loss(p,y),parameters)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    if name==TS_FULL:
        reference.eval()
        with torch.no_grad():torch.testing.assert_close(p,reference(x),rtol=0,atol=0)


def test_ts_mode_exact_five_runs_and_old_modes_preserved():
    from cmgm.scripts.formal_main_innovation_ablation import TS_PROTOCOL,GLOBAL_PROTOCOL,PROTOCOL,training_order,suite_names
    r={'protocol':TS_PROTOCOL}
    assert training_order(r)==(TS_FULL,*TS_NAMES[:-1])
    assert len(set(training_order(r)))==5 and suite_names(r)==TS_NAMES
    assert suite_names({'protocol':GLOBAL_PROTOCOL})==GLOBAL_NAMES
    assert suite_names({'protocol':PROTOCOL})==NAMES
    with pytest.raises(ValueError):MainInnovationAblation('w/o Balanced Readout',DATA,mode='ts-global-expert')
    old=MainInnovationAblation('w/o EdgeAttnMixHop',DATA)
    assert hasattr(old,'standard_gcn1') and not hasattr(old,'ordinary_mixhop1')


@pytest.mark.parametrize('name',[TS_FULL,'w/o Adaptive Regime Routing'])
def test_ts_native_training_loss_and_prediction_only_selection(name,monkeypatch):
    from cmgm.training.train import train_epoch,validate_epoch
    from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
    from torch.utils.data import DataLoader,TensorDataset
    seed_all();m=MainInnovationAblation(name,DATA,mode='ts-global-expert');m.switching_latent_transformer.set_epoch(20)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0  # Only this deterministic unit fixture.
    x,y=torch.randn(2,20,30,21),torch.randn(2,4,24)*.02
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    pred=m(x);pred_loss=prediction_loss(pred,y);aux=m.auxiliary_loss()
    if name==TS_FULL:assert aux.item()>0
    else:assert aux.item()==0
    expected=float((pred_loss+aux).detach())
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:None)  # No fitted run or optimizer update.
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    edge,weight=torch.empty((2,0),dtype=torch.long),torch.empty(0)
    assert train_epoch(m,loader,edge,weight,opt,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))==pytest.approx(expected,abs=1e-9)
    assert validate_epoch(m,loader,edge,weight,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))==pytest.approx(float(pred_loss.detach()),abs=1e-9)


@pytest.mark.parametrize('deltas,case',[( (.01,.02,.003), 'Case A'),((.01,-.02,0),'Case B'),((-.01,-.02,-.001),'Case C')])
def test_ts_report_five_rows_19_sections_and_signed_deltas(tmp_path,deltas,case):
    import csv,json
    from cmgm.scripts.formal_main_innovation_ablation import TS_PROTOCOL
    from cmgm.scripts.formal_main_ablation_report import report
    def entry(mae):return dict(path='fixture',sha256='fixture',metrics={'test':{'5':dict(MAE=mae,MSE=.02,RMSE=.02**.5,Hit=.5)}},
        mechanism={'global_mixture':dict(alpha_T=.49,alpha_S=.51)},sanity={},runtime={})
    r=dict(protocol=TS_PROTOCOL,results={n:entry(.1+d) for n,d in zip(TS_NAMES,(*deltas,.02,0))},
        reuse={n:{'reused':False} for n in TS_NAMES},references={},reference_results={},cross_architecture={},
        sanity={},initialization={},source_hashes={},data={'mapping':[]},status='COMPLETE')
    report(r,tmp_path)
    rows=list(csv.DictReader((tmp_path/'main_ablation_table.csv').open()))
    assert [row['Variant'] for row in rows]==list(TS_NAMES)
    assert float(rows[0]['DeltaMAE'])==pytest.approx(deltas[0])
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==19 and 'Q12.' in text and case in text
    assert 'post-hoc architecture-mechanism study' in text
    if case=='Case C':assert 'interaction expert alone is therefore insufficient' in text
    for p in tmp_path.glob('*.json'):json.loads(p.read_text())
