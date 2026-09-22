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
