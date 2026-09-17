import ast
import inspect
import pytest
import torch
from cmgm.models.formal_d0b_main_ablation import NAMES, NEW, MainInnovationAblation
from cmgm.scripts.formal_main_ablation_audit import init_audit,sanity

DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))

@pytest.mark.parametrize('name',NAMES)
def test_shared_init_legacy_equivalence_and_structural_checks(name):
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
    assert 'complementarity supported by both branch comparisons: NO' in (tmp_path/'REPORT.md').read_text() or 'complementarity supported by both branch comparisons: PENDING' in (tmp_path/'REPORT.md').read_text()
    assert len(NEW)==4
