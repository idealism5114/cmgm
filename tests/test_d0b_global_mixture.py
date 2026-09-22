import copy
import importlib
import json
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset,DataLoader
from cmgm.models.global_mixture_fusion import GlobalTwoExpertMixture,VARIANT
from cmgm.models.candidate_moe_fusion import VARIANT as DYNAMIC
from cmgm.scripts.d0b_global_mixture_audit import initialization,sanity,make_model
from cmgm.scripts.d0b_moe_audit import make_model as dynamic_model
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
DATA=dict(n_nodes=30,market_indices=dict(stock=(0,4),bond=(4,6),commodity=(6,30)))


def test_shared_expert_initialization_and_initial_function():
    static,a,e=initialization(DATA,torch.device('cpu'))
    assert a['PASS'] and e['PASS'] and a['parameter_delta']==-16832
    assert a['static_router_parameters']==2 and not a['unexpected_router_parameters']
    seed_all(42);dynamic=dynamic_model(DATA,DYNAMIC).eval();static.eval()
    x=torch.randn(2,20,30,21)
    torch.testing.assert_close(static(x),dynamic(x),rtol=0,atol=0)
    c=sanity(static,x,torch.randn(2,4,24)*.02)
    assert c['PASS'],c


def test_global_alpha_shape_invariance_wiring_and_gradients():
    f=GlobalTwoExpertMixture().eval()
    assert not any(isinstance(m,torch.nn.LayerNorm) for m in f.modules())
    s,t=torch.randn(4,64),torch.randn(4,64)
    with torch.no_grad():f.global_mixture_logits.copy_(torch.tensor([.3,-.7]))
    h=f(s,t);a=f.last['alpha'].clone()
    assert a.shape==(2,)
    torch.testing.assert_close(h,a[0]*f.temporal_expert(t)+a[1]*f.interaction_expert(torch.cat([s,t],-1)))
    perm=torch.tensor([3,0,1,2]);torch.testing.assert_close(f(s[perm],t[perm]),h[perm])
    torch.testing.assert_close(f(s[:1],t[:1]),h[:1]);f(s*2,t-5)
    assert torch.equal(a,f.last['alpha'])
    f(s,t).square().mean().backward()
    for group in (f.temporal_expert,f.interaction_expert):
        assert sum(p.grad.abs().sum() for p in group.parameters())>0
    assert torch.isfinite(f.global_mixture_logits.grad).all() and f.global_mixture_logits.grad.abs().sum()>0


def test_native_loss_selection_boundary(monkeypatch):
    from cmgm.training.train import train_epoch,validate_epoch
    seed_all();m=make_model(DATA);m.switching_latent_transformer.set_epoch(20)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    x,y=torch.randn(2,20,30,21),torch.randn(2,4,24)*.02
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    pred=m(x);prediction=prediction_loss(pred,y)
    expected=float((prediction+m.switching_latent_transformer.switch_loss()).detach())
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:None)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    edge,weight=torch.empty((2,0),dtype=torch.long),torch.empty(0)
    assert train_epoch(m,loader,edge,weight,opt,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))==pytest.approx(expected,abs=1e-9)
    assert validate_epoch(m,loader,edge,weight,torch.nn.HuberLoss(delta=.02),torch.device('cpu'))==pytest.approx(float(prediction.detach()),abs=1e-9)


def test_common_training_restores_selected_global_logits(tmp_path,monkeypatch):
    # Synthetic epoch callbacks test checkpoint semantics, not an experiment fit.
    tr=importlib.import_module('cmgm.training.train')
    m=make_model(DATA);loader=DataLoader(TensorDataset(torch.zeros(2,20,30,21),torch.zeros(2,4,24)),batch_size=2)
    step=[0]
    def fake_train(model,*args,**kw):
        step[0]+=1
        with torch.no_grad():model.global_mixture_fusion.global_mixture_logits.copy_(torch.tensor([step[0]*.1,0]))
        return .01
    def fake_val(model,*args,**kw):
        model._last_val5_diagnostic=dict(MAE=1/step[0],MSE=.01)
        return [.03,.02,.04][step[0]-1]
    monkeypatch.setattr(tr,'train_epoch',fake_train);monkeypatch.setattr(tr,'validate_epoch',fake_val)
    path=tmp_path/'best.pt'
    history=tr.train(m,loader,loader,torch.empty((2,0),dtype=torch.long),torch.empty(0),torch.device('cpu'),
        num_epochs=3,patience=10,checkpoint_path=str(path),checkpoint_metadata={'test':True})
    assert history['best_epoch']==2
    torch.testing.assert_close(m.global_mixture_fusion.global_mixture_logits,torch.tensor([.2,0.]))
    for row in history['global_weight_history']:assert row['train']==row['val']
    cp=torch.load(path,weights_only=False)
    assert cp['best_epoch']==2 and cp['model_state_dict']['global_mixture_fusion.global_mixture_logits'][0]==pytest.approx(.2)


def test_frozen_train_mean_intervention_and_report(tmp_path):
    from cmgm.scripts.d0b_global_mixture_evaluation import intervention
    from cmgm.scripts.formal_v2_protocol import metrics,sha
    from cmgm.scripts.d0b_global_mixture_report import report,DYNAMIC as LABEL
    seed_all();m=dynamic_model(DATA,DYNAMIC).eval()
    with torch.no_grad():m.candidate_moe_fusion.router.network[-1].weight.normal_(std=.03)
    x=torch.randn(4,20,30,21);y=torch.randn(4,4,24)*.02
    # TRAIN includes final incomplete batch under source loader, full loader must restore it.
    data=dict(DATA,loaders={s:DataLoader(TensorDataset(x*(i+1),y if s=='test' else torch.full_like(y,float('nan'))),batch_size=3,drop_last=s=='train') for i,s in enumerate(('train','val','test'))})
    with torch.no_grad():
        m(x);expected=m.candidate_moe_fusion.last['pi'].double().mean(0).float()
        p=m(x*3);c=m.candidate_moe_fusion.last
        fixed=m.head((expected.view(1,2,1)*c['experts']).sum(1)).reshape_as(p)
    path=tmp_path/'dynamic.pt';torch.save(dict(model_state_dict=m.state_dict()),path)
    ref=dict(checkpoint=str(path),checkpoint_sha256=sha(path),metrics=dict(test=metrics(p.detach().numpy(),y.numpy())),source_report='fixture',source_report_sha256='fixture',best_epoch=1)
    iv=intervention(data,ref,torch.device('cpu'),tmp_path)
    assert iv['train_count']==4 and iv['checkpoint_unchanged']
    np.testing.assert_array_equal(iv['weights'],expected.numpy())
    np.testing.assert_allclose(np.load(tmp_path/'same_checkpoint_predictions.npz')['fixed_mean'],fixed.detach().numpy(),atol=0,rtol=0)
    assert sha(path)==ref['checkpoint_sha256'] and set(iv['fixed_mean'])=={'1','5','10','20'}
    r=dict(controls={LABEL:ref,'TemporalOnly':ref,'Adaptive Gated Fusion / Full D0B':ref},config={},source_hashes={},initialization={},sanity={},data={},status='PREPARED',intervention=iv)
    report(r,tmp_path)
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    assert text.count('\n## ')==16 and 'Q9.' in text and 'PENDING' in text
    for path in tmp_path.glob('*.json'):json.loads(path.read_text())
