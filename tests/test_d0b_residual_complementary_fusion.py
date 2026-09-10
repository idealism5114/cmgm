import importlib
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cmgm.config import MULTI_HORIZONS
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_fusion_diagnostics import BASE,VARIANT,assert_backbone,initialization_check,gradient_diagnostics,structural_sanity,snapshot,collect,ablation_analysis
from cmgm.scripts.d0b_fusion_report import build_comparison,write_report,classify
from cmgm.training.train import validate_epoch,make_loss,_prediction_loss


def model(variant=VARIANT):
 torch.manual_seed(42);return HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=variant)


def test_real_count_initial_exactness_and_only_expected_modules():
 torch.manual_seed(42);m=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
 assert_backbone(m);r=initialization_check(m,torch.randn(2,20,284,21))
 assert r['PASS'] and (r['D0B_params'],r['New_params'],r['delta_params'])==(520549,526725,6176)
 assert all(r[k]==0 for k in ('shared_parameter_init_max_diff','shared_parameter_mismatch_count','h_base_max_diff','residual_max_abs','fused_base_max_diff','prediction_max_diff'))
 assert len(list(m.complementary_fusion_residual.parameters()))==3
 assert isinstance(m.complementary_fusion_residual[1],nn.ReLU) and m.complementary_fusion_residual[-1].bias is None


def test_raw_global_input_no_detach_and_original_gate_and_head():
 m=model().eval();x=torch.randn(3,20,9,21);y=torch.randn(3,4,3)*.03
 initial=gradient_diagnostics(m,(x,y));assert initial['norms']['residual_final']>0 and initial['norms']['residual_first']==0
 with torch.no_grad():m.complementary_fusion_residual[-1].weight.fill_(.01)
 captured=[];gate_inputs=[]
 hooks=[m.complementary_fusion_residual.register_forward_pre_hook(lambda mod,args:captured.append(args[0])),m.gate_fc.register_forward_pre_hook(lambda mod,args:gate_inputs.append(args[0]))]
 try:
  pred=m(x);torch.testing.assert_close(captured[0],gate_inputs[0],rtol=0,atol=0)
  g=torch.autograd.grad(_prediction_loss(m,pred,y,make_loss()),captured[0])[0]
  assert g[:,:64].abs().sum()>0 and g[:,64:].abs().sum()>0
 finally:
  for h in hooks:h.remove()
 with torch.no_grad():
  row=snapshot(m,x);s,t=row['h_spatial'],row['h_temporal'];gate=torch.sigmoid(m.gate_fc(torch.cat([s,t],-1)))
  base=gate*m.lstm_proj(t)+(1-gate)*m.gcn_proj(s)
  residual=m.complementary_fusion_residual(torch.cat([s,t],-1))
  expected=m.head(base+residual).view_as(pred)
  torch.testing.assert_close(expected,pred,rtol=0,atol=0)
 assert gradient_diagnostics(m,(x,y))['norms']['residual_first']>0
 assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('active',[False,True])
def test_component_causality_relabeling_batch_and_rng(active):
 m=model();x=torch.randn(3,20,9,21)
 if active:
  with torch.no_grad():m.complementary_fusion_residual[-1].weight.normal_(0,.01)
 rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in m.state_dict().items()}
 check=structural_sanity(m,x);assert check['PASS'],check
 assert check['fusion']['fusion_input_shape']==[3,128] and check['fusion']['residual_shape']==[3,64]
 torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
 for k,v in state.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)


def test_zero_control_head_input_and_full_loader_report(tmp_path):
 m=model().eval();b=model(BASE).eval()
 with torch.no_grad():m.complementary_fusion_residual[-1].weight.fill_(.01)
 x=torch.randn(5,20,9,21);y=torch.randn(5,4,3)*.03
 loaders={s:DataLoader(TensorDataset(x,y),batch_size=3,drop_last=True) for s in ('train','val','test')}
 base,bs=collect(b,loaders,torch.device('cpu'));new,ns=collect(m,loaders,torch.device('cpu'))
 assert len(new['train']['target'])==5
 for split in new:np.testing.assert_array_equal(base[split]['prediction'],new[split]['zero_prediction'])
 a=ablation_analysis(base,new);assert a['test']['representation_effect']['MAE']==0
 assert set(a['test']['metrics'])=={'OriginalD0B','Full','ZeroResidual'}
 r=dict(status='SYNTHETIC TEST',training_executed=True,sanity=dict(PASS=True,shared_init=initialization_check(model(),x)),gradients={},
   residual={'best':{'final_weight_norm':.01},'D0B_full_loader':bs,'New_full_loader':ns},ablations=a)
 r.update(build_comparison(base,new,['焦煤','原油','低硫燃料油'],r,tmp_path))
 r.update(best_epoch=1,train_time_seconds=0.,best_formal_val_objective=.01,history=dict(val5_diagnostic=[dict(MAE=.01)],best_val5_epoch=1,best_val5_mae=.01),baseline_reference=dict(PASS=True,actual=r['metrics']['D0B']['test']))
 write_report(r,tmp_path);text=(tmp_path/'REPORT.md').read_text()
 assert '16. Primary classification:' in text and 'Direct vs representation effect' in text
 assert len(r['commodity_metrics'])==3 and len(r['target_magnitude_groups'])==8
 i=r['leave_one_commodity_out']['removed_index'];keep=np.arange(3)!=i
 expected=np.abs(new['test']['prediction'][:,keep]-new['test']['target'][:,keep]).mean()
 assert r['leave_one_commodity_out']['New_TEST_MAE']==pytest.approx(expected)
 r.pop('metrics');write_report(r,tmp_path)
 assert 'No primary classification before formal evaluation' in (tmp_path/'REPORT.md').read_text()


class Predictions(nn.Module):
 def __init__(self,variant):super().__init__();self.variant=variant;self.graph_learner=True
 def forward(self,x,debug=False):return x


def test_secondary_logging_preserves_original_loss():
 p=torch.arange(40,dtype=torch.float32).view(5,4,2)*.001;y=torch.zeros_like(p)
 loader=DataLoader(TensorDataset(p,y),batch_size=3);a=Predictions(BASE);b=Predictions(VARIANT)
 args=(loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),make_loss(),torch.device('cpu'))
 assert validate_epoch(a,*args)==validate_epoch(b,*args)
 assert _prediction_loss(a,p,y,make_loss())==_prediction_loss(b,p,y,make_loss())
 assert b._last_val5_diagnostic['MAE']==pytest.approx(p[:,MULTI_HORIZONS.index(5)].double().abs().mean().item())


def test_formal_selection_uses_original_objective(tmp_path,monkeypatch):
 mod=importlib.import_module('cmgm.training.train');m=nn.Linear(1,1);m.variant=VARIANT
 val=iter([3.,2.,4.]);secondary=iter([.3,.4,.1]);seen=[]
 def validate(model,*args,**kwargs):
  model._last_val5_diagnostic=dict(MAE=next(secondary),MSE=1.,count=10);return next(val)
 monkeypatch.setattr(mod,'train_epoch',lambda *a,**k:1.);monkeypatch.setattr(mod,'validate_epoch',validate)
 original=mod.optim.lr_scheduler.ReduceLROnPlateau.step
 def step(scheduler,value,*a,**k):seen.append(value);return original(scheduler,value,*a,**k)
 monkeypatch.setattr(mod.optim.lr_scheduler.ReduceLROnPlateau,'step',step)
 h=mod.train(m,[],[],torch.empty(2,0),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(tmp_path/'formal.pt'))
 assert seen==[3.,2.,4.] and h['best_epoch']==2 and h['best_val5_epoch']==3
 assert torch.load(tmp_path/'formal.pt',weights_only=False)['best_epoch']==2


@pytest.mark.parametrize('case',['A','B','C','D','E','F','G'])
def test_case_priority_including_zero_residual_training_damage(case):
 base=dict(MAE=1.,MSE=1.,RMSE=1.,Hit=.5);new=dict(base,MAE=.8,MSE=.8);zero=dict(base,MAE=.9,MSE=.9)
 if case=='B':zero=base.copy()
 if case=='C':zero=new.copy()
 if case=='D':new['MSE']=1.1
 if case=='E':new=dict(base,MAE=1.1,MSE=1.1);zero=base.copy()
 if case=='F':zero=dict(base,MAE=1.2,MSE=1.2) # F wins even if Full itself improves
 if case=='G':zero=base.copy();new=base.copy()
 metrics={'D0B':{s:base for s in ('val','test')},'ResidualComplementaryFusion':{s:new for s in ('val','test')}}
 r=dict(sanity=dict(PASS=True),residual={'best':{'final_weight_norm':0. if case=='G' else .1},'New_full_loader':{'test':{'residual_base_norm_ratio':0. if case=='G' else .1}}},
   ablations={s:{'metrics':{'Full':new,'ZeroResidual':zero,'OriginalD0B':base}} for s in ('val','test')})
 assert classify(metrics,r,dict(delta_MAE=-.1))['primary_case']=='Case '+case
