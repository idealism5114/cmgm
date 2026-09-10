"""Controlled zero-init pre-GNN path, causal controls and legacy regressions."""
import importlib
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cmgm.config import MULTI_HORIZONS
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_pregnn_diagnostics import (
 BASE,VARIANT,initialization_check,assert_backbone,structural_sanity,gradient_diagnostics,local_controls,snapshot,commodity_order_check,
)
from cmgm.scripts.d0b_pregnn_analysis import collect_with_controls,primary_arrays,residual_analysis,ablation_analysis
from cmgm.scripts.d0b_pregnn_report import build_comparison,write_report
from cmgm.training.train import validate_epoch,make_loss,_prediction_loss


def model(variant=VARIANT):
 torch.manual_seed(42)
 return HeteroMixHopCMGM(9,3,n_stock=4,n_bond=2,variant=variant)


def test_full_size_init_count_zero_exactness():
 torch.manual_seed(42);m=HeteroMixHopCMGM(284,24,n_stock=248,n_bond=12,variant=VARIANT)
 x=torch.randn(2,20,284,21)
 r=initialization_check(m,x);assert_backbone(m)
 assert r['PASS'] and r['D0B_params']==520549 and r['PreGNN_params']==524805
 assert r['delta_params']==4256 and r['eval_prediction_max_diff']==r['residual_max_abs']==r['base_final_max_diff']==0
 assert r['shared_parameter_init_max_diff']==r['mismatch_count']==0
 assert len(list(m.pregnn_local_residual.parameters()))==3


def test_pre_nodes_api_keeps_legacy_outputs_and_gradients():
 m=model(BASE).eval();x=torch.randn(2,20,9,21)
 a=m._temp_weighted_spatial(x)
 ga=torch.autograd.grad(a.square().mean(),tuple(m.type_proj.parameters()))
 b,post,pre=m._temp_weighted_spatial(x,return_nodes=True,return_pre_nodes=True)
 gb=torch.autograd.grad(b.square().mean(),tuple(m.type_proj.parameters()))
 torch.testing.assert_close(a,b,atol=0,rtol=0)
 for left,right in zip(ga,gb):torch.testing.assert_close(left,right,atol=0,rtol=0)
 c,oldpost=m._temp_weighted_spatial(x,return_nodes=True)
 torch.testing.assert_close(c,b,atol=0,rtol=0);torch.testing.assert_close(oldpost,post,atol=0,rtol=0)
 assert not torch.allclose(pre,post)
 h=m.switching_latent_transformer(x)
 legacy=m._market_token_predict(a,h);optional,fused=m._market_token_predict(a,h,return_fused=True)
 torch.testing.assert_close(legacy,optional,atol=0,rtol=0)
 torch.testing.assert_close(m.head(fused).view_as(legacy),legacy,atol=0,rtol=0)


def test_zero_init_gradients_then_live_local_and_global_inputs():
 m=model();x=torch.randn(3,20,9,21);y=torch.randn(3,4,3)*.03
 r=gradient_diagnostics(m,(x,y));assert r['norms']['residual_final']>0 and r['norms']['residual_first']==0
 for name in ('type_proj','temporal_score','attn_mixhop1','attn_mixhop2','fusion','head'):assert r['norms'][name]>0
 # Synthetic nonzero fixture, NOT an optimizer update or training experiment.
 with torch.no_grad():m.pregnn_local_residual[-1].weight.fill_(.01)
 captured=[]
 hook=m.pregnn_local_residual.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
 try:
  m.eval();p=m(x);loss=_prediction_loss(m,p,y,make_loss())
  g=torch.autograd.grad(loss,captured[0],retain_graph=True)[0]
  assert g[:,:,:64].abs().sum()>0 and g[:,:,64:].abs().sum()>0
 finally:hook.remove()
 active=gradient_diagnostics(m,(x,y));assert active['norms']['residual_first']>0
 assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('active',[False,True])
def test_causal_batch_relabeling_controls_and_state_preservation(active):
 m=model();x=torch.randn(3,20,9,21)
 if active:
  with torch.no_grad():m.pregnn_local_residual[-1].weight.normal_(0,.01)
 rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in m.state_dict().items()}
 checks=structural_sanity(m,x)
 assert checks['PASS'],checks
 m.eval()
 with torch.no_grad():native=snapshot(m,x);controls=local_controls(m)
 for name,pack in controls.items():torch.testing.assert_close(pack['base_pred'],native['base_pred'],rtol=0,atol=0)
 torch.testing.assert_close(controls['zero']['prediction'],native['base_pred'],rtol=0,atol=0)
 if active:
  assert (controls['shuffle']['residual']-native['residual']).abs().max()>1e-6
  assert (controls['mean']['prediction']-native['prediction']).abs().max()>1e-6
 else:
  for pack in controls.values():torch.testing.assert_close(pack['prediction'],native['prediction'],rtol=0,atol=0)
 torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
 for k,v in state.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)


def test_ordering_checks_actual_raw_target_columns_and_detects_corruption():
 from cmgm.data.data_loader import MarketSequenceDataset
 m=model();prices=np.arange(50*9).reshape(50,9).astype(float)+100
 mi=dict(stock=(0,4),bond=(4,6),commodity=(6,9))
 ds=MarketSequenceDataset(prices,mi,20,feature_matrix=np.ones((50,9,21)),raw_prices=prices,target_type='return',horizons=MULTI_HORIZONS)
 data=dict(market_indices=mi,feature_names=[str(i) for i in range(9)],loaders={'train':DataLoader(ds,batch_size=3)})
 r=commodity_order_check(data,m);assert r['PASS'] and r['mapping'][1]['node_index']==7 and r['mapping'][1]['output_index']==1
 class Corrupted(MarketSequenceDataset):
  def __getitem__(self,i):
   x,y=super().__getitem__(i);return x,y.flip(-1)
 bad=Corrupted(prices,mi,20,feature_matrix=np.ones((50,9,21)),raw_prices=prices,target_type='return',horizons=MULTI_HORIZONS)
 data['loaders']['train']=DataLoader(bad,batch_size=3)
 assert not commodity_order_check(data,m)['PASS']


class Predictions(nn.Module):
 def __init__(self,variant):super().__init__();self.variant=variant;self.graph_learner=True
 def forward(self,x,debug=False):return x


def test_secondary_val5_does_not_change_loss():
 p=torch.arange(40,dtype=torch.float32).view(5,4,2)*.001;y=torch.zeros_like(p)
 loader=DataLoader(TensorDataset(p,y),batch_size=3);a=Predictions(BASE);b=Predictions(VARIANT)
 args=(loader,torch.empty(2,0,dtype=torch.long),torch.empty(0),make_loss(),torch.device('cpu'))
 assert validate_epoch(a,*args)==validate_epoch(b,*args)
 assert _prediction_loss(a,p,y,make_loss())==_prediction_loss(b,p,y,make_loss())
 e=p[:,MULTI_HORIZONS.index(5)].double()
 assert b._last_val5_diagnostic['MAE']==pytest.approx(e.abs().mean().item())


def test_mocked_formal_checkpoint_keeps_original_monitor(tmp_path,monkeypatch):
 mod=importlib.import_module('cmgm.training.train');m=nn.Linear(1,1);m.variant=VARIANT
 values=iter([3.,2.,4.]);secondary=iter([.3,.4,.1]);seen=[]
 def validate(model,*args,**kwargs):
  model._last_val5_diagnostic=dict(MAE=next(secondary),MSE=1.,count=10);return next(values)
 monkeypatch.setattr(mod,'train_epoch',lambda *a,**k:1.)
 monkeypatch.setattr(mod,'validate_epoch',validate)
 original=mod.optim.lr_scheduler.ReduceLROnPlateau.step
 def step(scheduler,value,*a,**k):seen.append(value);return original(scheduler,value,*a,**k)
 monkeypatch.setattr(mod.optim.lr_scheduler.ReduceLROnPlateau,'step',step)
 hist=mod.train(m,[],[],torch.empty(2,0),torch.empty(0),torch.device('cpu'),num_epochs=3,checkpoint_path=str(tmp_path/'formal.pt'))
 assert seen==[3.,2.,4.] and hist['best_epoch']==2 and hist['best_val5_epoch']==3
 assert torch.load(tmp_path/'formal.pt',weights_only=False)['best_epoch']==2


def test_full_loader_controls_residual_report_and_error_arithmetic(tmp_path):
 m=model().eval()
 with torch.no_grad():m.pregnn_local_residual[-1].weight.fill_(.01)
 x=torch.randn(5,20,9,21);y=torch.randn(5,4,3)*.03
 loaders={s:DataLoader(TensorDataset(x,y),batch_size=3,drop_last=True) for s in ('train','val','test')}
 arrays,permutation=collect_with_controls(m,loaders,torch.device('cpu'))
 assert len(arrays['train']['target'])==5 # must retain incomplete last batch
 base={s:dict(prediction=v['base'][:,MULTI_HORIZONS.index(5)],target=v['target'][:,MULTI_HORIZONS.index(5)]) for s,v in arrays.items()}
 names=['焦煤','低硫燃料油','小麦'];res=residual_analysis(arrays,names);ablation=ablation_analysis(base,arrays)
 assert all(p['base_pred_max_diff']==0 for p in ablation['test']['impacts'].values())
 assert ablation['test']['representation_training_effect']['MAE']==0
 r=dict(status='SYNTHETIC TEST ONLY',training_executed=True,sanity=dict(PASS=True,shared_init=dict(delta_params=4256,shared_parameter_init_max_diff=0,mismatch_count=0,eval_prediction_max_diff=0,residual_max_abs=0,base_final_max_diff=0),commodity_order=dict(PASS=True)),
   gradients={},residual={'best':{'final_weight_norm':.1},'full_loader':res},ablations=ablation,shuffle_permutation=permutation)
 r.update(build_comparison(base,primary_arrays(arrays),names,r,tmp_path))
 assert len(r['commodity_metrics'])==3 and len(r['target_magnitude_groups'])==8
 removed=r['leave_one_commodity_out']['MAE']['removed_index'];keep=np.arange(3)!=removed
 expected=np.mean(np.abs(arrays['test']['native'][:,1,keep]-arrays['test']['target'][:,1,keep]))
 assert r['leave_one_commodity_out']['MAE']['splits']['test']['PreGNNLocalSkip']['MAE']==pytest.approx(expected)
 r.update(best_epoch=1,train_time_seconds=0.,best_formal_val_objective=.01,history=dict(val5_diagnostic=[dict(MAE=.01)],best_val5_epoch=1,best_val5_mae=.01),baseline_reference=dict(PASS=True,actual=r['metrics']['D0B']['test']))
 write_report(r,tmp_path);text=(tmp_path/'REPORT.md').read_text()
 assert '17. Primary classification:' in text and 'Direct vs representation effect' in text
 r.pop('metrics');write_report(r,tmp_path)
 assert 'No primary case before formal evaluation' in (tmp_path/'REPORT.md').read_text()


@pytest.mark.parametrize('case',['A','B','C','D','E','F','G'])
def test_mechanism_classification_requires_controls_and_reports_corner_case(case):
 from cmgm.scripts.d0b_pregnn_report import classify
 baseline=dict(MAE=1.,MSE=1.,RMSE=1.,Hit=.5)
 new=dict(baseline,MAE=.8,MSE=.8)
 zero=dict(baseline,MAE=.9,MSE=.9)
 if case=='B':zero=baseline.copy()
 if case=='C':zero=new.copy()
 if case=='D':new['MSE']=1.1
 if case=='E':new.update(MAE=1.1,MSE=1.1)
 if case=='F':new=baseline.copy();zero=baseline.copy()
 metrics={'D0B':{'val':baseline,'test':baseline},'PreGNNLocalSkip':{'val':new,'test':new}}
 impacts={name:dict(prediction_mean_diff=0. if case=='F' else .01) for name in ('ShuffledLocal','MeanLocal')}
 r=dict(sanity=dict(PASS=True),residual={'best':{'final_weight_norm':0. if case=='F' else .1},
   'full_loader':{'test':dict(primary_5d=dict(mean_abs=0. if case=='F' else .01),mean_abs_residual_over_base_5d=0. if case=='F' else .2)}},
   ablations={'test':dict(metrics={'Full':new,'ZeroResidual':zero,'OriginalD0B':baseline,
     'ShuffledLocal':dict(baseline,MAE=1.2,MSE=1.2),'MeanLocal':dict(baseline,MAE=1.2,MSE=1.2)},impacts=impacts)})
 leave={'MAE':{'splits':{'test':{'delta':{'MAE':.01 if case=='G' else -.1}}}}}
 assert classify(metrics,r,leave)['primary_case']=='Case '+case
 if case=='A':
  r['ablations']['test']['impacts']['MeanLocal']['prediction_mean_diff']=0.
  assert classify(metrics,r,leave)['primary_case'] is None # active residual alone is not local-information evidence
