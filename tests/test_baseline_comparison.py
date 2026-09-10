import copy
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cmgm.models.comparison_baselines import BaselineMultimodalAdapter,ORDER,make_model,ITransformerBaseline
from cmgm.scripts.baseline_protocol import parameter_counts,sanity,prediction_loss,validation,train_one,evaluate
from cmgm.scripts.baseline_report import write_report

MI=dict(stock=(0,4),bond=(4,6),commodity=(6,30))


def test_neutral_adapter_population_std_and_layout():
 a=BaselineMultimodalAdapter(4,2);x=torch.arange(2*20*30*21,dtype=torch.float32).reshape(2,20,30,21)
 tokens=a(x);assert tokens.shape==(2,20,28,21) and sum(p.numel() for p in a.parameters())==0
 torch.testing.assert_close(tokens[:,:,0],x[:,:,:4].mean(2));torch.testing.assert_close(tokens[:,:,1],x[:,:,:4].std(2,unbiased=False))
 torch.testing.assert_close(tokens[:,:,2],x[:,:,4:6].mean(2));torch.testing.assert_close(tokens[:,:,3],x[:,:,4:6].std(2,unbiased=False))
 torch.testing.assert_close(tokens[:,:,4:],x[:,:,6:],atol=0,rtol=0)
 seq=a.sequence(x)
 for i in range(24):torch.testing.assert_close(seq[:,:,(4+i)*21:(5+i)*21],x[:,:,6+i],atol=0,rtol=0)
 singleton=BaselineMultimodalAdapter(1,1)(torch.ones(2,20,26,21))
 assert torch.count_nonzero(singleton[:,:,1])==torch.count_nonzero(singleton[:,:,3])==0


@pytest.mark.parametrize('name',ORDER)
def test_shapes_causal_batch_and_no_prohibited_modules(name):
 torch.manual_seed(42);m=make_model(name,MI);x=torch.randn(3,20,30,21)
 original={k:v.clone() for k,v in m.state_dict().items()};result=sanity(m,x)
 assert result['PASS'],result
 assert m(x).shape==(3,4,24)
 for k,v in original.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
 forbidden=('MarketAware','EdgeAttn','Switching','Balanced','TempWeighted')
 assert all(not any(word in type(mod).__name__ for word in forbidden) for mod in m.modules())
 assert all(p.grad is None for p in m.parameters())
 if name=='Linear':assert parameter_counts(m)['trainable']==11760*96+96
 if name=='MTGNN':assert m.graph.num_nodes==28
 if name=='VanillaTransformer':assert m.position_encoding.shape==(20,128)


def test_itransformer_commodity_feature_token_offsets():
 values=torch.arange(588,dtype=torch.float32)[None,:,None].expand(2,588,128)
 pooled=ITransformerBaseline.commodity_pool(values)
 for i in range(24):torch.testing.assert_close(pooled[:,i],torch.full((2,128),float((4+i)*21+10)))


def test_mtgnn_node_readout_and_gru_final_hidden_semantics():
 torch.manual_seed(42);x=torch.randn(2,20,30,21)
 m=make_model('MTGNN',MI).eval()
 with torch.no_grad():expected=m.head(m.temporal_states(x)[:,-1,4:]).permute(0,2,1);torch.testing.assert_close(m(x),expected)
 m=make_model('GRU',MI).eval()
 with torch.no_grad():
  _,h=m.gru(m.adapter.sequence(x));torch.testing.assert_close(m(x),m.head(h[-1]).view(2,4,24))


def test_loss_and_validation_match_legacy_batch_mean_objective():
 from cmgm.training.train import _prediction_loss,make_loss
 class Identity(nn.Module):
  def forward(self,x):return x
 p=torch.arange(5*4*24,dtype=torch.float32).reshape(5,4,24)*.0001;y=torch.zeros_like(p);m=Identity()
 torch.testing.assert_close(prediction_loss(p,y),_prediction_loss(m,p,y,make_loss()),atol=0,rtol=0)
 loader=DataLoader(TensorDataset(p,y),batch_size=3)
 value,secondary=validation(m,loader,torch.device('cpu'))
 expected=(prediction_loss(p[:3],y[:3])+prediction_loss(p[3:],y[3:]))/2
 assert value==pytest.approx(float(expected))
 idx=(1,5,10,20).index(5);assert secondary['MAE']==pytest.approx(float(p[:,idx].double().abs().mean()))


def test_formal_selection_and_optimizer_protocol_without_parameter_updates(tmp_path,monkeypatch):
 import cmgm.scripts.baseline_protocol as protocol
 class Fixture(nn.Module):
  def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(4,24)*.01)
  def forward(self,x):return self.p.unsqueeze(0).expand(len(x),-1,-1)
 m=Fixture();before=m.p.detach().clone();calls=[];values=iter([3.,2.,4.]);secondary=iter([.3,.4,.1])
 monkeypatch.setattr(protocol,'validation',lambda *a:(next(values),dict(MAE=next(secondary),MSE=.1)))
 def no_update(opt,*a,**k):
  assert opt.param_groups[0]['lr']==1e-4 and opt.param_groups[0]['weight_decay']==1e-5;calls.append(1)
 monkeypatch.setattr(torch.optim.Adam,'step',no_update)
 loader=DataLoader(TensorDataset(torch.zeros(2,1),torch.zeros(2,4,24)),batch_size=2)
 summary,history=train_one(m,loader,loader,torch.device('cpu'),tmp_path/'fixture.pt',dict(model='mock fixture'),max_epochs=3)
 assert summary['best_epoch']==2 and summary['secondary_best_val5_epoch']==3 and len(calls)==3
 torch.testing.assert_close(m.p,before,atol=0,rtol=0)
 saved=torch.load(tmp_path/'fixture.pt',weights_only=False);assert saved['best_val_loss']==2. and saved['training_complete']


def test_unified_zero_hit_includes_zero_targets_all_horizons():
 m=make_model('ZeroReturn',MI);x=torch.ones(5,20,30,21);y=torch.ones(5,4,24)*.1;y[0]=0
 result=evaluate(m,{'test':DataLoader(TensorDataset(x,y),batch_size=3)},torch.device('cpu'),[str(i) for i in range(24)])
 assert len(result['per_commodity'])==24
 for values in result['metrics']['test'].values():
  assert values['Hit']==.2 and values['MAE']==pytest.approx(.08) and values['RMSE']**2==pytest.approx(values['MSE'])


def test_report_ranks_negative_improvement_truthfully(tmp_path):
 r=dict(status='SYNTHETIC',protocol={},models={},sanity={},model_status={},reference_check=dict(PASS=True))
 for name,mae in zip((*ORDER,'D0B'),(1.,.8,.7,.6,.5,.4,.55)):
  metrics={s:{str(h):dict(MAE=mae,MSE=mae**2,RMSE=mae,Hit=.5) for h in (1,5,10,20)} for s in ('train','val','test')}
  r['models'][name]=dict(parameters=dict(trainable=1,nontrainable=0,buffer_elements=0),training=dict(best_epoch=2,train_seconds=1.,seconds_per_epoch_mean=.5),input_view='fixture',metrics=metrics,
      per_commodity=[dict(commodity=str(i),MAE=mae,MSE=mae**2) for i in range(24)])
  if name!='D0B':r['sanity'][name]=dict(PASS=True);r['model_status'][name]='COMPLETE'
 write_report(r,tmp_path);text=(tmp_path/'REPORT.md').read_text()
 assert 'Strongest baseline: MTGNN' in text and 'D0B rank: 3 of 7' in text and 'The baseline outperforms D0B' in text
 assert r['tables']['overall'][0]['Model']=='MTGNN' and r['tables']['overall'][0]['D0B_improvement_percent']<0
 assert len(r['tables']['all_horizons'])==84


def test_completed_training_recovery_validates_identity_without_optimizer(tmp_path,monkeypatch):
 from cmgm.scripts.baseline_comparison import completed_training,PROTOCOL
 monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('Recovery must not create optimizer'))
 path=tmp_path/'linear_best.pt';audit=dict(split_fingerprint={'train':'fixture'})
 metadata=dict(model='Linear',protocol=PROTOCOL,data_fingerprint=audit['split_fingerprint'])
 torch.save(dict(training_complete=True,metadata=metadata,training_summary={'best_epoch':2}),path)
 assert completed_training(path,'Linear',audit)['training_summary']['best_epoch']==2
 with pytest.raises(ValueError):completed_training(path,'GRU',audit)
 with pytest.raises(ValueError):completed_training(path,'Linear',dict(split_fingerprint={}))
 torch.save(dict(training_complete=False),path)
 assert completed_training(path,'Linear',audit) is None


def test_flush_saves_current_tables_not_stale_tables(tmp_path,monkeypatch):
 import json
 import cmgm.scripts.baseline_report as report
 from cmgm.scripts.baseline_comparison import flush
 r=dict(sanity={},tables={'old':1})
 monkeypatch.setattr(report,'write_report',lambda result,out:result.update(tables={'current':2}))
 flush(r,tmp_path)
 assert json.loads((tmp_path/'results.json').read_text())['tables']=={'current':2}


@pytest.mark.parametrize('mse',[0.,1e-20,.000912875418,123456789.123,1e100])
def test_rmse_identity_tolerance_scales_with_float64_mse(mse):
 import math
 from cmgm.scripts.baseline_protocol import check_metric_identity
 check_metric_identity(dict(MSE=mse,RMSE=math.sqrt(mse)))
 if mse:
  with pytest.raises(AssertionError):check_metric_identity(dict(MSE=mse,RMSE=math.sqrt(mse)*1.001))


def test_rmse_identity_rejects_nonfinite_metrics():
 from cmgm.scripts.baseline_protocol import check_metric_identity
 for value in (float('nan'),float('inf')):
  with pytest.raises(FloatingPointError):check_metric_identity(dict(MSE=value,RMSE=value))
