"""Full-information V2 models; only lossless input layouts before model layers."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from cmgm.config import MULTI_HORIZONS

CLASSICAL = ('Ridge Regression', 'Random Forest', 'XGBoost')
NEURAL = ('LSTM', 'TCN', 'Vanilla Transformer', 'Graph WaveNet', 'MTGNN', 'D0B')
ORDER = CLASSICAL + NEURAL
KEYS = dict(zip(ORDER, ('ridge','rf','xgb','lstm','tcn','transformer','graphwavenet','mtgnn','d0b')))


def input_view(name, x):
    if x.ndim != 4 or x.shape[1] != 20 or x.shape[-1] != 21:
        raise ValueError('Expected the original B,20,N,21 tensor')
    if name in CLASSICAL:
        return x.flatten(1)
    if name in ('LSTM','TCN','Vanilla Transformer'):
        return x.flatten(2)
    if name in ('Graph WaveNet','MTGNN'):
        return x.permute(0,3,2,1).contiguous()
    if name == 'D0B':
        return x
    raise ValueError(name)


def restore_input(name, view, shape):
    return view.permute(0,3,2,1).contiguous() if name in ('Graph WaveNet','MTGNN') else view.reshape(shape)


class SequenceModel(nn.Module):
    def output(self, h):
        return self.head(h).reshape(len(h),len(MULTI_HORIZONS),24)


class LSTM(SequenceModel):
    def __init__(self,n):
        super().__init__()
        self.lstm = nn.LSTM(n*21,128,2,dropout=.1,batch_first=True)
        self.head = nn.Linear(128,96)
    def temporal_states(self,x):
        return self.lstm(input_view('LSTM',x))[0]
    def forward(self,x):
        _,(h,_) = self.lstm(input_view('LSTM',x))
        return self.output(h[-1])


class CausalConv(nn.Conv1d):
    def forward(self,x):
        return super().forward(F.pad(x,((self.kernel_size[0]-1)*self.dilation[0],0)))


class TCNBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.body = nn.Sequential(CausalConv(128,128,3,dilation=dilation),nn.ReLU(),nn.Dropout(.1),
                                  CausalConv(128,128,3,dilation=dilation),nn.ReLU(),nn.Dropout(.1))
    def forward(self,x):
        return F.relu(x+self.body(x))


class TCN(SequenceModel):
    def __init__(self,n):
        super().__init__()
        self.projection = nn.Conv1d(n*21,128,1)
        self.blocks = nn.Sequential(*(TCNBlock(d) for d in (1,2,4)))
        self.head = nn.Linear(128,96)
    def temporal_states(self,x):
        return self.blocks(self.projection(input_view('TCN',x).transpose(1,2))).transpose(1,2)
    def forward(self,x):
        return self.output(self.temporal_states(x)[:,-1])


class Transformer(SequenceModel):
    def __init__(self,n):
        super().__init__()
        self.projection = nn.Linear(n*21,128)
        p = torch.arange(20)[:,None]
        f = torch.exp(torch.arange(0,128,2)*(-math.log(10000.)/128))
        pe = torch.zeros(20,128);pe[:,0::2]=torch.sin(p*f);pe[:,1::2]=torch.cos(p*f)
        self.register_buffer('position',pe)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,.1,activation='relu',batch_first=True),2,enable_nested_tensor=False)
        self.head = nn.Linear(128,96)
    def temporal_states(self,x):
        h = self.projection(input_view('Vanilla Transformer',x))+self.position
        mask = torch.ones(20,20,dtype=torch.bool,device=x.device).triu(1)
        return self.encoder(h,mask=mask)
    def forward(self,x):
        return self.output(self.temporal_states(x)[:,-1])


class OfficialGraph(nn.Module):
    def __init__(self,name,n,commodity_indices,device):
        super().__init__();self.name=name
        self.register_buffer('commodity_indices',torch.tensor(commodity_indices,dtype=torch.long))
        if name=='Graph WaveNet':
            from third_party.baselines.graphwavenet.model import gwnet
            self.core=gwnet(device,n,supports=None,gcn_bool=True,addaptadj=True,in_dim=21,out_dim=4)
        else:
            from third_party.baselines.mtgnn.net import gtnet
            self.core=gtnet(True,True,2,n,device,seq_length=20,in_dim=21,out_dim=4)
    def forward(self,x):
        full = self.core(input_view(self.name,x))
        if full.shape[:3] != (len(x),4,x.shape[2]):
            raise ValueError('Official graph output is not B,4,N,L')
        if self.name=='MTGNN' and full.shape[-1]!=1:
            raise ValueError('MTGNN final temporal dimension must be one')
        return full[:,:,:,-1].index_select(2,self.commodity_indices)


def make_neural(name,data,device):
    n=data['n_nodes'];mi=data['market_indices'];cs,ce=mi['commodity']
    if ce-cs!=24:raise ValueError('Expected 24 commodity targets')
    if name=='D0B':
        from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
        m=HeteroMixHopCMGM(n,24,n_stock=mi['stock'][1],n_bond=mi['bond'][1]-mi['bond'][0],variant='switching_latent_balanced_readout')
    elif name in ('Graph WaveNet','MTGNN'):
        m=OfficialGraph(name,n,list(range(cs,ce)),device)
    else:
        m={'LSTM':LSTM,'TCN':TCN,'Vanilla Transformer':Transformer}[name](n)
    return m.to(device)
