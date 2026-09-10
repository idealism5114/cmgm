"""Fixed neutral-input baselines. No D0B encoder/fusion/attention components."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from cmgm.graph.adaptive_graph import AdaptiveGraphLearner
from cmgm.models.model import MixHopPropagation

HORIZONS=(1,5,10,20)
TRAINABLE=('Linear','GRU','VanillaTransformer','iTransformer','MTGNN')
ORDER=('ZeroReturn',)+TRAINABLE


class BaselineMultimodalAdapter(nn.Module):
    """Population std (correction=0); token-major then feature-major flatten."""
    def __init__(self,n_stock=248,n_bond=12,n_commodities=24,features=21):
        super().__init__()
        self.n_stock=n_stock;self.n_bond=n_bond;self.n_commodities=n_commodities;self.features=features
        if min(n_stock,n_bond)<1 or n_commodities!=24 or features!=21:raise ValueError('Expected nonempty markets and 24 commodities ×21 features')
    def forward(self,x):
        if x.ndim!=4 or x.shape[2:]!=(self.n_stock+self.n_bond+self.n_commodities,self.features):raise ValueError('Expected (B,T,N,21) with declared market ordering')
        stock=x[:,:,:self.n_stock];bond=x[:,:,self.n_stock:self.n_stock+self.n_bond];comm=x[:,:,self.n_stock+self.n_bond:]
        summaries=torch.stack([stock.mean(2),stock.std(2,unbiased=False),bond.mean(2),bond.std(2,unbiased=False)],dim=2)
        return torch.cat([summaries,comm],dim=2)
    def sequence(self,x):return self(x).flatten(2)


class Base(nn.Module):
    def __init__(self,**market):
        super().__init__();self.adapter=BaselineMultimodalAdapter(**market)
    def check(self,p,b):
        if p.shape!=(b,4,24):raise ValueError(f'Wrong baseline output shape {p.shape}')
        return p


class ZeroReturn(Base):
    def forward(self,x):return x.new_zeros((len(x),4,24))


class LinearBaseline(Base):
    def __init__(self,**market):super().__init__(**market);self.head=nn.Linear(20*588,96)
    def forward(self,x):return self.check(self.head(self.adapter.sequence(x).flatten(1)).view(len(x),4,24),len(x))


class GRUBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.gru=nn.GRU(588,128,num_layers=2,dropout=.1,batch_first=True);self.head=nn.Linear(128,96)
    def temporal_states(self,x):return self.gru(self.adapter.sequence(x))[0]
    def forward(self,x):
        _,hidden=self.gru(self.adapter.sequence(x))
        return self.check(self.head(hidden[-1]).view(len(x),4,24),len(x))


class VanillaTransformerBaseline(Base):
    def __init__(self,**market):
        super().__init__(**market);self.input_projection=nn.Linear(588,128)
        position=torch.arange(20,dtype=torch.float32)[:,None];frequency=torch.exp(torch.arange(0,128,2)*(-math.log(10000.)/128))
        pe=torch.zeros(20,128);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position_encoding',pe)
        layer=nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(128,96)
    def temporal_states(self,x):
        h=self.input_projection(self.adapter.sequence(x))+self.position_encoding[:x.shape[1]]
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        return self.encoder(h,mask=mask)
    def forward(self,x):return self.check(self.head(self.temporal_states(x)[:,-1]).view(len(x),4,24),len(x))


class ITransformerBaseline(Base):
    """Specified inverted-token baseline, not a verbatim official iTransformer reproduction."""
    def __init__(self,**market):
        super().__init__(**market);self.history_projection=nn.Linear(20,128)
        layer=nn.TransformerEncoderLayer(128,4,256,.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(128,4)
    @staticmethod
    def commodity_pool(encoded):
        if encoded.shape[1:]!=(588,128):raise ValueError('Expected exactly 588 variate tokens')
        return encoded.reshape(len(encoded),28,21,128)[:,4:].mean(dim=2)
    def forward(self,x):
        inverted=self.adapter.sequence(x).transpose(1,2)
        if inverted.shape[1:]!=(588,20):raise ValueError('iTransformer requires 588 observed-history tokens of length 20')
        encoded=self.encoder(self.history_projection(inverted))
        return self.check(self.head(self.commodity_pool(encoded)).permute(0,2,1),len(x))


class GraphTemporalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.left_padding=2*dilation
        self.filter=nn.Conv2d(64,64,(1,3),dilation=(1,dilation))
        self.gate=nn.Conv2d(64,64,(1,3),dilation=(1,dilation))
        self.forward_graph=MixHopPropagation(64,64,K=2,beta=.05)
        self.reverse_graph=MixHopPropagation(64,64,K=2,beta=.05)
    def forward(self,x,A):
        # x=(B,T,N,C); padding touches time only, never batch/node.
        padded=F.pad(x.permute(0,3,2,1),(self.left_padding,0,0,0))
        temporal=(torch.tanh(self.filter(padded))*torch.sigmoid(self.gate(padded))).permute(0,3,2,1)
        return F.relu(x+self.forward_graph(temporal,A)+self.reverse_graph(temporal,A.T))


class MTGNNBaseline(Base):
    """Simplified MTGNN-style baseline with reused independent graph primitives."""
    def __init__(self,**market):
        super().__init__(**market);self.input_projection=nn.Linear(21,64)
        self.graph=AdaptiveGraphLearner(28,embed_dim=10,alpha=.5,top_k=10)
        self.blocks=nn.ModuleList([GraphTemporalBlock(1),GraphTemporalBlock(2)]);self.head=nn.Linear(64,4)
    def temporal_states(self,x):
        h=self.input_projection(self.adapter(x));A=self.graph()
        for block in self.blocks:h=block(h,A)
        return h
    def forward(self,x):return self.check(self.head(self.temporal_states(x)[:,-1,4:]).permute(0,2,1),len(x))


MODELS=dict(ZeroReturn=ZeroReturn,Linear=LinearBaseline,GRU=GRUBaseline,VanillaTransformer=VanillaTransformerBaseline,iTransformer=ITransformerBaseline,MTGNN=MTGNNBaseline)


def make_model(name,market_indices):
    if name not in MODELS:raise ValueError('Only the six prespecified baselines are supported')
    return MODELS[name](n_stock=market_indices['stock'][1],n_bond=market_indices['bond'][1]-market_indices['bond'][0],n_commodities=market_indices['commodity'][1]-market_indices['commodity'][0])
