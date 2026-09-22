"""Device-agnostic CrossGNN core adaptation; per-sample observed-window scale discovery."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .msgnet import sample_fft


class CrossGraphPropagation(nn.Module):
    def __init__(self,axis):
        super().__init__();self.axis=axis;self.projection=nn.Linear(24,8);self.dropout=nn.Dropout(.05)

    def forward(self,x,a):
        def propagate(v):
            return torch.einsum('btnd,btw->bwnd',v,a) if self.axis=='time' else torch.einsum('btnd,nm->btmd',v,a)
        first=propagate(x);second=propagate(first)
        return self.dropout(F.gelu(self.projection(torch.cat([x,first,second],-1))))


class CrossGNNCore(nn.Module):
    def __init__(self):
        super().__init__();self.anti_ood=False;self.scale_number=4;self.tk=10
        self.timevec1=nn.Parameter(torch.randn(40,1));self.timevec2=nn.Parameter(torch.randn(1,40))
        self.nodevec1=nn.Parameter(torch.randn(28,1));self.nodevec2=nn.Parameter(torch.randn(1,28))
        self.start=nn.Linear(1,8);self.time_graph=CrossGraphPropagation('time');self.variable_graph=CrossGraphPropagation('variable')
        self.refine=nn.Linear(16,1);self.dropout=nn.Dropout(.05);self.horizon_head=nn.Linear(20,4)

    def variable_adjacency(self):
        scores=F.relu(self.nodevec1@self.nodevec2)
        positive=scores>=scores.topk(3,dim=-1).values[:,-1:]
        negative=scores<=scores.kthvalue(3,dim=-1).values[:,None]
        # Official signed construction: positive softmax(score), negative softmax(1/(score+1)).
        pos=scores.masked_fill(~positive,-torch.inf).softmax(-1)
        neg=(1/(scores+1)).masked_fill(~negative,-torch.inf).softmax(-1)
        return pos-neg

    def time_adjacency(self,periods):
        scores=F.relu(self.timevec1@self.timevec2);masks=[]
        for row in periods.tolist():
            mask=torch.ones_like(scores,dtype=torch.bool);start=0
            for period in row:
                end=min(40,start+20//period);length=end-start
                if length<=0:break
                k=min(length,max(self.tk//period,5));block=scores[:,start:end]
                mask[:,start:end]=block<block.topk(k,dim=-1).values[:,-1:];start=end
                if start==40:break
            # Padded slots follow reference logic and remain valid zero-valued observed-window slots.
            mask[:,start:]=False
            indices=torch.arange(39,device=scores.device)
            mask[indices,indices+1]=False;mask[indices+1,indices]=False
            masks.append(mask)
        # Reference column normalization: for each destination normalize across source time nodes.
        return scores.unsqueeze(0).masked_fill(torch.stack(masks),-torch.inf).softmax(dim=1)

    @staticmethod
    def multiscale(x,periods):
        samples=[]
        for i,row in enumerate(periods.tolist()):
            views=[F.avg_pool1d(x[i:i+1].transpose(1,2),p,stride=p).transpose(1,2) for p in row]
            combined=torch.cat(views,1)[:,:40]
            samples.append(F.pad(combined,(0,0,0,40-combined.shape[1])))
        return torch.cat(samples,0)

    def forward(self,x):
        if x.shape[1:]!=(20,28):raise ValueError('CrossGNN expects B,20,28')
        frequencies,amplitudes=sample_fft(x,self.scale_number)
        periods=torch.cat([torch.ones_like(frequencies[:,:1]),torch.div(20+frequencies-1,frequencies,rounding_mode='floor')],1)
        multiscale=self.multiscale(x,periods);time_a=self.time_adjacency(periods);var_a=self.variable_adjacency()
        origin=self.start(multiscale.unsqueeze(-1));h=origin+self.time_graph(origin,time_a)
        h=h+self.variable_graph(h,var_a)
        states=self.dropout(self.refine(torch.cat([origin,h],-1)).squeeze(-1))[:,:20]
        self.last=dict(frequencies=frequencies.detach(),periods=periods.detach(),multi_scale=multiscale.detach(),
                       time_adjacency=time_a.detach(),variable_adjacency=var_a.detach(),fft_amplitudes=amplitudes.detach())
        return self.horizon_head(states.transpose(1,2)).transpose(1,2)
