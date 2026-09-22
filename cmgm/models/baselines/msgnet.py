"""MSGNet core adapted from YoZhibo/MSGNet. No external repository required at runtime."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_fft(x,k):
    """Select non-DC frequencies per sample, never by batch-average amplitude."""
    amplitude=torch.fft.rfft(x,dim=1).abs().mean(-1)
    if k>amplitude.shape[1]-1:raise ValueError('Not enough nonzero FFT bins')
    # Stable tie breaking makes zero/constant windows well-defined without selecting DC.
    frequencies=amplitude[:,1:].argsort(dim=-1,descending=True,stable=True)[:,:k]+1
    return frequencies,amplitude.gather(1,frequencies)


class ScaleGraph(nn.Module):
    """Official latent-to-node graph mapping, MixHop and residual graph refinement."""
    def __init__(self):
        super().__init__()
        self.nodevec1=nn.Parameter(torch.randn(28,10));self.nodevec2=nn.Parameter(torch.randn(10,28))
        self.start=nn.Conv2d(1,32,(32-28+1,1))
        self.mix=nn.Conv2d(3*32,32,1)
        self.end=nn.Conv2d(32,20,(1,20));self.linear=nn.Linear(28,32);self.norm=nn.LayerNorm(32)

    def adjacency(self):return F.relu(self.nodevec1@self.nodevec2).softmax(-1)

    def forward(self,x):
        a=self.adjacency();self.last_adjacency=a.detach()
        a=a+torch.eye(28,device=x.device,dtype=x.dtype);a=a/a.sum(1,keepdim=True)
        origin=self.start(x.transpose(1,2).unsqueeze(1));h=origin;hops=[h]
        for _ in range(2):
            h=.3*origin+.7*torch.einsum('bcnt,mn->bcmt',h,a);hops.append(h)
        h=F.gelu(self.mix(torch.cat(hops,1)))
        h=self.end(h).squeeze(-1)  # Keep B axis even for a single sample.
        return self.norm(x+self.linear(h))


class ScaleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=nn.MultiheadAttention(32,8,dropout=.1,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.Dropout(.1))
        self.dropout=nn.Dropout(.1);self.norm1=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
        self.outnorm=nn.LayerNorm(32)

    def forward(self,x):
        # Canonical within-period causal attention; full-window FFT itself is not prefix causal.
        mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device).triu(1)
        attended=self.attention(x,x,x,attn_mask=mask,need_weights=False)[0]
        x=self.norm1(x+self.dropout(attended));x=self.norm2(x+self.ffn(x))
        return F.gelu(self.outnorm(x))


class MSGNetCore(nn.Module):
    def __init__(self):
        super().__init__();self.top_k=3
        # Canonical value embedding and absolute positions, no timestamp covariates.
        self.embedding=nn.Conv1d(28,32,3,padding=1,padding_mode='circular',bias=False)
        nn.init.kaiming_normal_(self.embedding.weight,mode='fan_in',nonlinearity='leaky_relu')
        pos=torch.arange(20).float()[:,None];freq=torch.exp(torch.arange(0,32,2)*(-math.log(10000.)/32))
        pe=torch.zeros(20,32);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe);self.dropout=nn.Dropout(.1)
        self.graphs=nn.ModuleList([ScaleGraph() for _ in range(3)])
        self.attentions=nn.ModuleList([ScaleAttention() for _ in range(3)])
        self.norm=nn.LayerNorm(32);self.output_projection=nn.Linear(32,28)
        self.horizon_head=nn.Linear(20,4);self.output_dropout=nn.Dropout(.1)

    def forward(self,x):
        if x.shape[1:]!=(20,28):raise ValueError('MSGNet expects B,20,28')
        h=self.dropout(self.embedding(x.transpose(1,2)).transpose(1,2)+self.position)
        frequencies,amplitudes=sample_fft(h,self.top_k);periods=torch.div(20,frequencies,rounding_mode='floor')
        paths=[]
        for rank,(graph,attention) in enumerate(zip(self.graphs,self.attentions)):
            g=graph(h);result=torch.zeros_like(g)
            # Batch grouping is solely an execution optimization; each sample selects its own periods.
            for period in torch.unique(periods[:,rank]).tolist():
                indices=torch.where(periods[:,rank]==period)[0];group=g.index_select(0,indices)
                length=math.ceil(20/period)*period;group=F.pad(group,(0,0,0,length-20))
                v=attention(group.reshape(-1,period,32)).reshape(len(indices),length,32)[:,:20]
                result=result.index_copy(0,indices,v)
            paths.append(result)
        weights=amplitudes.softmax(-1)
        encoded=self.norm(h+(torch.stack(paths,-1)*weights[:,None,None,:]).sum(-1))
        nodes=self.output_projection(encoded).transpose(1,2)
        prediction=self.output_dropout(self.horizon_head(nodes)).transpose(1,2)
        self.last=dict(frequencies=frequencies.detach(),periods=periods.detach(),scale_weights=weights.detach(),fft_amplitudes=amplitudes.detach())
        return prediction  # Slots are 1/5/10/20d, without input-level de-normalization.
