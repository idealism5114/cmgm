"""Four information-source prediction experts; TRAIN-only router-only utility."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

VARIANT = 'switching_latent_balanced_4source_utility_moe'
LABELS = ('LongMemory', 'Microstate', 'SpatialGraph', 'Joint')
EPS, TAU, LAMBDA_ROUTE = 1e-8, 1.0, .1

class SourceRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(3)])
        self.network = nn.Sequential(nn.Linear(192,64), nn.ReLU(), nn.Linear(64,4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, long, micro, spatial):
        return self.network(torch.cat([n(x) for n,x in zip(self.norms,(long,micro,spatial))],-1)).softmax(-1)

@torch.no_grad()
def utility_targets(predictions, target):
    if predictions.shape != (len(target),4,4,24) or target.shape[1:] != (4,24):
        raise ValueError('Expected expert-major B,4,4,24 and B,4,24 targets')
    error = F.huber_loss(predictions,target[:,None].expand_as(predictions),delta=.02,reduction='none').mean(-1).sum(-1)
    q = torch.softmax(-error/(error.mean(-1,keepdim=True)+EPS)/TAU,-1)
    return q.detach(), error.mean().detach(), error.detach()

class FourSourcePredictiveMoE(nn.Module):
    def __init__(self, shared_head):
        super().__init__()
        self.experts = nn.ModuleList([copy.deepcopy(shared_head) for _ in range(3)])
        self.experts.append(nn.Sequential(nn.Linear(192,64),nn.ReLU(),nn.Dropout(.3),copy.deepcopy(shared_head[3])))
        self.router = SourceRouter()

    def forward(self, long, micro, spatial):
        if any(x.shape != (len(long),64) for x in (long,micro,spatial)):
            raise ValueError('Sources must have shape B,64')
        self.components = (long,micro,spatial)
        inputs = (*self.components,torch.cat(self.components,-1))
        self.predictions = torch.stack([e(x).reshape(-1,4,24) for e,x in zip(self.experts,inputs)],1)
        self.pi = self.router(*self.components)
        return (self.pi[:,:,None,None]*self.predictions).sum(1)

    def utility_loss(self,target):
        if not self.training:
            raise ValueError('Utility optimization is TRAIN-only')
        q,scale,error = utility_targets(self.predictions,target)
        pi_aux = self.router(*(x.detach() for x in self.components))
        raw = (torch.special.xlogy(q,q)-q*pi_aux.clamp_min(torch.finfo(pi_aux.dtype).tiny).log()).sum(-1).mean()
        loss = LAMBDA_ROUTE*scale*raw
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite utility')
        return loss, dict(raw_utility_KL=float(raw.detach()),utility_scale=float(scale),weighted_utility_loss=float(loss.detach()),
                         q=q, pi_aux=pi_aux.detach(),expert_error=error)

class SourceAccumulator:
    def __init__(self):
        self.rows=[];self.pi=[];self.q=[]

    def update(self,model,prediction_loss,total_loss,diagnostics=None):
        f=model.four_source_moe;b=model.switching_latent_transformer
        self.pi.append(f.pi.detach().cpu())
        row=dict(prediction_loss=float(prediction_loss),total_loss=float(total_loss),
                 raw_switch_KL=float(b.regime_filter._last_switch_loss.detach()),weighted_switch_KL=float(b.switch_loss().detach()))
        if diagnostics is not None:
            self.q.append(diagnostics['q'].cpu())
            row.update({k:diagnostics[k] for k in ('raw_utility_KL','utility_scale','weighted_utility_loss')})
        self.rows.append(row)

    def summary(self):
        result={k:sum(r[k] for r in self.rows)/len(self.rows) for k in self.rows[0]}
        for label,rows in (('pi',self.pi),('q',self.q)):
            if rows:
                p=torch.cat(rows).double()
                result[label]=dict(mean=p.mean(0).tolist(),std=p.std(0,unbiased=False).tolist(),entropy=float(-(torch.special.xlogy(p,p)).sum(-1).mean()))
        result['utility_applied']=bool(self.q)
        return result
