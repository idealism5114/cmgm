"""Candidate-aware dense T/ST fusion; no schedule or auxiliary objective."""
import torch
from torch import nn

VARIANT = 'switching_latent_balanced_candidate_2expert_moe'


class TemporalResidualExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,64))

    def forward(self,t):
        return t+self.mlp(t)


class SpatialTemporalInteractionExpert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,64))


class CandidateAwareRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_norm=nn.LayerNorm(64)
        self.interaction_norm=nn.LayerNorm(64)
        self.network=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,e_t,e_st):
        u_t=self.temporal_norm(e_t);u_st=self.interaction_norm(e_st)
        difference=(u_t-u_st).abs();product=u_t*u_st
        inputs=torch.cat([u_t,u_st,difference,product],dim=-1)
        logits=self.network(inputs)
        self.last={k:v.detach() for k,v in dict(u_T=u_t,u_ST=u_st,abs_diff=difference,
                   product=product,router_input=inputs,router_logits=logits).items()}
        return logits.softmax(dim=-1)


class CandidateAwareTwoExpertFusion(nn.Module):
    """Experts consume projected branches; only the router normalizes candidates."""
    def __init__(self):
        super().__init__()
        self.temporal_expert=TemporalResidualExpert()
        self.interaction_expert=SpatialTemporalInteractionExpert()
        self.router=CandidateAwareRouter()

    def forward(self,s,t):
        if s.ndim!=2 or s.shape!=t.shape or s.shape[-1]!=64:
            raise ValueError('Projected branches must both have shape (B,64)')
        e_t=self.temporal_expert(t)
        e_st=self.interaction_expert(torch.cat([s,t],dim=-1))
        pi=self.router(e_t,e_st)
        experts=torch.stack([e_t,e_st],dim=1)
        fused=(pi.unsqueeze(-1)*experts).sum(dim=1)
        self.last={**self.router.last,**{k:v.detach() for k,v in dict(s=s,t=t,e_T=e_t,e_ST=e_st,
                   pi=pi,experts=experts,h_moe=fused).items()}}
        return fused


class CandidateRoutingAccumulator:
    """Detached sample-pooled diagnostics; never contributes to training loss."""
    def __init__(self):
        self.rows=[]

    def update(self,fusion):
        self.rows.append(fusion.last['pi'].double().cpu())

    def summary(self):
        if not self.rows:raise ValueError('No routing observations')
        p=torch.cat(self.rows)
        result=dict(samples=len(p),routing_entropy=float(-(p*(p+1e-8).log()).sum(-1).mean()))
        for i,label in enumerate(('T','ST')):
            for key,value in [('mean',p[:,i].mean()),('std',p[:,i].std(unbiased=False)),
                              ('min',p[:,i].min()),('max',p[:,i].max())]:
                result[f'{key}_pi_{label}']=float(value)
        return result


BOTTLENECK16_VARIANT = 'switching_latent_balanced_candidate_2expert_moe_bottleneck16'


class CandidateAwareBottleneck16Fusion(CandidateAwareTwoExpertFusion):
    """Only expert capacity changes; preserve the original router RNG sequence.

    All original classes above remain unchanged, including state keys and defaults.
    Build the complete original fusion BEFORE replacing experts. No trained weights
    are copied, no auxiliary objective or routing warm-up is introduced.
    """
    def __init__(self):
        super().__init__()
        # Preserve the original residual forward: t + MLP(t).
        self.temporal_expert.mlp = nn.Sequential(
            nn.Linear(64, 16), nn.ReLU(), nn.Dropout(.1), nn.Linear(16, 64))
        self.interaction_expert = nn.Sequential(
            nn.Linear(128, 16), nn.ReLU(), nn.Dropout(.1), nn.Linear(16, 64))
