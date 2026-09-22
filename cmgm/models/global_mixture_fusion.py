"""Static router control for the candidate representation MoE."""
import torch
from torch import nn
from cmgm.models.candidate_moe_fusion import TemporalResidualExpert,SpatialTemporalInteractionExpert

VARIANT='switching_latent_balanced_candidate_2expert_global_mixture'


class GlobalTwoExpertMixture(nn.Module):
    def __init__(self):
        super().__init__()
        # Exact Dynamic construction order for the two unchanged experts.
        self.temporal_expert=TemporalResidualExpert()
        self.interaction_expert=SpatialTemporalInteractionExpert()
        self.global_mixture_logits=nn.Parameter(torch.zeros(2))

    def forward(self,s,t):
        if s.ndim!=2 or s.shape!=t.shape or s.shape[-1]!=64:raise ValueError('Expected B×64 projected branches')
        e_t=self.temporal_expert(t);e_st=self.interaction_expert(torch.cat([s,t],-1))
        alpha=self.global_mixture_logits.softmax(dim=0)
        fused=(alpha.view(1,2,1)*torch.stack([e_t,e_st],dim=1)).sum(dim=1)
        self.last={k:v.detach() for k,v in dict(s=s,t=t,e_T=e_t,e_ST=e_st,alpha=alpha,
                   global_logits=self.global_mixture_logits,fused=fused).items()}
        return fused

    @torch.no_grad()
    def weight_diagnostics(self):
        a=self.global_mixture_logits.softmax(0);l=self.global_mixture_logits
        return dict(alpha_T=float(a[0]),alpha_ST=float(a[1]),global_logit_T=float(l[0]),global_logit_ST=float(l[1]))
