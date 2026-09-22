"""Prediction-level dual experts with TRAIN-only, router-only utility supervision."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANT = 'switching_latent_balanced_utility_routed_moe'
EPS = 1e-8


class TemporalPredictiveExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,64))

    def forward(self,t):
        return t+self.mlp(t)


class SpatialTemporalPredictiveExpert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,64))


class UtilityAwareRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_norm=nn.LayerNorm(64)
        self.interaction_norm=nn.LayerNorm(64)
        self.network=nn.Sequential(nn.Linear(263,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,e_t,e_st,regime_prob,regime_prior):
        if regime_prob.shape!=(len(e_t),3) or regime_prior.shape!=regime_prob.shape:
            raise ValueError('Router requires the final observed K=3 posterior/prior')
        p=regime_prob.detach();prior=regime_prior.detach()
        gap=(p-prior).abs();entropy=-(p*(p+EPS).log()).sum(-1,keepdim=True)
        u=self.temporal_norm(e_t);v=self.interaction_norm(e_st)
        difference=(u-v).abs();product=u*v
        inputs=torch.cat([u,v,difference,product,p,gap,entropy],-1)
        logits=self.network(inputs);pi=logits.softmax(-1)
        self.context_requires_grad={k:t.requires_grad for k,t in dict(prob=p,prior=prior,gap=gap,entropy=entropy).items()}
        self.last={k:t.detach() for k,t in dict(u_T=u,u_ST=v,abs_diff=difference,product=product,
                   regime_prob=p,regime_prior=prior,regime_gap=gap,regime_entropy=entropy,
                   router_input=inputs,router_logits=logits,pi=pi).items()}
        return pi


class UtilityRoutedDualExpertFusion(nn.Module):
    """Heads stay on the main model. No representation mixture is predicted."""
    def __init__(self):
        super().__init__()
        self.temporal_expert=TemporalPredictiveExpert()
        self.interaction_expert=SpatialTemporalPredictiveExpert()
        self.router=UtilityAwareRouter()

    def forward(self,s,t,regime_prob,regime_prior):
        if s.ndim!=2 or s.shape!=t.shape or s.shape[-1]!=64:
            raise ValueError('Projected branches must be B×64')
        e_t=self.temporal_expert(t);e_st=self.interaction_expert(torch.cat([s,t],-1))
        pi=self.router(e_t,e_st,regime_prob,regime_prior)
        self._route_inputs=(e_t,e_st,regime_prob.detach(),regime_prior.detach())
        self.last={**self.router.last,**{k:v.detach() for k,v in dict(s=s,t=t,e_T=e_t,e_ST=e_st).items()}}
        return e_t,e_st,pi

    def router_only_pi(self):
        # q.detach alone would NOT isolate utility gradients: pi also depends on
        # live experts. Re-evaluate only the deterministic router on detached
        # candidates. Values match the original pi; all auxiliary gradients stop
        # before experts/heads/shared branches, including regime inference.
        return self.router(*(v.detach() for v in self._route_inputs))


def per_sample_huber(pred,target):
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[1:]!=(4,24):
        raise ValueError('Utility requires aligned B×4×24, all four horizons')
    return F.huber_loss(pred,target,delta=.02,reduction='none').mean(-1).sum(-1)


@torch.no_grad()
def utility_target(pred_t,pred_st,target):
    lt=per_sample_huber(pred_t,target);lst=per_sample_huber(pred_st,target)
    advantage=(lt-lst)/(lt+lst+EPS)
    q_st=(1+advantage)/2
    q=torch.stack([1-q_st,q_st],-1).detach()
    scale=((lt+lst)/2).mean().detach()
    return q,scale,lt.detach(),lst.detach()


def supervised_router_loss(model,target):
    if model.variant!=VARIANT or not model.training:
        raise ValueError('Utility supervision is TRAIN-only and utility-variant-only')
    pred_t=model._last_pred_T;pred_st=model._last_pred_ST;pi=model._last_utility_pi
    if any(not bool(torch.isfinite(v).all()) for v in (pred_t,pred_st,pi,target)):
        raise FloatingPointError('Implementation failure: nonfinite utility inputs')
    q,scale,lt,lst=utility_target(pred_t,pred_st,target)
    route_pi=model.utility_moe_fusion.router_only_pi()
    raw=(q*((q+EPS).log()-(route_pi+EPS).log())).sum(-1).mean()
    loss=scale*raw
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(route_pi).all()):
        raise FloatingPointError('Implementation failure: nonfinite routing loss')
    model._last_utility_q=q;model._last_route_pi=route_pi;model._last_route_scale=scale
    d=dict(raw_route_KL=float(raw.detach()),route_scale=float(scale),scaled_route_loss=float(loss.detach()),
           mean_q_T=float(q[:,0].mean()),mean_q_ST=float(q[:,1].mean()),std_q_ST=float(q[:,1].std(unbiased=False)),
           mean_pi_T=float(pi[:,0].detach().mean()),mean_pi_ST=float(pi[:,1].detach().mean()),
           mean_expert_loss_T=float(lt.mean()),mean_expert_loss_ST=float(lst.mean()),
           fraction_q_ST_gt_05=float((q[:,1]>.5).float().mean()))
    return loss,d


class UtilityRoutingAccumulator:
    def __init__(self):
        self.pi=[];self.q=[];self.utility=[]

    def update(self,model,diagnostics=None):
        self.pi.append(model._last_utility_pi.detach().double().cpu())
        if diagnostics is not None:
            self.q.append(model._last_utility_q.double().cpu());self.utility.append(diagnostics)

    def summary(self):
        p=torch.cat(self.pi);r=dict(samples=len(p),routing_entropy=float(-(p*(p+EPS).log()).sum(-1).mean()))
        for i,label in enumerate(('T','ST')):
            r.update({f'{k}_pi_{label}':float(v) for k,v in dict(mean=p[:,i].mean(),std=p[:,i].std(unbiased=False),min=p[:,i].min(),max=p[:,i].max()).items()})
        for q,label in ((.1,'P10'),(.5,'P50'),(.9,'P90')):r[label+'_pi_ST']=float(torch.quantile(p[:,1],q))
        if self.utility:
            for k in self.utility[0]:r[k]=sum(d[k] for d in self.utility)/len(self.utility)
            q=torch.cat(self.q)
            r.update(mean_q_T=float(q[:,0].mean()),mean_q_ST=float(q[:,1].mean()),std_q_ST=float(q[:,1].std(unbiased=False)),
                     fraction_q_ST_gt_05=float((q[:,1]>.5).double().mean()))
        return r
