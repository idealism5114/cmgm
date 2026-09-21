"""Three sample-routed dense experts for D0B fusion only."""
import torch
from torch import nn

VARIANT = 'switching_latent_balanced_moe_fusion'
BALANCE_COEFFICIENT = 1e-4
BALANCE_EPS = 1e-8


class ResidualExpert64(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64,64))

    def forward(self, x):
        return x + self.mlp(x)


class InteractionExpert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(128,64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64,64))


class FusionMoERouter(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,3))


class SpatialTemporalMoEFusion(nn.Module):
    """Expert order T/S/ST. Projections remain owned by the original D0B model.

    Warm-up applies consistently in train/eval at the explicitly selected epoch.
    The epoch is a persistent nontrainable buffer, so a best checkpoint restores
    its exact routing weights even when its best epoch precedes epoch 10.
    """
    def __init__(self):
        super().__init__()
        self.temporal_expert = ResidualExpert64()
        self.spatial_expert = ResidualExpert64()
        self.interaction_expert = InteractionExpert()
        self.router = FusionMoERouter()
        self.register_buffer('epoch', torch.tensor(1, dtype=torch.long))
        self._balance_loss = None

    def set_epoch(self, epoch):
        if isinstance(epoch,bool) or int(epoch)!=epoch or epoch<0:
            raise ValueError('MoE epoch must be a nonnegative integer (0 is a sanity-only uniform route)')
        self.epoch.fill_(int(epoch))

    @property
    def gamma(self):
        return min(1., int(self.epoch.item()) / 10.)

    def forward(self, h_spatial, h_temporal, spatial_proj, temporal_proj):
        expected=(h_spatial.shape[0],64)
        if any(tuple(x.shape)!=expected for x in (h_spatial,h_temporal,spatial_proj,temporal_proj)):
            raise ValueError('All raw/projected fusion inputs must have shape (B,64)')
        # Router sees raw branches; experts see their original projected branches.
        logits=self.router(torch.cat([h_spatial,h_temporal],dim=-1))
        pi=logits.softmax(dim=-1)
        effective=(1-self.gamma)*torch.full_like(pi,1/3)+self.gamma*pi
        experts=torch.stack([
            self.temporal_expert(temporal_proj),
            self.spatial_expert(spatial_proj),
            self.interaction_expert(torch.cat([spatial_proj,temporal_proj],dim=-1)),
        ],dim=1)
        fused=(effective.unsqueeze(-1)*experts).sum(dim=1)
        mean_pi=pi.mean(dim=0)
        self._balance_loss=(mean_pi*((mean_pi+BALANCE_EPS)/(1/3)).log()).sum()
        self.last_inputs={k:v.detach() for k,v in dict(h_spatial=h_spatial,h_temporal=h_temporal,
                                                     s=spatial_proj,t=temporal_proj).items()}
        self.last_logits=logits.detach()
        self.last_pi=pi.detach()
        self.last_effective_pi=effective.detach()
        self.last_experts=experts.detach()
        self.last_fused=fused.detach()
        return fused

    def balance_loss(self):
        if self._balance_loss is None:
            raise RuntimeError('MoE balance loss requires a forward pass')
        return self._balance_loss


class RoutingAccumulator:
    """Observation-pooled router summaries; mean batch balance matches the loss."""
    def __init__(self):
        self.count=0
        self.pi_sum=torch.zeros(3,dtype=torch.float64)
        self.entropy_sum=0.
        self.balance_sum=0.
        self.batches=0

    def update(self, fusion):
        p=fusion.last_pi.double().cpu()
        self.count+=len(p);self.pi_sum+=p.sum(0)
        self.entropy_sum+=float(-(p*(p+BALANCE_EPS).log()).sum())
        self.balance_sum+=float(fusion.balance_loss().detach())
        self.batches+=1

    def summary(self, fusion):
        mean=(self.pi_sum/max(self.count,1)).tolist()
        return dict(mean_pi_T=mean[0],mean_pi_S=mean[1],mean_pi_ST=mean[2],
                    routing_entropy=self.entropy_sum/max(self.count,1),
                    balance_loss=self.balance_sum/max(self.batches,1),gamma_warmup=fusion.gamma,
                    samples=self.count,batches=self.batches)
