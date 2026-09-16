"""Thirteen preregistered D0B controls; instantiate full D0B before bypass flags."""
import torch
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM

NAMES=('FullD0B-Control','CommodityOnly','w/o Stock','w/o Bond','w/o Spatial Branch','w/o Temporal Branch',
       'w/o Graph Propagation','w/o TempWeighted','w/o Markov Switching','w/o Microstate','w/o Switch KL',
       'w/o Balanced Readout','w/o Adaptive Fusion Gate')
KEYS=('full_d0b_control','commodity_only','no_stock','no_bond','no_spatial','no_temporal','no_graph','no_tempweighted',
      'no_markov_switching','no_microstate','no_switch_kl','no_balanced_readout','no_adaptive_gate')
KEY=dict(zip(NAMES,KEYS))
MASKS={'CommodityOnly':('stock','bond'),'w/o Stock':('stock',),'w/o Bond':('bond',)}


class FormalD0BAblation(HeteroMixHopCMGM):
    def __init__(self,name,data):
        if name not in NAMES:raise ValueError(name)
        mi=data['market_indices'];cs,ce=mi['commodity']
        if ce-cs!=24:raise ValueError('Exactly 24 aligned commodity targets required')
        super().__init__(data['n_nodes'],24,n_stock=mi['stock'][1],n_bond=mi['bond'][1]-mi['bond'][0],
                         variant='switching_latent_balanced_readout')
        # No new modules/parameters/RNG consumption after complete native construction.
        self.ablation_name=name;self.market_indices=mi
        self.disable_switch_kl=name in ('w/o Switch KL','w/o Temporal Branch')
        b=self.switching_latent_transformer
        b.formal_uniform_switching=name=='w/o Markov Switching'
        b.formal_bypass_balance=name=='w/o Balanced Readout'
        self.components={}

    def mask_input(self,x):
        if self.ablation_name not in MASKS:return x
        out=x.clone()
        for market in MASKS[self.ablation_name]:
            a,b=self.market_indices[market];out[:,:,a:b,:]=0.
        return out

    def forward(self,x,edge_index=None,edge_weight=None,debug=False):
        x=self.mask_input(x);name=self.ablation_name;b=self.switching_latent_transformer
        self.components={}
        if name!='w/o Spatial Branch':
            hs,pre=self._temp_weighted_spatial(x,return_pre_nodes=True,
                uniform_time=name=='w/o TempWeighted',bypass_graph=name=='w/o Graph Propagation')
            self.components.update(h_spatial=hs.detach(),H_pre=pre.detach())
        if name!='w/o Temporal Branch':
            ht=b(x,zero_readout_component='Z' if name=='w/o Microstate' else None)
            self.components['h_temporal']=ht.detach()
        if name=='w/o Spatial Branch':
            fused=self.lstm_proj(ht);self.components['t']=fused.detach()
        elif name=='w/o Temporal Branch':
            fused=self.gcn_proj(hs);self.components['s']=fused.detach()
        else:
            gate=hs.new_full(hs.shape,.5) if name=='w/o Adaptive Fusion Gate' else torch.sigmoid(self.gate_fc(torch.cat([hs,ht],dim=-1)))
            t=self.lstm_proj(ht);s=self.gcn_proj(hs)
            fused=gate*t+(1-gate)*s
            self.components.update(gate=gate.detach(),s=s.detach(),t=t.detach())
        self.components['fused']=fused.detach()
        return self.head(fused).reshape(len(x),self.n_horizons,self.n_commodities)

    def auxiliary_loss(self):
        if self.disable_switch_kl:return next(self.parameters()).new_zeros(())
        return self.switching_latent_transformer.switch_loss()
