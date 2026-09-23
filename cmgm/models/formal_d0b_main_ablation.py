"""Revised main-innovation controls. Historical/native model code stays untouched."""
import torch
from torch.nn import functional as F
from torch import nn
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.candidate_moe_fusion import VARIANT as CANDIDATE
from cmgm.models.global_mixture_fusion import GlobalTwoExpertMixture, VARIANT as GLOBAL, TS_VARIANT
from cmgm.models.model import MixHopPropagation
from cmgm.models.switching_latent_transformer import LongMemoryTransformer, RegimeLatentTransition

FULL = 'Full Candidate-Aware 2-Expert MoE'
NAMES = ('w/o Spatial Temporal Weighting', 'w/o Adaptive Graph', 'w/o EdgeAttnMixHop',
         'w/o Relative Position Encoding', 'w/o Adaptive Regime Routing',
         'w/o Regime-Specific Transitions', 'w/o Microstate', 'w/o Balanced Readout',
         'w/o MoE Fusion', 'w/o Candidate-Aware Routing', FULL)
KEY = dict(zip(NAMES, ('no_spatial_temporal_weighting', 'no_adaptive_graph', 'no_edge_attn_mixhop',
    'no_rpe', 'no_adaptive_regime_routing', 'no_regime_specific_transition', 'no_microstate',
    'no_balanced_readout', 'no_moe_fusion', 'no_candidate_aware_routing', 'full_candidate_2expert_moe')))
REUSE = {FULL: CANDIDATE, 'w/o Candidate-Aware Routing': GLOBAL}
NEW = tuple(n for n in NAMES if n not in REUSE)
DEFINITIONS = dict(zip(NAMES, (
    'H_pre=mean(H_seq,dim=1); remaining Candidate MoE Full unchanged',
    'A=ones(N,N); graph learner inactive; both native EdgeAttnMixHop blocks and Candidate MoE retained',
    'Same learned directed A into two single-hop GCN(64,64) layers: D^(-1/2)(A+I)D^(-1/2)H W + b; Candidate MoE retained',
    'Zero BaseRPE bias; causal Transformer and Candidate MoE retained',
    'Uniform p/prior; three distinct generators and Candidate MoE retained; KL mathematically zero',
    'Adaptive p and native KL retained; G0 shared across three candidates; Candidate MoE retained',
    'Post-normalization effective micro readout zero; recurrence, KL and Candidate MoE retained',
    'Bypass only long/micro LayerNorm; all readout projections and Candidate MoE retained',
    'Remove experts/router; Linear(128,64)([W_s h_s || W_t h_t]) into unchanged shared head',
    'Exact same two experts, two zero-initialized global logits, softmax(2); no candidate router',
    'Unmodified native switching_latent_balanced_candidate_2expert_moe')))


GLOBAL_FULL = 'Full Global Dual-Expert Mixture'
GLOBAL_NAMES = ('w/o Adaptive Regime Routing', 'w/o Regime-Specific Transitions',
                'w/o Microstate', GLOBAL_FULL)
GLOBAL_DEFINITIONS = {n: DEFINITIONS[n].replace('Candidate MoE', 'Global Dual-Expert Mixture')
                      for n in GLOBAL_NAMES[:-1]}
GLOBAL_DEFINITIONS[GLOBAL_FULL] = 'Unmodified native ' + GLOBAL
KEY[GLOBAL_FULL] = 'full_global_dual_expert_mixture'
REUSE[GLOBAL_FULL] = GLOBAL


TS_FULL = 'Full T+S Global Dual-Expert Mixture'
TS_NAMES = (*GLOBAL_NAMES[:-1], 'w/o EdgeAttnMixHop', TS_FULL)
TS_DEFINITIONS = {n: GLOBAL_DEFINITIONS[n].replace('Global Dual-Expert Mixture', 'T+S Global Dual-Expert Mixture')
                  for n in GLOBAL_NAMES[:-1]}
TS_DEFINITIONS['w/o EdgeAttnMixHop'] = 'Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05) blocks; branch-specific T+S global experts retained'
TS_DEFINITIONS[TS_FULL] = 'Unmodified native ' + TS_VARIANT
KEY[TS_FULL] = 'full_ts_global_dual_expert_mixture'


class StandardGraphPropagation(nn.Module):
    """Single-hop GCN aggregation on the unchanged learned weighted adjacency.

    A[i,j] sends node j into i. Add self-loops and use row-degree two-sided
    normalization; do NOT symmetrize the learned directed graph itself.
    No attention, hop concatenation/summation, beta recurrence or residual.
    """
    def __init__(self,in_dim=64,out_dim=64):
        super().__init__()
        self.linear=nn.Linear(in_dim,out_dim)

    def forward(self,x,A):
        augmented=A+torch.eye(A.shape[0],device=A.device,dtype=A.dtype)
        inv_degree=augmented.sum(dim=1).clamp_min(1e-8).rsqrt()
        normalized=inv_degree[:,None]*augmented*inv_degree[None,:]
        return self.linear(normalized @ x)


class NoRPELongMemory(LongMemoryTransformer):
    def relative_bias(self, tokens):
        batch, time, _ = tokens.shape
        if time > self.max_len:
            raise ValueError('Observed sequence exceeds native maximum length')
        bias = tokens.new_zeros(batch, self.n_heads, time, time)
        positions = torch.arange(time, device=tokens.device)
        self.last_relative_delta = positions[:, None] - positions[None, :]
        self.last_base_relative_bias = bias.detach()
        return bias


class SharedRegimeTransition(RegimeLatentTransition):
    def forward(self, h_t, z_prev, probabilities):
        shared = self.generators[0](torch.cat([h_t, z_prev], dim=-1))
        candidates = shared.unsqueeze(1).expand(-1, self.K, -1)
        return torch.einsum('bk,bkd->bd', probabilities, candidates), candidates


class MainInnovationAblation(HeteroMixHopCMGM):
    def __init__(self, name, data, mode='candidate'):
        if mode not in ('candidate','global-temporal','ts-global-expert'):raise ValueError(mode)
        if name not in (TS_NAMES if mode=='ts-global-expert' else GLOBAL_NAMES if mode=='global-temporal' else NAMES):
            raise ValueError(name)
        # Construct every native module before any replacement, preserving shared RNG order.
        mi=data['market_indices']
        if mi['commodity'][1]-mi['commodity'][0]!=24:raise ValueError('Expected 24 ordered commodity targets')
        super().__init__(data['n_nodes'],24,n_stock=mi['stock'][1]-mi['stock'][0],
                         n_bond=mi['bond'][1]-mi['bond'][0],variant=TS_VARIANT if mode=='ts-global-expert' else GLOBAL if mode=='global-temporal' else CANDIDATE)
        self.ablation_mode=mode
        self.ablation_name=name;self.market_indices=mi;self.disable_switch_kl=False
        self.components={}
        self.main_name = name
        branch = self.switching_latent_transformer
        branch.formal_uniform_switching=name=='w/o Adaptive Regime Routing'
        branch.formal_bypass_balance=name=='w/o Balanced Readout'
        if name=='w/o MoE Fusion':
            # Full construction (including experts/router) precedes this new Linear.
            self.simple_fusion=nn.Linear(128,64)
            del self.candidate_moe_fusion
            self.variant=CANDIDATE+'_formal_linear_fusion'
        elif name=='w/o Candidate-Aware Routing':
            # Move initialized experts unchanged, without consuming any further RNG.
            fusion=GlobalTwoExpertMixture.__new__(GlobalTwoExpertMixture)
            nn.Module.__init__(fusion)
            fusion.temporal_expert=self.candidate_moe_fusion.temporal_expert
            fusion.interaction_expert=self.candidate_moe_fusion.interaction_expert
            fusion.global_mixture_logits=nn.Parameter(torch.zeros(2))
            self.global_mixture_fusion=fusion
            del self.candidate_moe_fusion
            self.variant=GLOBAL
        if name == 'w/o Relative Position Encoding':
            # Same existing module/parameters; only the overridable bias method changes.
            branch.long_memory.__class__ = NoRPELongMemory
        elif name == 'w/o Regime-Specific Transitions':
            branch.latent_transition.__class__ = SharedRegimeTransition
        elif name == 'w/o EdgeAttnMixHop':
            if mode=='ts-global-expert':
                self.ordinary_mixhop1=MixHopPropagation(64,64,K=2,beta=.05)
                self.ordinary_mixhop2=MixHopPropagation(64,64,K=2,beta=.05)
            else:
                self.standard_gcn1 = StandardGraphPropagation(64,64)
                self.standard_gcn2 = StandardGraphPropagation(64,64)

    def forward(self,x,edge_index=None,edge_weight=None,debug=False):
        hs,pre=self._temp_weighted_spatial(x,return_pre_nodes=True,
            uniform_time=self.main_name=='w/o Spatial Temporal Weighting')
        ht=self.switching_latent_transformer(x,
            zero_readout_component='Z' if self.main_name=='w/o Microstate' else None)
        s=self.gcn_proj(hs);t=self.lstm_proj(ht)
        if self.main_name=='w/o MoE Fusion':
            fused=self.simple_fusion(torch.cat([s,t],dim=-1))
        elif hasattr(self,'global_mixture_fusion'):
            fused=self.global_mixture_fusion(s,t)
        else:
            fused=self.candidate_moe_fusion(s,t)
        self.components={k:v.detach() for k,v in dict(h_spatial=hs,h_temporal=ht,H_pre=pre,s=s,t=t,fused=fused).items()}
        return self.head(fused).reshape(len(x),self.n_horizons,self.n_commodities)

    def auxiliary_loss(self):
        return self.switching_latent_transformer.switch_loss()

    def _temp_weighted_spatial(self, x, return_nodes=False, return_pre_nodes=False,
                               uniform_time=False, bypass_graph=False):
        if self.main_name not in ('w/o Adaptive Graph','w/o EdgeAttnMixHop'):
            return super()._temp_weighted_spatial(x, return_nodes, return_pre_nodes,
                                                 uniform_time, bypass_graph)
        assert not uniform_time and not bypass_graph
        seq = torch.stack([self.type_proj(x[:,t], self.n_stock, self.n_bond)
                           for t in range(x.shape[1])], dim=1)
        alpha = F.softmax(self.temporal_score(seq), dim=1)
        pre = (seq * alpha).sum(dim=1)
        self.last_alpha = alpha.squeeze(-1).detach()
        if self.main_name == 'w/o Adaptive Graph':
            A = pre.new_ones(pre.shape[1], pre.shape[1])
            h = self.attn_mixhop2(F.relu(self.attn_mixhop1(pre, A)), A)
        else:
            A = self.graph_learner()
            if self.ablation_mode=='ts-global-expert':
                h=self.ordinary_mixhop2(F.relu(self.ordinary_mixhop1(pre,A)),A)
            else:
                h = self.standard_gcn2(F.relu(self.standard_gcn1(pre, A)), A)
        self.last_ablation_adjacency = A.detach()
        h = self.gcn_norm(h)
        pooled = self.type_pool(h)
        if return_pre_nodes:
            return (pooled,h,pre) if return_nodes else (pooled,pre)
        return (pooled,h) if return_nodes else pooled
