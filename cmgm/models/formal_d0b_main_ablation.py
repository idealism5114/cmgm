"""Revised main-innovation controls. Historical/native model code stays untouched."""
import torch
from torch.nn import functional as F
from cmgm.models.formal_d0b_ablation import FormalD0BAblation
from cmgm.models.model import MixHopPropagation
from cmgm.models.switching_latent_transformer import LongMemoryTransformer, RegimeLatentTransition

NAMES = ('w/o Spatial Branch', 'w/o Temporal Branch', 'w/o Spatial Temporal Attention',
         'w/o Adaptive Graph', 'w/o EdgeAttnMixHop', 'w/o Relative Position Encoding',
         'w/o Adaptive Regime Routing', 'w/o Regime-Specific Transitions',
         'w/o Microstate', 'w/o Balanced Readout', 'w/o Adaptive Fusion', 'Full D0B')
KEY = dict(zip(NAMES, ('no_spatial', 'no_temporal', 'no_spatial_temporal_attention',
    'no_adaptive_graph', 'no_edge_attn_mixhop', 'no_rpe', 'no_adaptive_regime_routing',
    'no_regime_specific_transition', 'no_microstate', 'no_balanced_readout', 'no_adaptive_fusion', 'full_d0b')))
REUSE = dict(zip((NAMES[i] for i in (0,1,2,6,8,9,10,11)),
    ('w/o Spatial Branch','w/o Temporal Branch','w/o TempWeighted','w/o Markov Switching',
     'w/o Microstate','w/o Balanced Readout','w/o Adaptive Fusion Gate','FullD0B-Control')))
NEW = tuple(n for n in NAMES if n not in REUSE)
DEFINITIONS = dict(zip(NAMES, (
    'h_fused=W_t h_temporal; spatial path inactive; original head',
    'h_fused=W_s h_spatial; temporal path and Switch KL inactive; original head',
    'H_pre=mean(H_seq,dim=1); graph and both EdgeAttnMixHop blocks retained',
    'A=ones(N,N); graph learner inactive; both native EdgeAttnMixHop blocks retained',
    'Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05) blocks',
    'Zero BaseRPE bias; causal mask and complete Transformer retained',
    'Uniform p/prior; three distinct generators retained; KL mathematically zero',
    'Adaptive p and native KL retained; generator[0] shared across all three candidates',
    'Effective post-normalization micro readout zero; original recurrence and KL retained',
    'Bypass only long/micro LayerNorm; all readout projections retained',
    'h_fused=.5 W_t h_temporal+.5 W_s h_spatial; original projections/head',
    'Unmodified native switching_latent_balanced_readout')))


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


class MainInnovationAblation(FormalD0BAblation):
    def __init__(self, name, data):
        if name not in NAMES:
            raise ValueError(name)
        # Construct every native module before any replacement, preserving shared RNG order.
        super().__init__(REUSE.get(name, 'FullD0B-Control'), data)
        self.main_name = name
        branch = self.switching_latent_transformer
        if name == 'w/o Relative Position Encoding':
            # Same existing module/parameters; only the overridable bias method changes.
            branch.long_memory.__class__ = NoRPELongMemory
        elif name == 'w/o Regime-Specific Transitions':
            branch.latent_transition.__class__ = SharedRegimeTransition
        elif name == 'w/o EdgeAttnMixHop':
            self.ordinary_mixhop1 = MixHopPropagation(64,64,K=2,beta=.05)
            self.ordinary_mixhop2 = MixHopPropagation(64,64,K=2,beta=.05)

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
            h = self.ordinary_mixhop2(F.relu(self.ordinary_mixhop1(pre, A)), A)
        self.last_ablation_adjacency = A.detach()
        h = self.gcn_norm(h)
        pooled = self.type_pool(h)
        if return_pre_nodes:
            return (pooled,h,pre) if return_nodes else (pooled,pre)
        return (pooled,h) if return_nodes else pooled
