"""Global complementary fusion checks and the sole zero-residual control."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from cmgm.config import MULTI_HORIZONS
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0e_diagnostics import diagnostic_context,model_arguments,flattened_gradients,prediction_loss
from cmgm.scripts.d0b_hybrid_graph_prior import full_sanity
from cmgm.training.metric_standard import population_metrics

BASE='switching_latent_balanced_readout'
VARIANT='switching_latent_balanced_residual_complementary_fusion'
DISPLAY='D0B-ResidualComplementaryFusion'


def assert_backbone(m):
    b=m.switching_latent_transformer
    assert m.variant==VARIANT and m.use_gcn and m.use_edge_attn and m.use_gate and m.use_lstm and not m.use_mixhop
    assert b.balanced_readout and not any((b.use_latent_memory,b.use_regime_relative_memory,b.use_dynamic_slope,b.use_balanced_transition_input,b.horizon_specific_state_readout))
    assert not b.regime_filter.learnable_sticky_alpha and b.regime_filter.sticky_alpha==.5
    assert not getattr(m,'disable_switch_kl',False)
    assert not hasattr(m,'pregnn_local_residual') and not hasattr(m,'commodity_residual_head')
    for layer in (m.attn_mixhop1,m.attn_mixhop2):
        assert not layer.qk_norm and layer.graph_prior_heads is None and layer.cross_mask is None and not layer.hard_mask
        assert layer.n_heads==8 and layer.prior_scale==.5 and layer.K==2 and layer.beta==.05


def snapshot(m,x):
    """Capture the actual shared gate/head interfaces without replacing forward."""
    capture={}
    def gate_hook(module,args,out):
        capture['input']=args[0].detach().clone();capture['gate']=torch.sigmoid(out.detach())
    def head_hook(module,args):capture['head_input']=args[0].detach().clone()
    hooks=[m.gate_fc.register_forward_hook(gate_hook),m.head.register_forward_pre_hook(head_hook)]
    try:prediction=m(x)
    finally:
        for hook in hooks:hook.remove()
    s,t=capture['input'].chunk(2,dim=-1)
    # ZeroResidual must use the ACTUAL native h_base, never a reconstructed approximation.
    base=m.last_fusion_base if m.variant==VARIANT else capture['head_input']
    residual=m.last_fusion_residual if m.variant==VARIANT else torch.zeros_like(base)
    return {k:v.detach().clone() for k,v in dict(h_spatial=s,h_temporal=t,gate=capture['gate'],fusion_input=capture['input'],
        h_base=base,fusion_residual=residual,h_fused=capture['head_input'],prediction=prediction).items()}


def initialization_check(m,x,seed=42):
    with diagnostic_context(m),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        b=HeteroMixHopCMGM(variant=BASE,**model_arguments(m)).to(device=x.device,dtype=x.dtype).eval()
        left,right=b.state_dict(),m.state_dict();assert set(left).issubset(right)
        assert all(left[k].shape==right[k].shape for k in left)
        differences=[float((v-right[k]).abs().max()) for k,v in left.items()]
        extra=sorted(set(right)-set(left));a,c=snapshot(b,x),snapshot(m,x)
        counts=[sum(p.numel() for p in model.parameters()) for model in (b,m)]
        r=dict(D0B_params=counts[0],New_params=counts[1],delta_params=counts[1]-counts[0],expected_delta=6176,
            shared_parameter_init_max_diff=max(differences),shared_parameter_mismatch_count=sum(v!=0 for v in differences),extra_tensors=extra,
            h_base_max_diff=float((a['h_base']-c['h_base']).abs().max()),residual_max_abs=float(c['fusion_residual'].abs().max()),
            fused_base_max_diff=float((c['h_fused']-c['h_base']).abs().max()),prediction_max_diff=float((c['prediction']-a['prediction']).abs().max()))
        r['PASS']=r['delta_params']==6176 and r['shared_parameter_mismatch_count']==0 and r['h_base_max_diff']<1e-7 and r['prediction_max_diff']<1e-7 and r['residual_max_abs']==r['fused_base_max_diff']==0 and extra==['complementary_fusion_residual.0.bias','complementary_fusion_residual.0.weight','complementary_fusion_residual.2.weight']
        return r


def summary(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)),P5=float(np.quantile(x,.05)),
        P10=float(np.quantile(x,.1)),P50=float(np.quantile(x,.5)),P90=float(np.quantile(x,.9)),P95=float(np.quantile(x,.95)),max=float(x.max()))


def representation_stats(rows):
    norms={k:np.linalg.norm(rows[k].astype(float),axis=-1) for k in ('h_spatial','h_temporal','h_base','fusion_residual','h_fused')}
    def cosine(a,b):
        a=rows[a].astype(float);b=rows[b].astype(float)
        return np.sum(a*b,-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
    return dict(norms={k:summary(v) for k,v in norms.items()},residual_base_norm_ratio=float(norms['fusion_residual'].mean()/(norms['h_base'].mean()+1e-12)),
        mean_abs_residual=float(np.abs(rows['fusion_residual']).mean()),cos_spatial_temporal=summary(cosine('h_spatial','h_temporal')),
        cos_base_fused=summary(cosine('h_base','h_fused')),gate=summary(rows['gate']))


def gradient_diagnostics(m,batch):
    groups={k:list(getattr(m,k).parameters()) for k in ('type_proj','temporal_score','attn_mixhop1','attn_mixhop2','head','gate_fc','switching_latent_transformer')}
    groups['residual_first']=list(m.complementary_fusion_residual[0].parameters());groups['residual_final']=list(m.complementary_fusion_residual[-1].parameters())
    with diagnostic_context(m),torch.enable_grad():
        pred=m(batch[0]);loss=prediction_loss(pred,batch[1]);g=flattened_gradients(loss,groups)
        r=m.last_fusion_residual.double();base=m.last_fusion_base.double()
        return dict(prediction_loss=float(loss.detach()),norms={k:float(v.double().norm()) for k,v in g.items()},
            first_weight_norm=float(m.complementary_fusion_residual[0].weight.detach().norm()),final_weight_norm=float(m.complementary_fusion_residual[-1].weight.detach().norm()),
            mean_abs_residual=float(r.abs().mean()),mean_residual_L2=float(r.norm(dim=-1).mean()),
            residual_base_norm_ratio=float(r.norm(dim=-1).mean()/(base.norm(dim=-1).mean()+1e-12)),
            method='Fixed TRAIN batch, eval, sum four Huber losses; autograd.grad, no .grad mutation or optimizer step. Zero initial W2 gives zero W1 gradient and nonzero W2 gradient.')


def _component_sanity(m,x):
    with diagnostic_context(m),torch.no_grad():
        native=snapshot(m,x);order=torch.arange(len(x)-1,-1,-1,device=x.device)
        perm=snapshot(m,x[order]);single=snapshot(m,x[:1])
        batch={k:float((v-native[k][order]).abs().max()) for k,v in perm.items()}
        single={k:float((v-native[k][:1]).abs().max()) for k,v in single.items()}
        changed=x.clone();changed[:,10:]=-3*changed[:,10:]+9
        a,b=snapshot(m,x[:,:10]),snapshot(m,changed[:,:10])
        prefix={k:float((a[k]-b[k]).abs().max()) for k in a}
        # The model interface receives ONLY the observed window, never future features/targets.
        future=torch.cat([x,x[:,:5]+100],1);future[:,20:]*=-5
        window=snapshot(m,future[:,:20]);window={k:float((window[k]-native[k]).abs().max()) for k in native}
        markets={};start=0
        for name,size in zip(('stock','bond','commodity'),(m.n_stock,m.n_bond,m.n_commodities)):
            node=torch.arange(m.num_nodes,device=x.device);node[start:start+size]=node[start:start+size].flip(0)
            output=torch.arange(m.n_commodities,device=x.device)
            if name=='commodity':output=output.flip(0)
            copy=HeteroMixHopCMGM(variant=VARIANT,**model_arguments(m)).to(device=x.device,dtype=x.dtype).eval()
            state={k:v.clone() for k,v in m.state_dict().items()}
            for k in ('graph_learner.E1','graph_learner.E2'):state[k]=state[k][node]
            for k in ('head.3.weight','head.3.bias'):
                v=state[k];state[k]=v.reshape(m.n_horizons,m.n_commodities,*v.shape[1:])[:,output].reshape_as(v)
            copy.load_state_dict(state);mapped=snapshot(copy,x[:,:,node])
            markets[name]={k:float((v-(native[k][:,:,output] if k=='prediction' else native[k])).abs().max()) for k,v in mapped.items()};start+=size
        gate_base=native['gate']*m.lstm_proj(native['h_temporal'])+(1-native['gate'])*m.gcn_proj(native['h_spatial'])
        gate_error=float((native['h_base']-gate_base).abs().max())
        formula=float((native['h_fused']-native['h_base']-native['fusion_residual']).abs().max())
        # Compare native addition directly too, to avoid subtraction cancellation.
        addition=float((native['h_fused']-(native['h_base']+native['fusion_residual'])).abs().max())
        r=dict(batch_permutation=batch,single_sample=single,prefix_forecast_cutoff10=prefix,legal_window_future_perturbation=window,
            within_market_relabeling=markets,fusion_input_shape=list(native['fusion_input'].shape),residual_shape=list(native['fusion_residual'].shape),
            gate_formula_max_error=gate_error,residual_addition_max_error=addition,residual_subtraction_error=formula,
            definition='Global fusion uses only observed-window representations. Prefix spatial forecasts aggregate only that prefix; full-window outputs can legitimately depend on its observed suffix. Relabel graph embeddings and original commodity output rows together.')
        r['PASS']=addition==0 and gate_error<1e-6 and max([v for group in (batch,single,prefix,window) for v in group.values()]+[v for group in markets.values() for v in group.values()])<1e-6
        return r


def structural_sanity(m,x):
    temporal=full_sanity(m,x);fusion=_component_sanity(m,x)
    if not fusion['PASS'] and fusion['residual_addition_max_error']==0 and max(fusion['batch_permutation'].values())<1e-6 and max(fusion['prefix_forecast_cutoff10'].values())<1e-6:
        with diagnostic_context(m):
            copy=HeteroMixHopCMGM(variant=VARIANT,**model_arguments(m)).to(device=x.device,dtype=torch.float64)
            copy.load_state_dict(m.state_dict());audit=_component_sanity(copy,x.double())
        fusion['float64_roundoff_audit']=audit;fusion['PASS']=audit['PASS']
        fusion['note']='Raw FP32 errors retained; separate FP64 audit of representation/relabeling roundoff, default model precision unchanged.'
    return dict(temporal=temporal,fusion=fusion,PASS=temporal['PASS'] and fusion['PASS'])


@torch.no_grad()
def collect(m,loaders,device):
    arrays={};stats={};idx=MULTI_HORIZONS.index(5)
    with diagnostic_context(m):
        for split,source in loaders.items():
            ps=[];ys=[];zeros=[];representations={}
            for batch in DataLoader(source.dataset,batch_size=64,shuffle=False,drop_last=False):
                row=snapshot(m,batch[0].to(device))
                ps.append(row['prediction'][:,idx].cpu().numpy());ys.append(batch[1][:,idx].numpy())
                if m.variant==VARIANT:
                    zero=m.head(row['h_base']).view(len(batch[0]),len(MULTI_HORIZONS),m.n_commodities)
                    zeros.append(zero[:,idx].cpu().numpy())
                for k,v in row.items():
                    if k!='prediction':representations.setdefault(k,[]).append(v.cpu().numpy())
            arrays[split]=dict(prediction=np.concatenate(ps),target=np.concatenate(ys))
            if zeros:arrays[split]['zero_prediction']=np.concatenate(zeros)
            stats[split]=representation_stats({k:np.concatenate(v) for k,v in representations.items()})
    return arrays,stats


def ablation_analysis(base,new):
    result={}
    for split,v in new.items():
        np.testing.assert_array_equal(base[split]['target'],v['target'])
        metrics=dict(OriginalD0B=population_metrics(base[split]['prediction'],v['target']),Full=population_metrics(v['prediction'],v['target']),ZeroResidual=population_metrics(v['zero_prediction'],v['target']))
        result[split]=dict(metrics=metrics,direct_effect={k:metrics['Full'][k]-metrics['ZeroResidual'][k] for k in ('MAE','MSE','RMSE','Hit')},
            representation_effect={k:metrics['ZeroResidual'][k]-metrics['OriginalD0B'][k] for k in ('MAE','MSE','RMSE','Hit')},
            prediction_mean_impact=float(np.abs(v['prediction']-v['zero_prediction']).mean()),prediction_max_impact=float(np.abs(v['prediction']-v['zero_prediction']).max()))
    return result
