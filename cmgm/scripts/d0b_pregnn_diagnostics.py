"""Read-only checks and local-state interventions for the pre-GNN skip."""
import numpy as np
import torch
from cmgm.config import MULTI_HORIZONS
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0e_diagnostics import diagnostic_context,model_arguments,flattened_gradients,prediction_loss
from cmgm.scripts.d0b_hybrid_graph_prior import full_sanity

BASE='switching_latent_balanced_readout'
VARIANT='switching_latent_balanced_pregnn_local_skip'
DISPLAY='D0B-PreGNNCommodityLocalSkip'


def assert_backbone(model):
    b=model.switching_latent_transformer
    assert model.variant==VARIANT and model.use_gcn and model.use_edge_attn and model.use_gate and model.use_lstm and not model.use_mixhop
    assert b.balanced_readout and not b.use_latent_memory and not b.use_regime_relative_memory
    assert not b.use_dynamic_slope and not b.use_balanced_transition_input and not b.horizon_specific_state_readout
    assert not b.regime_filter.learnable_sticky_alpha and b.regime_filter.sticky_alpha==.5
    assert not getattr(model,'disable_switch_kl',False)
    for layer in (model.attn_mixhop1,model.attn_mixhop2):
        assert not layer.qk_norm and layer.graph_prior_heads is None and layer.cross_mask is None and not layer.hard_mask
        assert layer.n_heads==8 and layer.prior_scale==.5 and layer.K==2 and layer.beta==.05
    assert list(model.pregnn_local_residual.children())[2].bias is None


def snapshot(model,x):
    pred=model(x)
    return {name:value.clone() for name,value in dict(H_pre=model.last_pregnn_nodes,h_comm_pre=model.last_pregnn_comm,
        base_pred=model.last_base_pred,residual=model.last_residual,prediction=pred.detach()).items()}


def initialization_check(model,x,seed=42):
    with diagnostic_context(model),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline=HeteroMixHopCMGM(variant=BASE,**model_arguments(model)).to(device=x.device,dtype=x.dtype).eval()
        left,right=baseline.state_dict(),model.state_dict()
        assert set(left).issubset(right)
        assert all(left[k].shape==right[k].shape for k in left)
        diff={k:float((v-right[k]).abs().max()) for k,v in left.items()}
        extras=sorted(set(right)-set(left))
        counts=[sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        pred=baseline(x);native=snapshot(model,x)
        result=dict(D0B_params=counts[0],PreGNN_params=counts[1],delta_params=counts[1]-counts[0],expected_delta=4256,
            shared_parameter_init_max_diff=max(diff.values()),mismatch_count=sum(v!=0 for v in diff.values()),extra_tensors=extras,
            eval_prediction_max_diff=float((native['prediction']-pred).abs().max()),
            residual_max_abs=float(native['residual'].abs().max()),base_final_max_diff=float((native['prediction']-native['base_pred']).abs().max()))
        result['PASS']=(result['delta_params']==4256 and result['mismatch_count']==0 and result['eval_prediction_max_diff']<1e-7
            and result['residual_max_abs']==result['base_final_max_diff']==0 and extras==['pregnn_local_residual.0.bias','pregnn_local_residual.0.weight','pregnn_local_residual.2.weight'])
        return result


def commodity_order_check(data,model):
    cs,ce=data['market_indices']['commodity'];names=data['feature_names']
    assert cs==model.n_stock+model.n_bond and ce==model.num_nodes and ce-cs==model.n_commodities
    assert len(names)==model.num_nodes and len(set(names[cs:ce]))==model.n_commodities
    errors={}
    for split,loader in data['loaders'].items():
        ds=loader.dataset
        assert (ds.commodity_start,ds.commodity_end)==(cs,ce) and ds.horizons==MULTI_HORIZONS and ds.target_type=='return'
        maximum=0.
        for index in range(len(ds)):
            end=index+ds.seq_len-1
            expected=np.stack([np.clip(ds.raw_prices[end+h,cs:ce]/np.maximum(np.abs(ds.raw_prices[end,cs:ce]),1e-8)-1.,-1.,1.).astype(np.float32) for h in MULTI_HORIZONS])
            maximum=max(maximum,float(np.max(np.abs(ds[index][1].numpy()-expected))))
        errors[split]=maximum
    rows=[dict(commodity_name=str(names[cs+i]),node_index=cs+i,target_index=i,output_index=i,
        flat_head_rows=[h*model.n_commodities+i for h in range(len(MULTI_HORIZONS))]) for i in range(model.n_commodities)]
    return dict(mapping=rows,all_origins_target_reconstruction_max_error=errors,PASS=all(v==0 for v in errors.values()),
        evidence='feature_names and market_indices are emitted by the actual merged price-column order; independently reconstruct every horizon target from the same indexed raw price columns. The existing head reshape is (B,horizon,commodity); local output permutes (B,commodity,horizon) to this order.')


def gradient_diagnostics(model,batch):
    groups={name:list(getattr(model,name).parameters()) for name in ('type_proj','temporal_score','attn_mixhop1','attn_mixhop2','head')}
    groups['residual_first']=list(model.pregnn_local_residual[0].parameters());groups['residual_final']=list(model.pregnn_local_residual[-1].parameters())
    groups['fusion']=list(model.gate_fc.parameters())+list(model.lstm_proj.parameters())+list(model.gcn_proj.parameters())
    with diagnostic_context(model),torch.enable_grad():
        pred=model(batch[0]);loss=prediction_loss(pred,batch[1])
        grads=flattened_gradients(loss,groups)
        return dict(prediction_loss=float(loss.detach()),norms={k:float(v.double().norm()) for k,v in grads.items()},
            first_weight_norm=float(model.pregnn_local_residual[0].weight.detach().norm()),
            final_weight_norm=float(model.pregnn_local_residual[-1].weight.detach().norm()),
            mean_abs_residual=float(model.last_residual.abs().mean()),
            method='Fixed TRAIN batch, eval, sum four Huber losses, autograd.grad; no optimizer step or .grad mutation. Initial zero final weight implies expected zero first-layer gradient, with a live final-layer gradient.')


def fixed_permutation(n,device):
    order=torch.randperm(n,generator=torch.Generator().manual_seed(42))
    if n>1 and torch.equal(order,torch.arange(n)):order=order.roll(1)
    return order.to(device)


def local_controls(model,permutation=None):
    """Reuse ONE native forward's frozen global path. Never re-run GNN/head."""
    assert not model.training
    base=model.last_base_pred;comm=model.last_pregnn_comm;fused=model.last_pregnn_fused
    if permutation is None:permutation=fixed_permutation(model.n_commodities,comm.device)
    global_context=fused[:,None,:].expand(-1,model.n_commodities,-1)
    result=dict(native=dict(prediction=base+model.last_residual,residual=model.last_residual,base_pred=base),
        zero=dict(prediction=base,residual=torch.zeros_like(model.last_residual),base_pred=base))
    with torch.no_grad():
        for key,local in [('shuffle',comm[:,permutation]),('mean',comm.mean(dim=1,keepdim=True).expand_as(comm))]:
            r=model.pregnn_local_residual(torch.cat([global_context,local],-1)).permute(0,2,1)
            result[key]=dict(prediction=base+r,residual=r,base_pred=base)
    return result


def _local_sanity(model,x):
    with diagnostic_context(model),torch.no_grad():
        native=snapshot(model,x);order=torch.arange(len(x)-1,-1,-1,device=x.device)
        perm=snapshot(model,x[order]);single=snapshot(model,x[:1])
        batch={k:float((v-native[k][order]).abs().max()) for k,v in perm.items()}
        single={k:float((v-native[k][:1]).abs().max()) for k,v in single.items()}
        # Prefix forecasts may only aggregate the observed prefix, not the full 20-step window.
        changed=x.clone();changed[:,10:]=-2*changed[:,10:]+7
        left=snapshot(model,x[:,:10]);right=snapshot(model,changed[:,:10])
        prefix={k:float((v-right[k]).abs().max()) for k,v in left.items()}
        future=torch.cat([x,x[:,:5]*-3+11],1);future_changed=future.clone();future_changed[:,20:]+=100
        window=snapshot(model,future_changed[:,:20])
        window_errors={k:float((native[k]-window[k]).abs().max()) for k in native}
        relabel={};start=0
        for market,size in zip(('stock','bond','commodity'),(model.n_stock,model.n_bond,model.n_commodities)):
            node=torch.arange(model.num_nodes,device=x.device);node[start:start+size]=node[start:start+size].flip(0)
            output=torch.arange(model.n_commodities,device=x.device)
            if market=='commodity':output=output.flip(0)
            other=HeteroMixHopCMGM(variant=VARIANT,**model_arguments(model)).to(device=x.device,dtype=x.dtype).eval()
            state={k:v.clone() for k,v in model.state_dict().items()}
            for key in ('graph_learner.E1','graph_learner.E2'):state[key]=state[key][node]
            for key in ('head.3.weight','head.3.bias'):
                v=state[key];state[key]=v.reshape(model.n_horizons,model.n_commodities,*v.shape[1:])[:,output].reshape_as(v)
            other.load_state_dict(state);mapped=snapshot(other,x[:,:,node])
            expected=dict(H_pre=native['H_pre'][:,node],h_comm_pre=native['h_comm_pre'][:,output],
                **{k:native[k][:,:,output] for k in ('base_pred','residual','prediction')})
            relabel[market]={k:float((mapped[k]-expected[k]).abs().max()) for k in expected};start+=size
        # Independently reconstruct pre-GNN state and compare first GNN input.
        observed=[];hook=model.attn_mixhop1.register_forward_pre_hook(lambda m,args:observed.append(args[0].detach().clone()))
        try:current=snapshot(model,x)
        finally:hook.remove()
        seq=torch.stack([model.type_proj(x[:,t],model.n_stock,model.n_bond) for t in range(x.shape[1])],1)
        expected=(seq*model.temporal_score(seq).softmax(1)).sum(1)
        source_error=float((current['H_pre']-expected).abs().max());gnn_input_error=float((current['H_pre']-observed[0]).abs().max())
        controls=local_controls(model)
        base_errors={k:float((v['base_pred']-current['base_pred']).abs().max()) for k,v in controls.items()}
        result=dict(batch_permutation=batch,single_sample=single,prefix_forecast_cutoff10=prefix,legal_window_future_perturbation=window_errors,
            within_market_relabeling=relabel,H_pre_shape=list(native['H_pre'].shape),h_comm_pre_shape=list(native['h_comm_pre'].shape),
            independently_reconstructed_pre_max_diff=source_error,first_GNN_input_max_diff=gnn_input_error,control_base_max_diff=base_errors,
            causality_scope='Temporal trajectories use the existing masked-prefix test. Local/global spatial forecasts aggregate ONLY the observed prefix or forecast window; perturbations inside a fully observed window may legitimately change its final prediction. No full-window output invariance to an observed suffix is claimed.')
        values=list(batch.values())+list(single.values())+list(prefix.values())+list(window_errors.values())+[v for rows in relabel.values() for v in rows.values()]+[source_error,gnn_input_error]+list(base_errors.values())
        result['PASS']=max(values)<1e-6
        return result


def structural_sanity(model,x):
    temporal=full_sanity(model,x);local=_local_sanity(model,x)
    if not local['PASS'] and max(local['batch_permutation'].values())<1e-6 and max(local['prefix_forecast_cutoff10'].values())<1e-6 and max(local['legal_window_future_perturbation'].values())<1e-6:
        # Audit representation-scale floating-point roundoff on a separate double copy.
        with diagnostic_context(model):
            copy=HeteroMixHopCMGM(variant=VARIANT,**model_arguments(model)).to(device=x.device,dtype=torch.float64)
            copy.load_state_dict(model.state_dict());audit=_local_sanity(copy,x.double())
        local['float64_roundoff_audit']=audit
        local['PASS']=audit['PASS']
        local['precision_note']='Raw FP32 errors retained. Independent FP64 copy audits single-sample/relabeling roundoff on large pre-GNN features; no forward precision or batch/prefix threshold change.'
    return dict(temporal=temporal,local=local,PASS=temporal['PASS'] and local['PASS'])
