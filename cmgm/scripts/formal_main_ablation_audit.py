"""Candidate-MoE shared initialization, causal and single-mechanism audits."""
import torch
from cmgm.models.formal_d0b_main_ablation import MainInnovationAblation,FULL,CANDIDATE,GLOBAL,GLOBAL_FULL,TS_VARIANT,TS_FULL
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss


def native(data,variant=CANDIDATE):
    mi=data['market_indices']
    return HeteroMixHopCMGM(data['n_nodes'],24,n_stock=mi['stock'][1]-mi['stock'][0],
        n_bond=mi['bond'][1]-mi['bond'][0],variant=variant)


def init_audit(name,data,device,x,mode='candidate'):
    seed_all(42);full=native(data,TS_VARIANT if mode=='ts-global-expert' else GLOBAL if mode=='global-temporal' else CANDIDATE).to(device)
    seed_all(42);model=MainInnovationAblation(name,data,mode=mode).to(device)
    a=dict(full.named_parameters());b=dict(model.named_parameters());diffs={};removed=[];mismatch=[]
    for key,p in a.items():
        target=key
        if name=='w/o Candidate-Aware Routing' and key.startswith('candidate_moe_fusion.'):
            if key.startswith('candidate_moe_fusion.router.'):
                removed.append(key);continue
            target=key.replace('candidate_moe_fusion.','global_mixture_fusion.',1)
        if name=='w/o MoE Fusion' and key.startswith('candidate_moe_fusion.'):
            removed.append(key);continue
        q=b.get(target)
        if q is None or p.shape!=q.shape:mismatch.append(key);continue
        diffs[key]=float((p-q).detach().abs().max())
        if diffs[key]!=0:mismatch.append(key)
    expected_extra=(('ordinary_mixhop',) if mode=='ts-global-expert' else ('standard_gcn',)) if name=='w/o EdgeAttnMixHop' else ('simple_fusion.',) if name=='w/o MoE Fusion' else ('global_mixture_fusion.',) if name=='w/o Candidate-Aware Routing' else ()
    extra=sorted(set(b)-set(a));unexpected=[k for k in extra if not k.startswith(expected_extra)]
    check=dict(shared_parameter_initial_max_abs_diff=max(diffs.values()),shared_parameter_count=sum(a[k].numel() for k in diffs),
        mismatch_count=len(mismatch)+len(unexpected),mismatches=mismatch,unexpected=unexpected,removed_tensors=removed,extra_tensors=extra,
        full_total_instantiated=sum(p.numel() for p in a.values()),total_instantiated=sum(p.numel() for p in b.values()),
        shared_differences=diffs,PASS=not mismatch and not unexpected)
    if name in (FULL,GLOBAL_FULL,TS_FULL,'w/o Candidate-Aware Routing'):
        seed_all(42);reference=native(data,TS_VARIANT if name==TS_FULL else GLOBAL if name!=FULL else CANDIDATE).to(device)
        reference.eval();model.eval()
        state_ok=all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items()) and reference.state_dict().keys()==model.state_dict().keys()
        with torch.no_grad():diff=float((reference(x)-model(x)).abs().max())
        check.update(native_state_exact=state_ok,native_forward_max_diff=diff)
        check['PASS'] &= state_ok and diff<1e-7
    if hasattr(model,'global_mixture_fusion'):
        check['initial_alpha_max_error']=float((model.global_mixture_fusion.global_mixture_logits.softmax(0)-.5).abs().max())
        check['PASS'] &= check['initial_alpha_max_error']==0
    if mode=='ts-global-expert':
        seed_all(42);old=native(data,GLOBAL).to(device)
        old_params=dict(old.named_parameters())
        shared={k:float((p-old_params[k]).detach().abs().max()) for k,p in model.named_parameters()
                if k in old_params and p.shape==old_params[k].shape}
        check['tst_shared_initialization']=dict(max_abs_diff=max(shared.values()),mismatch_count=sum(v!=0 for v in shared.values()),
            parameter_count=sum(dict(model.named_parameters())[k].numel() for k in shared),PASS=all(v==0 for v in shared.values()))
        check['PASS'] &= check['tst_shared_initialization']['PASS']
    return model,check


def _base_sanity(m,x,y):
    m.eval();name=m.main_name;b=m.switching_latent_transformer;calls={};handles=[]
    tracked={'graph':m.graph_learner,'attn1':m.attn_mixhop1,'attn2':m.attn_mixhop2,'temporal_score':m.temporal_score,
        'LN_H':b.long_memory_norm,'LN_Z':b.micro_state_norm,**{f'G{i}':g for i,g in enumerate(b.latent_transition.generators)}}
    for key,module in tracked.items():
        calls[key]=0
        def hook(mod,inputs,output,key=key):calls[key]+=1
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():p=m(x)
    finally:
        for handle in handles:handle.remove()
    comps={k:v.clone() for k,v in m.components.items()};err=lambda a,c:float((a-c).detach().abs().max())
    checks=dict(call_counts=calls,output_shape=list(p.shape),finite=bool(torch.isfinite(p).all()))
    with torch.no_grad():
        perm=torch.arange(len(x)-1,-1,-1,device=x.device)
        checks['batch_permutation']=err(m(x[perm]),p[perm]);checks['single_sample']=err(m(x[:1]),p[:1])
        m(x);states={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
        future=x.clone();future[:,10:]=future[:,10:]*-3+7;m(future)
        checks['causal_prefix']={k:err(v[:,:10],getattr(b,k)[:,:10]) for k,v in states.items()}
        H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
        h,z=b.last_long_memory[:,:10],b.last_latent_states[:,:10]
        zero='Z' if name=='w/o Microstate' else None
        old=b.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]),zero_component=zero)
        long=b.last_h_long_effective.clone();micro=b.last_h_micro_effective.clone()
        new=b.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]),zero_component=zero)
        checks['causal_prefix'].update(h_temporal=err(old,new),h_long=err(long,b.last_h_long_effective),h_micro=err(micro,b.last_h_micro_effective))
        checks['causality_scope']='E/H/p/Z and readouts prefix10; spatial pooling and final forecast depend on the full legal observed window.'
        checks['prediction_observed_prefix']=err(m(x[:,:10]),m(torch.cat([x[:,:10],x[:,10:]+7],1)[:,:10]))
        m(x)
        if name=='w/o Spatial Temporal Weighting':
            seq=torch.stack([m.type_proj(x[:,t],m.n_stock,m.n_bond) for t in range(x.shape[1])],1)
            checks['uniform_time_error']=err(comps['H_pre'],seq.mean(dim=1));checks['score_bypass']=calls['temporal_score']==0
        if name=='w/o Adaptive Regime Routing':
            checks['uniform_p_error']=err(b.last_regime_probabilities,torch.full_like(b.last_regime_probabilities,1/3))
            checks['uniform_prior_error']=err(b.last_regime_priors,torch.full_like(b.last_regime_priors,1/3))
            checks['three_generators']=all(calls[f'G{i}']==x.shape[1] for i in range(3))
        if name=='w/o Microstate':checks['zero_micro']=float(b.last_h_micro_effective.abs().max())
        if name=='w/o Balanced Readout':checks['LN_bypass']=calls['LN_H']==calls['LN_Z']==0
    params=list(m.named_parameters());pred=m(x)
    grads=torch.autograd.grad(prediction_loss(pred,y),[v for _,v in params],allow_unused=True)
    checks['gradient_finite']=all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    checks['parameter_counts']=dict(total_instantiated=sum(p.numel() for _,p in params),
        active_prediction_path_tensor_elements=sum(p.numel() for (_,p),g in zip(params,grads) if g is not None),
        nonzero_prediction_gradient_elements=sum(int(torch.count_nonzero(g)) for g in grads if g is not None),
        active_parameter_names=[k for (k,_),g in zip(params,grads) if g is not None],
        scope='Autograd-connected tensor count; zero-valued gradients may still be connected.')
    epoch=b.regime_filter.current_epoch
    b.set_epoch(20);m(x);aux=m.auxiliary_loss()
    checks['switch_KL']=dict(value=float(aux.detach()),requires_grad=aux.requires_grad,beta=b.regime_filter.current_beta)
    if name=='w/o Adaptive Regime Routing':checks['KL_zero']=abs(float(aux.detach()))<1e-8
    b.set_epoch(epoch)
    errors=[checks[k] for k in ('batch_permutation','single_sample','prediction_observed_prefix','uniform_time_error','uniform_p_error','uniform_prior_error','zero_micro') if k in checks]+list(checks['causal_prefix'].values())
    flags=[v for v in checks.values() if isinstance(v,bool)]
    checks['PASS']=all(flags) and pred.shape==(len(x),4,24) and max(errors)<1e-6
    return checks


def sanity(model,x,y):
    result=_base_sanity(model,x,y)
    name=model.main_name;b=model.switching_latent_transformer
    checks={};calls={};handles=[];adjacencies=[]
    tracked={**{f'edge{l}.{p}':getattr(getattr(model,f'attn_mixhop{l}'),p) for l in (1,2) for p in ('q','k','v')},
             **{f'transformer{i}.{p}':getattr(layer.attention,p) for i,layer in enumerate(b.long_memory.layers) for p in ('q','k','v')},
             'long_readout':b.long_memory_readout,'micro_readout':b.micro_state_readout,'state_readout':b.state_readout}
    for layer in (1,2):
        for prefix in ('standard_gcn','ordinary_mixhop'):
            key=f'{prefix}{layer}'
            if hasattr(model,key):tracked[key]=getattr(model,key)
    for key,module in tracked.items():
        calls[key]=0
        def hook(module,inputs,output,key=key):
            calls[key]+=1
            if key.startswith(('standard_gcn','ordinary_mixhop')):adjacencies.append(inputs[1])
        handles.append(module.register_forward_hook(hook))
    if name=='w/o Adaptive Graph':
        for layer in (model.attn_mixhop1,model.attn_mixhop2):layer.capture_attention=True
    try:
        with torch.no_grad():model(x)
        for handle in handles:handle.remove()
        with torch.no_grad():
            c=model.components
            checks['branch_shapes']=all(c[k].shape==(len(x),64) for k in ('s','t','h_spatial','h_temporal','fused'))
            if name=='w/o MoE Fusion':
                checks['moe_absent']=not hasattr(model,'candidate_moe_fusion') and not hasattr(model,'global_mixture_fusion')
                checks['linear_only']=type(model.simple_fusion) is torch.nn.Linear and model.simple_fusion.in_features==128 and model.simple_fusion.out_features==64
                checks['simple_fusion_parameter_count']=sum(p.numel() for p in model.simple_fusion.parameters())
                checks['fusion_error']=float((c['fused']-model.simple_fusion(torch.cat([c['s'],c['t']],-1))).abs().max())
            else:
                static=hasattr(model,'global_mixture_fusion')
                f=model.global_mixture_fusion if static else model.candidate_moe_fusion
                d={k:v.clone() for k,v in f.last.items()}
                ts=hasattr(f,'spatial_expert')
                second='e_S' if ts else 'e_ST'
                checks['expert_shapes']=all(d[k].shape==(len(x),64) for k in ('e_T',second))
                checks['temporal_wiring_error']=float((d['e_T']-f.temporal_expert(c['t'])).abs().max())
                if ts:
                    from cmgm.models.candidate_moe_fusion import SpatialTemporalInteractionExpert
                    checks['spatial_wiring_error']=float((d['e_S']-f.spatial_expert(c['s'])).abs().max())
                    checks['temporal_residual_error']=float((d['e_T']-c['t']-f.temporal_expert.mlp(c['t'])).abs().max())
                    checks['spatial_residual_error']=float((d['e_S']-c['s']-f.spatial_expert.mlp(c['s'])).abs().max())
                    counts=[sum(p.numel() for p in expert.parameters()) for expert in (f.temporal_expert,f.spatial_expert)]
                    checks['expert_parameters']=dict(Temporal=counts[0],Spatial=counts[1])
                    checks['expert_symmetry']=counts[0]==counts[1] and repr(f.temporal_expert.mlp)==repr(f.spatial_expert.mlp)
                    checks['no_interaction_expert']=not hasattr(f,'interaction_expert') and not any(isinstance(v,SpatialTemporalInteractionExpert) for v in f.modules())
                    f(c['s']+2,c['t']);checks['temporal_independent_of_spatial_error']=float((f.last['e_T']-d['e_T']).abs().max())
                    f(c['s'],c['t']+2);checks['spatial_independent_of_temporal_error']=float((f.last['e_S']-d['e_S']).abs().max())
                    checks['branch_specific_expert_wiring']=all(checks[k]<1e-6 for k in ('temporal_wiring_error','spatial_wiring_error','temporal_independent_of_spatial_error','spatial_independent_of_temporal_error'))
                else:
                    checks['interaction_wiring_error']=float((d['e_ST']-f.interaction_expert(torch.cat([c['s'],c['t']],-1))).abs().max())
                weights=d['alpha'] if static else d['pi']
                checks['weight_shape']=weights.shape==((2,) if static else (len(x),2))
                checks['weight_sum_error']=float((weights.sum(-1)-1).abs().max())
                expected=(weights.reshape(-1,2,1)*torch.stack([d['e_T'],d[second]],1)).sum(1)
                checks['fusion_error']=float((c['fused']-expected).abs().max())
                if static:
                    checks['candidate_router_absent']=not hasattr(f,'router') and not hasattr(model,'candidate_moe_fusion')
                    checks['global_logits_shape']=f.global_mixture_logits.shape==(2,)
                    checks['global_logits_trainable']=f.global_mixture_logits.requires_grad
                    checks['router_parameters_absent']=not any('router' in k or 'temporal_norm' in k or 'interaction_norm' in k for k,_ in f.named_parameters())
                    model(x*2+1);checks['input_invariant_alpha_error']=float((f.last['alpha']-weights).abs().max())
                else:
                    perm=torch.arange(len(x)-1,-1,-1,device=x.device);model(x[perm])
                    checks['pi_permutation_error']=float((f.last['pi']-weights[perm]).abs().max())
                    model(x[:1]);checks['pi_single_sample_error']=float((f.last['pi']-weights[:1]).abs().max())
                model(x)
        active=result['parameter_counts']['active_parameter_names']
        if hasattr(model,'global_mixture_fusion'):
            checks['global_logits_prediction_connected']='global_mixture_fusion.global_mixture_logits' in active
        cc=result['call_counts']
        if name=='w/o Adaptive Graph':
            checks['ones_error']=float((model.last_ablation_adjacency-1).abs().max())
            checks['graph_inactive']=cc['graph']==0 and not any(k.startswith('graph_learner') for k in active)
            checks['edge_qkv_active']=all(calls[f'edge{l}.{p}']>0 for l in (1,2) for p in ('q','k','v'))
            checks['content_attention_softmax_error']=max(float((d['attention']-d['content'].softmax(dim=2)).abs().max()) for layer in (model.attn_mixhop1,model.attn_mixhop2) for d in layer.last_attention_diagnostics)
            d=model.attn_mixhop1.last_attention_diagnostics[0]['attention']
            checks['sample_attention_difference']=float((d[0]-d[-1]).abs().max())
            checks['attention_uniform_deviation']=float((d-1/x.shape[2]).abs().max())
            checks['note']='Native log(1+1e-6) is a constant prior shift, not literal zero; softmax is content-only to numerical tolerance.'
        elif name=='w/o EdgeAttnMixHop':
            checks['learned_graph_active']=cc['graph']>0 and any(k.startswith('graph_learner') for k in active)
            checks['same_A_both_blocks']=len(adjacencies)==2 and adjacencies[0] is adjacencies[1] and torch.equal(adjacencies[0].detach(),model.last_ablation_adjacency)
            if model.ablation_mode=='ts-global-expert':
                from cmgm.models.model import MixHopPropagation
                checks['ordinary_mixhop_blocks_active']=calls['ordinary_mixhop1']==calls['ordinary_mixhop2']==1
                checks['ordinary_mixhop_exact']=all(type(getattr(model,f'ordinary_mixhop{i}')) is MixHopPropagation and getattr(model,f'ordinary_mixhop{i}').K==2 and getattr(model,f'ordinary_mixhop{i}').beta==.05 for i in (1,2))
                checks['no_gcn_replacement']=not hasattr(model,'standard_gcn1')
            else:
                checks['standard_gcn_blocks_active']=calls['standard_gcn1']==calls['standard_gcn2']==1
                checks['single_hop_linear_only']=all(type(getattr(model,f'standard_gcn{i}').linear) is torch.nn.Linear and len(list(getattr(model,f'standard_gcn{i}').children()))==1 for i in (1,2))
                checks['no_mixhop_modules']=not any('ordinary_mixhop' in k for k,_ in model.named_modules())
            checks['edge_qkv_inactive']=all(calls[f'edge{l}.{p}']==0 for l in (1,2) for p in ('q','k','v')) and not any(k.startswith('attn_mixhop') for k in active)
        elif name=='w/o Relative Position Encoding':
            checks['rpe_bias_error']=float(b.long_memory.last_base_relative_bias.abs().max())
            checks['rpe_prediction_inactive']=not any(k.endswith('base_rpe') for k in active)
            checks['transformer_qkv_active']=all(v>0 for k,v in calls.items() if k.startswith('transformer'))
            checks['causal_mask_active']=all(bool(torch.isneginf(layer.attention.last_attention_logits[...,torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)]).all()) for layer in b.long_memory.layers)
        elif name=='w/o Regime-Specific Transitions':
            c=b.last_latent_candidates
            checks['candidate12_error']=float((c[:,:,0]-c[:,:,1]).abs().max())
            checks['candidate13_error']=float((c[:,:,0]-c[:,:,2]).abs().max())
            checks['shared_G_only']=cc['G0']==x.shape[1] and cc['G1']==cc['G2']==0
            checks['unused_generators_inactive']=not any(k.startswith(('switching_latent_transformer.latent_transition.generators.1','switching_latent_transformer.latent_transition.generators.2')) for k in active)
            checks['adaptive_p_retained']=not b.formal_uniform_switching
            checks['p_uniform_deviation']=float((b.last_regime_probabilities-1/3).abs().max())
            checks['KL_retained']=result['switch_KL']['requires_grad'] and result['switch_KL']['beta']==.0005
        elif name=='w/o Adaptive Regime Routing':
            gs=b.latent_transition.generators
            checks['generator_parameters_distinct']=all(not torch.equal(gs[0][0].weight,g[0].weight) for g in gs[1:])
        elif name=='w/o Microstate':
            checks['recurrence_retained']=all(cc[f'G{i}']==x.shape[1] for i in range(3))
            checks['KL_retained']=result['switch_KL']['requires_grad'] and result['switch_KL']['beta']==.0005
        elif name=='w/o Balanced Readout':
            checks['readout_projections_active']=all(calls[k]>0 for k in ('long_readout','micro_readout','state_readout'))
        flags=[v for v in checks.values() if isinstance(v,bool)]
        errors=[v for k,v in checks.items() if k.endswith('_error')]
        result['revised_structure']=checks
        result['revised_call_counts']=calls
        result['PASS'] &= all(flags) and all(v<1e-6 for v in errors)
    finally:
        for h in handles:h.remove()
        for layer in (model.attn_mixhop1,model.attn_mixhop2):
            layer.capture_attention=False;layer.last_attention_diagnostics=[]
    return result
