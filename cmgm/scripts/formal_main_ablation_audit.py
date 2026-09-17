"""Revised structural checks extend (and do not change) the previous formal audit."""
import torch
from cmgm.models.formal_d0b_main_ablation import MainInnovationAblation, REUSE
from cmgm.models.formal_d0b_ablation import FormalD0BAblation
from cmgm.scripts.formal_ablation_audit import sanity as legacy_sanity
from cmgm.scripts.baseline_protocol import seed_all


def init_audit(name, data, device, x):
    seed_all(42); full = FormalD0BAblation('FullD0B-Control',data).to(device)
    seed_all(42); model = MainInnovationAblation(name,data).to(device)
    a,b=full.state_dict(),model.state_dict()
    mismatches=[k for k,v in a.items() if k not in b or v.shape!=b[k].shape or not torch.equal(v,b[k])]
    extra=sorted(set(b)-set(a))
    expected_extra=name=='w/o EdgeAttnMixHop'
    check=dict(shared_parameter_initial_max_abs_diff=max(float((v-b[k]).abs().max()) for k,v in a.items()),
        mismatch_count=len(mismatches),mismatches=mismatches,extra_tensors=extra,
        full_total_instantiated=sum(p.numel() for p in full.parameters()),
        total_instantiated=sum(p.numel() for p in model.parameters()),
        PASS=not mismatches and (all(k.startswith('ordinary_mixhop') for k in extra) if expected_extra else not extra))
    if name in REUSE:
        seed_all(42); old=FormalD0BAblation(REUSE[name],data).to(device)
        old.eval();model.eval()
        with torch.no_grad():diff=float((old(x)-model(x)).abs().max())
        check['legacy_forward_max_diff']=diff;check['PASS'] &= diff==0
    return model,check


def sanity(model,x,y):
    result=legacy_sanity(model,x,y)
    name=model.main_name;b=model.switching_latent_transformer
    checks={};calls={};handles=[];adjacencies=[]
    tracked={**{f'edge{l}.{p}':getattr(getattr(model,f'attn_mixhop{l}'),p) for l in (1,2) for p in ('q','k','v')},
             **{f'transformer{i}.{p}':getattr(layer.attention,p) for i,layer in enumerate(b.long_memory.layers) for p in ('q','k','v')},
             'long_readout':b.long_memory_readout,'micro_readout':b.micro_state_readout,'state_readout':b.state_readout}
    for layer in (1,2):
        key=f'ordinary_mixhop{layer}'
        if hasattr(model,key):tracked[key]=getattr(model,key)
    for key,module in tracked.items():
        calls[key]=0
        def hook(module,inputs,output,key=key):
            calls[key]+=1
            if key.startswith('ordinary_mixhop'):adjacencies.append(inputs[1])
        handles.append(module.register_forward_hook(hook))
    if name=='w/o Adaptive Graph':
        for layer in (model.attn_mixhop1,model.attn_mixhop2):layer.capture_attention=True
    try:
        with torch.no_grad():model(x)
        active=result['parameter_counts']['active_parameter_names']
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
            checks['ordinary_blocks_active']=calls['ordinary_mixhop1']==calls['ordinary_mixhop2']==1
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
