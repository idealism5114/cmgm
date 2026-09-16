"""Read-only, bounded mechanism and shared-initialization checks."""
import torch
from cmgm.models.formal_d0b_ablation import NAMES,MASKS,FormalD0BAblation
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss


def init_audit(name,data,device):
    seed_all(42);full=FormalD0BAblation(NAMES[0],data).to(device)
    seed_all(42);m=FormalD0BAblation(name,data).to(device)
    a=dict(full.named_parameters());b=dict(m.named_parameters());rows=[]
    for key,p in a.items():
        q=b.get(key);error=float((p-q).abs().max()) if q is not None and q.shape==p.shape else None
        rows.append(dict(name=key,shape=list(p.shape),max_abs_diff=error))
    mismatch=sum(v['max_abs_diff']!=0 for v in rows)+len(set(b)-set(a))
    result=dict(parameters=rows,total_instantiated=sum(p.numel() for p in b.values()),
        max_abs_diff=max((v['max_abs_diff'] or 0) for v in rows),mismatch_count=mismatch,PASS=mismatch==0)
    del full;return m,result


def sanity(m,x,y):
    m.eval();name=m.ablation_name;branch=m.switching_latent_transformer
    checks={};calls={};handles=[]
    tracked={'graph':m.graph_learner,'attn1':m.attn_mixhop1,'attn2':m.attn_mixhop2,'temporal_score':m.temporal_score,
        'LN_H':branch.long_memory_norm,'LN_Z':branch.micro_state_norm,'gate':m.gate_fc,
        **{f'G{i}':g for i,g in enumerate(branch.latent_transition.generators)}}
    for key,module in tracked.items():
        calls[key]=0
        def hook(mod,inputs,output,key=key):calls[key]+=1
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():p=m(x)
    for h in handles:h.remove()
    comps={k:v.clone() for k,v in m.components.items()};checks['call_counts']=calls
    checks.update(output_shape=list(p.shape),finite=bool(torch.isfinite(p).all()))
    with torch.no_grad():
        perm=torch.arange(len(x)-1,-1,-1,device=x.device)
        checks['batch_permutation']=float((m(x[perm])-p[perm]).abs().max())
        checks['single_sample']=float((m(x[:1])-p[:1]).abs().max())
        if name!='w/o Temporal Branch':
            m(x);states={k:getattr(branch,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
            future=x.clone();future[:,10:]=future[:,10:]*-3+7;m(future)
            checks['causal_prefix']={k:float((v[:,:10]-getattr(branch,k)[:,:10]).abs().max()) for k,v in states.items()}
            H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
            h,z=branch.last_long_memory[:,:10],branch.last_latent_states[:,:10]
            a=branch.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]),zero_component='Z' if name=='w/o Microstate' else None)
            longs=branch.last_h_long_effective.clone();micros=branch.last_h_micro_effective.clone()
            b=branch.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]),zero_component='Z' if name=='w/o Microstate' else None)
            checks['causal_prefix'].update(h_temporal=float((a-b).abs().max()),h_long=float((longs-branch.last_h_long_effective).abs().max()),h_micro=float((micros-branch.last_h_micro_effective).abs().max()))
        checks['causality_scope']='E/H/p/Z and effective readouts prefix10 where present. Forecast head and spatial pool use the entire legal observed window; no full-window prediction invariance to changing observed future-within-window values is claimed.'
        checks['prediction_observed_prefix']=float((m(x[:,:10])-m(torch.cat([x[:,:10],x[:,10:]+7],dim=1)[:,:10])).abs().max())
        if name in MASKS:
            masked=m.mask_input(x);checks['masking']={market:float((masked[:,:,a:b]-x[:,:,a:b]).abs().max()) for market,(a,b) in m.market_indices.items()}
            checks['mask_exact']=all(torch.count_nonzero(masked[:,:,a:b])==0 if market in MASKS[name] else torch.equal(masked[:,:,a:b],x[:,:,a:b]) for market,(a,b) in m.market_indices.items())
        if name=='w/o Graph Propagation':checks['graph_bypass']=calls['graph']==calls['attn1']==calls['attn2']==0
        if name=='w/o TempWeighted':
            seq=torch.stack([m.type_proj(x[:,t],m.n_stock,m.n_bond) for t in range(x.shape[1])],dim=1)
            checks['uniform_time_error']=float((comps['H_pre']-seq.mean(dim=1)).abs().max());checks['score_bypass']=calls['temporal_score']==0
        m(x)
        if name=='w/o Markov Switching':
            checks['uniform_p_error']=float((branch.last_regime_probabilities-1/3).abs().max())
            checks['three_generators']=all(calls[f'G{i}']==x.shape[1] for i in range(3))
        if name=='w/o Microstate':checks['zero_micro']=float(branch.last_h_micro_effective.abs().max())
        if name=='w/o Balanced Readout':checks['LN_bypass']=calls['LN_H']==calls['LN_Z']==0
        if name in ('w/o Spatial Branch','w/o Temporal Branch','w/o Adaptive Fusion Gate'):
            expected=comps['t'] if name=='w/o Spatial Branch' else comps['s'] if name=='w/o Temporal Branch' else .5*comps['t']+.5*comps['s']
            checks['fusion_formula_error']=float((comps['fused']-expected).abs().max());checks['gate_bypass']=calls['gate']==0
    p=m(x);loss=prediction_loss(p,y);params=list(m.named_parameters())
    gradients=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    defined=[n for (n,_),g in zip(params,gradients) if g is not None]
    counts=dict(total_instantiated=sum(p.numel() for _,p in params),active_prediction_path_tensor_elements=sum(p.numel() for (_,p),g in zip(params,gradients) if g is not None),
        defined_prediction_gradient_elements=sum(g.numel() for g in gradients if g is not None),nonzero_prediction_gradient_elements=sum(int(torch.count_nonzero(g)) for g in gradients if g is not None),
        active_parameter_names=defined,scope='Active path counted at whole parameter-tensor granularity via autograd connectivity; nonzero counts are fixed-batch element counts, not theoretical capacity.')
    inactive=[]
    if name=='w/o Spatial Branch':inactive=['type_proj','temporal_score','graph_learner','attn_mixhop','gcn_norm','type_pool','gcn_proj','gate_fc']
    if name=='w/o Temporal Branch':inactive=['switching_latent_transformer','lstm_proj','gate_fc']
    if name=='w/o Graph Propagation':inactive=['graph_learner','attn_mixhop']
    if name=='w/o TempWeighted':inactive=['temporal_score']
    if name=='w/o Markov Switching':inactive=['switching_latent_transformer.regime_filter']
    if name=='w/o Microstate':inactive=['switching_latent_transformer.regime_filter','switching_latent_transformer.latent_transition','switching_latent_transformer.micro_state']
    if name=='w/o Balanced Readout':inactive=['switching_latent_transformer.long_memory_norm','switching_latent_transformer.micro_state_norm']
    if name=='w/o Adaptive Fusion Gate':inactive=['gate_fc']
    checks['inactive_gradient_violations']=[n for (n,_),g in zip(params,gradients) if any(n.startswith(prefix) for prefix in inactive) and g is not None and bool(torch.count_nonzero(g))]
    branch.set_epoch(20);m(x);aux=m.auxiliary_loss()
    checks['switch_KL']=dict(value=float(aux.detach()),requires_grad=aux.requires_grad,beta=branch.regime_filter.current_beta,
        reason='disabled with entire temporal branch' if name=='w/o Temporal Branch' else 'beta contribution zero' if name=='w/o Switch KL' else 'fixed-uniform p/prior: KL=0, no learnable routing gradient' if name=='w/o Markov Switching' else 'unchanged D0B KL')
    if m.disable_switch_kl:checks['KL_zero']=float(aux.detach())==0 and not aux.requires_grad
    checks['gradient_finite']=all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
    checks['parameter_counts']=counts
    errors=[checks[k] for k in ('batch_permutation','single_sample','prediction_observed_prefix')]+list(checks.get('causal_prefix',{}).values())
    errors += [checks[k] for k in ('uniform_time_error','uniform_p_error','zero_micro','fusion_formula_error') if k in checks]
    flags=[checks[k] for k in ('mask_exact','graph_bypass','score_bypass','three_generators','LN_bypass','gate_bypass','KL_zero','gradient_finite') if k in checks]
    checks['PASS']=checks['finite'] and p.shape==(len(x),4,24) and max(errors)<1e-6 and all(flags) and not checks['inactive_gradient_violations']
    branch.set_epoch(1)
    return checks
