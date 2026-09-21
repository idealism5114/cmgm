"""Read-only initialization, fusion wiring and causal-state audits."""
import torch
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.moe_fusion import VARIANT,BALANCE_COEFFICIENT
from cmgm.scripts.baseline_protocol import seed_all,prediction_loss
BASE='switching_latent_balanced_readout'


def make_model(data,variant=VARIANT):
    mi=data['market_indices']
    return HeteroMixHopCMGM(data['n_nodes'],24,n_stock=mi['stock'][1]-mi['stock'][0],
                           n_bond=mi['bond'][1]-mi['bond'][0],variant=variant)


def initialization(data,device):
    seed_all(42);baseline=make_model(data,BASE).to(device)
    seed_all(42);model=make_model(data).to(device)
    a=dict(baseline.named_parameters());b=dict(model.named_parameters())
    shared=sorted(set(a)&set(b));removed=sorted(set(a)-set(b));added=sorted(set(b)-set(a))
    diffs={n:float((a[n]-b[n]).detach().abs().max()) for n in shared}
    result=dict(baseline_parameters=sum(p.numel() for p in a.values()),moe_parameters=sum(p.numel() for p in b.values()),
        shared_max_abs_diff=max(diffs.values()),mismatch_count=sum(v!=0 for v in diffs.values()),
        removed_parameters=removed,new_parameters=added,shared_parameter_differences=diffs,
        initialization='Native D0B constructed first; old gate removed only in MoE; new experts/router use standard nn.Linear initialization.')
    result['delta_parameters']=result['moe_parameters']-result['baseline_parameters']
    result['PASS']=result['shared_max_abs_diff']==0 and result['delta_parameters']==25027 and removed==['gate_fc.bias','gate_fc.weight'] and all(n.startswith('moe_fusion.') for n in added)
    return model,result


def sanity(model,x,y,require_initial_activity=True):
    model.eval();f=model.moe_fusion;b=model.switching_latent_transformer
    epoch=int(f.epoch);state={k:v.clone() for k,v in model.state_dict().items()}
    try:
        with torch.no_grad():
            p=model(x);pi=f.last_pi.clone();eff=f.last_effective_pi.clone()
            inputs={k:v.clone() for k,v in f.last_inputs.items()}
            experts=f.last_experts.clone();fused=f.last_fused.clone()
            result=dict(shapes={**{k:list(v.shape) for k,v in inputs.items()},'experts':list(experts.shape),'router_logits':list(f.last_logits.shape),
                               'pi':list(pi.shape),'pi_eff':list(eff.shape),'h_moe':list(fused.shape),'prediction':list(p.shape)},
                finite=bool(torch.isfinite(p).all()),pi_normalization_error=float((pi.sum(-1)-1).abs().max()),
                effective_normalization_error=float((eff.sum(-1)-1).abs().max()),sample_pi_difference=float((pi[0]-pi[-1]).abs().max()),
                fusion_formula_error=float((fused-(eff[...,None]*experts).sum(1)).abs().max()),balance_loss=float(f.balance_loss()))
            result['expert_formula_error']=max(float((experts[:,0]-f.temporal_expert(inputs['t'])).abs().max()),
                float((experts[:,1]-f.spatial_expert(inputs['s'])).abs().max()),
                float((experts[:,2]-f.interaction_expert(torch.cat([inputs['s'],inputs['t']],-1))).abs().max()))
            result['router_raw_input_error']=float((pi-f.router(torch.cat([inputs['h_spatial'],inputs['h_temporal']],-1)).softmax(-1)).abs().max())
            result['projections_preserved_error']=max(float((inputs['s']-model.gcn_proj(inputs['h_spatial'])).abs().max()),float((inputs['t']-model.lstm_proj(inputs['h_temporal'])).abs().max()))
            perm=torch.arange(len(x)-1,-1,-1,device=x.device)
            result['batch_permutation_error']=float((model(x[perm])-p[perm]).abs().max())
            result['single_sample_error']=float((model(x[:1])-p[:1]).abs().max())
            model(x);states={k:getattr(b,k).clone() for k in ('last_market_tokens','last_long_memory','last_regime_probabilities','last_latent_states')}
            future=x.clone();future[:,10:]=future[:,10:]*-3+7;model(future)
            result['causal_prefix']={k:float((v[:,:10]-getattr(b,k)[:,:10]).abs().max()) for k,v in states.items()}
            H,Z=states['last_long_memory'][:,:10],states['last_latent_states'][:,:10]
            h,z=b.last_long_memory[:,:10],b.last_latent_states[:,:10]
            old=b.readout(H.reshape(-1,H.shape[-1]),Z.reshape(-1,Z.shape[-1]));old_long=b.last_h_long.clone();old_micro=b.last_h_micro.clone()
            new=b.readout(h.reshape(-1,h.shape[-1]),z.reshape(-1,z.shape[-1]))
            result['causal_prefix'].update(h_temporal=float((old-new).abs().max()),h_long=float((old_long-b.last_h_long).abs().max()),h_micro=float((old_micro-b.last_h_micro).abs().max()))
            result['warmup']={}
            for e in (0,1,2,10,20):
                model.set_moe_epoch(e);model(x)
                expected=(1-min(1,e/10))/3+min(1,e/10)*f.last_pi
                result['warmup'][str(e)]=dict(gamma=f.gamma,max_formula_error=float((expected-f.last_effective_pi).abs().max()))
            uniform=torch.full((3,),1/3,device=x.device,dtype=torch.float64)
            result['uniform_balance']=float((uniform*((uniform+1e-8)/(1/3)).log()).sum())
        model.set_moe_epoch(1)
        pred=model(x);loss=prediction_loss(pred,y)+b.switch_loss()+BALANCE_COEFFICIENT*model.moe_balance_loss()
        params=list(model.named_parameters());grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
        result['gradient_norms']={k:sum(float(g.detach().double().square().sum()) for (n,_),g in zip(params,grads) if n.startswith('moe_fusion.'+k) and g is not None)**.5 for k in ('temporal_expert','spatial_expert','interaction_expert','router')}
        result['initial_activity_required']=require_initial_activity
        result['dense_autograd_connected']=all(any(n.startswith('moe_fusion.'+key) and g is not None for (n,_),g in zip(params,grads)) for key in result['gradient_norms'])
        result['gradient_finite']=all(g is None or bool(torch.isfinite(g).all()) for g in grads)
        errors=[v for k,v in result.items() if k.endswith('_error')]+list(result['causal_prefix'].values())+[v['max_formula_error'] for v in result['warmup'].values()]
        result['PASS']=result['finite'] and p.shape==(len(x),4,24) and pi.shape==(len(x),3) and experts.shape==(len(x),3,64) and (not require_initial_activity or result['sample_pi_difference']>0) and max(errors)<1e-6 and result['balance_loss']>=-1e-6 and abs(result['uniform_balance'])<1e-6 and result['gradient_finite'] and result['dense_autograd_connected'] and (not require_initial_activity or all(v>0 for v in result['gradient_norms'].values()))
        result['causality_scope']='Prefix10 E/H/p/Z and long/micro/temporal readouts; spatial and fusion forecast use only the complete legal observed window.'
        return result
    finally:
        model.set_moe_epoch(epoch)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
        assert all(p.grad is None for p in model.parameters())
