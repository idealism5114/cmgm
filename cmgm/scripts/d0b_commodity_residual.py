"""D0B + one zero-start, shared linear commodity residual. No training in probes."""
from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.data.data_loader import set_seed
from cmgm.training.train import COMMODITY_RESIDUAL_VARIANT as VARIANT, _prediction_loss, make_loss
from cmgm.training.metric_standard import STANDARD, population_metrics
from cmgm.scripts.d0b_regime_routing_diagnostics import (
    ROOT, array, difference, probability_stats, specialization, horizon_metrics, run_intervention, sha256,
)
from cmgm.scripts.d0b_5d_only_diagnostics import prediction_losses_from_arrays
from cmgm.scripts.d0e_diagnostics import diagnostic_context, model_arguments, micro_diagnostics, transition_diagnostics
from cmgm.scripts.d0b_grouped_diagnostics import extra_fixed_diagnostics
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint

BASE_VARIANT='switching_latent_balanced_readout'
DISPLAY='D0B-CommodityResidualAdapter'
NEW='CommodityResidual'
HORIZONS=(1,5,10,20)
TRACE_KEYS=('E','H','prior','evidence','p','candidates','Z','h_long','h_micro','h_temporal',
            'h_nodes','h_comm','h_spatial','gate','fused','base_pred','prediction')


def is_adapter(model):
    return model.variant==VARIANT


def assert_backbone(model):
    b=model.switching_latent_transformer
    assert model.variant in (BASE_VARIANT,VARIANT)
    assert b.balanced_readout and b.K==3 and b.z_dim==64
    assert not any((b.horizon_specific_state_readout,b.use_dynamic_slope,b.use_balanced_transition_input,
                    b.use_latent_memory,b.use_regime_relative_memory,b.regime_filter.learnable_sticky_alpha,
                    getattr(model,'disable_switch_kl',False)))
    assert b.regime_filter.sticky_alpha_value()==.5 and b.regime_filter.tau==1.
    assert b.regime_filter.beta_max==5e-4 and b.regime_filter.warmup_epochs==20
    if is_adapter(model):
        assert model.commodity_residual_head.bias is None
        assert model.commodity_residual_alpha.ndim==0
        assert (model.commodity_residual_head.in_features,model.commodity_residual_head.out_features)==(64,4)


def reference(model,x):
    """Capture the single real forward. No second spatial or temporal evaluation."""
    captured={}
    def nodes_hook(module,inputs,output):captured['h_nodes']=output.detach().clone()
    def gate_hook(module,inputs,output):
        captured['h_spatial']=inputs[0][...,:64].detach().clone()
        captured['gate']=output.sigmoid().detach().clone()
    def head_hook(module,inputs):captured['fused']=inputs[0].detach().clone()
    handles=[model.gcn_norm.register_forward_hook(nodes_hook),model.gate_fc.register_forward_hook(gate_hook),
             model.head.register_forward_pre_hook(head_hook)]
    try:prediction=model(x)
    finally:
        for h in handles:h.remove()
    b=model.switching_latent_transformer
    attrs={'E':'last_market_tokens','H':'last_long_memory','prior':'last_regime_priors',
           'evidence':'last_regime_evidence','p':'last_regime_probabilities','candidates':'last_latent_candidates',
           'Z':'last_latent_states','h_long':'last_h_long','h_micro':'last_h_micro','h_temporal':'last_h_temporal'}
    trace={k:getattr(b,v).clone() for k,v in attrs.items()}
    trace.update(captured,prediction=prediction,h_comm=captured['h_nodes'][:,model.n_stock+model.n_bond:])
    trace.update(base_pred=model.last_base_pred.clone() if is_adapter(model) else prediction,
                 raw=model.last_commodity_residual_raw.clone() if is_adapter(model) else torch.zeros_like(prediction),
                 effective=model.last_commodity_residual_effective.clone() if is_adapter(model) else torch.zeros_like(prediction))
    trace.update(q=trace['p'],A=b.transition_matrix())
    return trace


def commodity_permutation(n,seed=42,device='cpu'):
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed))
    if n>1 and torch.equal(order,torch.arange(n)):order=order.roll(1)
    return order.to(device)


def commodity_control(model,trace,mode,permutation):
    """Change residual inputs only; never mutate alpha, weights or the base path."""
    base=trace['base_pred']
    if mode=='alpha=0' or not is_adapter(model):return base
    nodes=trace['h_comm']
    if mode=='shuffled':nodes=nodes[:,permutation]
    elif mode=='mean-commodity':nodes=nodes.mean(dim=1,keepdim=True).expand_as(nodes)
    else:raise ValueError(mode)
    raw=model.commodity_residual_head(nodes).permute(0,2,1)
    return base+model.commodity_residual_alpha*raw


def temporal_control(model,trace,mode):
    spec={'mode':'uniform'} if mode=='uniform' else {'zero_component':'Z' if mode=='zero-micro' else 'H'}
    changed=run_intervention(model,trace['h_spatial'],trace,spec,None,None)
    torch.testing.assert_close(changed['p'],trace['p'],rtol=0,atol=0)
    # The spatial residual has no H/Z dependence; retain it in every control.
    return changed['prediction']+trace['effective']


def residual_statistics(trace):
    raw,effective,base=[array(trace[k]).astype(np.float64) for k in ('raw','effective','base_pred')]
    return {'raw_mean_abs':float(np.abs(raw).mean()),'raw_std':float(raw.std()),
            'raw_norm':float(np.linalg.norm(raw.reshape(len(raw),-1),axis=-1).mean()),
            'effective_mean_abs':float(np.abs(effective).mean()),
            'effective_base_ratio':float(np.abs(effective).mean()/(np.abs(base).mean()+1e-8)),
            'horizons':{str(h):{'raw_mean_abs':float(np.abs(raw[:,i]).mean()),
                              'effective_mean_abs':float(np.abs(effective[:,i]).mean()),
                              'effective_base_ratio':float(np.abs(effective[:,i]).mean()/(np.abs(base[:,i]).mean()+1e-8)),
                              **{p:float(np.percentile(np.abs(effective[:,i]),q)) for p,q in [('P50',50),('P90',90),('P95',95)]},
                              'max':float(np.abs(effective[:,i]).max())} for i,h in enumerate(HORIZONS)}}


def parameter_groups(model):
    b=model.switching_latent_transformer
    modules={'Market Encoder':[b.market_encoder],'LongMemory Transformer':[b.long_memory],
             'regime evidence':[b.regime_filter.regime_evidence],'generators':[b.latent_transition],
             'BalancedReadout':[b.long_memory_readout,b.micro_state_readout,b.long_memory_norm,b.micro_state_norm,b.state_readout],
             'spatial branch':[model.type_proj,model.temporal_score,model.graph_learner,model.attn_mixhop1,
                               model.attn_mixhop2,model.gcn_norm,model.type_pool],
             'gate':[model.gate_fc],'fusion projections':[model.lstm_proj,model.gcn_proj],
             'base head':[model.head]}
    groups={name:[p for module in parts for p in module.parameters()] for name,parts in modules.items()}
    groups['transition logits']=[b.regime_filter.transition_logits]
    groups['Base RPE']=[b.long_memory.base_rpe]
    groups.update({f'G{i}':list(g.parameters()) for i,g in enumerate(b.latent_transition.generators)})
    if is_adapter(model):
        groups['commodity_residual_alpha']=[model.commodity_residual_alpha]
        groups['commodity_residual_head.weight']=[model.commodity_residual_head.weight]
    return groups


def gradient_probe(model,batch):
    groups=parameter_groups(model);parameters=list(model.parameters());slots={id(p):i for i,p in enumerate(parameters)}
    x,y=[v.to(next(model.parameters()).device) for v in batch[:2]]
    rows={}
    with diagnostic_context(model):
        prediction=model(x)
        losses={str(h):make_loss()(prediction[:,i],y[:,i]) for i,h in enumerate(HORIZONS)}
        losses['prediction_sum']=_prediction_loss(model,prediction,y,make_loss())
        losses['total']=losses['prediction_sum']+model.switching_latent_transformer.switch_loss()
        for index,(h,loss) in enumerate(losses.items()):
            grads=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=index<len(losses)-1)
            norms={name:float(torch.sqrt(sum((torch.zeros_like(p) if grads[slots[id(p)]] is None else grads[slots[id(p)]]).detach().double().square().sum() for p in ps)))
                   for name,ps in groups.items()}
            alpha=None
            if is_adapter(model):alpha=float(grads[slots[id(model.commodity_residual_alpha)]].detach())
            rows[h]={'loss':float(loss.detach()),'norms':norms,'signed_alpha_gradient':alpha}
    return {'objectives':rows,'method':'eval raw Huber delta=.02; per horizon, prediction sum and total; autograd.grad; no optimizer step',
            'initial_zero_note':'At alpha=0, residual-head loss gradients are exactly zero; alpha gradients can be nonzero. Adam weight decay still applies normally.',
            'acceptance_note':'Gradient conflict is descriptive and is never an acceptance/rejection criterion.'}


def initialization_check(model,batch,seed=42):
    assert_backbone(model)
    x,y=[v.to(next(model.parameters()).device) for v in batch[:2]]
    with diagnostic_context(model),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(x.device).eval()
        old,new=dict(baseline.named_parameters()),dict(model.named_parameters())
        assert set(new)-set(old)=={'commodity_residual_alpha','commodity_residual_head.weight'}
        assert all(new[k].shape==p.shape for k,p in old.items())
        diffs={k:float((new[k]-p).abs().max()) for k,p in old.items()}
        before,after=reference(baseline,x),reference(model,x)
        forward={k:difference(after[k],before[k]) for k in TRACE_KEYS}
        count=[sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        losses={}
        for name,active,trace in [('D0B',baseline,before),(NEW,model,after)]:
            b=active.switching_latent_transformer
            pred=_prediction_loss(active,trace['prediction'],y,make_loss()).item()
            epoch=b.regime_filter.current_epoch
            b.set_epoch(20);switch=b.switch_loss().item();b.set_epoch(epoch)
            losses[name]={'prediction':pred,'raw_horizons':{str(h):make_loss()(trace['prediction'][:,i],y[:,i]).item() for i,h in enumerate(HORIZONS)},
                          'switch_epoch20':switch,'total_epoch20':pred+switch}
        result={'seed':seed,'fixed_TRAIN_shape':list(x.shape),'prediction_shape':list(after['prediction'].shape),
                'h_nodes_shape':list(after['h_nodes'].shape),'h_comm_shape':list(after['h_comm'].shape),
                'D0B_params':count[0],'new_params':count[1],'difference':count[1]-count[0],'expected_difference':257,
                'shared_parameter_count':len(old),'shared_parameter_numel':count[0],
                'max_abs_diff':max(diffs.values()),'mismatch_count':sum(v!=0 for v in diffs.values()),
                'forward_differences':forward,'losses':losses,'initial_alpha':float(model.commodity_residual_alpha),
                'initial_residual':residual_statistics(after)}
    result['gradients']=gradient_probe(model,batch)
    head=result['gradients']['objectives']['prediction_sum']['norms']['commodity_residual_head.weight']
    result['PASS']=(result['difference']==257 and result['mismatch_count']==0 and result['initial_alpha']==0.
                    and max(v['max'] for v in forward.values())<=2e-6 and head==0.
                    and result['initial_residual']['effective_mean_abs']==0.
                    and abs(losses[NEW]['prediction']-losses['D0B']['prediction'])<=2e-8)
    if not result['PASS']:raise AssertionError(result)
    print(f'[{DISPLAY} shared init] '+json.dumps(result),flush=True)
    return result


def fixed_sanity(model,x,raise_on_failure=True):
    """Temporal full-sequence causality and spatial window-prefix evaluation.

    Commodity relabeling also permutes the original output rows, so the complete
    forecast and the residual are equivariant. Residual-only shuffle is a
    different, deliberately misaligned intervention.
    """
    b=model.switching_latent_transformer;cut=x.shape[1]//2
    def trajectory(values):
        b(values)
        trace={k:getattr(b,v).clone() for k,v in {'E':'last_market_tokens','H':'last_long_memory','p':'last_regime_probabilities','Z':'last_latent_states'}.items()}
        readouts={k:[] for k in ('h_long','h_micro','h_temporal')}
        for t in range(values.shape[1]):
            output=b.readout(trace['H'][:,t],trace['Z'][:,t])
            readouts['h_long'].append(b.last_h_long.clone());readouts['h_micro'].append(b.last_h_micro.clone())
            readouts['h_temporal'].append(output)
        trace.update({k:torch.stack(v,1) for k,v in readouts.items()})
        return trace
    with diagnostic_context(model),torch.no_grad():
        before=trajectory(x);changed=x.clone();changed[:,cut:]=-2*changed[:,cut:]+7
        after=trajectory(changed)
        temporal={k:difference(before[k][:,:cut],after[k][:,:cut])['max'] for k in before}
        prefix=reference(model,x[:,:cut]);prefix_changed=reference(model,changed[:,:cut])
        spatial_prefix={k:difference(prefix[k],prefix_changed[k])['max'] for k in ('h_nodes','h_comm','h_spatial','gate','fused','base_pred','prediction')}
        prefix_consistency={k:difference(prefix[k],before[k][:,:cut])['max'] for k in ('E','H','p','Z')}
        native=reference(model,x)
        order=torch.arange(len(x)-1,-1,-1,device=x.device)
        permuted=reference(model,x[order]);single=reference(model,x[:1])
        keys=('h_nodes','h_comm','base_pred','prediction','raw')
        batch={k:difference(permuted[k],native[k][order])['max'] for k in keys}
        one={k:difference(single[k],native[k][:1])['max'] for k in keys}
        markets={};start=0
        for market,size in zip(('stock','bond','commodity'),(model.n_stock,model.n_bond,model.n_commodities)):
            node_order=torch.arange(model.num_nodes,device=x.device)
            node_order[start:start+size]=node_order[start:start+size].flip(0)
            output_order=(torch.arange(model.n_commodities-1,-1,-1,device=x.device) if market=='commodity'
                          else torch.arange(model.n_commodities,device=x.device))
            relabeled=HeteroMixHopCMGM(variant=model.variant,**model_arguments(model)).to(device=x.device,dtype=x.dtype).eval()
            state={k:v.clone() for k,v in model.state_dict().items()}
            for key in ('graph_learner.E1','graph_learner.E2'):state[key]=state[key][node_order]
            for key in ('head.3.weight','head.3.bias'):
                v=state[key];state[key]=v.reshape(4,model.n_commodities,*v.shape[1:])[:,output_order].reshape_as(v)
            relabeled.load_state_dict(state,strict=True)
            rel=reference(relabeled,x[:,:,node_order])
            markets[market]={
                'nodes_equivariance_max':difference(rel['h_nodes'],native['h_nodes'][:,node_order])['max'],
                'temporal_invariance_max':difference(rel['h_temporal'],native['h_temporal'])['max'],
                'base_equivariance_max':difference(rel['base_pred'],native['base_pred'][:,:,output_order])['max'],
                'residual_equivariance_max':difference(rel['raw'],native['raw'][:,:,output_order])['max'],
                'full_equivariance_max':difference(rel['prediction'],native['prediction'][:,:,output_order])['max']}
            start+=size
        base=native['base_pred'].clone();perm=commodity_permutation(model.n_commodities,device=x.device)
        cached_base=model.last_base_pred.clone() if is_adapter(model) else None
        commodity_control(model,native,'shuffled',perm)
        base_diff=difference(base,native['base_pred'])['max']
        if is_adapter(model):base_diff=max(base_diff,difference(cached_base,model.last_base_pred)['max'])
        values=[*temporal.values(),*spatial_prefix.values(),*prefix_consistency.values(),*batch.values(),*one.values(),base_diff,
                *[v for row in markets.values() for v in row.values()]]
        result={'PASS':max(values)<=3e-6,'prefix_cutoff':cut,'temporal_causality':temporal,
                'spatial_window_prefix':spatial_prefix,'temporal_prefix_vs_full':prefix_consistency,
                'batch_permutation':batch,'single_sample':one,'within_market':markets,
                'residual_shuffle_base_max_diff':base_diff,
                'causality_scope':'Spatial nodes, pooling and base prediction are window-end states. Spatial prefix test recomputes truncated windows; full-window spatial states may use all observed timesteps, never future targets.'}
        if not result['PASS'] and raise_on_failure:raise AssertionError(result)
        return result


def precision_sanity(model,x):
    with diagnostic_context(model):
        low=fixed_sanity(model,x,raise_on_failure=False)
        if low['PASS']:result=low
        else:
            high_model=HeteroMixHopCMGM(variant=model.variant,**model_arguments(model)).to(device=x.device,dtype=torch.float64).eval()
            high_model.load_state_dict(model.state_dict(),strict=True)
            high=fixed_sanity(high_model,x.double())
            # Audit all numeric invariance errors, including node-level batch errors.
            def errors(d):
                return [v for key,v in d.items() if isinstance(v,float)] + [v for value in d.values() if isinstance(value,dict) for v in errors(value)]
            if max(errors(high))>1e-10:raise AssertionError(high)
            result={'PASS':True,'float32':low,'float64_copy':high,'float64_max_error':max(errors(high)),
                    'audit_tolerance':1e-10,'note':'Original FP32 discrepancies retained; independent FP64 numerical audit, no model/precision/RNG changes.'}
        print(f'[{DISPLAY} causality/batch/relabeling] '+json.dumps(result),flush=True)
        return result


def epoch_probe(model,batch,stage,initial_weight):
    with diagnostic_context(model),torch.no_grad():
        trace=reference(model,batch[0].to(next(model.parameters()).device))
        weight=model.commodity_residual_head.weight
        result={'stage':stage,'alpha':float(model.commodity_residual_alpha),
                'residual_head_weight_norm':float(weight.norm()),
                'residual_head_distance_from_init':float((weight-weight.new_tensor(initial_weight)).norm()),
                **residual_statistics(trace)}
    result['gradients']=gradient_probe(model,batch)
    print(f'[{DISPLAY} {stage}] '+json.dumps(result),flush=True)
    return result


def node_statistics(nodes):
    nodes=np.asarray(nodes,dtype=np.float64);norm=np.linalg.norm(nodes,axis=-1)
    i,j=np.triu_indices(nodes.shape[1],1)
    unit=nodes/np.maximum(norm[...,None],1e-12)
    pair=np.einsum('bid,bjd->bij',unit,unit)[:,i,j]
    return {'mean_norm':float(norm.mean()),'std_norm':float(norm.std()),
            'within_sample_pairwise_cosine_mean':float(pair.mean()),'within_sample_pairwise_cosine_std':float(pair.std()),
            'cross_commodity_feature_variance':float(nodes.var(axis=1).mean()),
            'cross_sample_feature_variance':float(nodes.var(axis=0).mean()),
            'definition':'population variance; pairwise cosine excludes diagonal; per-sample commodity pairs'}


def variation(values):
    a=np.asarray(values,dtype=np.float64)
    return {'mean':float(a.mean()),'std':float(a.std()),'CV':float(a.std()/(a.mean()+1e-12)),'min':float(a.min()),'max':float(a.max())}


@torch.no_grad()
def collect_split(model,loader,split,output,names,permutation):
    full=DataLoader(loader.dataset,batch_size=loader.batch_size,shuffle=False,drop_last=False)
    modes=('alpha=0','shuffled','mean-commodity','zero-micro','zero-long','uniform')
    values={k:[] for k in ('native','target','raw','effective','base_pred',*modes)}
    cache={k:[] for k in ('p','prior','candidates','h_comm','gate')}
    micro,rpe,counts,projection_weights=[],[],[],[]
    base_control_errors=[]
    for batch in full:
        trace=reference(model,batch[0].to(next(model.parameters()).device))
        counts.append(len(batch[0]));values['native'].append(array(trace['prediction']));values['target'].append(array(batch[1]))
        for k in ('raw','effective','base_pred'):values[k].append(array(trace[k]))
        for k in cache:cache[k].append(array(trace[k]))
        micro.append(micro_diagnostics(model,trace));extra=extra_fixed_diagnostics(model,trace)
        rpe.append(extra['base_rpe']);projection_weights.append(extra['readout_weights'])
        base=trace['base_pred'].clone()
        for mode in modes:
            pred=(commodity_control(model,trace,mode,permutation) if mode in ('alpha=0','shuffled','mean-commodity')
                  else temporal_control(model,trace,mode))
            values[mode].append(array(pred))
        base_control_errors.append(difference(trace['base_pred'],base)['max'])
        assert base_control_errors[-1]==0
        print(f'[{DISPLAY} {model.variant} {split}] {sum(counts)}/{len(full.dataset)}',flush=True)
    arrays={k:np.concatenate(v) for k,v in values.items()};c={k:np.concatenate(v) for k,v in cache.items()}
    np.savez_compressed(output/f'{split.lower()}_predictions.npz',**arrays)
    impact={mode:{str(h):{'mean':float(np.abs(arrays['native'][:,i].astype(np.float64)-arrays[mode][:,i]).mean()),
                              'max':float(np.abs(arrays['native'][:,i].astype(np.float64)-arrays[mode][:,i]).max())} for i,h in enumerate(HORIZONS)} for mode in modes}
    metrics={k:horizon_metrics(arrays[k],arrays['target']) for k in ('native',*modes)}
    idx=HORIZONS.index(5)
    commodities={mode:[{'name':str(name),**population_metrics(arrays[mode][:,idx,j],arrays['target'][:,idx,j])} for j,name in enumerate(names)]
                 for mode in ('native','alpha=0')}
    corrections=[{'name':str(name),'mean_abs_effective':float(np.abs(arrays['effective'][:,idx,j]).mean()),
                  'std_effective':float(arrays['effective'][:,idx,j].std())} for j,name in enumerate(names)]
    def weighted(rows):
        return {k:float(np.average([r[k] for r in rows],weights=counts)) for k,v in rows[0].items() if isinstance(v,(int,float))}
    # residual_statistics also accepts tensors; use torch views of concatenated population arrays.
    residual=residual_statistics({k:torch.from_numpy(arrays[k]) for k in ('raw','effective','base_pred')})
    return {'samples':sum(counts),'batches':len(counts),'native_metrics':metrics.pop('native'),'control_metrics':metrics,
            'normal_regime':probability_stats(c['p'],prior=c['prior']),'candidate_specialization':specialization(c['candidates'],c['p']),
            'micro':weighted(micro),'base_rpe':weighted(rpe),'projection_weights':weighted(projection_weights),
            'gate':{'mean':float(c['gate'].mean()),'std':float(c['gate'].std()),'min':float(c['gate'].min()),'max':float(c['gate'].max())},
            'residual':residual,'impacts':impact,'commodity_nodes':node_statistics(c['h_comm']),
            'micro_long_ratio':{str(h):impact['zero-micro'][str(h)]['mean']/(impact['zero-long'][str(h)]['mean']+1e-8) for h in HORIZONS},
            'RoutingFraction':{str(h):impact['uniform'][str(h)]['mean']/(impact['zero-micro'][str(h)]['mean']+1e-8) for h in HORIZONS},
            'commodity_5d':commodities,'commodity_corrections':corrections,
            'strongest_corrections':sorted(corrections,key=lambda v:v['mean_abs_effective'],reverse=True)[:5],
            'weakest_corrections':sorted(corrections,key=lambda v:v['mean_abs_effective'])[:5],
            'commodity_MSE':{mode:variation([v['MSE'] for v in rows]) for mode,rows in commodities.items()},
            'prediction_losses':prediction_losses_from_arrays(arrays['native'],arrays['target'],full.batch_size),
            'base_max_diff_during_residual_controls':max(base_control_errors)}


def compare_commodities(before,after):
    rows=[]
    for b,a,correction in zip(before['commodity_5d']['native'],after['commodity_5d']['native'],after['commodity_corrections']):
        assert b['name']==a['name']==correction['name']
        rows.append({'name':b['name'],'D0B':b,NEW:a,'delta_MAE':a['MAE']-b['MAE'],'delta_MSE':a['MSE']-b['MSE'],
                     'effective_mean_abs':correction['mean_abs_effective']})
    x=np.array([v['D0B']['MSE'] for v in rows]);improvement=-np.array([v['delta_MSE'] for v in rows])
    nonconstant=x.std()>1e-15 and improvement.std()>1e-15
    order=sorted(rows,key=lambda v:v['delta_MSE'])
    # Error redistribution: compare fixed bottom/top baseline-error halves; descriptive only.
    ranking=np.argsort(x);half=len(x)//2
    groups={name:{'baseline_MSE':float(x[indices].mean()),'mean_MSE_improvement':float(improvement[indices].mean()),
                  'relative_improvement':float(improvement[indices].sum()/(x[indices].sum()+1e-12))}
            for name,indices in [('low_error',ranking[:half]),('high_error',ranking[half:])]}
    return {'rows_sorted_by_delta_MSE':order,'most_improved':order[:5],'most_worsened':order[-5:][::-1],
            'Pearson_baseline_MSE_vs_improvement':float(np.corrcoef(x,improvement)[0,1]) if nonconstant else None,
            'Spearman_baseline_MSE_vs_improvement':float(spearmanr(x,improvement).statistic) if nonconstant else None,
            'baseline_error_halves':groups,'interpretation':'Descriptive cross-commodity association, not causal evidence.'}


def performance_comparison(models):
    result={}
    for s in ('TRAIN','VAL','TEST'):
        result[s]={}
        for h in map(str,HORIZONS):
            b=models['D0B']['splits'][s]['native_metrics'][h];a=models[NEW]['splits'][s]['native_metrics'][h]
            result[s][h]={'D0B':b,NEW:a,**{f'delta_{k}':a[k]-b[k] for k in ('MAE','MSE')},
                          **{f'relative_{k}':(a[k]-b[k])/b[k] if b[k] else None for k in ('MAE','MSE')}}
    return result


def case_assessment(comparison,activity,commodity_comparison,healthy=True):
    mae=[comparison[s]['5']['relative_MAE'] for s in ('VAL','TEST')]
    mse=[comparison[s]['5']['relative_MSE'] for s in ('VAL','TEST')]
    tol=.001  # Fixed descriptive 0.1% convention, never a tuning loop.
    active=activity['functional_evidence'];rejected=activity['near_zero_and_tiny']
    groups=commodity_comparison['TEST']['baseline_error_halves']
    redistribution=groups['high_error']['relative_improvement']>.01 and groups['low_error']['relative_improvement']<0
    if any(v is None or not np.isfinite(v) for v in mae+mse) or not healthy:
        case,reason='Case E','Invalid metric comparison or unhealthy mechanism; no robust acceptance evidence.'
    elif all(v<0 for v in mae) and any(v>tol for v in mse):
        case,reason='Case F','MAE gain with a material MSE tail-error tradeoff; retain D0B.'
    elif all(abs(v)<tol for v in mae) and redistribution:
        case,reason='Case G','Aggregate tied, with localized benefit for difficult commodities and error redistribution; retain D0B.'
    elif all(abs(v)<tol for v in mae+mse) and rejected:
        case,reason='Case D','Optimizer essentially rejects the extra correction; no material commodity residual bottleneck.'
    elif all(v<0 for v in mae) and active:
        case,reason='Case A','Both 5d MAEs improve, without material MSE harm, with active and commodity-aligned residual effects.'
    elif all(v<0 for v in mae) and rejected:
        case,reason='Case B','Performance improvement without functional residual evidence.'
    elif all(v>0 for v in mae) and active:
        case,reason='Case C','Commodity-specific correction is learnable but harms generalization; global pooling provides useful regularization.'
    else:
        case,reason='Case E','No robust joint performance/mechanism evidence across VAL and TEST; retain D0B.'
    return {'case':case,'reason':reason,'candidate_for_human_review':case=='Case A',
            'MAE_relative_VAL_TEST':mae,'MSE_relative_VAL_TEST':mse,'negligible_relative_convention':tol,
            'redistribution':redistribution,'healthy':healthy,
            'note':'Single-seed descriptive case; no significance claim or automatic model replacement. Gradient conflict is not an acceptance criterion.'}


def validate_protocol(args):
    expected={'seed':42,'batch_size':64,'seq_len':20,'epochs':200,'patience':10}
    if any(getattr(args,k)!=v for k,v in expected.items()):raise ValueError(f'Controlled run requires {expected}')
    if (config.MULTI_HORIZONS!=list(HORIZONS) or config.TARGET_HORIZON!=5 or config.FEATURE_DIM!=21
        or config.LEARNING_RATE!=1e-4 or config.WEIGHT_DECAY!=1e-5 or config.LOSS_TYPE!='huber' or config.HUBER_DELTA!=.02):
        raise ValueError('Original D0B objective and protocol must remain intact')


def make_model(data,variant=VARIANT):
    m=data['market_indices']
    return HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=m['stock'][1]-m['stock'][0],
                           n_bond=m['bond'][1]-m['bond'][0],feat_dim=config.FEATURE_DIM,variant=variant)


def prepare(args,data,device):
    validate_protocol(args);set_seed(args.seed)
    model=make_model(data)
    with torch.random.fork_rng(devices=[]):fixed=next(iter(data['loaders']['train']))
    initial=initialization_check(model,fixed,args.seed)
    if initial['D0B_params']!=520549 or initial['new_params']!=520806:raise ValueError('Parameter baseline changed')
    metadata={'variant':VARIANT,'display_name':DISPLAY,'seed':args.seed,
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'parameter_count':initial['new_params'],'initialization':initial,
              'initial_residual_weight':model.commodity_residual_head.weight.detach().cpu().tolist(),
              'initial_transition_logits':model.switching_latent_transformer.regime_filter.transition_logits.detach().cpu().tolist(),
              'commodity_residual':True,'commodity_residual_head':'Linear(64,4,bias=False)',
              'commodity_residual_shared_across_commodities':True,'commodity_residual_alpha_init':0.,
              'loss':'sum four-horizon Huber delta=.02','switch_beta_max':5e-4,'warmup_epochs':20,
              'metric_standard':STANDARD,'forecast_horizons':list(HORIZONS),
              'protocol':{'seed':42,'batch_size':64,'seq_len':20,'epochs':200,'patience':10,
                          'optimizer':'Adam','lr':1e-4,'weight_decay':1e-5,'scheduler':'ReduceLROnPlateau',
                          'validation_selection':'original prediction-only horizon sum; mean of batch means'},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'cmgm/models/hetero_mixhop_model.py',
                    ROOT/'cmgm/models/switching_latent_transformer.py',ROOT/'cmgm/training/train.py',ROOT/'cmgm/training/metric_standard.py']}}
    return model.to(device),metadata


def checkpoint_report(model,payload,data,d0b_path,output,checkpoint_path,seed=42):
    assert model.variant==VARIANT;assert_backbone(model)
    output.mkdir(parents=True,exist_ok=False)
    paths={'D0B':Path(d0b_path),NEW:Path(checkpoint_path)};hashes={k:sha256(p) for k,p in paths.items()}
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    models={};names=data['feature_names'][slice(*data['market_indices']['commodity'])]
    perm=commodity_permutation(model.n_commodities,seed,next(model.parameters()).device)
    with diagnostic_context(model):
        fixed=next(iter(data['loaders']['test']))
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(next(model.parameters()).device)
        base_payload=_load_checkpoint(baseline,paths['D0B'],next(model.parameters()).device)
        for label,active,record in [('D0B',baseline,base_payload),(NEW,model,payload)]:
            assert_backbone(active)
            active.switching_latent_transformer.set_epoch(record.get('best_epoch',1))
            directory=output/label;directory.mkdir()
            with diagnostic_context(active):
                sanity=precision_sanity(active,fixed[0].to(next(active.parameters()).device))
                with torch.no_grad():
                    trace=reference(active,fixed[0].to(next(active.parameters()).device))
                    np.savez_compressed(directory/'fixed_native.npz',**{k:array(v) for k,v in trace.items()},X=array(fixed[0]),target=array(fixed[1]))
                splits={s.upper():collect_split(active,loader,s.upper(),directory,names,perm) for s,loader in data['loaders'].items()}
                metadata=record.get('metadata',{});weight=getattr(active,'commodity_residual_head',None)
                init=metadata.get('initial_residual_weight')
                models[label]={'params':sum(p.numel() for p in active.parameters()),'best_epoch':record.get('best_epoch'),
                               'train_time':metadata.get('training_elapsed_seconds',record.get('history',{}).get('training_elapsed_seconds')),
                               'history':record.get('history',{}),'metadata':metadata,'sanity':sanity,'splits':splits,
                               'transition':transition_diagnostics(active,metadata.get('initial_transition_logits')),
                               'gradients':gradient_probe(active,fixed),
                               'alpha':float(active.commodity_residual_alpha.detach()) if is_adapter(active) else 0.,
                               'residual_head_weight_norm':float(weight.weight.detach().norm()) if weight is not None else None,
                               'residual_head_distance_from_init':float((weight.weight.detach()-weight.weight.new_tensor(init)).norm()) if init is not None else None}
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(sha256(paths[k])==digest for k,digest in hashes.items())
    comparison=performance_comparison(models)
    commodities={s:compare_commodities(models['D0B']['splits'][s],models[NEW]['splits'][s]) for s in ('VAL','TEST')}
    ratios={s:{mode:models[NEW]['splits'][s]['impacts'][mode]['5']['mean']/(models[NEW]['splits'][s]['native_metrics']['5']['MAE']+1e-12)
               for mode in ('alpha=0','shuffled','mean-commodity')} for s in ('VAL','TEST')}
    alpha=abs(models[NEW]['alpha']);threshold=.001
    activity={'alpha_abs':alpha,'impact_over_native_MAE':ratios,'alpha_near_zero_convention':1e-4,
              'impact_ratio_convention':threshold,
              'functional_evidence':alpha>1e-4 and all(v['alpha=0']>threshold and v['shuffled']>threshold for v in ratios.values()),
              'near_zero_and_tiny':alpha<=1e-4 and all(max(v.values())<threshold for v in ratios.values())}
    warnings={s:max(v['normal_regime']['occupancy'])>=.99 and v['normal_regime']['entropy']<=.1 and v['normal_regime']['mean_max']>=.99
              for s,v in models[NEW]['splits'].items()}
    report={'display_name':DISPLAY,'variant':VARIANT,'metric_standard':STANDARD,'metadata':payload.get('metadata',{}),
            'checkpoint_paths':{k:str(p.resolve()) for k,p in paths.items()},'checkpoint_sha256':hashes,
            'fixed_TEST_shape':list(fixed[0].shape),'commodity_permutation':perm.cpu().tolist(),
            'models':models,'comparison':comparison,'commodity_comparison':commodities,'activity':activity,
            'regime_over_specialization_warning':warnings,
            'assessment':case_assessment(comparison,activity,commodities,not any(warnings.values())),
            'integrity':{'parameters_unchanged':True,'checkpoint_files_unchanged':True,
                         'max_RMSE_squared_minus_MSE':max(abs(v['RMSE']**2-v['MSE']) for m in models.values() for s in m['splits'].values()
                                                          for metrics in [s['native_metrics'],*s['control_metrics'].values()] for v in metrics.values())}}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    from cmgm.scripts.d0b_commodity_residual_report import write_report
    write_report(report,output/'REPORT.md')
    print(f'[{DISPLAY} REPORT] {output / "REPORT.md"}',flush=True)
    return report


def run_commodity_residual(args,device,data):
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant
    from cmgm.training.train import train
    validate_protocol(args)
    if not args.d0b_checkpoint.is_file():raise FileNotFoundError(args.d0b_checkpoint)
    path=_checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    if path.exists():raise FileExistsError(f'{path} exists; use checkpoint-only diagnostics; no overwrite or second training')
    model,metadata=prepare(args,data,device)
    with diagnostic_context(model):
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(device)
        _load_checkpoint(baseline,args.d0b_checkpoint,device)
        metadata['D0B_checkpoint_sha256']=sha256(args.d0b_checkpoint);del baseline
        fixed=next(iter(data['loaders']['train']));test=next(iter(data['loaders']['test']))
        metadata['fixed_TEST_sanity']=precision_sanity(model,test[0].to(device))
    output=args.commodity_residual_report_dir/time.strftime('%Y%m%d_%H%M%S');output.mkdir(parents=True,exist_ok=False)
    (output/'preflight.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    # Exactly one training call; no D0B retraining and no checkpoint initialization.
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata,
          epoch_diagnostic=lambda active,stage:epoch_probe(active,fixed,stage,metadata['initial_residual_weight']))
    payload=_load_checkpoint(model,path,device)
    report=checkpoint_report(model,payload,data,args.d0b_checkpoint,output/'checkpoint_diagnostics',path,args.seed)
    from cmgm.training.evaluate import compute_metrics,inverse_transform_predictions
    values=np.load(output/'checkpoint_diagnostics'/NEW/'test_predictions.npz')
    idx=HORIZONS.index(5);p,y=values['native'][:,idx],values['target'][:,idx]
    mn=compute_metrics(p,y)
    po,yo=inverse_transform_predictions(p,y,data['norm_stats'],data['raw_prices_test'],data['market_indices'],target_type=config.TARGET_TYPE)
    return {'variant':DISPLAY,'params':report['models'][NEW]['params'],'time':payload['history']['training_elapsed_seconds'],
            **{k:mn[k] for k in ('MAE','MSE','RMSE','Hit_Ratio')},'vs_zero_pct':(mn['MAE']/float(np.abs(y).mean())-1)*100,
            'mn':mn,'mo':compute_metrics(po,yo),'report_path':str(output/'checkpoint_diagnostics'/'REPORT.md')}


def preflight(args,data,device):
    model,metadata=prepare(args,data,device)
    with diagnostic_context(model):
        fixed=next(iter(data['loaders']['test']))
        metadata['fixed_TEST_shape']=list(fixed[0].shape)
        metadata['fixed_TEST_sanity']=precision_sanity(model,fixed[0].to(device))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'preflight.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    from cmgm.scripts.d0b_commodity_residual_report import write_preflight
    write_preflight(metadata,args.output/'REPORT.md')
    print(f'[{DISPLAY} preflight] PASS; no formal training; {args.output}',flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description='Commodity residual preflight / checkpoint-only diagnostics; training via main_ablation')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--no-cuda',action='store_true');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    args.seed=42;args.batch_size=64;args.seq_len=20;args.epochs=200;args.patience=10
    if args.output.exists():raise FileExistsError(args.output)
    device=torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    from cmgm.scripts.main_ablation import build_data
    data=build_data(args)
    if args.preflight:preflight(args,data,device)
    else:
        model=make_model(data).to(device);payload=_load_checkpoint(model,args.checkpoint,device)
        checkpoint_report(model,payload,data,args.d0b_checkpoint,args.output,args.checkpoint)


if __name__=='__main__':main()
