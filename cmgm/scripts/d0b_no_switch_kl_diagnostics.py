"""D0B NoSwitchKL integration: original prediction objective, zero KL weight."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import (
    NO_SWITCH_KL_VARIANT as VARIANT, _effective_switch_loss, _prediction_loss,
    _raw_regime_statistics, make_loss,
)
from cmgm.scripts.d0b_5d_only_diagnostics import (
    BASE_VARIANT, TRACE_KEYS, assert_backbone, full_reference, collect_model,
)
from cmgm.scripts.d0b_grouped_diagnostics import extra_fixed_diagnostics
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0b_regime_routing_diagnostics import ROOT, array, difference, sha256
from cmgm.scripts.d0e_diagnostics import (
    diagnostic_context, model_arguments, gradient_groups, flattened_gradients, functional_values,
)

DISPLAY = 'D0B-NoSwitchKL'
ALL_TRACES = tuple(dict.fromkeys(TRACE_KEYS + ('evidence',)))


def initialization_check(model, batches, seed=42):
    assert model.variant == VARIANT and model.disable_switch_kl
    assert_backbone(model)
    device=next(model.parameters()).device
    with diagnostic_context(model), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(device).eval()
        left,right=dict(baseline.named_parameters()),dict(model.named_parameters())
        assert left.keys()==right.keys() and all(left[k].shape==right[k].shape for k in left)
        diffs={k:(left[k]-right[k]).abs().max().item() for k in left}
        counts=[sum(p.numel() for p in m.parameters()) for m in (baseline,model)]
        result={'seed':seed,'D0B_params':counts[0],'NoSwitchKL_params':counts[1],
                'difference':counts[1]-counts[0],'max_abs_diff':max(diffs.values()),
                'mismatch_count':sum(v!=0 for v in diffs.values()),'batches':{}}
        b_branch,n_branch=baseline.switching_latent_transformer,model.switching_latent_transformer
        original_epoch=n_branch.regime_filter.current_epoch
        try:
            for split,batch in batches.items():
                x,y=[v.to(device) for v in batch[:2]]
                before,after=full_reference(baseline,x),full_reference(model,x)
                forward={k:difference(after[k],before[k]) for k in ALL_TRACES}
                b_pred=_prediction_loss(baseline,before['prediction'],y,make_loss())
                n_pred=_prediction_loss(model,after['prediction'],y,make_loss())
                sanity={}
                for epoch in (1,20):
                    b_branch.set_epoch(epoch);n_branch.set_epoch(epoch)
                    weighted=_effective_switch_loss(baseline,b_branch)
                    actual=_effective_switch_loss(model,n_branch)
                    b_total,n_total=b_pred+weighted,n_pred+actual
                    residual=((b_total-n_total)-weighted).abs().item()
                    sanity[str(epoch)]={'reference_epoch':epoch,'prediction_loss':b_pred.item(),
                        'switch_raw_KL':b_branch.regime_filter._last_switch_loss.item(),
                        'beta_D0B':b_branch.regime_filter.current_beta,'beta_effective_NoSwitchKL':0.,
                        'weighted_switch_loss_D0B':weighted.item(),'weighted_switch_loss_NoSwitchKL':actual.item(),
                        'D0B_total':b_total.item(),'NoSwitch_total':n_total.item(),
                        'total_difference':(b_total-n_total).item(),'sanity_diff':residual}
                    assert actual.item()==0. and not actual.requires_grad
                    torch.testing.assert_close(b_total-n_total,weighted,rtol=1e-5,atol=1e-8)
                result['batches'][split]={'shape':list(x.shape),'forward_differences':forward,
                    'D0B_prediction_loss':b_pred.item(),'NoSwitch_prediction_loss':n_pred.item(),
                    'prediction_loss_abs_diff':(b_pred-n_pred).abs().item(),'total_loss_sanity':sanity}
        finally:
            n_branch.set_epoch(original_epoch)
        result['PASS']=(result['difference']==0 and result['mismatch_count']==0 and
                        all(v['prediction_loss_abs_diff']==0 and all(d['max']<=2e-6 for d in v['forward_differences'].values())
                            for v in result['batches'].values()))
        print(f'[{DISPLAY} shared init] {json.dumps(result)}',flush=True)
        if not result['PASS']:raise AssertionError('NoSwitchKL differs from D0B before loss ablation')
        return result


def fixed_regime_probe(model,batch,stage):
    with diagnostic_context(model),torch.no_grad():
        model(batch[0].to(next(model.parameters()).device))
        b=model.switching_latent_transformer
        result={'stage':stage,'batch_shape':list(batch[0].shape),**_raw_regime_statistics(b),
                'beta_effective':0. if getattr(model,'disable_switch_kl',False) else b.regime_filter.current_beta,
                'weighted_switch_loss':_effective_switch_loss(model,b).item(),
                'reference_schedule_beta':b.regime_filter.current_beta,'mode':'fixed TRAIN eval probe'}
    print(f'[{DISPLAY} KL trajectory {stage}] {json.dumps(result)}',flush=True)
    return result


def switch_gradient_probe(model,batch):
    """Actual pred/total and counterfactual original-schedule KL; no optimizer."""
    x,y=[v.to(next(model.parameters()).device) for v in batch[:2]]
    b=model.switching_latent_transformer;groups=gradient_groups(model)
    with diagnostic_context(model),torch.enable_grad():
        prediction=model(x)
        pred=_prediction_loss(model,prediction,y,make_loss())
        raw=b.regime_filter._last_switch_loss
        counterfactual=b.switch_loss()  # original schedule at this checkpoint epoch
        effective=_effective_switch_loss(model,b)
        total=pred+effective
        gp=flattened_gradients(pred,groups,retain_graph=True)
        gs=flattened_gradients(counterfactual,groups,retain_graph=True)
        gr=flattened_gradients(raw,groups,retain_graph=True)
        gt=flattened_gradients(total,groups)
        rows={}
        for name,p in gp.items():
            s=gs[name];pn,sn=p.norm().item(),s.norm().item()
            rows[name]={'prediction_norm':pn,'total_norm':gt[name].norm().item(),
                        'counterfactual_switch_norm':sn,'raw_KL_norm':gr[name].norm().item(),
                        'cos_prediction_switch':float(torch.dot(p,s).item()/(pn*sn)) if pn>0 and sn>0 else None,
                        'switch_prediction_ratio':sn/pn if pn>0 else None,
                        'total_prediction_max_diff':(gt[name]-p).abs().max().item()}
        result={'epoch':b.regime_filter.current_epoch,'prediction_loss':pred.item(),
                'raw_KL':raw.item(),'actual_weighted_switch_loss':effective.item(),'actual_total_loss':total.item(),
                'beta_effective':0. if getattr(model,'disable_switch_kl',False) else b.regime_filter.current_beta,
                'counterfactual_beta':b.regime_filter.current_beta,'counterfactual_weighted_switch_loss':counterfactual.item(),
                'counterfactual_policy':'Original D0B beta schedule at this checkpoint best epoch (zero at epoch 1); raw-KL gradients also reported',
                'evidence_weight_norm':b.regime_filter.regime_evidence.weight.norm().item(),
                'evidence_bias_norm':b.regime_filter.regime_evidence.bias.norm().item(),
                'modules':rows,'mode':'eval fixed TEST; prediction=sum four raw Huber losses; no optimizer step'}
    print(f'[{DISPLAY} {model.variant} KL gradient contribution] {json.dumps(result)}',flush=True)
    return result


def collapse_flags(item):
    result={}
    for split,data in item['splits'].items():
        p=data['normal_regime']
        # Descriptive thresholds only; not used to change optimization.
        concentrated=[i for i,v in enumerate(p['occupancy']) if v>.95]
        hard=p['entropy']<.1 and p['mean_max']>.99 and p['min_probability']<1e-4
        candidate=data['candidate_specialization']['pairwise']
        identical=all(v['mean_abs']<1e-6 for v in candidate.values())
        result[split]={'argmax_concentration_warning':bool(concentrated),'states_over_95pct':concentrated,
                      'hard_posterior_warning':hard,'near_identical_candidate_warning':identical,
                      'entropy':p['entropy'],'mean_p':p['mean'],'mean_max_p':p['mean_max'],
                      'min_probability':p['min_probability'],
                      'definition':'occupancy>95% alone is not collapse; hard warning: H<0.1, mean max>0.99, min p<1e-4; all candidate pair mean abs<1e-6 flags near identity'}
    return result


def assess(models,comparison):
    effect,tie=.001,1e-6
    values=[comparison[s]['5']['relative_change'] for s in ('VAL','TEST')]
    base,new=models['D0B'],models['NoSwitchKL']
    routing={s:functional_values(new,s,5)[4]-functional_values(base,s,5)[4] for s in ('VAL','TEST')}
    posterior_close=all(abs(new['splits'][s]['normal_regime'][k]-base['splits'][s]['normal_regime'][k]) <= 1e-4
                        for s in ('VAL','TEST') for k in ('entropy','temporal_L1','posterior_prior_KL','mean_max','margin'))
    routing_close=all(abs(v)<=1e-4 for v in routing.values())
    healthy=not any(v['hard_posterior_warning'] or v['near_identical_candidate_warning'] for v in new['collapse_checks'].values())
    result={'case':None,'retain_D0B':True,'routing_fraction_5d_delta':routing,
            'mechanism_checks_healthy':healthy,'candidate_replacement_justified':False,
            'threshold_note':'0.1% relative MAE for materiality; 1e-6 for numerical tie; mechanism near-equality absolute tolerance 1e-4. Reporting conventions only.'}
    if any(v is None or not np.isfinite(v) for v in values):
        result['reason']='Invalid reference metrics; no case assigned.'
    elif all(abs(v)<=tie for v in values):
        if posterior_close and routing_close:
            result.update(case='Case D',reason='Performance, posterior dynamics and routing are essentially unchanged; current KL has little measured effect.')
        elif not posterior_close and not routing_close:
            result.update(case='Case E',reason='KL changes regime geometry and routing without a forecasting gain.')
        else:
            result['reason']='Tied performance with only one mechanism changing; supplied Case D/E conditions are not fully met.'
    elif all(v < -tie for v in values):
        if any(v> -effect for v in values):
            result.update(case='Case B',reason='Both splits improve but at least one by <0.1%; not a demonstrated major bottleneck.')
        elif all(v>0 for v in routing.values()) and healthy and all(m['sanity']['PASS'] for m in models.values()):
            result.update(case='Case A',retain_D0B=False,candidate_replacement_justified=True,
                          reason='Removing KL improves both 5d splits and functional routing without severe posterior/candidate warning; supports over-regularization.')
        else:
            result['reason']='Both splits improve, but routing/health conditions for Case A fail; do not infer functional regime improvement.'
    elif all(v>tie for v in values):
        result.update(case='Case C',reason='Switch KL benefits generalization; stronger differentiation is not a forecasting improvement.')
    elif values[1]<-tie and values[0]>tie:
        result.update(case='Case F',reason='TEST-only improvement: no robust evidence; retain D0B.')
    else:
        result['reason']='Split directions do not provide robust improvement; supplied A–F conditions do not cover this outcome. Retain D0B.'
    return result


def load_no_switch(model,path,device):
    payload=_load_checkpoint(model,Path(path),device)
    m=payload.get('metadata',{})
    if m.get('variant')!=VARIANT or m.get('switch_kl_enabled') is not False or m.get('beta_effective')!=0:
        raise ValueError('NoSwitchKL checkpoint must identify variant, switch_kl_enabled=false and beta_effective=0')
    return payload


def checkpoint_report(model,payload,data,d0b_path,output,seed=42,initialization=None,checkpoint_path=None):
    assert model.variant==VARIANT and model.disable_switch_kl
    output.mkdir(parents=True,exist_ok=False)
    paths={'D0B':Path(d0b_path)}
    if checkpoint_path is not None:paths['NoSwitchKL']=Path(checkpoint_path)
    hashes={k:sha256(p) for k,p in paths.items()}
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    with diagnostic_context(model):
        fixed=next(iter(data['loaders']['test']));device=next(model.parameters()).device
        baseline=HeteroMixHopCMGM(variant=BASE_VARIANT,**model_arguments(model)).to(device)
        saved=_load_checkpoint(baseline,Path(d0b_path),device)
        if saved.get('metadata',{}).get('variant',BASE_VARIANT)!=BASE_VARIANT:
            raise ValueError('Reference must be native D0B')
        models={}
        for label,active,stored in (('D0B',baseline,saved),('NoSwitchKL',model,payload)):
            active.switching_latent_transformer.set_epoch(stored.get('best_epoch',1))
            item=collect_model(active,stored,data,output/label,fixed,seed,label=DISPLAY,extra_fixed=extra_fixed_diagnostics)
            if any(v!=0. for v in item['sanity']['causality'].values()):
                raise AssertionError('NoSwitchKL study requires exact zero future-prefix differences')
            item['switch_gradients']=switch_gradient_probe(active,fixed)
            item['collapse_checks']=collapse_flags(item)
            models[label]=item
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(sha256(paths[k])==h for k,h in hashes.items())
    comparison={}
    for split in ('TRAIN','VAL','TEST'):
        comparison[split]={}
        for h in map(str,config.MULTI_HORIZONS):
            b,n=[models[k]['splits'][split]['native_metrics'][h] for k in ('D0B','NoSwitchKL')]
            delta=n['MAE']-b['MAE']
            comparison[split][h]={'D0B':b,'NoSwitchKL':n,'delta_MAE':delta,'relative_change':delta/b['MAE'] if b['MAE']>0 else None}
    report={'variant':VARIANT,'objective':'sum_1d_5d_10d_20d','switch_kl_enabled':False,'beta_effective':0.,
            'seed':seed,'fixed_TEST_batch_shape':list(fixed[0].shape),'initialization':initialization,
            'checkpoint_paths':{k:str(p.resolve()) for k,p in paths.items()},'checkpoint_sha256':hashes,
            'metadata':payload.get('metadata',{}),'models':models,'comparison':comparison,
            'assessment':assess(models,comparison),'integrity':{'parameters_unchanged':True,'checkpoint_files_unchanged':True}}
    (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    from cmgm.scripts.d0b_no_switch_kl_report import write_report
    write_report(report,output/'REPORT.md')
    print(f'[{DISPLAY} REPORT] {output / "REPORT.md"}',flush=True)
    return report


def run_no_switch_kl(args,device,data):
    from cmgm.training.train import train
    from cmgm.scripts.main_ablation import _checkpoint_path_for_variant,evaluate_primary_horizon,print_diagnostics
    from cmgm.training.evaluate import compute_metrics
    if not args.d0b_checkpoint.is_file():raise FileNotFoundError(f'D0B reference missing: {args.d0b_checkpoint}')
    if args.epochs<1:raise ValueError('Training requires at least one epoch')
    reference_meta=torch.load(args.d0b_checkpoint,map_location='cpu',weights_only=True).get('metadata',{})
    if reference_meta.get('variant',BASE_VARIANT)!=BASE_VARIANT:
        raise ValueError('Reference checkpoint must identify native D0B')
    set_seed(args.seed);market=data['market_indices']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=market['stock'][1]-market['stock'][0],
                          n_bond=market['bond'][1]-market['bond'][0],feat_dim=config.FEATURE_DIM,variant=VARIANT)
    with torch.random.fork_rng(devices=[]):
        fixed={s.upper():next(iter(data['loaders'][s])) for s in ('train','test')}
    initial=initialization_check(model,fixed,args.seed)
    model=model.to(device);model._experiment_seed=args.seed
    branch=model.switching_latent_transformer
    branch._initial_transition_logits=branch.regime_filter.transition_logits.detach().cpu().clone()
    sources=[Path(__file__),ROOT/'cmgm/training/train.py',ROOT/'cmgm/models/hetero_mixhop_model.py',
             ROOT/'cmgm/models/switching_latent_transformer.py',ROOT/'cmgm/scripts/main_ablation.py']
    metadata={'variant':VARIANT,'display_name':DISPLAY,'switch_kl_enabled':False,'beta_effective':0.,
              'seed':args.seed,'seq_len':args.seq_len,'initialization_check':initial,
              'initial_transition_logits':array(branch._initial_transition_logits).tolist(),
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
              'protocol':{'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
                          'lr':config.LEARNING_RATE,'weight_decay':config.WEIGHT_DECAY,'Huber_delta':config.HUBER_DELTA,
                          'horizons':config.MULTI_HORIZONS,'optimizer':'Adam','scheduler':'ReduceLROnPlateau',
                          'prediction_objective':'sum_1d_5d_10d_20d','reference_beta_max':5e-4,'reference_warmup_epochs':20}}
    path=_checkpoint_path_for_variant(VARIANT,args.checkpoint_dir)
    if path.resolve()==args.d0b_checkpoint.resolve():raise ValueError('New checkpoint must not overwrite D0B reference')
    start=time.time()
    train(model,data['loaders']['train'],data['loaders']['val'],torch.empty(2,0,dtype=torch.long),torch.zeros(0),device,
          num_epochs=args.epochs,patience=args.patience,checkpoint_path=str(path),checkpoint_metadata=metadata,
          epoch_diagnostic=lambda active,stage:fixed_regime_probe(active,fixed['TRAIN'],stage))
    payload=load_no_switch(model,path,device)
    normalized,original,target=evaluate_primary_horizon(model,data['loaders']['test'],data,device)
    legacy=print_diagnostics(model,VARIANT,data['loaders'],device)
    output=args.no_switch_report_dir/time.strftime('%Y%m%d_%H%M%S')
    report=checkpoint_report(model,payload,data,args.d0b_checkpoint,output,args.seed,initial,path)
    zero=compute_metrics(np.zeros_like(target),target)
    return {'variant':DISPLAY,'params':initial['NoSwitchKL_params'],'time':time.time()-start,
            'MAE':normalized['MAE'],'RMSE':normalized['RMSE'],'Hit_Ratio':normalized['Hit_Ratio'],
            'vs_zero_pct':(normalized['MAE']/zero['MAE']-1)*100,'mn':normalized,'mo':original,
            'diagnostics':legacy,'report_path':str(output/'REPORT.md'),'case':report['assessment']['case']}


def main():
    parser=argparse.ArgumentParser(description='NoSwitchKL checkpoint-only comparison; no training')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--d0b-checkpoint',type=Path,default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=config.RANDOM_SEED)
    parser.add_argument('--batch-size',type=int,default=config.BATCH_SIZE)
    parser.add_argument('--seq-len',type=int,default=config.SEQ_LEN)
    parser.add_argument('--no-cuda',action='store_true')
    args=parser.parse_args();set_seed(args.seed)
    device=torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    from cmgm.scripts.main_ablation import build_data
    data=build_data(SimpleNamespace(batch_size=args.batch_size,seq_len=args.seq_len,seed=args.seed));market=data['market_indices']
    model=HeteroMixHopCMGM(data['n_nodes'],data['n_commodities'],n_stock=market['stock'][1]-market['stock'][0],
                          n_bond=market['bond'][1]-market['bond'][0],variant=VARIANT).to(device)
    payload=load_no_switch(model,args.checkpoint,device)
    checkpoint_report(model,payload,data,args.d0b_checkpoint,args.output,args.seed,
                      payload.get('metadata',{}).get('initialization_check'),args.checkpoint)


if __name__=='__main__':main()
