"""Minimal complementary-fusion error reporting, no selection feedback."""
import numpy as np
import pandas as pd
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0b_risk_probe_analysis import table


def leave_one_commodity_out(base,new,names,rows):
    selected=max(rows,key=lambda v:v['absolute_error_improvement_sum']);i=selected['commodity_index']
    keep=np.arange(len(names))!=i
    b=population_metrics(base['test']['prediction'][:,keep],base['test']['target'][:,keep])
    n=population_metrics(new['test']['prediction'][:,keep],new['test']['target'][:,keep])
    return dict(removed_commodity=str(names[i]),removed_index=i,D0B_TEST_MAE=b['MAE'],New_TEST_MAE=n['MAE'],delta_MAE=n['MAE']-b['MAE'],
        note='One supplementary TEST arithmetic exclusion of the largest MAE-improving commodity; no retraining or tuning.')


def classify(metrics,r,leave_out):
    if not r['sanity']['PASS']:return dict(primary_case=None,interpretation='DO NOT INTERPRET PERFORMANCE: sanity failed')
    b,n=metrics['D0B'],metrics['ResidualComplementaryFusion'];a=r['ablations']
    delta={s:{k:n[s][k]/b[s][k]-1 for k in ('MAE','MSE')} for s in ('val','test')}
    zero={s:{k:a[s]['metrics']['ZeroResidual'][k]/b[s][k]-1 for k in ('MAE','MSE')} for s in ('val','test')}
    close_full_zero=all(abs(a[s]['metrics']['Full'][k]/a[s]['metrics']['ZeroResidual'][k]-1)<=.001 for s in ('val','test') for k in ('MAE','MSE'))
    zero_tied=all(abs(v)<=.001 for d in zero.values() for v in d.values())
    active=r['residual']['best']['final_weight_norm']>1e-6 and r['residual']['New_full_loader']['test']['residual_base_norm_ratio']>1e-3
    damage=any(zero[s]['MAE']>.001 for s in ('val','test'))
    gain=delta['val']['MAE']<0 and delta['test']['MAE']<0
    if damage:
        case='Case F';text='The extra fusion branch degrades the trained zero-residual base path on at least one evaluation split. This is evidence of adverse shared-model training effects; the decomposition does not uniquely localize them to a particular encoder.'
    elif gain and delta['test']['MSE']>.001:
        case='Case D';text='Complementary fusion redistributes errors but is not a clean replacement.'
    elif gain and delta['test']['MSE']<=.001 and zero_tied and active:
        case='Case B';text='Global complementary residual directly improves fusion; the trained zero-residual path is near original D0B.'
    elif gain and delta['test']['MSE']<=.001 and all(zero[s]['MAE']<0 for s in ('val','test')) and close_full_zero:
        case='Case C';text='The added global path mainly improves training of the shared model rather than direct residual output.'
    elif gain and delta['test']['MSE']<=.001 and active:
        case='Case A';text='This run supports useful global spatial-temporal complementary information beyond the competitive gate.'
    elif active and all(a[s]['metrics']['Full']['MAE']>a[s]['metrics']['ZeroResidual']['MAE'] for s in ('val','test')):
        case='Case E';text='The direct fusion residual harms forecasting; competitive gated fusion may provide useful regularization.'
    elif not active and r['residual']['best']['final_weight_norm']<=1e-6 and r['residual']['New_full_loader']['test']['residual_base_norm_ratio']<=1e-3 and close_full_zero and all(abs(v)<=.001 for d in delta.values() for v in d.values()):
        case='Case G';text='Optimizer effectively rejects complementary fusion; the current gated fusion appears sufficient.'
    else:
        case=None;text='Mixed evidence outside the supplied A–G definitions; no forced success classification.'
    return dict(primary_case=case,interpretation=text,mechanism_status='active' if active else 'weak',zero_residual_relative_deltas=zero,
        stop_incremental_patching=delta['val']['MAE']>0 and delta['test']['MAE']>0,
        single_commodity_caveat=delta['test']['MAE']<0 and leave_out['delta_MAE']>=0,
        classification_notes='Reporting conventions only: material/near-tie error 0.1%; active residual requires final weight norm >1e-6 and residual/base L2 ratio >0.1%. Case F has priority if zero-residual MAE worsens materially on either split; inspect both raw deltas. B/C refine A. No criterion controls training/selection or statistical-significance claims. Shared-path changes also include the trained head; they cannot prove an encoder alone was damaged.')


def improvement_shares(values):
    values=np.asarray(values,dtype=float);positive=np.maximum(values,0);negative=np.maximum(-values,0)
    def parts(v):
        total=v.sum();ordered=np.sort(v)[::-1]
        return dict(total=float(total),largest_share=float(ordered[0]/total) if total else None,
                    top3_share=float(ordered[:3].sum()/total) if total else None)
    net=float(values.sum())
    return dict(net_improvement_sum=net,gross_improvements=parts(positive),gross_degradations=parts(negative),
        largest_positive_share_of_net=float(positive.max()/net) if net>0 else None,
        top3_positive_share_of_net=float(np.sort(positive)[-3:].sum()/net) if net>0 else None,
        definition='positive = D0B error minus ResidualComplementaryFusion error; gross shares use only positive improvements (degradations separately). Net shares can exceed 100% when other commodities worsen; undefined if net improvement <=0.')


def build_comparison(base,local,names,diagnostics,out):
    metrics={label:{split:population_metrics(row['prediction'],row['target']) for split,row in arrays.items()}
             for label,arrays in [('D0B',base),('ResidualComplementaryFusion',local)]}
    overall=[]
    for split in ('train','val','test'):
        np.testing.assert_array_equal(base[split]['target'],local[split]['target'])
        for key in ('MAE','MSE','RMSE','Hit'):
            b,h=metrics['D0B'][split][key],metrics['ResidualComplementaryFusion'][split][key]
            overall.append(dict(split=split,metric=key,D0B=b,ResidualComplementaryFusion=h,delta=h-b,
                relative_percent=100*(h-b)/b if b else None,unit='fraction (Hit% = 100*Hit)' if key=='Hit' else 'raw return units'))
    commodity=[];b,h=base['test'],local['test'];be=b['prediction'].astype(float)-b['target'];he=h['prediction'].astype(float)-h['target']
    for i,name in enumerate(names):
        bm,hm=population_metrics(b['prediction'][:,i],b['target'][:,i]),population_metrics(h['prediction'][:,i],h['target'][:,i])
        commodity.append(dict(commodity=str(name),commodity_index=i,D0B_MAE=bm['MAE'],ResidualComplementaryFusion_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
            D0B_MSE=bm['MSE'],ResidualComplementaryFusion_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE'],
            absolute_error_improvement_sum=float((np.abs(be[:,i])-np.abs(he[:,i])).sum()),
            squared_error_improvement_sum=float((be[:,i]**2-he[:,i]**2).sum())))
    thresholds=dict(zip(('P75','P90','P95'),np.quantile(np.abs(base['train']['target']).astype(float),[.75,.9,.95]).tolist()))
    groups=[]
    for split in ('val','test'):
        y=base[split]['target'];a=np.abs(y)
        masks={'<=P75':a<=thresholds['P75'],'P75-P90':(a>thresholds['P75'])&(a<=thresholds['P90']),'>P90':a>thresholds['P90'],'>P95':a>thresholds['P95']}
        for name,mask in masks.items():
            if not mask.any():
                groups.append(dict(split=split,group=name,count=0));continue
            bm=population_metrics(base[split]['prediction'][mask],y[mask]);hm=population_metrics(local[split]['prediction'][mask],y[mask])
            groups.append(dict(split=split,group=name,count=int(mask.sum()),D0B_MAE=bm['MAE'],ResidualComplementaryFusion_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
                D0B_MSE=bm['MSE'],ResidualComplementaryFusion_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE']))
    for name,rows in [('overall_metrics',overall),('commodity_metrics',commodity),('target_magnitude_groups',groups)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    contributions={key:improvement_shares([c[key] for c in commodity]) for key in ('absolute_error_improvement_sum','squared_error_improvement_sum')}
    leave_out=leave_one_commodity_out(base,local,names,commodity)
    return dict(leave_one_commodity_out=leave_out,metrics=metrics,overall_metrics=overall,commodity_metrics=commodity,target_magnitude_groups=groups,
        train_target_thresholds=thresholds,contribution_shares=contributions,
        top5_improved=sorted(commodity,key=lambda x:x['delta_MAE'])[:5],top5_worsened=sorted(commodity,key=lambda x:x['delta_MAE'],reverse=True)[:5],
        focus_commodities=[c for c in commodity if any(k in c['commodity'] for k in ('焦煤','焦炭','燃料油','原油','低硫燃料油'))],
        **classify(metrics,diagnostics,leave_out))





def write_report(r,out):
    shared=r.get('sanity',{}).get('shared_init',{})
    lines=['# D0B-ResidualComplementaryFusion','',r['status'],'',
        '## 1. Controlled change','',
        'Baseline: D0B-BalancedLatentReadout. Original competitive gated fusion fully retained. Only change: raw global [h_spatial,h_temporal] → Linear(128,32) → ReLU → Linear(32,64,bias=False), zero-init final layer, then add to h_base before the original head. No alpha, gate, normalization, dropout, local/horizon/graph/regime-specific additions. Original raw Q/K and all graph priors retained.','',
        f"SHA: {r.get('git_sha')}; exact working-file hashes in results.json. Checkpoint: {r.get('baseline_checkpoint')}",'',
        '## 2. Parameter/init sanity','',str(shared),'',
        '## 3. Initial exactness','',
        table(['h_base difference','Residual max abs','Fused vs base max diff','Prediction max diff','PASS'],[[shared.get(k) for k in ('h_base_max_diff','residual_max_abs','fused_base_max_diff','prediction_max_diff','PASS')]]),'',
        'Eval-mode equality is required; independent train-mode dropout calls need not match. Residual input is not detached.','',
        '## 4. Structural sanity','',str(r.get('sanity')),'',
        'Temporal masked prefix and observed-window spatial/fusion prefix checks are separate. A fully observed window may legitimately use its suffix. All component batch/single/relabeling raw errors are retained, including any independent FP64 roundoff audit. No default forward precision change.','',
        f"Native pooled reference: {r.get('baseline_reference')}",'',
        '## 5. Activation','',
        table(['Stage','First weight norm','Final weight norm','First grad','Final grad','Mean abs residual','Mean residual L2','Residual/base L2'],
            [[s,g['first_weight_norm'],g['final_weight_norm'],g['norms']['residual_first'],g['norms']['residual_final'],g['mean_abs_residual'],g['mean_residual_L2'],g['residual_base_norm_ratio']] for s,g in r.get('gradients',{}).items()]),'',
        'Fixed TRAIN batch, eval, four Huber losses summed, autograd.grad only; no extra optimizer step or .grad mutation. Zero initial final weights imply zero first-layer gradient; final gradient must be live. Complete module gradient norms: gradient_diagnostics.json.','']
    complete=r.get('training_executed') and 'metrics' in r
    if not complete:
        lines+=['Formal training/evaluation pending; no trained mechanism, performance or Case A–G is assigned.','',
            '## Required 16 answers','']
        answers=[f"Native D0B TEST5: {r.get('baseline_reference')}",f"Parameter delta: {shared.get('delta_params')} (expected 6176)",
            f"Shared initialization max diff: {shared.get('shared_parameter_init_max_diff')}; mismatches: {shared.get('shared_parameter_mismatch_count')}",
            f"Initial residual max abs: {shared.get('residual_max_abs')}",f"Initial prediction max diff: {shared.get('prediction_max_diff')}",
            'Original gate input, projections, sigmoid and temporal/spatial orientation retained.',
            'Residual input is exactly raw [h_spatial,h_temporal], shape (B,128).',f"Core sanity PASS: {r.get('sanity',{}).get('PASS')}",
            'Trained activation pending.','VAL5 MAE comparison pending.','TEST5 MAE comparison pending.','TEST5 MSE comparison pending.',
            'Direct Full−ZeroResidual effect pending.','ZeroResidual−D0B training effect pending.','Commodity contributions pending.',
            'No primary classification before formal evaluation. STOP.']
        lines += [f'{i}. {v}' for i,v in enumerate(answers,1)]
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return
    hist=r['history'];best=r['best_epoch'];b,n=r['metrics']['D0B'],r['metrics']['ResidualComplementaryFusion']
    lines += [table(['Formal best epoch','Train seconds','Formal val objective','VAL5 MAE at formal best','Secondary best VAL5 epoch','Secondary best VAL5 MAE'],
        [[best,r['train_time_seconds'],r['best_formal_val_objective'],hist['val5_diagnostic'][best-1]['MAE'],hist['best_val5_epoch'],hist['best_val5_mae']]]),'',
        'Only original multi-horizon validation controls scheduler, early stopping and formal checkpoint. Secondary VAL5 never selects TEST weights. Unreached diagnostic epochs remain absent.','',
        '## 6. Main result','',
        table(['Model','Split','MAE','MSE','RMSE','Hit%'],[[name,s,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for name,splits in r['metrics'].items() for s,m in splits.items() if s in ('val','test')]),'',
        table(['Split','Metric','D0B','New','Absolute delta','Relative %'],[[v[k] for k in ('split','metric','D0B','ResidualComplementaryFusion','delta','relative_percent')] for v in r['overall_metrics'] if v['split']!='train']),'',
        'Pooled origin × commodity metrics; MSE directly averages squared errors and RMSE=sqrt(MSE). Unmasked sign Hit includes zero targets. Hit deltas in the delta table are fractions, multiply by 100 for percentage points.','',
        'RMSE squared checks: '+str({label:{s:abs(m['RMSE']**2-m['MSE']) for s,m in splits.items()} for label,splits in r['metrics'].items()}),'',
        '## 7. Direct vs representation effect','',
        table(['Split','Object','MAE','MSE','RMSE','Hit%'],[[s,label,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for s,a in r['ablations'].items() for label,m in a['metrics'].items() if s in ('val','test')]),'',
        table(['Split','Effect','MAE delta','MSE delta','RMSE delta','Hit delta'],[[s,k,*[a[k][metric] for metric in ('MAE','MSE','RMSE','Hit')]] for s,a in r['ablations'].items() for k in ('direct_effect','representation_effect') if s in ('val','test')]),'',
        'DirectEffect=Full−ZeroResidual; RepresentationEffect=ZeroResidual−OriginalD0B. Negative error delta means improvement. ZeroResidual reuses the trained h_base and the same original head in eval mode, setting only the fusion residual to zero. This identifies a changed trained base path, including the head, without proving an encoder alone caused the change.','',
        'Zero-residual relative deltas vs original D0B: '+str(r['zero_residual_relative_deltas']),'',
        '## 8. Fusion mechanism','']
    norm_rows=[];cos_rows=[];gate_rows=[]
    for label,key in [('D0B','D0B_full_loader'),('New','New_full_loader')]:
        for split,stats in r['residual'][key].items():
            for name,value in stats['norms'].items():norm_rows.append([label,split,name,*[value[k] for k in ('mean','std','median','P90','P95','max')],stats['residual_base_norm_ratio']])
            for name in ('cos_spatial_temporal','cos_base_fused'):
                value=stats[name];cos_rows.append([label,split,name,*[value[k] for k in ('mean','std','P5','P50','P95')]])
            value=stats['gate'];gate_rows.append([label,split,*[value[k] for k in ('mean','std','P10','P50','P90')]])
    lines += [table(['Model','Split','State L2','Mean','Std','Median','P90','P95','Max','Residual/base L2 ratio'],norm_rows),'',
        table(['Model','Split','Cosine','Mean','Std','P5','P50','P95'],cos_rows),'',
        table(['Model','Split','Gate mean','Gate std','Gate P10','Gate P50','Gate P90'],gate_rows),'',
        'These are descriptive distributions, not tuning targets. Residual magnitude or direction changes alone are not forecasting success.','',
        '## 9. Commodity results','',
        table(['Commodity','D0B MAE','New MAE','Delta MAE','D0B MSE','New MSE','Delta MSE'],[[v[k] for k in ('commodity','D0B_MAE','ResidualComplementaryFusion_MAE','delta_MAE','D0B_MSE','ResidualComplementaryFusion_MSE','delta_MSE')] for v in r['commodity_metrics']]),'',
        'Top5 improved by MAE (signed deltas): '+str(r['top5_improved']),'',
        'Top5 degraded by MAE (signed deltas): '+str(r['top5_worsened']),'',
        '焦煤/焦炭/原油/燃料油/低硫燃料油: '+str(r['focus_commodities']),'',
        'Gross improvement/degradation and largest/top3 shares: '+str(r['contribution_shares']),'',
        'One leave-one-out arithmetic: '+str(r['leave_one_commodity_out']),'',
        f"Single-commodity caveat (TEST MAE advantage disappears after exclusion): {r['single_commodity_caveat']}",'',
        '## 10. Error magnitude','',f"TRAIN thresholds: {r['train_target_thresholds']}; >P95 is nested within >P90.",'',
        table(['Split','Group','Count','D0B MAE','New MAE','Delta MAE','D0B MSE','New MSE','Delta MSE'],[[v.get(k) for k in ('split','group','count','D0B_MAE','ResidualComplementaryFusion_MAE','delta_MAE','D0B_MSE','ResidualComplementaryFusion_MSE','delta_MSE')] for v in r['target_magnitude_groups']]),'',
        '## Required 16 answers','']
    answers=[f"Native D0B TEST5 reproduced: {r['baseline_reference']['PASS']}; {r['baseline_reference']['actual']}",
        f"Parameter delta: {shared['delta_params']}",f"Shared init diff: {shared['shared_parameter_init_max_diff']}; mismatches: {shared['shared_parameter_mismatch_count']}",
        f"Initial residual: {shared['residual_max_abs']}",f"Initial prediction max diff: {shared['prediction_max_diff']}",
        'Original gated fusion is fully retained and its formula is checked against captured head inputs.',
        'Residual input is exactly raw global [h_spatial,h_temporal].',f"Core sanity PASS: {r['sanity']['PASS']}",
        f"Branch status: {r['mechanism_status']}; measured activation table above.",f"VAL5 MAE improves: {n['val']['MAE']<b['val']['MAE']}",
        f"TEST5 MAE improves: {n['test']['MAE']<b['test']['MAE']}",f"TEST5 MSE delta: {n['test']['MSE']-b['test']['MSE']}",
        'Direct residual effect: '+str({s:a['direct_effect'] for s,a in r['ablations'].items() if s!='train'}),
        'Representation-training effect: '+str({s:a['representation_effect'] for s,a in r['ablations'].items() if s!='train'}),
        f"Commodity concentration: {r['contribution_shares']}; leave-one-out: {r['leave_one_commodity_out']}",
        f"Primary classification: {r['primary_case']}; {r['interpretation']}"]
    lines += [f'{i}. {v}' for i,v in enumerate(answers,1)]
    lines += ['',r['classification_notes'],'',r['interpretation'],
        '**STOP D0B incremental patching**' if r['stop_incremental_patching'] else 'STOP this one controlled experiment.',
        'No second fusion variant or automatic backbone redesign. Future direction, if needed: genuinely redesigned backbone / information architecture rather than another local D0B adapter.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
