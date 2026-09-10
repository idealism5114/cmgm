"""Read-only error decomposition and explicit mechanism-aware reporting."""
import numpy as np
import pandas as pd
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0b_risk_probe_analysis import table


def leave_one_commodity_out(base,local,names,rows):
    result={}
    for metric,key in [('MAE','absolute_error_improvement_sum'),('MSE','squared_error_improvement_sum')]:
        selected=max(rows,key=lambda v:v[key]);index=selected['commodity_index'];mask=np.arange(len(names))!=index
        splits={}
        for split in ('val','test'):
            b=population_metrics(base[split]['prediction'][:,mask],base[split]['target'][:,mask])
            n=population_metrics(local[split]['prediction'][:,mask],local[split]['target'][:,mask])
            splits[split]=dict(D0B=b,PreGNNLocalSkip=n,delta={k:n[k]-b[k] for k in ('MAE','MSE','RMSE','Hit')})
        result[metric]=dict(removed_commodity=str(names[index]),removed_index=index,removed_improvement_sum=selected[key],splits=splits,
            description='Supplementary arithmetic, remove the TEST largest improving commodity for this metric; no retraining or model selection. Reuse this same exclusion in VAL.')
    return result


def classify(metrics,r,leave_out):
    if not r['sanity']['PASS']:return dict(primary_case=None,interpretation='DO NOT INTERPRET PERFORMANCE: sanity failed')
    b,n=metrics['D0B'],metrics['PreGNNLocalSkip']
    dv=n['val']['MAE']/b['val']['MAE']-1;dt=n['test']['MAE']/b['test']['MAE']-1;dm=n['test']['MSE']/b['test']['MSE']-1
    a=r['ablations']['test'];res=r['residual']['full_loader']['test']['primary_5d']['mean_abs'];weight=r['residual']['best']['final_weight_norm']
    scale=r['residual']['full_loader']['test']['mean_abs_residual_over_base_5d']
    impacts=a['impacts'];active=scale>1e-3 and weight>1e-6
    shuffle=impacts['ShuffledLocal']['prediction_mean_diff']>1e-6
    mean=impacts['MeanLocal']['prediction_mean_diff']>1e-6
    controls_support=shuffle and mean and all(any(a['metrics'][c][k]>a['metrics']['Full'][k] for k in ('MAE','MSE')) for c in ('ShuffledLocal','MeanLocal'))
    zero=a['metrics']['ZeroResidual'];original=a['metrics']['OriginalD0B'];full=a['metrics']['Full']
    zero_tied=all(abs(zero[k]/original[k]-1)<=.001 for k in ('MAE','MSE'))
    full_zero_tied=all(abs(full[k]/zero[k]-1)<=.001 for k in ('MAE','MSE'))
    corner=dt<0 and leave_out['MAE']['splits']['test']['delta']['MAE']>=0
    if corner:
        case='Case G';text='The pooled MAE advantage disappears after excluding its largest improving commodity; apparent benefit is not broad commodity-local generalization.'
    elif dv<0 and dt<0 and dm>.001:
        case='Case D';text='Local correction redistributes errors but is not a clean replacement.'
    elif dv<0 and dt<0 and dm<=.001 and zero_tied and active and controls_support:
        case='Case B';text='The local skip works primarily as an inference-time commodity-local correction; the independently trained zero-residual base is near D0B.'
    elif dv<0 and dt<0 and dm<=.001 and zero['MAE']<original['MAE'] and zero['MSE']<=original['MSE']*1.001 and full_zero_tied:
        case='Case C';text='The local path mainly improves training of the shared representation rather than direct residual output.'
    elif dv<0 and dt<0 and dm<=.001 and active and controls_support:
        case='Case A';text='Pre-GNN commodity-local information provides useful incremental signal beyond the global D0B representation.'
    elif dv>0 and dt>0 and active and shuffle:
        case='Case E';text='The branch is active and shuffle-sensitive, but direct local access harms D0B generalization; global pooling may provide useful regularization.'
    elif scale<=1e-3 and weight<=1e-6 and not shuffle and full_zero_tied:
        case='Case F';text='The optimizer effectively rejects the pre-GNN local correction path.'
    else:
        case=None;text='Mixed evidence outside the supplied A–G definitions; no forced success or claim of negative transfer. No follow-up experiment.'
    return dict(primary_case=case,interpretation=text,mechanism_status='active' if active else 'weak',controls_support_local_information=controls_support,
        classification_notes='Descriptive conventions only: near-tie/material-MSE 0.1%; residual/base magnitude >0.1% and final weight norm >1e-6 for activation; control prediction impact >1e-6. Case G takes priority, then tradeoff, then B/C mechanistic specializations before A. None of these thresholds affects training, checkpoints or tuning. One run does not establish statistical significance.')


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
        definition='positive = D0B error minus PreGNNLocalSkip error; gross shares use only positive improvements (degradations separately). Net shares can exceed 100% when other commodities worsen; undefined if net improvement <=0.')


def build_comparison(base,local,names,diagnostics,out):
    metrics={label:{split:population_metrics(row['prediction'],row['target']) for split,row in arrays.items()}
             for label,arrays in [('D0B',base),('PreGNNLocalSkip',local)]}
    overall=[]
    for split in ('train','val','test'):
        np.testing.assert_array_equal(base[split]['target'],local[split]['target'])
        for key in ('MAE','MSE','RMSE','Hit'):
            b,h=metrics['D0B'][split][key],metrics['PreGNNLocalSkip'][split][key]
            overall.append(dict(split=split,metric=key,D0B=b,PreGNNLocalSkip=h,delta=h-b,
                relative_percent=100*(h-b)/b if b else None,unit='fraction (Hit% = 100*Hit)' if key=='Hit' else 'raw return units'))
    commodity=[];b,h=base['test'],local['test'];be=b['prediction'].astype(float)-b['target'];he=h['prediction'].astype(float)-h['target']
    for i,name in enumerate(names):
        bm,hm=population_metrics(b['prediction'][:,i],b['target'][:,i]),population_metrics(h['prediction'][:,i],h['target'][:,i])
        commodity.append(dict(commodity=str(name),commodity_index=i,D0B_MAE=bm['MAE'],PreGNNLocalSkip_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
            D0B_MSE=bm['MSE'],PreGNNLocalSkip_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE'],
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
            groups.append(dict(split=split,group=name,count=int(mask.sum()),D0B_MAE=bm['MAE'],PreGNNLocalSkip_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
                D0B_MSE=bm['MSE'],PreGNNLocalSkip_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE']))
    for name,rows in [('overall_metrics',overall),('commodity_metrics',commodity),('target_magnitude_groups',groups)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    contributions={key:improvement_shares([c[key] for c in commodity]) for key in ('absolute_error_improvement_sum','squared_error_improvement_sum')}
    leave_out=leave_one_commodity_out(base,local,names,commodity)
    return dict(leave_one_commodity_out=leave_out,metrics=metrics,overall_metrics=overall,commodity_metrics=commodity,target_magnitude_groups=groups,
        train_target_thresholds=thresholds,contribution_shares=contributions,
        top5_improved=sorted(commodity,key=lambda x:x['delta_MAE'])[:5],top5_worsened=sorted(commodity,key=lambda x:x['delta_MAE'],reverse=True)[:5],
        focus_commodities=[c for c in commodity if any(k in c['commodity'] for k in ('焦煤','焦炭','燃料油','原油','低硫燃料油','小麦'))],
        **classify(metrics,diagnostics,leave_out))




def write_report(r,out):
    shared=r.get('sanity',{}).get('shared_init',{})
    lines=['# D0B-PreGNNCommodityLocalSkip','',r['status'],'',
        '## 1. Controlled change','',
        'Baseline: D0B-BalancedLatentReadout. Only architecture change: pre-GNN TempWeighted commodity-local states enter ONE shared residual head [fused global, local] → Linear(128,32) → ReLU → Linear(32,4,bias=False), last layer zero initialized. No alpha, gate, embedding or commodity-specific parameters. Full original D0B global spatial/temporal/switching/fusion/head retained, with raw Q/K and all 8 original graph priors. SEQ_LEN=20.','',
        f"Diagnostic SHA: {r.get('git_sha')}; working file hashes: results.json. Baseline checkpoint: {r.get('baseline_checkpoint')}",'',
        '## 2. Parameter/init sanity','',str(shared),'',
        '## 3. Initial exactness','',
        table(['D0B vs new prediction max diff','Residual max abs','Base vs final max diff','PASS'],[[shared.get('eval_prediction_max_diff'),shared.get('residual_max_abs'),shared.get('base_final_max_diff'),shared.get('PASS')]]),'',
        'Eval mode is used. Train-mode independent dropout calls need not match. Residual inputs stay attached to autograd; only diagnostic copies detach.','',
        '## 4. Structural sanity','',str(r.get('sanity')),'',
        'H_pre is independently reconstructed from type projection plus node-wise temporal softmax aggregation and compared with the captured input to the first GNN. No post-GNN state is used by the local path. Temporal masked-prefix causality and local prefix/window causality are distinct: an observed suffix can legitimately affect a full-window spatial prediction. Commodity relabeling permutes graph embeddings and original output rows; the single shared residual head is not permuted.','',
        f"Native pooled reference: {r.get('baseline_reference')}",'',
        table(['Commodity','Node index','Target index','Output index'],[[v[k] for k in ('commodity_name','node_index','target_index','output_index')] for v in r.get('sanity',{}).get('commodity_order',{}).get('mapping',[])]),'',
        '## 5. Training and activation','']
    stages=[]
    for stage,g in r.get('gradients',{}).items():
        stages.append([stage,g['final_weight_norm'],g['first_weight_norm'],g['norms']['residual_first'],g['norms']['residual_final'],g['mean_abs_residual']])
    lines += [table(['Stage','Final weight norm','First weight norm','First gradient norm','Final gradient norm','Fixed TRAIN mean abs residual'],stages),'',
        'At zero initialization the first residual layer gradient is expected to be zero; the final layer must have a nonzero gradient. Early stage diagnostics test subsequent activation without an extra update. All prediction-only gradients use fixed TRAIN inputs, eval, sum four Huber losses and autograd.grad without .grad mutation.','',str(r.get('gradients')),'']
    complete=r.get('training_executed') and 'metrics' in r
    if not complete:
        lines += ['Formal training/evaluation not completed; trained residual activation, local controls and Case A–G are unavailable.','',
            '## Required 17 answers','']
        answers=[f"Native D0B TEST5: {r.get('baseline_reference')}",f"Parameter delta: {shared.get('delta_params')} (expected 4256).",
            f"Shared init max diff: {shared.get('shared_parameter_init_max_diff')}; mismatch_count={shared.get('mismatch_count')}",
            f"Initial eval prediction diff: {shared.get('eval_prediction_max_diff')}",f"Initial residual max abs: {shared.get('residual_max_abs')}",
            'H_pre source is audited against both independent TempWeighted reconstruction and first GNN input.',
            f"Ordering PASS: {r.get('sanity',{}).get('commodity_order',{}).get('PASS')}",f"Structural sanity PASS: {r.get('sanity',{}).get('PASS')}",
            'Trained branch activation pending.','VAL5 improvement pending.','TEST5 improvement pending.','TEST5 MSE comparison pending.',
            'Trained zero-residual base vs D0B pending.','Trained local-shuffle effect pending.','Trained mean replacement effect pending.',
            'Single-commodity contribution and exclusion arithmetic pending.','No primary case before formal evaluation. STOP.']
        lines += [f'{i}. {answer}' for i,answer in enumerate(answers,1)]
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return
    hist=r['history'];epoch=r['best_epoch']
    lines += [table(['Formal best epoch','Train seconds','Formal val objective','VAL5 MAE at formal best','Secondary best VAL5 epoch','Secondary best VAL5 MAE'],
        [[epoch,r['train_time_seconds'],r['best_formal_val_objective'],hist['val5_diagnostic'][epoch-1]['MAE'],hist['best_val5_epoch'],hist['best_val5_mae']]]),'',
        'Original multi-horizon validation controls scheduler, early stopping and the only formal checkpoint. VAL5 history is secondary logging. Missing stage epochs indicate early stopping before that epoch.','',
        '## 6. Main result','',
        table(['Model','Split','MAE','MSE','RMSE','Hit%'],[[name,s,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for name,splits in r['metrics'].items() for s,m in splits.items() if s in ('val','test')]),'',
        table(['Split','Metric','D0B','PreGNNLocalSkip','Absolute delta','Relative %'],[[v[k] for k in ('split','metric','D0B','PreGNNLocalSkip','delta','relative_percent')] for v in r['overall_metrics'] if v['split']!='train']),'',
        'Metrics pool origin × commodity. MSE is mean squared residual, RMSE=sqrt(MSE); sign Hit includes zero targets. Hit deltas in the delta table are fractions (multiply by 100 for percentage points).','',
        'RMSE squared checks: '+str({n:{s:abs(m['RMSE']**2-m['MSE']) for s,m in split.items()} for n,split in r['metrics'].items()}),'',
        '## 7. Direct vs representation effect','',
        table(['Split','Model/control','MAE','MSE','RMSE','Hit%'],[[s,n,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for s,v in r['ablations'].items() for n,m in v['metrics'].items() if s in ('val','test')]),'',
        'Direct effect (Full minus ZeroResidual), representation-training effect (ZeroResidual minus OriginalD0B); negative error delta means improvement: '+str({s:{k:v[k] for k in ('direct_effect','representation_training_effect')} for s,v in r['ablations'].items()}),'',
        'The trained base path is independently optimized and need not equal the original checkpoint. This decomposition describes output and training effects; it does not isolate a unique shared module as causal.','',
        '## 8. Local-state mechanism','',
        f"Fixed commodity permutation, seed 42: {r['shuffle_permutation']}",'',
        table(['Split','Control','Prediction mean change','Prediction max change','Residual mean change','Residual max change','Control residual mean abs','Base max diff'],[[s,n,*[p[k] for k in ('prediction_mean_diff','prediction_max_diff','residual_mean_diff','residual_max_diff','mean_abs_control_residual','base_pred_max_diff')]] for s,v in r['ablations'].items() for n,p in v['impacts'].items() if s in ('val','test')]),'',
        'All local interventions reuse the exact native global path; no GNN, temporal, fusion or base head recomputation. Mean replacement uses each origin’s own cross-commodity mean. These controls never select a checkpoint.','',
        'Residual distribution, per-horizon and per-commodity signed/absolute statistics and correction direction: '+str(r['residual']['full_loader']),'',
        '## 9. Commodity decomposition','',
        table(['Commodity','D0B MAE','New MAE','Delta MAE','D0B MSE','New MSE','Delta MSE'],[[v[k] for k in ('commodity','D0B_MAE','PreGNNLocalSkip_MAE','delta_MAE','D0B_MSE','PreGNNLocalSkip_MSE','delta_MSE')] for v in r['commodity_metrics']]),'',
        'Top5 improvements by MAE, retaining signed deltas: '+str(r['top5_improved']),'',
        'Top5 degradations by MAE, retaining signed deltas: '+str(r['top5_worsened']),'',
        'Focus: 焦煤、焦炭、原油、燃料油、低硫燃料油、小麦: '+str(r['focus_commodities']),'',
        'Net/gross improvement/degradation, largest improving/degrading and top3 shares: '+str(r['contribution_shares']),'',
        'Leave-one-commodity-out arithmetic: '+str(r['leave_one_commodity_out']),'',
        '## 10. Target magnitude decomposition','',
        f"TRAIN thresholds: {r['train_target_thresholds']}. >P95 is nested within >P90; these are not additive disjoint groups.",'',
        table(['Split','Group','Count','D0B MAE','New MAE','Delta MAE','D0B MSE','New MSE','Delta MSE'],[[v.get(k) for k in ('split','group','count','D0B_MAE','PreGNNLocalSkip_MAE','delta_MAE','D0B_MSE','PreGNNLocalSkip_MSE','delta_MSE')] for v in r['target_magnitude_groups']]),'',
        '## Required 17 answers','']
    b,n=r['metrics']['D0B'],r['metrics']['PreGNNLocalSkip'];a=r['ablations']['test']
    answers=[f"Native D0B TEST5 reproduced: {r['baseline_reference']['PASS']}; {r['baseline_reference']['actual']}",
        f"Parameter delta: {shared['delta_params']} (expected 4256).",f"Shared initialization: {shared['shared_parameter_init_max_diff']}; mismatches={shared['mismatch_count']}",
        f"Initial eval prediction diff: {shared['eval_prediction_max_diff']}",f"Initial residual max abs: {shared['residual_max_abs']}",
        'H_pre is the independently verified TempWeighted aggregation and exact first-GNN input.',f"Commodity ordering PASS: {r['sanity']['commodity_order']['PASS']}",
        f"All core sanity PASS: {r['sanity']['PASS']}",f"Branch status: {r['mechanism_status']}; activation history above.",
        f"VAL5 MAE improved: {n['val']['MAE']<b['val']['MAE']}",f"TEST5 MAE improved: {n['test']['MAE']<b['test']['MAE']}",
        f"TEST5 MSE delta: {n['test']['MSE']-b['test']['MSE']}",f"Trained zero-residual base vs D0B: {a['representation_training_effect']}",
        f"Shuffle impact: {a['impacts']['ShuffledLocal']}; metric changes above. Sensitivity is descriptive, not statistical significance.",
        f"Mean replacement impact: {a['impacts']['MeanLocal']}; a magnitude change alone does not prove useful local information.",
        f"Commodity dominance: {r['leave_one_commodity_out']}",f"Primary classification: {r['primary_case']}; {r['interpretation']}"]
    lines += [f'{i}. {answer}' for i,answer in enumerate(answers,1)]
    lines += ['',r['classification_notes'],'',r['interpretation'],
        'This preserves the full global GNN; success can support complementary local/global information, not a claim that GNN destroys information.',
        'Late local residual adapters should stop.' if n['val']['MAE']>=b['val']['MAE'] and n['test']['MAE']>=b['test']['MAE'] else 'No automatic follow-up experiment.',
        'STOP. Do not implement bigger adapters, embeddings, gates, GRUs, fusion redesign or other models.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
