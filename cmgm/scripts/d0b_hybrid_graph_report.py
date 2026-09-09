"""Uniform pooled evaluation and bounded reporting; no model selection feedback."""
import numpy as np
import pandas as pd
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0b_risk_probe_analysis import table


def classify(metrics,attention,sanity):
    if not sanity:return dict(primary_case=None,interpretation='DO NOT INTERPRET PERFORMANCE: core sanity failed')
    gaps=[abs(attention[l]['free_entropy']-attention[l]['graph_entropy']) for l in ('layer1','layer2')]
    masses=[abs(attention[l]['free_topk_mass']-attention[l]['graph_topk_mass']) for l in ('layer1','layer2')]
    active=max(gaps+masses)>1e-3
    d={s:{k:metrics['Hybrid'][s][k]/metrics['D0B'][s][k]-1 for k in ('MAE','MSE')} for s in ('val','test')}
    val,test=d['val']['MAE'],d['test']['MAE'];mse=d['test']['MSE']
    if val<0 and test<0 and mse<=.001 and active:
        case='Case A';text='VAL/TEST MAE improve with active differentiation and no >0.1% relative TEST MSE worsening; supports complementary priors/content on this single run.'
    elif val<0 and test<0 and mse>.001:
        case='Case B';text='MAE improves but TEST MSE worsens by more than 0.1%; error-distribution tradeoff, not a clean replacement.'
    elif val<0 and test>=0:
        case='Case C';text='VAL-only improvement; benefit did not generalize to TEST.'
    elif val>0 and test>0 and mse>0 and active:
        case='Case D';text='Mechanism active but forecasting errors worsen; results are consistent with useful regularization from the static prior.'
    elif not active and max(abs(v) for s in d.values() for v in s.values())<=.001:
        case='Case E';text='Near-baseline errors and weak functional head differentiation; redundant under this diagnostic.'
    elif val>0 and test>0 and d['val']['MSE']>0 and mse>0:
        case='Case F';text='VAL and TEST forecast errors worsen; do not continue this spatial route.'
    else:
        case=None;text='Mixed result does not satisfy the supplied A–F definitions; manual scientific review required, no forced success or new experiment.'
    return dict(primary_case=case,interpretation=text,mechanism_status='active' if active else 'weak',
        forecasting_value='positive' if case=='Case A' else ('tradeoff' if case in ('Case B','Case C') else 'negative or unestablished'),
        classification_notes='Descriptive reporting tolerances only, never training/selection/tuning: entropy/topk mass gap >1e-3 for active; 0.1% relative error for near-tie/material-MSE convention. Full raw values retained. One seed does not establish statistical robustness.',
        attention_entropy_gaps=gaps,attention_mass_gaps=masses)


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
        definition='positive = D0B error minus Hybrid error; gross shares use only positive improvements (degradations separately). Net shares can exceed 100% when other commodities worsen; undefined if net improvement <=0.')


def build_comparison(base,hybrid,names,attention,sanity,out):
    metrics={label:{split:population_metrics(row['prediction'],row['target']) for split,row in arrays.items()}
             for label,arrays in [('D0B',base),('Hybrid',hybrid)]}
    overall=[]
    for split in ('train','val','test'):
        np.testing.assert_array_equal(base[split]['target'],hybrid[split]['target'])
        for key in ('MAE','MSE','RMSE','Hit'):
            b,h=metrics['D0B'][split][key],metrics['Hybrid'][split][key]
            overall.append(dict(split=split,metric=key,D0B=b,Hybrid=h,delta=h-b,
                relative_percent=100*(h-b)/b if b else None,unit='fraction (Hit% = 100*Hit)' if key=='Hit' else 'raw return units'))
    commodity=[];b,h=base['test'],hybrid['test'];be=b['prediction'].astype(float)-b['target'];he=h['prediction'].astype(float)-h['target']
    for i,name in enumerate(names):
        bm,hm=population_metrics(b['prediction'][:,i],b['target'][:,i]),population_metrics(h['prediction'][:,i],h['target'][:,i])
        commodity.append(dict(commodity=str(name),commodity_index=i,D0B_MAE=bm['MAE'],Hybrid_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
            D0B_MSE=bm['MSE'],Hybrid_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE'],
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
            bm=population_metrics(base[split]['prediction'][mask],y[mask]);hm=population_metrics(hybrid[split]['prediction'][mask],y[mask])
            groups.append(dict(split=split,group=name,count=int(mask.sum()),D0B_MAE=bm['MAE'],Hybrid_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
                D0B_MSE=bm['MSE'],Hybrid_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE']))
    for name,rows in [('overall_metrics',overall),('commodity_metrics',commodity),('target_magnitude_groups',groups)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    contributions={key:improvement_shares([c[key] for c in commodity]) for key in ('absolute_error_improvement_sum','squared_error_improvement_sum')}
    return dict(metrics=metrics,overall_metrics=overall,commodity_metrics=commodity,target_magnitude_groups=groups,
        train_target_thresholds=thresholds,contribution_shares=contributions,
        top5_improved=sorted(commodity,key=lambda x:x['delta_MAE'])[:5],top5_worsened=sorted(commodity,key=lambda x:x['delta_MAE'],reverse=True)[:5],
        focus_commodities=[c for c in commodity if any(k in c['commodity'] for k in ('焦煤','焦炭','燃料油','原油'))],
        **classify(metrics,attention,sanity))


def write_report(r,out):
    lines=['# D0B-HybridGraphPriorHeads','',r['status'],'',
        '## 1. Controlled change','',
        'Baseline: D0B-BalancedLatentReadout. Only architecture change: 8 graph-prior heads → 4 free-content heads + 4 graph-prior heads in BOTH EdgeAttnMixHop layers. No new QKV, unchanged adaptive graph, temporal branch, fusion, head, objective and formal checkpoint selection. SEQ_LEN=20.','',
        f"Initialization/parameter count: {r.get('sanity',{}).get('shared_init')}",'',
        '## 2. Sanity','',str(r.get('sanity')),'',f"Native checkpoint/reference: {r.get('baseline_checkpoint')}; {r.get('baseline_reference')}",'',
        'Read-only capture is pre-dropout and per MixHop hop. Free heads have zero direct graph bias; later content states can inherit earlier mixed graph information. Graph top-k is descriptive on the unchanged learned A, not a new hard mask.','']
    rows=[]
    for stage,value in r.get('attention',{}).items():
        for layer in ('layer1','layer2'):
            for h in value[layer]['hops']:
                rows.append([stage,layer,h['hop'],h['free_prior_error'],h['graph_prior_error'],
                    h['prior_formula_error'],h['exact_native_logit_recomposition_error'],h['float64_bias_subtraction_error'],h['partition_PASS']])
    lines += [table(['Stage','Layer','Hop','Free bias error','Native subtraction error','A-prior formula error','Exact native recomposition error','FP64 subtraction error','Partition PASS'],rows),'',
        'Large FP32 logits can make (content+bias)-content exceed 1e-6 through cancellation. Raw failure is retained above and in JSON; PASS requires zero free bias, exact native addition with independently recomputed A prior, a verified rounding bound, and FP64 same-operands subtraction below 1e-6. This is a diagnostic precision audit, not a change to model forward.','']
    if not r.get('training_executed'):
        lines += ['Formal training has not run. VAL/TEST Hybrid performance, learned mechanism and Case A–F are not available. No performance conclusion is fabricated.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return
    hist=r['history'];best=r['best_epoch'];best5=hist['val5_diagnostic'][best-1]
    lines += ['## 3. Training','',table(['Formal best epoch','Train time s','Formal best val objective','VAL5 MAE at formal best','Secondary best-VAL5 epoch','Secondary best-VAL5 MAE'],
        [[best,r['train_time_seconds'],r['best_formal_val_objective'],best5['MAE'],hist['best_val5_epoch'],hist['best_val5_mae']]]),'',
        'Only the multi-horizon validation selected checkpoint was saved/reloaded for formal TEST evaluation. Secondary VAL5 history never controls scheduler, early stopping, checkpoint or TEST selection.','',
        '## 4. Main result','',table(['Model','Split','MAE','MSE','RMSE','Hit%'],[[name,split,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for name,splits in r['metrics'].items() for split,m in splits.items() if split in ('val','test')]),'',
        table(['Split','Metric','D0B','Hybrid','Hybrid minus D0B','Relative %'],[[x[k] for k in ('split','metric','D0B','Hybrid','delta','relative_percent')] for x in r['overall_metrics'] if x['split']!='train']),'',
        'Hit deltas above are fractions; multiply by 100 for percentage-point differences. MAE/MSE/RMSE are pooled across origin × commodity, zero targets included in sign Hit.','',
        '## 5. Spatial mechanism','']
    rows=[]
    for stage,value in r['attention'].items():
        for layer in ('layer1','layer2'):
            l=value[layer];rows.append([stage,layer,l['free_entropy'],l['graph_entropy'],l['free_topk_mass'],l['graph_topk_mass']])
    lines += [table(['Stage','Layer','Free entropy','Graph entropy','Free top-k mass','Graph top-k mass'],rows),'',
        'Head cosine/correlation for free-free, graph-graph and free-graph, per hop and per stage: attention_head_diagnostics.json.','',
        '## 6. Graph learner','',str(r['graph']), '',str(r['gradients']['best']),'',
        'Graph gradient connectivity, norm and learned adjacency values above determine whether this path remains active; no inference from head count alone. Graph learner alpha is a learnable graph parameter; Markov sticky_alpha remains fixed .5.','',
        '## 7. Commodity decomposition','',table(['Commodity','D0B MAE','Hybrid MAE','Delta MAE','D0B MSE','Hybrid MSE','Delta MSE'],[[x[k] for k in ('commodity','D0B_MAE','Hybrid_MAE','delta_MAE','D0B_MSE','Hybrid_MSE','delta_MSE')] for x in r['commodity_metrics']]),'',
        'Top5 improvements (MAE ordering; retain signed deltas): '+str(r['top5_improved']),'',
        'Top5 degradations (MAE ordering; retain signed deltas): '+str(r['top5_worsened']),'',
        '焦煤/焦炭/燃料油/原油: '+str(r['focus_commodities']),'',
        'Contribution shares: '+str(r['contribution_shares']),'',
        '## 8. Error regime','',f"TRAIN thresholds: {r['train_target_thresholds']}; >P95 is a subset of >P90, not an additive fourth partition.",'',
        table(['Split','Group','Count','D0B MAE','Hybrid MAE','Delta MAE','D0B MSE','Hybrid MSE','Delta MSE'],[[x.get(k) for k in ('split','group','count','D0B_MAE','Hybrid_MAE','delta_MAE','D0B_MSE','Hybrid_MSE','delta_MSE')] for x in r['target_magnitude_groups']]),'',
        '## 9. Required 15 answers','']
    m=r['metrics'];b=m['D0B'];h=m['Hybrid'];a=r['attention']['best']
    answers=[f"Native TEST5 reference reproduced: {r['baseline_reference']['PASS']}; {r['baseline_reference']['actual']}",
        f"Parameter delta: {r['sanity']['shared_init']['delta_params']}",f"Shared init max diff: {r['sanity']['shared_init']['shared_parameter_init_max_diff']}",
        f"4+4 partition PASS: {a['PASS']}",'Free heads: zero direct bias checked per hop.',
        'Graph heads: unchanged prior_scale * log(clamp_min(A,0)+1e-6) checked per hop.',
        f"Causality/batch/relabeling PASS: {r['sanity']['PASS']}",
        f"VAL5 MAE improved: {h['val']['MAE']<b['val']['MAE']}",f"TEST5 MAE improved: {h['test']['MAE']<b['test']['MAE']}",
        f"TEST5 MSE delta: {h['test']['MSE']-b['test']['MSE']}",f"Head mechanism: {r.get('mechanism_status')}; {r.get('attention_entropy_gaps')}; {r.get('attention_mass_gaps')}",
        f"Graph learner: {r['graph']['best']}; gradients: {r['gradients']['best']}",
        'Commodity changes: see full signed-delta table and top5/focus groups above.',
        f"Single-commodity dominance: {r['contribution_shares']}",f"Primary classification: {r.get('primary_case')}; {r['interpretation']}"]
    lines += [f'{i}. {answer}' for i,answer in enumerate(answers,1)]
    lines += ['',f"Observed result: {r.get('primary_case')}; {r['interpretation']}",
        f"Mechanism status: {r.get('mechanism_status')}",f"Forecasting value: {r.get('forecasting_value')}",
        'Scientific interpretation: '+r['interpretation'],r.get('classification_notes',''),
        'Therefore: report this one controlled run; do not select a different head ratio or checkpoint on TEST.',
        'Do NOT yet implement: another ratio, dynamic graph, prior-scale tuning, graph redesign, new branch/head/loss or checkpoint-selection redesign. STOP.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
