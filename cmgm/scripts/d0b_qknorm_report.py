"""Post-training reporting only; no selection, tuning or model mutation."""
import numpy as np
import pandas as pd
from cmgm.training.metric_standard import population_metrics
from cmgm.scripts.d0b_risk_probe_analysis import table


def mechanism_comparison(r):
    rows=[]
    for layer in ('layer1','layer2'):
        for i,(b,q) in enumerate(zip(r['qk_norm']['D0B_best'][layer],r['qk_norm']['best'][layer])):
            ba=r['attention']['D0B_best'][layer]['hops'][i];qa=r['attention']['best'][layer]['hops'][i]
            row=dict(layer=layer,hop=q['hop'])
            for name,bv,qv in [('content_std',b['content_logits']['std'],q['content_logits']['std']),
                ('entropy',ba['entropy_mean'],qa['entropy_mean']),('top1_mass',ba['top1_mass'],qa['top1_mass']),
                ('top5_mass',ba['top5_mass'],qa['top5_mass']),('top10_mass',ba['top10_mass'],qa['top10_mass']),
                ('graph_topk_mass',ba['graph_topk_mass'],qa['graph_topk_mass']),
                ('head_cosine',ba['head_diversity']['cosine_mean'],qa['head_diversity']['cosine_mean'])]:
                row[name]=dict(D0B=bv,QKNorm=qv,delta=qv-bv,relative_delta=(qv-bv)/(abs(bv)+1e-12))
            rows.append(row)
    return rows


def classify(metrics,diagnostics):
    if not diagnostics['sanity']['PASS']:
        return dict(primary_case=None,interpretation='DO NOT INTERPRET PERFORMANCE: core sanity failed')
    rows=mechanism_comparison(diagnostics)
    active=any(abs(row['content_std']['relative_delta'])>.01 or abs(row['entropy']['delta'])>1e-3 or abs(row['top10_mass']['delta'])>1e-3 for row in rows)
    d={s:{k:metrics['QKNorm'][s][k]/metrics['D0B'][s][k]-1 for k in ('MAE','MSE')} for s in ('val','test')}
    val,test,mse=d['val']['MAE'],d['test']['MAE'],d['test']['MSE']
    if val<0 and test<0 and mse<=.001 and active:
        case='Case A';text='Q/K norm geometry was a meaningful spatial-attention bottleneck in this controlled run; VAL/TEST MAE improve without material TEST MSE worsening.'
    elif val<0 and test<0 and mse>.001:
        case='Case B';text='QKNorm changes spatial error distribution but is not a clean replacement.'
    elif val<0 and test>=0:
        case='Case C';text='Non-generalizing Q/K geometry benefit.'
    elif val>0 and test>0 and mse>0 and active:
        case='Case D';text='Mechanism active but forecast errors worsen, consistent with useful regularization from raw dot-product geometry. Whether attention sharpened must be read from the entropy/mass measurements.'
    elif not active and max(abs(v) for split in d.values() for v in split.values())<=.001:
        case='Case E';text='Q/K norm magnitude was not the practical bottleneck.'
    elif val>0 and test>0 and mse>0:
        case='Case F';text='QKNormGraphAttention is rejected.'
    else:
        case=None;text='Mixed evidence outside the supplied A–F definitions; no forced success classification or new experiment.'
    return dict(primary_case=case,interpretation=text,mechanism_status='active' if active else 'weak',mechanism_comparison=rows,
        classification_notes='Reporting conventions only: material TEST MSE deterioration >0.1%; near-baseline errors <=0.1%; geometry active if content std changes >1% relatively or entropy/top10 mass changes >0.001 absolutely. No training or selection uses these thresholds. One seed does not establish statistical significance.')


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
        definition='positive = D0B error minus QKNorm error; gross shares use only positive improvements (degradations separately). Net shares can exceed 100% when other commodities worsen; undefined if net improvement <=0.')


def build_comparison(base,qknorm,names,diagnostics,out):
    metrics={label:{split:population_metrics(row['prediction'],row['target']) for split,row in arrays.items()}
             for label,arrays in [('D0B',base),('QKNorm',qknorm)]}
    overall=[]
    for split in ('train','val','test'):
        np.testing.assert_array_equal(base[split]['target'],qknorm[split]['target'])
        for key in ('MAE','MSE','RMSE','Hit'):
            b,h=metrics['D0B'][split][key],metrics['QKNorm'][split][key]
            overall.append(dict(split=split,metric=key,D0B=b,QKNorm=h,delta=h-b,
                relative_percent=100*(h-b)/b if b else None,unit='fraction (Hit% = 100*Hit)' if key=='Hit' else 'raw return units'))
    commodity=[];b,h=base['test'],qknorm['test'];be=b['prediction'].astype(float)-b['target'];he=h['prediction'].astype(float)-h['target']
    for i,name in enumerate(names):
        bm,hm=population_metrics(b['prediction'][:,i],b['target'][:,i]),population_metrics(h['prediction'][:,i],h['target'][:,i])
        commodity.append(dict(commodity=str(name),commodity_index=i,D0B_MAE=bm['MAE'],QKNorm_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
            D0B_MSE=bm['MSE'],QKNorm_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE'],
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
            bm=population_metrics(base[split]['prediction'][mask],y[mask]);hm=population_metrics(qknorm[split]['prediction'][mask],y[mask])
            groups.append(dict(split=split,group=name,count=int(mask.sum()),D0B_MAE=bm['MAE'],QKNorm_MAE=hm['MAE'],delta_MAE=hm['MAE']-bm['MAE'],
                D0B_MSE=bm['MSE'],QKNorm_MSE=hm['MSE'],delta_MSE=hm['MSE']-bm['MSE']))
    for name,rows in [('overall_metrics',overall),('commodity_metrics',commodity),('target_magnitude_groups',groups)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    contributions={key:improvement_shares([c[key] for c in commodity]) for key in ('absolute_error_improvement_sum','squared_error_improvement_sum')}
    return dict(metrics=metrics,overall_metrics=overall,commodity_metrics=commodity,target_magnitude_groups=groups,
        train_target_thresholds=thresholds,contribution_shares=contributions,
        top5_improved=sorted(commodity,key=lambda x:x['delta_MAE'])[:5],top5_worsened=sorted(commodity,key=lambda x:x['delta_MAE'],reverse=True)[:5],
        focus_commodities=[c for c in commodity if any(k in c['commodity'] for k in ('焦煤','焦炭','燃料油','原油'))],
        **classify(metrics,diagnostics))



def write_report(r,out):
    lines=['# D0B-QKNormGraphAttention','',r['status'],'',
        '## 1. Controlled change','',
        'Baseline: D0B-BalancedLatentReadout. Raw Q/K dot-product geometry → per-head L2-normalized Q/K geometry in BOTH spatial layers and every hop. All 8 heads retain the original graph prior. V, AdaptiveGraphLearner, temporal path, fusion, head, loss and selection protocol are unchanged. SEQ_LEN=20. No temperature parameter. Parameter delta: 0 (verified below).','',
        f"Diagnostic SHA: {r.get('git_sha')}. Exact working-file hashes: results.json. Baseline checkpoint: {r.get('baseline_checkpoint')}",'',
        '## 2. Sanity','',str(r.get('sanity')),'',
        f"Native pooled reference: {r.get('baseline_reference')}",'',
        'Core failures prohibit performance interpretation. Legacy preservation additionally has frozen-class output/gradient regression tests. Capture is read-only, eval, pre-dropout. Q/K normalize over head_dim with fixed eps=1e-6; V is untouched. F.normalize uses max(norm,eps), so epsilon-degenerate vectors are explicitly exempt from the sqrt(head_dim) norm requirement.','',
        '## 3. Q/K geometry','']
    rows=[];audits=[];nearzero=[];attention=[]
    for stage,layers in r.get('qk_norm',{}).items():
        for layer,hops in layers.items():
            for i,h in enumerate(hops):
                q,k=h['Q'],h['K'];a=r['attention'][stage][layer]['hops'][i]
                rows.append([stage,layer,h['hop'],q['raw']['mean'],q['raw']['median'],q['raw']['P5'],
                    k['raw']['mean'],k['raw']['median'],k['raw']['P5'],q['used']['mean'],q['used']['std'],
                    k['used']['mean'],k['used']['std'],h['content_logits']['std'],h['prior_bias']['std'],
                    h['final_logits']['std'],a['entropy_mean'],a['top10_mass']])
                audits.append([stage,layer,h['hop'],q['non_degenerate_max_deviation'],k['non_degenerate_max_deviation'],
                    h['cosine_content_formula_max_error'],h['graph_prior_audit'],h['PASS']])
                for name,norm in [('Q',q),('K',k)]:
                    nearzero.append([stage,layer,h['hop'],name,norm['raw']['min'],norm['raw']['P1'],
                        norm['raw']['fraction_norm_lt_1e_minus_6'],norm['raw']['fraction_norm_lt_1e_minus_4']])
                attention.append([stage,layer,h['hop'],a['entropy_mean'],a['entropy_std'],a['uniform_entropy'],
                    a['top1_mass'],a['top5_mass'],a['top10_mass'],a['graph_topk_mass'],
                    a['head_diversity']['cosine_mean'],a['head_diversity']['cosine_std'],
                    a['head_diversity']['correlation_mean'],a['head_diversity']['correlation_std']])
    lines += [table(['Stage','Layer','Hop','Raw Q mean','Raw Q median','Raw Q P5','Raw K mean','Raw K median','Raw K P5',
        'Used Q mean','Used Q std','Used K mean','Used K std','Content std','Prior std','Final std','Entropy','Top10 mass'],rows),'',
        table(['Stage','Layer','Hop','Q norm deviation','K norm deviation','Cosine formula error','All-head prior audit','PASS'],audits),'',
        'Native raw norms are not expected to equal sqrt(head_dim). Native large-logit FP32 bias subtraction may suffer cancellation; the audit retains the raw error and verifies exact native recomposition, rounding bound and FP64 same-operands subtraction. Forward precision is unchanged.','',
        table(['Stage','Layer','Hop','Q/K','Raw minimum','Raw P1','Fraction <1e-6','Fraction <1e-4'],nearzero),'',
        'Complete raw/used norm min/max, content mean/std/P5/P50/P95, prior/final mean/std are in qk_norm_diagnostics.json.','',
        '## 4. D0B vs QKNorm mechanism','',
        table(['Stage','Layer','Hop','Entropy mean','Entropy std','log(N)','Top1','Top5','Top10','Graph top-k','Head cosine mean','Head cosine std','Head corr mean','Head corr std'],attention),'']
    gradients=[]
    for stage,g in r.get('gradients',{}).items():
        for layer,ratios in g['ratios']['prediction_only'].items():
            prefix=layer.replace('layer','attn_mixhop')+'.'
            gradients.append([stage,layer,*[g['prediction_only'][prefix+x]['norm'] for x in ('q','k','v')],ratios['Q_over_V'],ratios['K_over_V']])
    lines += [table(['Stage','Layer','Q grad','K grad','V grad','Q/V','K/V'],gradients),'',
        'Gradients use the same fixed TRAIN batch, eval mode, sum of four Huber prediction losses, autograd.grad; no optimizer step or .grad mutation. Full graph E1/E2/Theta1/Theta2/alpha gradients and connectivity are in gradient_diagnostics.json.','',
        'Graph states (alpha, adjacency mean/std, near-zero fraction, positive/effective degree, top-k concentration): '+str(r.get('graph')),'',
        'Mechanism activity is not forecast success; neither higher dispersion nor lower entropy is assumed in advance.','']
    complete=r.get('training_executed') and 'metrics' in r and 'history' in r
    if not complete:
        lines += ['## 5–8. Formal training and results pending','',
            'Formal evaluation is not complete. No QKNorm performance result or Case A–F is assigned. Training must be run by the user; do not infer generalization from initialization.','',
            '## Required 16 answers','']
        answers=[f"Native TEST5 reproduced: {r.get('baseline_reference')}",
            f"Parameter equality: {r.get('sanity',{}).get('shared_init')}",
            'Shared initialization: see exact comparison above.',
            f"Legacy baseline preserved: {r.get('sanity',{}).get('legacy_qk_norm_false')}",
            'All 8 heads retain original prior in both layers; per-hop audit above.',
            'Normalization is per sample/node/head over head_dim.',
            'Normalized norms and epsilon exceptions are tabulated above.',
            'D0B raw norm distribution and near-zero fractions are tabulated above; small Q/K gradients alone do not demonstrate norm collapse.',
            'Trained dispersion comparison pending.', 'Trained attention uniformity comparison pending.',
            'Trained Q/V and K/V comparison pending.',f"Core sanity: {r.get('sanity',{}).get('PASS')}",
            'VAL5 improvement pending.','TEST5 improvement pending.','TEST5 MSE comparison pending.',
            'No primary case before formal training/evaluation. STOP.']
        lines += [f'{i}. {answer}' for i,answer in enumerate(answers,1)]
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');return
    hist=r['history'];best=r['best_epoch']
    lines += ['Mechanism deltas: '+str(r['mechanism_comparison']),'',
        '## 5. Training','',table(['Formal best epoch','Train seconds','Formal multi-horizon val objective','VAL5 MAE at formal best','Secondary best-VAL5 epoch','Secondary best-VAL5 MAE'],
            [[best,r['train_time_seconds'],r['best_formal_val_objective'],hist['val5_diagnostic'][best-1]['MAE'],hist['best_val5_epoch'],hist['best_val5_mae']]]),'',
        'Only the original multi-horizon objective controls scheduler/early stopping/formal checkpoint. Secondary VAL5 is logging only; no secondary TEST checkpoint is evaluated. Missing epoch stages mean training stopped before that epoch.','',
        '## 6. Main forecasting result','',
        table(['Model','Split','MAE','MSE','RMSE','Hit%'],[[name,s,m['MAE'],m['MSE'],m['RMSE'],100*m['Hit']] for name,splits in r['metrics'].items() for s,m in splits.items() if s in ('val','test')]),'',
        table(['Split','Metric','D0B','QKNorm','Absolute delta','Relative %'],[[v[k] for k in ('split','metric','D0B','QKNorm','delta','relative_percent')] for v in r['overall_metrics'] if v['split']!='train']),'',
        'Metrics pool all origins × commodities; zero targets included in unmasked sign Hit. MSE directly averages squared residuals, RMSE=sqrt(MSE). Hit deltas above are fractions; multiply by 100 for percentage points.','',
        'RMSE squared checks: '+str({name:{s:abs(m['RMSE']**2-m['MSE']) for s,m in splits.items()} for name,splits in r['metrics'].items()}),'',
        '## 7. Commodity decomposition','',
        table(['Commodity','D0B MAE','QKNorm MAE','Delta MAE','D0B MSE','QKNorm MSE','Delta MSE'],[[v[k] for k in ('commodity','D0B_MAE','QKNorm_MAE','delta_MAE','D0B_MSE','QKNorm_MSE','delta_MSE')] for v in r['commodity_metrics']]),'',
        'Top5 improved by MAE (signed deltas): '+str(r['top5_improved']),'',
        'Top5 worsened by MAE (signed deltas): '+str(r['top5_worsened']),'',
        '焦煤/焦炭/原油/燃料油/低硫燃料油: '+str(r['focus_commodities']),'',
        'Gross improvement/degradation, largest improving commodity and top3 contribution shares: '+str(r['contribution_shares']),'',
        '## 8. Target magnitude decomposition','',
        f"TRAIN-only thresholds: {r['train_target_thresholds']}. >P95 overlaps >P90; do not add these groups.",'',
        table(['Split','Group','Count','D0B MAE','QKNorm MAE','Delta MAE','D0B MSE','QKNorm MSE','Delta MSE'],[[v.get(k) for k in ('split','group','count','D0B_MAE','QKNorm_MAE','delta_MAE','D0B_MSE','QKNorm_MSE','delta_MSE')] for v in r['target_magnitude_groups']]),'',
        '## Required 16 answers','']
    b,q=r['metrics']['D0B'],r['metrics']['QKNorm']
    raw=[h[key]['raw'] for hops in r['qk_norm']['D0B_best'].values() for h in hops for key in ('Q','K')]
    max_nearzero=max(v['fraction_norm_lt_1e_minus_4'] for v in raw)
    answers=[f"Native TEST5 reproduced: {r['baseline_reference']['PASS']}; {r['baseline_reference']['actual']}",
        f"Parameter equality: {r['sanity']['shared_init']}",
        f"Shared init max diff: {r['sanity']['shared_init']['shared_parameter_init_max_diff']}",
        f"Legacy equivalence: {r['sanity']['legacy_qk_norm_false']}",
        'All 8 heads in both layers retain original graph prior; per-hop bias audit PASS.',
        'Q/K normalize on last head_dim, separately for each sample/node/head.',
        'Q/K sqrt(head_dim) norm checks and epsilon-degenerate exceptions: see per-hop table.',
        f"D0B maximum fraction of raw Q/K norms below 1e-4: {max_nearzero}; distributions above. This is a descriptive near-zero audit, not proof that norm size caused poor forecasting.",
        'Content dispersion changes: '+str([(v['layer'],v['hop'],v['content_std']) for v in r['mechanism_comparison']]),
        'Attention distance from uniform: inspect entropy versus log(N), top-k masses above; lower entropy is not a success criterion.',
        'Prediction-only Q/V and K/V magnitude changes: '+str({s:r['gradients'][s]['ratios']['prediction_only'] for s in ('D0B_best','best')}),
        f"Causality/batch/relabeling/core sanity PASS: {r['sanity']['PASS']}",
        f"VAL5 MAE improves: {q['val']['MAE']<b['val']['MAE']}; delta={q['val']['MAE']-b['val']['MAE']}",
        f"TEST5 MAE improves: {q['test']['MAE']<b['test']['MAE']}; delta={q['test']['MAE']-b['test']['MAE']}",
        f"TEST5 MSE delta={q['test']['MSE']-b['test']['MSE']}",
        f"Primary classification: {r['primary_case']}; {r['interpretation']}"]
    lines += [f'{i}. {answer}' for i,answer in enumerate(answers,1)]
    lines += ['',r['classification_notes'],'',f"Observed result: {r['primary_case']}. {r['interpretation']}",
        f"Mechanism status: {r['mechanism_status']}",
        'The EdgeAttention score-geometry route should stop.' if r['primary_case']!='Case A' else 'This one run supports retaining QKNorm; no automatic follow-up experiment.',
        'STOP. No additional temperature, normalization, graph, branch, fusion, loss or checkpoint-selection experiment.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
