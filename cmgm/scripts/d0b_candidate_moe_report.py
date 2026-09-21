"""Fixed reporting only; oracle evidence never enters the model comparison table."""
import json
from cmgm.scripts.formal_main_ablation_report import table,csv_file
from cmgm.scripts.formal_v2_protocol import atomic_json


def report(r,out):
    ev=r.get('evaluation',{});ms=ev.get('metrics',{});test=ms.get('test',{}).get('5')
    routes=ev.get('routing',{});route=routes.get('test');expert=ev.get('experts',{});utility=ev.get('utility',{});oracle=ev.get('oracle',{})
    rows=[]
    for label,c in r['controls'].items():
        v=c['metrics']['test']['5'];rows.append(dict(Variant=label,MAE=v['MAE'],MSE=v['MSE'],RMSE=v['RMSE'],Hit_percent=100*v['Hit'],Source='existing formal artifact'))
    rows.append(dict(Variant='Candidate-Aware 2-Expert MoE',MAE=test['MAE'] if test else None,
        MSE=test['MSE'] if test else None,RMSE=test['RMSE'] if test else None,Hit_percent=100*test['Hit'] if test else None,
        Source='one seed42 run' if test else 'PENDING'))
    csv_file(out/'fusion_comparison.csv',rows)
    artifacts=dict(config=r['config'],source_hashes=r['source_hashes'],initialization_audit=r['initialization'],
        structural_sanity=dict(initial=r['sanity'],best=r.get('best_sanity'),data=r['data']),
        test_metrics=ms.get('test',{}),routing_diagnostics=routes,expert_diagnostics=expert,
        expert_utility_diagnostics=utility,oracle_diagnostic=oracle,reference_provenance=r['controls'],
        best_checkpoint_metadata=r.get('best_checkpoint_metadata',{}))
    if r.get('history'):
        artifacts.update(training_history=r['history'],routing_history=r['history']['candidate_routing_history'])
    for name,value in artifacts.items():atomic_json(out/(name+'.json'),value)
    for name in ('training_history','routing_history'):
        if not (out/(name+'.json')).exists():atomic_json(out/(name+'.json'),[])
    answers=[]
    for label in ('Adaptive Gated Fusion / Full D0B','3-Expert Dense MoE','TemporalOnly'):
        if not test:answers.append('PENDING: comparison with '+label+'.')
        else:
            ref=r['controls'][label]['metrics']['test']['5']['MAE'];delta=test['MAE']-ref
            answers.append(f'Against {label}: '+('lower' if delta<0 else 'equal' if delta==0 else 'higher')+f' TEST5 MAE; delta={delta:.10g}, relative delta={100*delta/ref:.6g}%.')
    if route:
        variation=f"std(pi_ST)={route['pi_ST_std']:.9g}, P90-P10={route['pi_ST_P90_minus_P10']:.9g}, max-min={route['pi_ST_range']:.9g}"
        answers += [f"TEST mean pi_T/pi_ST={route['mean']}.",
                    ('Routing is sample-invariant at evaluated precision. ' if route['pi_ST_range']==0 else 'Routing exhibits sample-dependent variation at the measured scale. ')+variation+'. No significance claim or artificial PASS cutoff.',
                    f"Interaction routing P10/P50/P90/P99={[route['quantiles'][str(q)][1] for q in (.1,.5,.9,.99)]}; min/max={[route['min'][1],route['max'][1]]}. These magnitudes describe whether high routing is substantial; hard occupancy alone is insufficient.",
                    f"Expert mean absolute representation difference={expert['mean_absolute_disagreement']:.9g}; cosine distribution={expert['cosine_similarity']}. Representation differences do not establish forecasting utility.",
                    f"Routing/advantage Spearman and Pearson: {utility['correlations']}."]
        groups={g['group']:g for g in utility['quantile_groups']}
        low,high=groups['Low'],groups['High']
        answers.append(f"High pi_ST fraction(A_ST>0)={high['fraction_A_ST_positive']:.9g}, low={low['fraction_A_ST_positive']:.9g}; high mean advantage={high['mean_A_ST']:.9g}. "+('Interaction wins more frequently in the high group.' if high['fraction_A_ST_positive']>low['fraction_A_ST_positive'] else 'High routing does not correspond to a higher interaction win fraction.')+' Rank ties retain chronological order and do not imply learned differentiation.')
        answers.append(f"EX-POST oracle TEST5 MAE={oracle['metrics']['MAE']:.10g}; candidate/MoE MAEs={oracle['reference_MAE']}; oracle-minus-reference={oracle['oracle_minus_reference_MAE']}. Uses TEST labels; not deployable.")
        refs=oracle['reference_MAE'];best_expert=min(refs['Temporal'],refs['Interaction']);gain=best_expert-oracle['metrics']['MAE']
        if gain>1e-12 and refs['LearnedMoE']>=best_expert:
            answers.append('The experts exhibit exploitable ex-post complementarity, but the learned Router does not successfully identify that complementarity from the available candidate representations. '+f'Oracle improves over the better single expert by {gain:.10g}, while learned MoE does not beat that expert. This is descriptive evidence only.')
        else:
            answers.append(f"Oracle improvement over the better single expert={gain:.10g}; learned MoE minus better expert={refs['LearnedMoE']-best_expert:.10g}. No claim that routing causally explains this gap; numerical tolerance 1e-12 only handles metric identity.")
        if gain<=1e-12:
            answers.append('The two expert candidates show limited sample-level predictive complementarity under the current architecture. Oracle has no measurable gain over the better single candidate at metric precision.')
        else:
            answers.append(f'Oracle offers {gain:.10g} MAE reduction ({100*gain/best_expert:.6g}%) over the better individual expert. Its practical magnitude is reported without a significance or human-chosen success cutoff.')
    else:
        answers += ['PENDING: TEST mean routing.','PENDING: routing variation.','PENDING: high interaction routing.',
                    'PENDING: expert representation differences.','PENDING: utility correlations.',
                    'PENDING: high vs low routing utility.','PENDING: oracle vs individual experts and learned MoE.',
                    'If oracle is useful but learned routing fails, report that available candidates did not support successful learned identification and STOP.',
                    'If oracle also shows little gain, report limited sample-level complementarity and STOP.']
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Experiment Objective','Do Temporal and Interaction experts possess sample-level complementarity, and can a candidate-aware router identify when interaction is useful? One predefined seed42 run; no tuning.')
    prior=r['controls']['3-Expert Dense MoE']
    section('Motivation from Previous 3-Expert MoE','Previous formal TEST5 and routing are read from audited artifacts:\n\n```json\n'+json.dumps(dict(metrics=prior['metrics']['test']['5'],routing=prior.get('routing',{}).get('test',{})),indent=2)+'\n```\n\nPrior reference provenance is preserved; no control is retrained.')
    section('Fixed D0B Components','Full input/preprocessing/splits/targets, TempWeighted spatial branch/AdaptiveGraph/EdgeAttnMixHop, temporal encoder/causal Transformer/Base RPE/Markov/K=3 latent recurrence/Balanced Readout/Switch KL, W_s/W_t and shared head keep native definitions. Shared weights train normally. Original gate and three-expert variants remain unchanged.')
    section('Two-Expert Architecture',r'''\[
t=W_th_t,\quad s=W_sh_s,\quad e_T=E_T(t),\quad e_{ST}=E_{ST}([s\Vert t]).
\]
T: residual Linear64→64/ReLU/Dropout(.1)/Linear64→64. ST: Linear128→64/ReLU/Dropout(.1)/Linear64→64, no residual. Both execute densely.
\[
h_{MoE}=\pi_T e_T+\pi_{ST}e_{ST},\qquad \hat Y=f_{head}(h_{MoE}).
\]
One unchanged Linear64→64/ReLU/Dropout(.3)/Linear64→96 head predicts B×4×24. No expert-specific training predictions.''')
    section('Candidate-Aware Router',r'''\[
u_T=LN_T(e_T),\quad u_{ST}=LN_{ST}(e_{ST}),\quad
r=[u_T\Vert u_{ST}\Vert|u_T-u_{ST}|\Vert u_T\odot u_{ST}],
\]
\[
[\pi_T,\pi_{ST}]=Softmax(R(r)).
\]
Router: Linear256→64/ReLU/Linear64→2. Last weight and bias initialize to zero, giving exact initial 50/50 routing. LayerNorm is router-only: fusion combines raw candidates. No warm-up, temperature, balance/entropy loss or target-aware routing.''')
    a=r['initialization']
    section('Initialization Audit',table([{k:a.get(k) for k in ('baseline_parameters','candidate_parameters','shared_parameter_count','new_parameter_count','delta_parameters','shared_max_abs_diff','mismatch_count','PASS')}])+'\nNative shared modules are constructed first, then the new variant removes its obsolete gate and creates experts/router. Initial predictions need not equal native D0B.')
    section('Structural Sanity',table([dict(Stage='initial',PASS=r['sanity'].get('PASS'),Device=r['sanity'].get('device')),dict(Stage='best',PASS=r.get('best_sanity',{}).get('PASS'))])+'\nShape, exact initial routing, candidate comparison wiring, raw-expert fusion, dense gradients, batch/single-sample consistency, commodity mapping/data fingerprints, and temporal prefix10 causality are audited. Initial RouterFirst/LayerNorm zero gradients are expected through the zero final weight; RouterFinal must receive nonzero gradient. No initial-uniform or nonzero-gradient requirement is imposed on trained routing concentration.')
    section('Training Protocol','```json\n'+json.dumps(r['config'],indent=2)+'\n```\n\n'+r'\[L=\sum_{h\in\{1,5,10,20\}}Huber_{.02}(\hat Y_h,Y_h)+L_{switch}.\]'+'\nNo MoE auxiliary loss. Checkpoint, scheduler and early stopping use the native prediction-only multi-horizon VAL Huber batch mean. VAL5 is secondary logging.\n\nBest checkpoint: '+json.dumps(r.get('best_checkpoint_metadata',{}),ensure_ascii=False))
    section('TEST Multi-Horizon Metrics',table([dict(Horizon=h,MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit']) for h,m in ms.get('test',{}).items()])+'\nPooled origins × 24 commodities; direct MSE, sqrt(MSE), unmasked sign agreement including zeros. Horizon indices use MULTI_HORIZONS lookup.')
    section('Fusion Comparison',table(rows)+'\nExisting values come from reports verified against checkpoint hashes/protocol/data. The oracle and individual expert diagnostics do not enter this formal table.')
    section('Routing Statistics',table([dict(Split=s,Mean=d['mean'],Std=d['std'],P10=d['quantiles']['0.1'],P90=d['quantiles']['0.9'],Min=d['min'],Max=d['max'],Entropy=d['routing_entropy'],HardOccupancy=d['hard_occupancy']) for s,d in routes.items()])+'\nAll requested quantiles are saved in routing_diagnostics.json. Final TRAIN includes every origin; epoch TRAIN uses native drop_last batches. No collapse verdict follows from entropy or occupancy alone.')
    section('Expert Representation Diagnostics','```json\n'+json.dumps(expert,indent=2)+'\n```')
    section('Expert Utility Alignment',r'''Only after freezing the best checkpoint and persisting formal predictions:
\[
\ell_T^{(i)}=MAE_{24}(Head(e_T^{(i)})_{5d},Y_{i,5d}),\quad
\ell_{ST}^{(i)}=MAE_{24}(Head(e_{ST}^{(i)})_{5d},Y_{i,5d}),
\]
\[
A_{ST}^{(i)}=\ell_T^{(i)}-\ell_{ST}^{(i)},\quad
\rho=Spearman(\pi_{ST},A_{ST}).
\]
Positive advantage means interaction has lower error. Shared frozen head, eval mode, no fit or update. Correlations are descriptive, not causal or oracle performance estimates.
'''+json.dumps(utility.get('correlations',{}),indent=2))
    section('Routing Quantile Analysis',table(utility.get('quantile_groups',[]))+'\nStable rank thirds by TEST pi_ST, with chronological tie breaking. TEST labels are used only to summarize post-hoc utility, never to set routing or select a model. Tied weights can yield group differences without routing differentiation.')
    section('Ex-Post Oracle Diagnostic',r'''\[
\hat Y_{oracle}^{(i)}=\begin{cases}Head(e_T^{(i)}),&\ell_T^{(i)}<\ell_{ST}^{(i)}\\Head(e_{ST}^{(i)}),&otherwise.\end{cases}
\]
**USES TEST LABELS. DIAGNOSTIC UPPER BOUND ONLY. NOT DEPLOYABLE. NOT A VALID PREDICTIVE MODEL.**
'''+ '\n```json\n'+json.dumps(oracle,indent=2)+'\n```')
    interpretation='PENDING: no formal training/evaluation result is available.' if not test else '\n\n'.join(answers[:3]+answers[10:])
    section('Interpretation',interpretation+'\n\nRouting weights are model-internal mixture coefficients, not causal attribution scores. Strong temporal preference can be appropriate; assess it jointly with variation and utility. Accept results and STOP regardless of outcome.')
    section('Required Answers','\n\n'.join(f'Q{i}. {answer}' for i,answer in enumerate(answers,1)))
    section('Limitations','Single seed42 fixed-design experiment; no statistical significance or multi-seed claim. Compared with three-expert MoE, expert count/router inputs/normalization/initialization and prescribed regularization all differ; any gain cannot be uniquely attributed to one of them. Expert-only shared-head predictions are post-hoc interventions, not independently trained predictors. Oracle is restricted to two candidates and sample-level 5d MAE; it is not a bound on arbitrary nonlinear MoE forecasts. No automatic follow-up experiment.\n\nSTOP')
    (out/'FINAL_REPORT.md').write_text('# D0B-CandidateAware2ExpertMoE\n\nStatus: '+r['status']+'\n\n'+''.join(sections),encoding='utf-8')
