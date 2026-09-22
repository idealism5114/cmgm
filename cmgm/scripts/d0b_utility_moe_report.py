"""Reporting only: forecasting, expert complementarity and dynamic-router value."""
import json
from cmgm.scripts.formal_main_ablation_report import table,csv_file
from cmgm.scripts.formal_v2_protocol import atomic_json


def metric_row(label,m):
    return dict(Variant=label,MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit'])


def report(r,out):
    ev=r.get('evaluation',{});ms=ev.get('metrics',{});test=ms.get('test',{}).get('5')
    routes=ev.get('routing',{});route=routes.get('test');alignment=ev.get('alignment',{});oracle=ev.get('oracle',{})
    experts=ev.get('expert_metrics',{});fixed=ev.get('fixed_router',{});regime=ev.get('regime',{})
    rows=[metric_row(n,c['metrics']['test']['5']) for n,c in r['controls'].items()]
    rows.append(metric_row('Utility-Routed Predictive MoE',test) if test else dict(Variant='Utility-Routed Predictive MoE',MAE=None,MSE=None,RMSE=None,Hit_percent=None))
    csv_file(out/'fusion_comparison.csv',rows)
    mechanism=[]
    if test:
        for label,m in [('Temporal Expert only',experts['Temporal']['5']),('ST Expert only',experts['Interaction']['5']),
                        ('Fixed TRAIN-Mean Mixture',fixed['metrics']['5']),('Dynamic Utility Router',test),
                        ('EX-POST ORACLE: TEST LABELS, NOT A MODEL',oracle['metrics']['5'])]:
            mechanism.append(metric_row(label,m))
    csv_file(out/'mechanism_comparison.csv',mechanism or [dict(Variant='PENDING',MAE=None)])
    artifacts=dict(config=r['config'],source_hashes=r['source_hashes'],initialization_audit=r['initialization'],
        structural_sanity=dict(initial=r['sanity'],best=r.get('best_sanity'),data=r['data']),
        best_checkpoint_metadata=r.get('best_checkpoint_metadata',{}),test_metrics=ms.get('test',{}),
        expert_metrics=experts,expert_diagnostics=ev.get('expert_diagnostics',{}),routing_diagnostics=routes,
        routing_utility_alignment=alignment,regime_routing_diagnostics=regime,fixed_router_control=fixed,
        oracle_diagnostic=oracle,reference_provenance=r['controls'])
    if r.get('history'):
        h=r['history'];rh=h['utility_routing_history']
        artifacts.update(training_history=h,routing_history=rh,utility_training_history=[dict(epoch=row['epoch'],**row['train']) for row in rh])
    for k,v in artifacts.items():atomic_json(out/(k+'.json'),v)
    for name in ('training_history','routing_history','utility_training_history'):
        if not (out/(name+'.json')).exists():atomic_json(out/(name+'.json'),[])
    answers=[]
    for label in ('Candidate-Aware 2-Expert Representation MoE','Adaptive Gated Fusion / Full D0B','TemporalOnly'):
        if not test:answers.append('PENDING: comparison with '+label+'.')
        else:
            ref=r['controls'][label]['metrics']['test']['5']['MAE'];delta=test['MAE']-ref
            answers.append(f'Compared with {label}: '+('lower' if delta<0 else 'equal' if delta==0 else 'higher')+f' TEST5 MAE; delta={delta:.10g}, relative delta={100*delta/ref:.6g}%.')
    if test:
        delta=fixed['dynamic_minus_fixed']['5']['MAE'];rho=alignment['primary_correlations']['Spearman']
        gain=min(oracle['reference_mean_huber'][k] for k in ('Temporal','Interaction'))-oracle['mean_four_horizon_huber']
        answers += [f"Standalone TEST5 predictive experts: Temporal={experts['Temporal']['5']}, Interaction={experts['Interaction']['5']}. All horizons appear below and in expert_metrics.json.",
            f"Under four-horizon Huber, Temporal wins {alignment['fraction_T_better']:.8g}, Interaction wins {alignment['fraction_ST_better']:.8g}, ties {alignment['fraction_ties']:.8g} of origins.",
            f"Oracle four-horizon Huber={oracle['mean_four_horizon_huber']:.10g}; improvements over individual experts={oracle['huber_improvement_over_experts']}. Oracle TEST5 MAE={oracle['metrics']['5']['MAE']:.10g}; selector was not optimized for 5d MAE.",
            f"TEST std(pi_ST)={route['pi_ST_std']:.9g}; P10/P50/P90={[route['quantiles'][str(q)][1] for q in (.1,.5,.9)]}; mean={route['mean']}, range={route['pi_ST_range']:.9g}. Variation magnitude alone is not success.",
            f"Primary four-horizon utility correlations: {alignment['primary_correlations']}. Supplemental 5d MAE: {alignment['five_day_MAE_correlations']}."]
        groups={g['group']:g for g in alignment['groups']};lo,hi=groups['Low'],groups['High']
        answers.append(f"High routing ST-win fraction={hi['fraction_ST_better']:.8g}, Low={lo['fraction_ST_better']:.8g}; high mean A={hi['mean_A']:.10g}. "+('ST wins more often in the high group.' if hi['fraction_ST_better']>lo['fraction_ST_better'] else 'Higher routing does not correspond to a higher ST-win frequency.')+' Equal-weight rank ties are not evidence of routing variation.')
        answers.append(f"Dynamic minus Fixed TRAIN-Mean TEST5 MAE={delta:.10g}. "+('Under the fixed protocol, sample-dependent routing provides predictive value beyond replacing the Router with its global TRAIN-average mixture.' if delta<0 else 'The learned dynamic Router does not outperform an equivalent fixed global mixture.'))
        answers.append(f"Descriptive regime groups={regime['groups']}; pi_ST/regime-entropy correlations={regime['entropy_correlation']}. No causal attribution.")
        if delta<0 and rho is not None and rho>0 and gain>1e-12:
            answers.append('Under this fixed single-run protocol, positive routing–utility alignment, expert winner switching and lower error than Fixed TRAIN-Mean provide evidence for conditional expert routing. This is not statistical significance or universal superiority.')
        else:
            answers.append('The full evidence required for true conditional expert routing is not established; mainly global expert mixing is the supported conservative interpretation. Forecast gains, if any, must be separated from dynamic routing value.')
        if gain<=1e-12:
            interpretation='The two predictive experts show limited sample-level winner switching, reducing the potential benefit of dynamic routing. Oracle gain is negligible at metric precision (1e-12).'
        elif delta<0:
            interpretation='The two experts exhibit sample-level complementarity, and the learned Router captures part of this conditional structure relative to the fixed TRAIN-mean intervention. The sign and magnitude of utility alignment must also be considered.'
        else:
            interpretation='Expert complementarity exists, but the Router does not successfully identify it from observable model states well enough to beat the fixed TRAIN-mean mixture.'
        interpretation+=f' Oracle gain over the better expert in four-horizon Huber is {gain:.10g}; dynamic-minus-fixed TEST5 MAE is {delta:.10g}. No subjective “large variance” success criterion.'
        previous=r['controls']['Candidate-Aware 2-Expert Representation MoE']['metrics']['test']['5']['MAE']
        if test['MAE']<previous and delta>=0:
            interpretation+=' The dual-expert architecture improves prediction, while the incremental value of sample-dependent routing is not supported.'
    else:
        answers+=['PENDING: standalone predictive experts.','PENDING: sample-level expert winners.','PENDING: oracle complementarity.',
                  'PENDING: routing variability/quantiles.','PENDING: routing–utility correlations.','PENDING: high-routing utility.',
                  'PENDING: dynamic versus fixed TRAIN-mean routing.','PENDING: regime-routing description.',
                  'PENDING: conditional routing versus mainly global mixing. No conclusion before evaluation.']
        interpretation='PENDING: no formal Utility-Routed MoE training or evaluation has been executed.'
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Experiment Objective','Can supervised expert utility produce useful sample-dependent routing between Temporal and Spatial–Temporal predictive experts? Forecasting success, expert complementarity and Router success are distinct outcomes.')
    previous=r['controls']['Candidate-Aware 2-Expert Representation MoE']
    section('Motivation from Candidate 2-Expert MoE','Audited prior control:\n\n```json\n'+json.dumps(dict(TEST5=previous['metrics']['test']['5'],routing=previous.get('routing',{}).get('test',{})),indent=2)+'\n```\nThe prior representation mixture is preserved. No historical controls are retrained.')
    section('Latest GitHub Base / Source Hashes','Base Git SHA: `'+r.get('git_sha','unknown')+'`.\n\n'+json.dumps(r.get('base_provenance',{}),indent=2)+'\n\nCurrent experiment source hashes:\n\n```json\n'+json.dumps(r['source_hashes'],indent=2)+'\n```')
    section('Fixed D0B Components','Native data/window20/full nodes/features21/targets/splits/preprocessing; spatial TempWeighted/AdaptiveGraph/EdgeAttnMixHop/type pooling; temporal market encoder/Transformer/Base RPE/Markov/K=3 latent transitions/microstate/Balanced Readout/Switch KL; gcn_proj/lstm_proj unchanged. Native head becomes Temporal head, with an independent initialized-identically Interaction copy. Architecture definitions are frozen; active weights still train under the main objective.')
    section('Predictive Expert Architecture',r'''\[
s=W_sh_s,\quad t=W_th_t,\quad e_T=E_T(t),\quad e_{ST}=E_{ST}([s\Vert t]),
\]
\[
\hat Y_T=P_T(e_T),\quad\hat Y_{ST}=P_{ST}(e_{ST}),\quad
\hat Y=\pi_T\hat Y_T+\pi_{ST}\hat Y_{ST}.
\]
T: residual Linear64/ReLU/Dropout(.1)/Linear64. ST: Linear128→64/ReLU/Dropout(.1)/Linear64, no residual. Each prediction head is Linear64→64/ReLU/Dropout(.3)/Linear64→96. Heads initialize identically via deepcopy but do not share parameters. No shared head after representation mixing.''')
    section('Utility-Aware Router',r'''\[
u_T=LN_T(e_T),\quad u_{ST}=LN_{ST}(e_{ST}),
\]
\[
r=[u_T\Vert u_{ST}\Vert|u_T-u_{ST}|\Vert u_T\odot u_{ST}\Vert p_T\Vert|p_T-p_T^{prior}|\Vert H(p_T)],
\]
\[
[\pi_T,\pi_{ST}]=Softmax(R(r)).
\]
Router-only normalization; Linear263→64/ReLU/Linear64→2/softmax. Final weight/bias zero: exact initial 50/50. Sample-level dense routing; no balance, entropy objective, temperature, warm-up or usage constraints.''')
    section('Regime Context','Use only final observed timestep posterior/prior from the unchanged temporal branch. Posterior, prior, absolute gap and entropy are detached before entering Router. Regime context has 3+3+1 dimensions; prior itself is used to compute the gap, not concatenated as an extra feature. Fusion utility gradients cannot update Markov inference.')
    section('Utility Supervision Objective',r'''\[
\ell_{k,i}=\sum_h\frac1{24}\sum_c Huber_{.02}(\hat Y_{k,i,h,c},Y_{i,h,c}),\quad
a_i=\frac{\ell_{T,i}-\ell_{ST,i}}{\ell_{T,i}+\ell_{ST,i}+10^{-8}},
\]
\[
q_{ST,i}=(1+a_i)/2,\quad q_{T,i}=1-q_{ST,i},\quad q=stopgrad(q),
\]
\[
L_{route}=\frac1B\sum_i KL(q_i\Vert\pi_i),\quad
s_{route}=stopgrad\left[\frac1B\sum_i(\ell_{T,i}+\ell_{ST,i})/2\right],
\]
\[
L_{route}^{scaled}=s_{route}L_{route},\qquad L=L_{pred}+L_{switch}+L_{route}^{scaled}.
\]
All four horizons, equal weight, unreduced Huber with commodity mean then horizon sum. Epsilon 1e-8 in utility and KL; no extra coefficient. TRAIN targets only. Crucially, detached q alone is insufficient: the utility objective recomputes the same deterministic Router probabilities on detached candidates/context. Thus auxiliary gradients update only Router parameters (including its LayerNorms); primary prediction gradients train experts/heads/shared branches normally. No expert loss is added separately.''')
    a=r['initialization']
    section('Initialization Audit',table([{k:a.get(k) for k in ('baseline_parameters','utility_parameters','shared_parameter_count','new_parameter_count','delta_parameters','shared_max_abs_diff','mismatch_count','head_init_max_diff','heads_independent','PASS')}])+'\nAll native shared modules are constructed first. Only new variant gate_fc is removed. Deepcopy consumes no RNG; new expert/router construction cannot alter native shared initialization.')
    section('Structural Sanity',table([dict(Stage='initial',PASS=r['sanity'].get('PASS'),Device=r['sanity'].get('device')),dict(Stage='best',PASS=r.get('best_sanity',{}).get('PASS'))])+'\nStructural report records shapes, mixture identity, 50/50 initialization, final causal regime context, detached q/scale/context, route-only gradient paths, prediction gradients, batch/single-sample consistency and prefix10 temporal causality. Router final-layer zero initialization can initially zero earlier router gradients; it must not be misclassified as a disconnected path.')
    section('Training Protocol','```json\n'+json.dumps(r['config'],indent=2)+'\n```\n\nTRAIN: prediction+native Switch KL+auto-scaled utility KL. VAL selection/early stop/scheduler: prediction-only four-horizon Huber. No routing auxiliary objective is computed in validation training logic. NaN/Inf invalidates the implementation run.\n\nBest checkpoint metadata:\n'+json.dumps(r.get('best_checkpoint_metadata',{}),ensure_ascii=False))
    section('TEST Multi-Horizon Metrics',table([dict(Horizon=h,**metric_row('Dynamic Utility MoE',m)) for h,m in ms.get('test',{}).items()])+'\nPooled all origins × 24 commodities, direct MSE and sqrt(MSE), unmasked sign Hit including zeros; horizon lookup via MULTI_HORIZONS.')
    section('Formal Model Comparison',table(rows)+'\nAll historical numbers are read from verified reports/checkpoints. Oracle is excluded from this table.')
    expert_rows=[]
    for n,hm in experts.items():
        for h,m in hm.items():expert_rows.append(dict(Horizon=h,**metric_row(n,m)))
    section('Predictive Expert Performance',table(expert_rows)+'\nTEST prediction disagreement and representation diagnostics:\n```json\n'+json.dumps(ev.get('expert_diagnostics',{}),indent=2)+'\n```')
    section('Routing Statistics',table([dict(Split=s,Mean=d['mean'],Std=d['std'],P10=d['quantiles']['0.1'],P50=d['quantiles']['0.5'],P90=d['quantiles']['0.9'],Entropy=d['routing_entropy'],HardOccupancy=d['hard_occupancy']) for s,d in routes.items()])+'\nSoft probabilities and population std/quantiles describe routing; neither large variance nor hard occupancy is a success criterion. TRAIN epoch logs retain native drop_last; final TRAIN diagnostics include every origin.')
    section('Routing–Utility Alignment','Primary correlations use four-horizon per-origin Huber advantage, identical to TRAIN utility definition; supplemental correlations use per-origin 5d MAE. Positive advantage means ST is better.\n\n'+json.dumps({k:alignment.get(k) for k in ('primary_correlations','five_day_MAE_correlations')},indent=2)+'\n\n'+table(alignment.get('groups',[]))+'\nGroups are stable pi_ST rank thirds with chronological tie handling; tied routing cannot itself establish sample differentiation.')
    fixed_rows=[]
    for h,m in fixed.get('metrics',{}).items():
        fixed_rows += [dict(Horizon=h,**metric_row('Fixed TRAIN-Mean',m)),dict(Horizon=h,**metric_row('Dynamic',ms['test'][h]))]
    section('Dynamic vs Fixed Router','Weights are frozen from mean learned probabilities over ALL TRAIN origins at the best checkpoint. No TRAIN labels, VAL tuning, TEST labels or refitting determine the weights.\n\n'+table(fixed_rows)+'\n```json\n'+json.dumps(fixed,indent=2)+'\n```')
    section('Regime–Routing Diagnostics','```json\n'+json.dumps(regime,indent=2)+'\n```\nThese are descriptive groups by final causal argmax(p_T) and entropy correlation, not causal effects.')
    section('Oracle Diagnostic','**EX-POST ORACLE ONLY. USES TEST LABELS. NOT DEPLOYABLE. NOT A MODEL. NOT FOR MODEL SELECTION.**\n\nChoose lower per-origin four-horizon Huber expert; ties select ST. Therefore its 5d metrics are NOT a 5d-specific oracle bound.\n\n'+table(mechanism)+'\n```json\n'+json.dumps(oracle,indent=2)+'\n```')
    section('Mechanism Interpretation',interpretation+'\n\nPredictive success, expert complementarity and conditional routing value must be assessed separately. Lower final MAE does not validate dynamic routing if a fixed TRAIN-mean mixture performs equally well or better.')
    section('Required Answers','\n\n'.join(f'Q{i}. {answer}' for i,answer in enumerate(answers,1)))
    section('Limitations','One fixed seed42 run. No significance or multi-seed claim. Compared with representation MoE, head independence, prediction-level mixture, regime context and prescribed supervision change together; the experiment does not isolate each component. Router coefficients are not causal contribution scores. The oracle is restricted to two candidate predictions and uses TEST labels. Fixed routing is an inference intervention on the same trained checkpoint, not a retrained model. Accept all outcomes and STOP; no automatic retry, new expert, loss or router.')
    (out/'FINAL_REPORT.md').write_text('# D0B-UtilityRoutedMoE\n\nStatus: '+r['status']+'\n\n'+''.join(sections),encoding='utf-8')
