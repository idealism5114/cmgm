"""Report both Router controls without conflating architecture with routing value."""
import json
from cmgm.scripts.formal_main_ablation_report import table,csv_file
from cmgm.scripts.formal_v2_protocol import atomic_json
DYNAMIC='Candidate-Aware 2-Expert Representation MoE'
STATIC='Candidate-Aware 2-Expert Global Static Mixture'


def report(r,out):
    ev=r.get('evaluation',{});iv=r.get('intervention',{});init=r['initialization']
    static=ev.get('metrics',{}).get('test',{});dyn=r['controls'][DYNAMIC]['metrics']['test']
    rows=[]
    for label,c in r['controls'].items():
        m=c['metrics']['test']['5']
        rows.append(dict(Variant=label,**{k:m[k] for k in ('MAE','MSE','RMSE')},Hit_percent=100*m['Hit'],Status='audited existing control'))
    m=static.get('5',{})
    rows.append(dict(Variant=STATIC,**{k:m.get(k) for k in ('MAE','MSE','RMSE')},Hit_percent=100*m['Hit'] if m else None,Status='complete' if m else 'PENDING: user launches one formal fit'))
    csv_file(out/'router_ablation_comparison.csv',rows)
    multi=[]
    for label,metrics in [(DYNAMIC,dyn),(STATIC,static),('Same-checkpoint Frozen TRAIN-Mean',iv.get('fixed_mean',{}))]:
        for h,values in metrics.items():multi.append(dict(Variant=label,Horizon=h,**values))
    csv_file(out/'multi_horizon_comparison.csv',multi)
    artifacts=dict(config=r['config'],source_hashes=r['source_hashes'],shared_initialization_audit=init,
        expert_initialization_audit=r.get('expert_initialization',{}),structural_sanity=dict(initial=r['sanity'],best=r.get('best_sanity'),data=r['data']),
        best_checkpoint_metadata=r.get('best_checkpoint_metadata',{}),test_metrics=static,
        dynamic_reference_provenance=r['controls'],dynamic_routing_statistics=iv.get('routing',{}),
        same_checkpoint_fixed_mean_metrics=iv)
    if r.get('history'):
        artifacts.update(training_history=r['history'],global_weight_history=r['history']['global_weight_history'])
    for name,value in artifacts.items():atomic_json(out/(name+'.json'),value)
    for name in ('training_history','global_weight_history'):
        if not (out/(name+'.json')).exists():atomic_json(out/(name+'.json'),[])
    weights=ev.get('best_global_weights',{});delta=m['MAE']-dyn['5']['MAE'] if m else None
    relative=100*delta/m['MAE'] if m else None
    within=iv.get('delta_within')
    # No unregistered numerical threshold for "approximately equal" is used to force a case.
    interpretation='PENDING: the retrained static control is required before answering whether input dependence adds value beyond both controls.'
    if m and iv:
        if delta<0:
            interpretation='Case D: The input-dependent Router is not necessary for the observed predictive gains under this protocol. The retrained static control has lower TEST5 MAE.'
        elif delta>0 and within>0:
            interpretation='Case A by strict metric ordering: Dynamic has lower MAE than both controls. Practical strength depends on the absolute deltas below; strict ordering alone does not establish material predictive value.'
        elif delta>0 and within==0:
            interpretation='Case B: The dynamic architecture trains a stronger representation system than the global-mixture control, but the final sample-to-sample routing variation itself contributes little at inference.'
        elif delta==0 and within==0:
            interpretation='Case C: Router input dependence is not supported as a material performance contributor.'
        else:
            interpretation='The two controls give mixed evidence: report their exact deltas independently. Dynamic routing is not supported as improving over both controls.'
    if iv:
        interpretation+=f"\n\nFrozen-checkpoint delta_within={within:.12g} ({iv['relative_delta_percent']:.9g}%). "+('FixedMean has lower error, so sample dependence does not improve this checkpoint\'s TEST5 MAE.' if within<0 else 'Dynamic has lower error at the evaluated precision.' if within>0 else 'The learned router is formally input-dependent, but its sample-to-sample variation contributes little predictive value at the frozen best checkpoint.')
    if m:interpretation+=f'\n\nRetrained-control delta_router={delta:.12g} ({relative:.9g}%).'
    interpretation+='\n\nNo practical-equivalence cutoff was preregistered. Small absolute differences must be described as small, even if strict metric ordering assigns Case A/D. Near-equivalence interpretations B/C need magnitude-aware reading rather than a fabricated significance threshold.'
    answers=[
        str(weights) if weights else 'PENDING: best static checkpoint alpha.',
        str(iv['weights']) if iv else 'PENDING: full TRAIN-mean Dynamic pi.',
        f'delta_router=Static-Dynamic={delta:.12g}; relative={relative:.9g}% (denominator Static).' if m else 'PENDING: static formal fit.',
        f'delta_within=FixedMean-Dynamic={within:.12g}; '+('Dynamic lower.' if within>0 else 'FixedMean lower.' if within<0 else 'Identical MAE.') if iv else 'PENDING: frozen intervention.',
        str({h:v['MAE'] for h,v in iv.get('fixed_minus_dynamic',{}).items()}) if iv else 'PENDING: four-horizon intervention.',
        interpretation,
        'If the reported within-checkpoint gap is practically negligible: The learned router is formally input-dependent, but its sample-to-sample variation contributes little predictive value at the frozen best checkpoint. No practical tolerance is invented here.',
        'Only if both controls show a clear meaningful error increase: Under the fixed single-run protocol, input-dependent routing contributes predictive value beyond global expert mixing. Strictly positive tiny deltas alone are not strong evidence.',
        ('PENDING: both controls must be available to distinguish dynamic conditional routing from dual-expert representation specialization with near-global soft mixing.' if not m else
         'Use both exact contrasts above: evidence against input dependence favors dual-expert representation specialization with near-global soft mixing; clear gains over both favor dynamic conditional routing. This interpretation does not change the model\'s dense soft MoE identity.')]
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Experiment Objective','Does input-dependent Candidate-Aware routing provide predictive value beyond a learned global mixture of exactly the same experts? One static seed42 fit plus one frozen-checkpoint inference intervention; no optimization campaign.')
    section('Motivation','The completed Dynamic Candidate model has small routing variation. Routing standard deviation is descriptive; performance against both static controls is the evidence of interest. Controls are read from audited formal artifacts, not prompt literals.')
    section('Dynamic Candidate Reference','```json\n'+json.dumps({k:r['controls'][DYNAMIC][k] for k in ('checkpoint','checkpoint_sha256','source_report','source_report_sha256','best_epoch')},indent=2)+'\n```\n'+table([dict(Horizon=h,**v) for h,v in dyn.items()])+'\nFresh reproduction errors and frozen-state/hash checks are in same_checkpoint_fixed_mean_metrics.json.')
    section('Static Global-Mixture Architecture',r'''\[
t=W_th_t,\quad s=W_sh_s,\quad e_T=t+MLP_T(t),\quad e_{ST}=MLP_{ST}([s\Vert t]).
\]
T: Linear64→64/ReLU/Dropout(.1)/Linear64→64 plus residual. ST: Linear128→64/ReLU/Dropout(.1)/Linear64→64, no residual.
\[
\alpha=Softmax(a),\quad a\in\mathbb R^2,\quad a_{init}=[0,0],\quad
h_i^{static}=\alpha_Te_{T,i}+\alpha_{ST}e_{ST,i},\quad \hat Y=Head(h_i^{static}).
\]
Alpha is exactly shape (2,), shared by all samples/splits. Candidate comparison features, router-only LayerNorm, and router MLP are absent. No extra prediction heads, warm-up or mixture regularization.''')
    section('Fixed Components','Original full-node input/preprocessing/splits/targets, spatial branch, temporal branch, Balanced Readout, W_s/W_t, experts and shared head definitions are preserved. Weights train normally. Candidate model source is immutable. Utility-Routed predictive MoE is not used. TRAIN/VAL/TEST fingerprints and commodity mapping are included in structural_sanity.json.')
    section('Initialization Audit',table([{k:init.get(k) for k in ('dynamic_parameters','static_parameters','parameter_delta','dynamic_router_parameters','static_router_parameters','shared_max_abs_diff','mismatch_count','PASS')}])+'\n```json\n'+json.dumps(r.get('expert_initialization',{}),indent=2)+'\n```\nNative shared modules first, identical Temporal/Interaction expert construction next, zero logits last. No dummy parameter matching.')
    section('Structural Sanity',table([dict(Stage='initial',PASS=r['sanity'].get('PASS')),dict(Stage='restored best',PASS=r.get('best_sanity',{}).get('PASS'))])+'\nIncludes vector alpha/input invariance, shapes, expert/projection/shared-head wiring, initial 50/50 identity, batch permutation, single sample, prefix10 temporal causality and prediction gradient to global logits. No optimizer step during audit.')
    section('Training Protocol','```json\n'+json.dumps(r['config'],indent=2)+'\n```\n'+r'\[L=\sum_{h\in\{1,5,10,20\}}Huber_{.02}(\hat Y_h,Y_h)+L_{switch}.\]'+'\nGlobal logits receive prediction gradients only (plus the unchanged optimizer weight decay). Selection/scheduler/early stopping: prediction-only multi-horizon validation Huber. VAL5 is logging only. No retries based on performance.')
    section('Learned Global Weights','Best weights: '+json.dumps(weights)+'\n\nEach epoch logs the same end-of-epoch logits/alpha under TRAIN and VAL in global_weight_history.json. Initial alpha is [.5,.5]. Best checkpoint restores the selected alpha; full-split evaluation verifies it stays identical.')
    section('TEST Multi-Horizon Metrics',table([dict(Horizon=h,**v) for h,v in static.items()])+'\nDirect pooled MAE/MSE, RMSE=sqrt(MSE), unmasked sign agreement including zeros; Hit is a fraction in JSON and percent in comparison CSV. Horizon lookup follows MULTI_HORIZONS.')
    section('Dynamic vs Retrained Static',table(rows)+'\n'+(f'Delta_router=Static-Dynamic={delta:.12g}; 100×delta/Static={relative:.9g}%.' if m else 'PENDING: static training is not yet run.'))
    section('Same-Checkpoint Fixed-Mean Intervention',r'''\[
\pi_i=Softmax(R(e_{T,i},e_{ST,i})),\quad h_i^{dyn}=\pi_{T,i}e_{T,i}+\pi_{ST,i}e_{ST,i},
\]
\[
\bar\pi^{TRAIN}=\frac1{N_{train}}\sum_i\pi_i,\quad
h_i^{mean}=\bar\pi_T^{TRAIN}e_{T,i}+\bar\pi_{ST}^{TRAIN}e_{ST,i}.
\]
Full TRAIN origins, eval mode, labels never used in mean extraction. Mean saved before TEST evaluation. Same immutable checkpoint/experts/backbone/head; representation mixing before the shared nonlinear head. No fit, fine-tuning or validation optimization.
'''+ '\nTRAIN count/mean: '+str((iv.get('train_count'),iv.get('weights')))+'\n\n'+table([dict(Horizon=h,Dynamic_MAE=iv['dynamic'][h]['MAE'],FixedMean_MAE=iv['fixed_mean'][h]['MAE'],**{'Delta_'+k:v for k,v in d.items()}) for h,d in iv.get('fixed_minus_dynamic',{}).items()]))
    section('Router Value Analysis',interpretation)
    section('MoE Interpretation','Dynamic remains a dense soft MoE: two independent experts, input-conditioned softmax router and weighted expert representations. Whether input dependence helps empirically is a separate question. Routing weights are mixture coefficients, not causal contribution scores. Neither small routing variance nor the MoE label answers the performance question.')
    section('Required Answers','\n\n'.join(f'Q{i}. {a}' for i,a in enumerate(answers,1)))
    section('Limitations','Seed42 controlled single runs, no statistical significance or robustness claim. Different router parameterizations consume different random draws and can lead to different training trajectories despite exact shared/expert initialization. The frozen-checkpoint intervention isolates inference sample dependence. Retained expert nonlinearities mean representation mixing is not a prediction-level ensemble. No new router/seed/loss/temperature or follow-up experiment.\n\nSTOP')
    (out/'FINAL_REPORT.md').write_text('# D0B Router Ablation: Dynamic vs Global Mixture\n\nStatus: '+r['status']+'\n\n'+''.join(sections),encoding='utf-8')
