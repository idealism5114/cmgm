"""Final Candidate-MoE main ablation tables; historical Adaptive-Gate results excluded."""
import csv
import json
from cmgm.models.formal_d0b_main_ablation import NAMES,DEFINITIONS,FULL,GLOBAL,GLOBAL_NAMES,GLOBAL_FULL,GLOBAL_DEFINITIONS,TS_VARIANT,TS_FULL,TS_NAMES,TS_DEFINITIONS
from cmgm.scripts.formal_v2_protocol import atomic_json
LEVELS=dict(zip(NAMES,('Spatial',)*3+('Temporal',)*5+('Fusion',)*2+('Full',)))
QUESTIONS=(
    'Removing spatial temporal weighting', 'Removing adaptive graph learning',
    'Replacing EdgeAttnMixHop by standard single-hop GCN', 'Removing relative position encoding',
    'Fixing uniform regime routing', 'Sharing regime transitions',
    'Removing microstate readout contribution', 'Bypassing Balanced Readout normalizations',
    'Replacing the whole Candidate MoE by Linear([s||t])',
    'Replacing candidate-aware routing by a learned global mixture')


def csv_file(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row)) or ['Status']
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fmt(v):
    if v is None:return 'Pending'
    if isinstance(v,float):return f'{v:.10g}'
    return str(v).replace('|',';').replace('\n',' ')


def table(rows,fields=None):
    if not rows:return 'Not available.\n'
    fields=fields or list(rows[0])
    return '| '+' | '.join(fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'+''.join('| '+' | '.join(fmt(row.get(k)) for k in fields)+' |\n' for row in rows)


def evidence(delta,relative):
    if delta is None:return 'PENDING','Awaiting the formal run; no conclusion.'
    if abs(relative)<.1:return 'MIXED','Near-tie under the single-run protocol; no strong mechanism conclusion.'
    if delta>0:return 'YES','The single-run ablation is consistent with a beneficial forecasting contribution from this mechanism.'
    return 'NO','The single-run experiment does not support a positive net forecasting contribution from this mechanism under the current protocol.'


def report(r,out):
    if r.get('protocol',{}).get('formal_reference')==TS_VARIANT:return ts_global_report(r,out)
    if r.get('protocol',{}).get('formal_reference')==GLOBAL:return global_temporal_report(r,out)
    results=r['results'];full=results.get(FULL,{}).get('metrics',{}).get('test',{}).get('5')
    rows=[];contribution=[];runtime=[];allh=[];commodity=[];status=[]
    for n in NAMES:
        entry=results.get(n,{});metric=entry.get('metrics',{}).get('test',{}).get('5',{})
        check=entry.get('sanity',r['sanity'].get(n,{}));counts=check.get('parameter_counts',{})
        delta=metric['MAE']-full['MAE'] if metric and full else None
        rel=100*delta/full['MAE'] if delta is not None else None
        rows.append(dict(Category=LEVELS[n],Variant=n,Params=counts.get('active_prediction_path_tensor_elements'),
            TEST5_MAE=metric.get('MAE'),MSE=metric.get('MSE'),RMSE=metric.get('RMSE'),
            Hit_percent=100*metric['Hit'] if metric else None,DeltaMAE=delta,RelativeDeltaMAE_percent=rel))
        if n!=FULL:
            supported,meaning=evidence(delta,rel)
            contribution.append(dict(Mechanism=n.removeprefix('w/o '),Ablation=n,Supported=supported,Evidence=meaning,DeltaMAE=delta))
        reuse=r['reuse'][n]
        status.append(dict(Variant=n,OldCheckpointFound=reuse['old_checkpoint_found'],DefinitionExactMatch=reuse['definition_exact_match'],
            Reuse=reuse.get('reuse_status','eligible pending fresh evaluation' if reuse['reused'] else 'REJECTED / new'),
            NeedNewTraining=n!=FULL and not reuse['reused'] and not bool(entry),SanityPASS=check.get('PASS'),Completed=bool(entry)))
        rt=entry.get('runtime',{})
        runtime.append(dict(Variant=n,TotalInstantiated=counts.get('total_instantiated'),
            ActivePredictionParameters=counts.get('active_prediction_path_tensor_elements'),BestEpoch=rt.get('best_epoch'),
            TrainSeconds=rt.get('train_seconds'),SecondsPerEpoch=rt.get('seconds_per_epoch'),Reused=entry.get('reused')))
        for split,hm in entry.get('metrics',{}).items():
            for h,m in hm.items():allh.append(dict(Variant=n,Split=split,Horizon=int(h),**m))
        for c in entry.get('per_commodity',[]):commodity.append(dict(Variant=n,**c))
    indexed={v['Variant']:v for v in rows}
    spatial=[indexed[n] for n in (*NAMES[:3],FULL)]
    temporal=[indexed[n] for n in (*NAMES[3:8],FULL)]
    fusion=[indexed[n] for n in (*NAMES[8:10],FULL)]
    for file,data in [('main_architecture_ablation',rows),('spatial_innovation_ablation',spatial),
        ('temporal_innovation_ablation',temporal),('moe_fusion_ablation',fusion),('multi_horizon_ablation',allh),
        ('per_commodity_ablation',commodity),('runtime_summary',runtime),('contribution_evidence',contribution)]:
        csv_file(out/(file+'.csv'),data)
    mechanism={n:e.get('mechanism',{}) for n,e in results.items()}
    for file,data in [('protocol',r['protocol']),('protocol_revision',r.get('revision',{})),('variant_definitions',DEFINITIONS),
        ('checkpoint_provenance',dict(reuse=r['reuse'],completed={n:{k:e.get(k) for k in ('path','sha256','reused','source_experiment')} for n,e in results.items()})),
        ('initialization_audit',r['initialization']),('structural_sanity',dict(initial=r['sanity'],best={n:e['sanity'] for n,e in results.items()},commodity_order=r['data']['mapping'])),
        ('causality_audit',{n:{k:c.get(k) for k in ('causal_prefix','causality_scope','prediction_observed_prefix','PASS')} for n,c in r['sanity'].items()}),
        ('mechanism_summary',mechanism)]:atomic_json(out/(file+'.json'),data)
    (out/'RUN_STATUS.md').write_text('# Candidate-MoE main ablation status\n\n'+r['status']+'\n\n'+table(status)+
        '\nOnly missing eligible main configurations may train. Full never trains. Historical Adaptive-Gate ablation numbers are not reused.\n',encoding='utf-8')
    answers=[]
    for i,(question,n) in enumerate(zip(QUESTIONS,NAMES[:-1]),1):
        row=indexed[n];delta=row['DeltaMAE'];rel=row['RelativeDeltaMAE_percent']
        answers.append(f'Q{i}. {question}: '+('PENDING; no result inferred.' if delta is None else
            f'TEST5 MAE delta = {delta:.12g}; relative = {rel:.8g}%. '+evidence(delta,rel)[1]))
    weights=mechanism.get('w/o Candidate-Aware Routing',{}).get('global_mixture')
    answers[-1]+=' Best global weights: '+json.dumps(weights,ensure_ascii=False)
    whole=indexed['w/o MoE Fusion'];router=indexed['w/o Candidate-Aware Routing']
    if whole['DeltaMAE'] is None or router['DeltaMAE'] is None:
        meaning='PENDING both fusion controls; do not infer whole-MoE contribution from routing weights.'
    else:
        meaning=''
        if whole['DeltaMAE']>0:
            meaning+='The dual-expert MoE fusion improves prediction relative to ordinary linear feature fusion under the fixed single-run protocol. '
        else:
            meaning+='The whole-MoE ablation does not support a predictive benefit over ordinary linear fusion in this run. '
        if abs(router['RelativeDeltaMAE_percent'])<r['protocol'].get('near_tie_relative_percent',.1):
            meaning+=('Most of the MoE gain appears to arise from expert representation specialization and soft expert fusion, while the incremental contribution of sample-dependent routing is limited.' if whole['DeltaMAE']>0 else
                      'The global-mixture control is a near tie; incremental sample-dependent routing value is limited, and a whole-MoE gain is not established.')
        elif router['DeltaMAE']>0:
            meaning+='Candidate-aware input-dependent routing provides additional predictive value relative to the retrained global control under this single-run protocol; this alone does not isolate frozen-checkpoint routing variation.'
        else:
            meaning+='The global-mixture control has lower MAE; sample-dependent routing is not supported as the main fusion gain source.'
    answers.append('Q11. '+meaning)
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Objective','Single-factor main-innovation ablations of the final Candidate-Aware 2-Expert MoE: 3 spatial + 5 temporal + 2 fusion controls + frozen Full. No branch-removal experiments, supplementary controls or tuning are part of this main table.')
    section('Final Full Model Definition','Full = switching_latent_balanced_candidate_2expert_moe. Native spatial and switching temporal branches, Balanced Readout, branch projections, temporal residual expert and interaction expert, candidate-only LayerNorm/comparison router, dense sample-level soft mixture and one unchanged shared head. Full is strictly loaded and freshly evaluated against its completed formal artifact; it never trains here.\n\n'+table(status))
    section('Data / Protocol Audit','```json\n'+json.dumps(r['protocol'],ensure_ascii=False,indent=2)+'\n```\n\nSame full historical tensor (B,20,N,21), TRAIN/VAL/TEST splits, preprocessing and return targets [1,5,10,20] with24 commodities. No input removal, refit or target rescaling. TRAIN shuffle=False/drop_last=True; full evaluation drop_last=False. Training = sum four Huber(.02) + native Switch KL. Scheduler, early stopping and selection use prediction-only four-horizon VAL Huber. VAL5 is logging only. No MoE loss or warmup.\n\nData/order fingerprints: `'+json.dumps(r['data'],ensure_ascii=False)+'`')
    section('Initialization Audit',table([dict(Variant=n,**{k:a.get(k) for k in ('shared_parameter_count','shared_parameter_initial_max_abs_diff','mismatch_count','PASS')}) for n,a in r['initialization'].items()])+ '\nAll shared modules are constructed as complete Candidate Full before interventions. Static experts are moved unchanged; its state and forward are checked against the native Global control. Simple linear fusion is created only after the complete Full. Removed router/expert tensors are explicitly listed in initialization_audit.json.')
    section('Structural Sanity',table([dict(Variant=n,Shape=c.get('output_shape'),Finite=c.get('finite'),BatchPermutation=c.get('batch_permutation'),SingleSample=c.get('single_sample'),PASS=c.get('PASS')) for n,c in r['sanity'].items()])+ '\nCausality checks use E/H/p/Z and effective readouts at prefix10. Final forecasts and spatial pooling may use the whole observed window. All initial checks must pass before any fit; restored best checkpoints pass again. Detailed active-call, parameter and fusion wiring audits are saved.')
    section('Spatial Ablations',table(spatial)+'\nUniform temporal mean changes only time weighting. Ones adjacency removes the learned prior but retains edge-content attention (log(1+epsilon) is a constant softmax-invariant shift). Two single-hop GCN layers retain the same learned directed graph: A_hat=A+I; D=row sums of A_hat; H_next=Linear(D^(-1/2) A_hat D^(-1/2) H). The adjacency itself is not symmetrized. No attention, multi-hop selection or beta recurrence remains. This tests the complete EdgeAttnMixHop operator against standard propagation, not edge attention alone. All three still end in Candidate MoE.')
    section('Temporal Ablations',table(temporal)+'\nNo RPE keeps the causal mask and Transformer. Uniform routing retains three different generators and has KL=0. Shared transitions retain adaptive p and native KL but use G0 for each candidate. No microstate nulls the post-normalization readout, not recurrence. No balancing bypasses only the two LayerNorms. All five retain Candidate MoE.')
    section('MoE Fusion Ablations',table(fusion)+'\nWhole-MoE control: h=Linear128→64([W_s h_s || W_t h_t]) into the unchanged shared head; experts/router absent. Router control: alpha=softmax(a), a initially [0,0], h=alpha_T e_T+alpha_ST e_ST; one global vector(2,), no sample input or router normalization/MLP. Eligible completed Global control is audited and freshly reproduced without retraining.')
    section('Full Multi-Horizon Results',table([row for row in allh if row['Variant']==FULL])+'\nAll variants and all splits/horizons are saved in multi_horizon_ablation.csv. Metrics pool all origins×24 commodities; Hit includes zero targets. MSE is directly pooled and RMSE=sqrt(MSE). Raw Hit is a fraction; main table shows percent.')
    section('Main TEST5 Ablation Table',table(rows)+'\nDelta = ablation−Full Candidate MoE; relative=100×delta/Full. Positive delta means removing the mechanism worsens MAE. Params counts autograd-connected prediction parameter tensors; total instantiated counts and historical/new runtime are below. Inactive original modules retained for initialization control are not counted as active capacity.\n\n'+table(runtime))
    section('Router / Regime / Readout Diagnostics','```json\n'+json.dumps(mechanism,ensure_ascii=False,indent=2)+'\n```\n\nRouting variation is descriptive, not a success criterion or causal attribution. Uniform-regime argmax occupancy [1,0,0] is deterministic tie breaking, not learned collapse. Shared-G candidate disagreement should vanish; readout norms verify normalization intervention.')
    section('Required Answers','\n\n'.join(answers))
    section('Limitations','Seed42 single-run evidence; no statistical-significance or multi-seed robustness claim. Near tie is a descriptive convention of absolute relative MAE delta <0.1%, with exact deltas always reported. The whole-MoE ablation changes both mechanism and model capacity; it evaluates the contribution of the complete MoE fusion subsystem rather than an exactly parameter-matched alternative. No dummy layers are added. Historical source hashes and any drift are explicit in checkpoint_provenance.json; native-state/initialization and fresh evaluation are required for reference reuse. Historical artifacts without a completion flag require an audited COMPLETE report and matching full terminated history; checkpoints remain read-only. Historical Adaptive-Gate ablations cannot establish effects under the new architecture. No supplementary reruns, model optimization or extra ablations follow. STOP.')
    text='# Candidate-Aware 2-Expert MoE — Formal Main-Innovation Ablation\n\nStatus: '+r['status']+'\n\n'+''.join(sections)
    for file in ('FINAL_REPORT.md','REPORT.md'):(out/file).write_text(text,encoding='utf-8')


def global_temporal_report(r,out):
    """Targeted four-row mode of the existing report; historical tables are explanatory."""
    results=r['results'];reference=results.get(GLOBAL_FULL,{})
    full=reference.get('metrics',{}).get('test',{}).get('5',{})
    full_weights=reference.get('mechanism',{}).get('global_mixture',{})
    rows=[];weights=[];allh=[];runtime=[];status=[];commodity=[]
    for n in GLOBAL_NAMES:
        e=results.get(n,{});m=e.get('metrics',{}).get('test',{}).get('5',{})
        delta=m['MAE']-full['MAE'] if m and full else None
        relative=100*delta/full['MAE'] if delta is not None else None
        rows.append(dict(Variant=n,TEST5_MAE=m.get('MAE'),DeltaMAE=delta,RelativeDelta_percent=relative,
            MSE=m.get('MSE'),RMSE=m.get('RMSE'),Hit_percent=100*m['Hit'] if m else None))
        mech=e.get('mechanism',{});w=mech.get('global_mixture',{})
        weights.append(dict(Variant=n,alpha_T=w.get('alpha_T'),alpha_ST=w.get('alpha_ST'),
            Delta_alpha_ST=w['alpha_ST']-full_weights['alpha_ST'] if w and full_weights else None,
            **mech.get('test_expert_norms',{})))
        check=e.get('sanity',r['sanity'].get(n,{}));count=check.get('parameter_counts',{})
        runtime.append(dict(Variant=n,TotalInstantiated=count.get('total_instantiated'),
            ActivePredictionParameters=count.get('active_prediction_path_tensor_elements'),
            **e.get('runtime',{}),Reused=e.get('reused')))
        status.append(dict(Variant=n,Reused=e.get('reused',r['reuse'][n]['reused']),
            InitPASS=r['initialization'].get(n,{}).get('PASS'),SanityPASS=check.get('PASS'),
            State='COMPLETE' if e else r.get('jobs',{}).get(n,{}).get('status','PENDING')))
        for split,hm in e.get('metrics',{}).items():
            for h,m in hm.items():allh.append(dict(Variant=n,Split=split,Horizon=int(h),**m))
        for c in e.get('per_commodity',[]):commodity.append(dict(Variant=n,**c))
    indexed={row['Variant']:row for row in rows}
    cross=[]
    for n in GLOBAL_NAMES[:-1]:
        row=dict(TemporalMechanism=n.removeprefix('w/o '))
        for label in ('Old Adaptive Fusion','Dynamic Candidate MoE'):
            row[label+' Delta_percent']=r.get('cross_architecture',{}).get(label,{}).get('relative_deltas',{}).get(n)
        row['Global Mixture Delta_percent']=indexed[n]['RelativeDelta_percent'];cross.append(row)
    for file,data in (('temporal_innovation_ablation',rows),('main_architecture_ablation',rows),
        ('cross_architecture_comparison',cross),('global_mixture_diagnostics',weights),
        ('multi_horizon_ablation',allh),('runtime_summary',runtime),('per_commodity_ablation',commodity)):
        csv_file(out/(file+'.csv'),data)
    mechanism={n:e.get('mechanism',{}) for n,e in results.items()}
    for file,data in (('protocol',r['protocol']),('source_hashes',r['source_hashes']),('variant_definitions',GLOBAL_DEFINITIONS),
        ('checkpoint_provenance',dict(git_sha=r.get('git_sha'),reuse=r['reuse'],completed={n:{k:e.get(k) for k in ('path','sha256','reused','source_experiment')} for n,e in results.items()})),
        ('cross_architecture_provenance',r.get('cross_architecture',{})),('initialization_audit',r['initialization']),
        ('structural_sanity',dict(initial=r['sanity'],best={n:e['sanity'] for n,e in results.items()},commodity_order=r['data']['mapping'])),
        ('mechanism_summary',mechanism),('global_mixture_diagnostics',weights)):
        atomic_json(out/(file+'.json'),data)
    (out/'RUN_STATUS.md').write_text('# Global temporal mechanism study\n\n'+r['status']+'\n\n'+table(status)+
        '\nOnly three temporal ablations may train, once each at seed42. Native Global Full must be reused. STOP after completion.\n',encoding='utf-8')
    eps=r['protocol'].get('near_tie_relative_percent',.1)
    regime=[indexed[n] for n in GLOBAL_NAMES[:2]]
    if any(row['DeltaMAE'] is None for row in regime):
        case='PENDING';support='PENDING';interpretation='The three new formal runs are not complete; no restoration or redundancy conclusion can be drawn.'
    else:
        restored=[row['RelativeDelta_percent']>=eps for row in regime]
        if all(restored):
            case='Case A';support='supported'
            interpretation=('Removing the second expert-level dynamic router restores the marginal contribution of temporal regime routing and regime-specific dynamics, consistent with routing redundancy in the previous dual-routing architecture. This is single-run descriptive evidence, not causal proof.')
        elif any(restored):
            case='Case B';support='partially supported'
            positive=GLOBAL_NAMES[restored.index(True)].removeprefix('w/o ')
            interpretation=('Removing expert-level routing restores part, but not all, of the contribution associated with temporal switching, indicating partial functional overlap rather than complete redundancy. Restored mechanism: '+positive+'.')
        else:
            case='Case C';support='not supported'
            interpretation=('Removing the second expert-level router does not restore the predictive value of the regime mechanisms; routing redundancy is therefore insufficient to explain the previous ablation reversal. The stronger dual-expert representation architecture, rather than expert routing alone, may change their marginal utility.')
    def answer_delta(n):
        row=indexed[n]
        if row['DeltaMAE'] is None:return 'PENDING formal training/evaluation.'
        return f"Delta MAE={row['DeltaMAE']:.12g}; relative delta={row['RelativeDelta_percent']:.9g}%. "+evidence(row['DeltaMAE'],row['RelativeDelta_percent'])[1]
    answers=[
        'Q1. Full Global TEST5 MAE: '+fmt(full.get('MAE'))+'. Fresh evaluation only; no hardcoded result.',
        'Q2. Full Global [alpha_T, alpha_ST]: '+str([full_weights.get('alpha_T'),full_weights.get('alpha_ST')]),
        'Q3. No adaptive regime routing: '+answer_delta(GLOBAL_NAMES[0]),
        'Q4. Restored routing contribution: '+answer_delta(GLOBAL_NAMES[0]),
        'Q5. Shared regime transition: '+answer_delta(GLOBAL_NAMES[1]),
        'Q6. Restored transition contribution: '+answer_delta(GLOBAL_NAMES[1]),
        'Q7. Microstate contribution: '+answer_delta(GLOBAL_NAMES[2]),
        'Q8. Cross-architecture changes (each relative to its own Full):\n\n'+table(cross),
        'Q9. Routing-redundancy interpretation: '+support+'; '+case+'. '+interpretation,
        'Q10. '+('A plausible division is temporal Markov routing for dynamic market-state adaptation, dual experts for representation specialization, and global mixture for stable integration. This interpretation is consistent with, not established causally by, the evidence.' if support in ('supported','partially supported') else 'That division of responsibilities is not established by the current evidence.'),
        'Q11. '+(interpretation if support=='not supported' else 'Non-restoration statement is not applicable yet or is addressed separately for the unsupported mechanism in Case B.')]
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Objective','Does removing expert-level dynamic routing restore the value of temporal regime mechanisms? Exactly three temporal controls and a reused Global Full; no extra ablations or model optimization.\n\n'+table(status))
    section('Routing Redundancy Hypothesis','Temporal regime routing and Candidate-Aware expert routing may supply overlapping adaptive capacity. This experiment removes only the latter at the Full-architecture level, then re-estimates three temporal marginal effects. Outcomes may support, partially support, or fail to support that hypothesis.')
    section('Global Dual-Expert Architecture',r'$s=W_s h_s,\ t=W_t h_t;\ e_T=E_T(t);\ e_{ST}=E_{ST}([s\Vert t])$.'+'\n\n'+r'$\alpha=Softmax(a),\ a\in\mathbb R^2;\ h=\alpha_T e_T+\alpha_{ST}e_{ST};\ \hat Y=Head(h)$.'+'\n\nNative '+GLOBAL+'. The temporal expert is residual; the interaction expert is not. One shared head. Alpha has shape (2,), shared across every sample. No Candidate Router, router-only LayerNorm, router MLP, warmup or auxiliary loss.')
    section('Fixed Components','Native spatial branch, input/preprocessing, market encoder, causal Transformer and RPE, branch projections, both experts, balanced readout and shared head remain fixed in definition except the stated single temporal intervention. No parameter matching or extra layers. The only remaining input-dependent routing mechanism is the temporal Markov regime router; the uniform-routing control removes its sample dependence too. This refers to regime/expert routing, not to ordinary content attention.')
    section('Formal Full Reference Audit','Full is never retrained. Strict state loading, historical completion/protocol/seed/data/source audit and fresh TRAIN/VAL/TEST reproduction are mandatory. Checkpoint and source report SHA256 remain unchanged. Legacy completion evidence is disclosed without modifying the checkpoint.\n\n```json\n'+json.dumps(r['reuse'][GLOBAL_FULL],ensure_ascii=False,indent=2)+'\n```')
    section('Initialization Audit',table([dict(Variant=n,**{k:a.get(k) for k in ('shared_parameter_count','shared_parameter_initial_max_abs_diff','mismatch_count','PASS')}) for n,a in r['initialization'].items()])+'\nDirect native Global construction precedes each intervention; no Candidate router is constructed, so its random initialization is not consumed. Every new control starts global logits at [0,0], alpha=[0.5,0.5], and learns its own weights. No copy/freeze of Full alpha.')
    section('Structural Sanity',table([dict(Variant=n,Shape=c.get('output_shape'),Finite=c.get('finite'),BatchPermutation=c.get('batch_permutation'),SingleSample=c.get('single_sample'),PASS=c.get('PASS')) for n,c in r['sanity'].items()])+'\nE/H/p/Z and effective readouts are checked at prefix10. Global weights have no batch dimension and are input invariant. Uniform posterior AND prior retain three distinct generators and KL=0. Shared-G retains adaptive p and native KL, with identical candidates. NoMicro retains recurrence and KL, nulling only post-normalization effective micro readout. All checks precede training and repeat on restored best checkpoints.')
    section('Training Protocol','```json\n'+json.dumps(r['protocol'],ensure_ascii=False,indent=2)+'\n```\n\nTraining objective = sum four equally weighted Huber(.02) + native Switch KL (beta_max=5e-4, warmup20). Uniform p/prior makes KL zero. No MoE-specific loss. Selection, scheduler and early stopping use prediction-only multihorizon VAL Huber. VAL5 is diagnostic only. Full data evaluation uses drop_last=False; train uses shuffle=False/drop_last=True. No gradients are clipped.\n\nData fingerprints/order: `'+json.dumps(r['data'],ensure_ascii=False)+'`')
    section('Full Global Results',table([row for row in allh if row['Variant']==GLOBAL_FULL])+'\nAll metrics are pooled over origins×24 commodities; RMSE=sqrt(MSE); Hit is unmasked including zero targets. CSV Hit is a fraction, main table Hit is percent.')
    section('Temporal Ablation Results',table(rows)+'\nDelta=ablation−Global Full; relative=100×delta/Global Full. Positive means removing the mechanism worsens MAE. All splits/four horizons are in multi_horizon_ablation.csv.\n\n'+table(runtime))
    section('Cross-Architecture Comparison',table(cross)+'\nExplanatory historical comparison only. Each column uses its own audited Full as denominator; historical artifacts/checkpoints are hash-checked and never retrained. Sources are recorded in cross_architecture_provenance.json. No historical numbers are substituted for the new Global controls.')
    section('Global Mixture Weight Diagnostics',table(weights)+'\nDelta alpha_ST=control−Global Full. Expert norms are mean TEST L2. These coefficients and scale shifts are descriptive, not causal attribution or evidence for automatic redesign.')
    section('Regime Diagnostics','```json\n'+json.dumps(mechanism,ensure_ascii=False,indent=2)+'\n```\n\nUniform argmax occupancy [1,0,0] is deterministic tie breaking under equal probabilities, not collapse. Shared-G candidate pairwise disagreement must be zero to numerical tolerance. Effective microstate nulling is audited separately from stored pre-intervention readout norms.')
    section('Required Answers','\n\n'.join(answers))
    section('Interpretation',case+' — '+support+'. '+interpretation+'\n\nMicrostate control: '+answer_delta(GLOBAL_NAMES[2])+' A positive material microstate delta would support retained memory value, distinct from regime-routing complexity.')
    section('Limitations','This experiment is a post-hoc mechanism study motivated by the observed interaction between temporal regime routing and expert-level routing. TEST results of previous experiments have already been observed; this is not pre-registered untouched final model selection. Single seed42, no statistical-significance, causal-proof or multi-seed robustness claim. The inherited descriptive near-tie convention is absolute relative MAE delta <0.1%; exact signed deltas are always provided and the convention is not a statistical test. Three retrained controls do not isolate every possible training-path interaction. Accept every result; no tuning, additional seed, router, loss, expert or ablation follows. STOP.')
    text='# Global Dual-Expert Temporal Routing Redundancy\n\nStatus: '+r['status']+'\n\n'+''.join(sections)
    for file in ('FINAL_REPORT.md','REPORT.md'):(out/file).write_text(text,encoding='utf-8')


def ts_global_report(r,out):
    """Five targeted T+S rows; existing architectures are contextual references only."""
    results=r['results'];full_entry=results.get(TS_FULL,{})
    full=full_entry.get('metrics',{}).get('test',{}).get('5',{})
    previous=r.get('reference_results',{}).get(GLOBAL_FULL,{})
    previous_metric=previous.get('metrics',{}).get('test',{}).get('5',{})
    full_weights=full_entry.get('mechanism',{}).get('global_mixture',{})
    rows=[];allh=[];weights=[];scales=[];status=[];runtime=[];commodity=[]
    for name in TS_NAMES:
        entry=results.get(name,{});metric=entry.get('metrics',{}).get('test',{}).get('5',{})
        delta=metric['MAE']-full['MAE'] if metric and full else None
        relative=100*delta/full['MAE'] if delta is not None else None
        rows.append(dict(Variant=name,TEST5_MAE=metric.get('MAE'),DeltaMAE=delta,RelativeDelta_percent=relative,
            MSE=metric.get('MSE'),RMSE=metric.get('RMSE'),Hit_percent=100*metric['Hit'] if metric else None))
        mechanism=entry.get('mechanism',{});w=mechanism.get('global_mixture',{})
        weights.append(dict(Variant=name,**w,Delta_alpha_S=w['alpha_S']-full_weights['alpha_S'] if w and full_weights else None))
        scales.append(dict(Variant=name,**mechanism.get('test_expert_norms',{})))
        check=entry.get('sanity',r['sanity'].get(name,{}));counts=check.get('parameter_counts',{})
        runtime.append(dict(Variant=name,TotalInstantiated=counts.get('total_instantiated'),
            ActivePredictionParams=counts.get('active_prediction_path_tensor_elements'),**entry.get('runtime',{})))
        status.append(dict(Variant=name,InitializationPASS=r['initialization'].get(name,{}).get('PASS'),
            SanityPASS=check.get('PASS'),State='COMPLETE' if entry else r.get('jobs',{}).get(name,{}).get('status','PENDING')))
        for split,hm in entry.get('metrics',{}).items():
            for horizon,m in hm.items():allh.append(dict(Variant=name,Split=split,Horizon=int(horizon),**m))
        for c in entry.get('per_commodity',[]):commodity.append(dict(Variant=name,**c))
    indexed={row['Variant']:row for row in rows}
    cross=[]
    for name in TS_NAMES[:-1]:
        row=dict(Mechanism=name.removeprefix('w/o '))
        for label in ('Old Adaptive Fusion','Dynamic Candidate MoE','T+ST Global Mixture'):
            value=r.get('cross_architecture',{}).get(label,{}).get('relative_deltas',{}).get(name)
            row[label+' Delta_percent']='N/A (not formally available)' if value is None else value
        row['T+S Global Mixture Delta_percent']=indexed[name]['RelativeDelta_percent'];cross.append(row)
    direct=[]
    for label,entry in (('T+ST Global Dual-Expert',previous),('T+S Global Dual-Expert',full_entry)):
        m=entry.get('metrics',{}).get('test',{}).get('5',{})
        counts=entry.get('sanity',{}).get('parameter_counts',{})
        direct.append(dict(Architecture=label,Parameters=counts.get('total_instantiated'),
            TEST5_MAE=m.get('MAE'),MSE=m.get('MSE'),RMSE=m.get('RMSE'),Hit_percent=100*m['Hit'] if m else None))
    direct_delta=full['MAE']-previous_metric['MAE'] if full and previous_metric else None
    direct_relative=100*direct_delta/previous_metric['MAE'] if direct_delta is not None else None
    for file,data in (('main_ablation_table',rows),('multi_horizon_ablation',allh),('cross_architecture_comparison',cross),
        ('ts_vs_tst_global_comparison',direct),('runtime_summary',runtime),('per_commodity_ablation',commodity)):
        csv_file(out/(file+'.csv'),data)
    mechanism={n:e.get('mechanism',{}) for n,e in results.items()}
    expert_audits={n:c.get('revised_structure',{}) for n,c in r['sanity'].items()}
    for file,data in (('config',r['protocol']),('protocol',r['protocol']),('source_hashes',r['source_hashes']),
        ('data_audit',r['data']),('variant_definitions',TS_DEFINITIONS),('initialization_audit',r['initialization']),
        ('expert_wiring_audit',expert_audits),('structural_sanity',dict(initial=r['sanity'],best={n:e.get('sanity') for n,e in results.items()})),
        ('best_checkpoint_metadata',{n:dict(path=e['path'],sha256=e['sha256'],**e.get('best_checkpoint_metadata',{})) for n,e in results.items()}),
        ('test_metrics',{n:e['metrics'].get('test',{}) for n,e in results.items()}),('expert_diagnostics',scales),
        ('global_weight_diagnostics',weights),('regime_diagnostics',{n:m.get('regime',{})|{'candidate_pairwise_mean_absolute_disagreement':m.get('candidate_pairwise_mean_absolute_disagreement')} for n,m in mechanism.items()}),
        ('reference_provenance',dict(references=r.get('references',{}),cross_architecture=r.get('cross_architecture',{}),
            fresh_reference_results=r.get('reference_results',{}))),('mechanism_summary',mechanism)):
        atomic_json(out/(file+'.json'),data)
    (out/'RUN_STATUS.md').write_text('# T+S Global targeted study\n\n'+r['status']+'\n\n'+table(status)+
        '\nExactly one new Full and four targeted controls; seed42 each. Historical T+ST models are never retrained. STOP after completion.\n',encoding='utf-8')
    eps=r['protocol'].get('near_tie_relative_percent',.1)
    temporal=[indexed[n] for n in TS_NAMES[:3]]
    if any(row['DeltaMAE'] is None for row in temporal):
        case='PENDING';support='PENDING';meaning='The new T+S formal runs have not completed; no hypothesis verdict is available.'
    else:
        positive=[row['RelativeDelta_percent']>=eps for row in temporal]
        if all(positive):
            case='Case A';support='supported'
            meaning=('Replacing the cross-branch interaction expert with branch-specific experts restores the marginal value of temporal mechanisms, consistent with the interaction expert absorbing part of the temporal representation-learning role. This does not prove that mechanism causally.')
        elif any(positive):
            case='Case B';support='partially supported'
            restored=', '.join(TS_NAMES[i].removeprefix('w/o ') for i,v in enumerate(positive) if v)
            meaning=('Branch-specific experts restore only part of the temporal mechanism contribution, indicating partial functional overlap between the former interaction expert and temporal representation learning. Restored under the descriptive convention: '+restored+'.')
        else:
            case='Case C';support='not supported'
            meaning=('Replacing the interaction expert with branch-specific experts does not restore the temporal mechanism contributions; the interaction expert alone is therefore insufficient to explain the previous ablation reversal.')
    def delta_answer(name):
        row=indexed[name]
        if row['DeltaMAE'] is None:return 'PENDING; do not infer a result.'
        return f"DeltaMAE={row['DeltaMAE']:.12g}; relative={row['RelativeDelta_percent']:.9g}%. "+evidence(row['DeltaMAE'],row['RelativeDelta_percent'])[1]
    answers=[
        'Q1. Full T+S TEST5 MAE: '+fmt(full.get('MAE')),
        'Q2. T+S minus T+ST Global Full: absolute MAE delta='+fmt(direct_delta)+'; relative delta='+fmt(direct_relative)+'%.',
        'Q3. Full T+S [alpha_T, alpha_S]: '+str([full_weights.get('alpha_T'),full_weights.get('alpha_S')]),
        'Q4. Full TEST expert norms and spatial/temporal ratio: '+json.dumps(full_entry.get('mechanism',{}).get('test_expert_norms',{})),
        *[f'Q{i+5}. {n}: '+delta_answer(n) for i,n in enumerate(TS_NAMES[:-1])],
        'Q9. Relative changes across architectures (each against its own Full):\n\n'+table(cross),
        'Q10. '+support+'; '+case+'. '+meaning,
        'Q11. '+('A consistent, non-causal interpretation is: spatial branch for cross-market relational representation; temporal branch for market-state temporal dynamics; Temporal and Spatial experts for their own branch refinement; global mixture for stable integration. The partially supported case only justifies this for the mechanisms that recover.' if support in ('supported','partially supported') else 'The proposed division of responsibilities is not established by the current results.'),
        'Q12. '+(meaning if support=='not supported' else 'Non-restoration conclusion is not applicable yet or is qualified by the per-mechanism evidence above.')]
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Objective','Does branch-specific T+S expert separation restore the marginal value of spatial and temporal mechanisms? One new Full plus exactly four targeted controls, no additional experiments.\n\n'+table(status))
    section('Motivation','Previous T+ST Global controls did not restore temporal mechanism contributions after deleting the Candidate Router. This study changes only the expert definition: E_ST([s||t]) becomes a branch-specific residual E_S(s). It does not search for a new router.')
    section('Previous T+ST Evidence','All historical numbers are read from completed, audited artifacts. Sources and checkpoint hashes are in reference_provenance.json. T+ST native Full is strictly loaded and freshly reproduced with the current evaluator.\n\n'+table(direct[:1])+'\n\n'+table(cross))
    section('T+S Global Dual-Expert Architecture',r'$s=W_s h_s,\quad t=W_t h_t;\quad e_T=t+F_T(t),\quad e_S=s+F_S(s)$.'+'\n\n'+r'$\alpha=Softmax(a),\ a\in\mathbb{R}^{2};\quad h=\alpha_T e_T+\alpha_S e_S;\quad \hat Y=Head(h)$.'+'\n\nBoth F are independent Linear64→64, ReLU, Dropout(.1), Linear64→64. Global logits start at [0,0], giving [0.5,0.5]. No expert sees both branches, no interaction expert or sample-dependent router. Shared head: Linear64→64, ReLU, Dropout(.3), Linear64→96, reshape (B,4,24).')
    section('Fixed Components','Native spatial type projection/time weighting/adaptive graph/EdgeAttnMixHop/norm/type pool and temporal market encoder/causal Transformer/RPE/Markov/K=3/transitions/recurrence/Balanced Readout are preserved outside each named intervention. W_s, W_t, shared head, data preprocessing and chronological splits are unchanged. Only new configurations train; original Candidate and Global checkpoints are read-only.')
    section('Expert Symmetry and Wiring Audit','```json\n'+json.dumps(expert_audits,ensure_ascii=False,indent=2)+'\n```\n\nBoth experts have identical layer dimensions, activations, dropout and residual structure; separate parameters. Wiring is checked by recomputation and independent branch perturbations. T cannot receive s and S cannot receive t. Alpha is global shape(2,), input invariant, trainable through prediction loss.')
    section('Initialization Audit',table([dict(Variant=n,**{k:a.get(k) for k in ('shared_parameter_count','shared_parameter_initial_max_abs_diff','mismatch_count','PASS')}) for n,a in r['initialization'].items()])+'\nConstruct the complete native T+S Full at seed42 before interventions. MixHop replacements are constructed last. Unrelated T+ST backbone/projections/head/Temporal expert initialization is also compared exactly. No Full alpha or trained parameters are copied.')
    section('Structural Sanity',table([dict(Variant=n,Shape=c.get('output_shape'),Finite=c.get('finite'),BatchPermutation=c.get('batch_permutation'),SingleSample=c.get('single_sample'),PASS=c.get('PASS')) for n,c in r['sanity'].items()])+'\nInitial and restored-best checks cover prefix10 E/H/p/Z/readout causality, finite predictions, (B,4,24), no batch mixing, full input and target order, expert wiring, global normalization, three distinct uniform-regime generators, shared-G identical candidates, and post-normalization microstate nulling.')
    section('Training Protocol','```json\n'+json.dumps(r['protocol'],ensure_ascii=False,indent=2)+'\n```\n\nExactly one seed42 run per configuration. Adam1e-4/WD1e-5/batch64/max200/patience10/ReduceLROnPlateau factor.5 patience5; train shuffle=False, drop_last=True. Objective is sum of four equal Huber(.02) plus native Switch KL (max5e-4/warmup20; uniform posterior/prior makes KL zero). No MoE auxiliary loss or clipping. Selection, scheduler and early stopping use prediction-only four-horizon VAL Huber; VAL5 is secondary logging.\n\nData/order fingerprints: `'+json.dumps(r['data'],ensure_ascii=False)+'`')
    section('Full T+S Multi-Horizon Results',table([row for row in allh if row['Variant']==TS_FULL])+'\nAll splits use full evaluation, drop_last=False. Pooled origins×24 commodities; Hit unmasked including zero targets; RMSE=sqrt(MSE). CSV Hit is a fraction, displayed main-table Hit is percent.')
    section('Four Targeted Ablations',table(rows)+'\nDelta=ablation−T+S Full, relative=100×delta/T+S Full. Uniform routing preserves independent G1/G2/G3 and sets posterior/prior uniform. Shared-G keeps adaptive p and native KL. NoMicro nulls only effective post-normalization readout. NoEdge replaces two EdgeAttnMixHop blocks by ordinary MixHopPropagation(64,64,K=2,beta=.05) on the same learned adjacency; no AdaptiveGraph removal. Every control keeps T+S experts.\n\n'+table(runtime))
    section('T+S vs T+ST Global Comparison',table(direct)+f'\nT+S−T+ST MAE delta={fmt(direct_delta)}; relative={fmt(direct_relative)}%. Post-hoc architecture comparison, not a significance claim.')
    section('Cross-Architecture Mechanism Comparison',table(cross)+'\nHistorical Edge rows use the audited ordinary-MixHop definitions, matching this study; the later GCN replacement is a different control and is not substituted. T+ST Global Edge is N/A unless an exact formal control exists. Historical effects are contextual, not new T+S measurements.')
    section('Global Weight Diagnostics',table(weights)+'\nDelta alpha_S=control−T+S Full. Each configuration learns its own global pair from zero logits; no freezing or VAL/TEST alpha fitting. Mixture coefficients are not causal feature contributions.')
    old_norms=previous.get('mechanism',{}).get('test_expert_norms',{})
    section('Expert Scale Diagnostics',table(scales)+'\nScale ratio=mean TEST ||e_S||2 / (mean TEST ||e_T||2 + 1e-8). Historical T+ST reference norms: `'+json.dumps(old_norms)+'`. Descriptive only; unequal scales never trigger normalization or retraining.')
    section('Regime Diagnostics','```json\n'+json.dumps(mechanism,ensure_ascii=False,indent=2)+'\n```\n\nUniform argmax occupancy [1,0,0] is deterministic tie breaking under uniform probabilities, not learned collapse. Shared-G candidate disagreement must vanish to numerical tolerance. Stored pre-intervention micro norms are distinguished from the audited effective zero readout.')
    section('Required Answers','\n\n'.join(answers))
    section('Interpretation',case+' — '+support+'. '+meaning+'\n\nSpatial mechanism: '+delta_answer('w/o EdgeAttnMixHop')+' All unfavorable or near-zero results are retained. No new design follows.')
    section('Limitations','This is a post-hoc architecture-mechanism study motivated by the observed loss of marginal contribution after introducing dual-expert fusion. TEST has been observed repeatedly; this is not pre-registered final model selection or untouched-test evaluation. Single seed42; no statistical significance, causal proof or multi-seed robustness claim. The inherited 0.1% relative-MAE near-tie convention is descriptive; exact deltas remain primary evidence. Replacing a non-residual 128D-input interaction expert with a residual 64D-input spatial expert changes both inductive structure and parameter count; this is not a parameter-matched isolation of cross-branch information alone. The original shared nonlinear head can still mix the combined representation. No experts, router, loss, normalization, K, seed or additional ablation are changed after observing results. STOP.')
    text='# T+S Global Dual-Expert — Targeted Mechanism Study\n\nStatus: '+r['status']+'\n\n'+''.join(sections)
    for file in ('FINAL_REPORT.md','REPORT.md'):(out/file).write_text(text,encoding='utf-8')
