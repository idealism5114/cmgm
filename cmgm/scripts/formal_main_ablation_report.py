"""Final Candidate-MoE main ablation tables; historical Adaptive-Gate results excluded."""
import csv
import json
from cmgm.models.formal_d0b_main_ablation import NAMES,DEFINITIONS,FULL
from cmgm.scripts.formal_v2_protocol import atomic_json
LEVELS=dict(zip(NAMES,('Spatial',)*3+('Temporal',)*5+('Fusion',)*2+('Full',)))
QUESTIONS=(
    'Removing spatial temporal weighting', 'Removing adaptive graph learning',
    'Replacing EdgeAttnMixHop by ordinary MixHop', 'Removing relative position encoding',
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
    for file,data in [('protocol',r['protocol']),('variant_definitions',DEFINITIONS),
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
    section('Spatial Ablations',table(spatial)+'\nUniform temporal mean changes only time weighting. Ones adjacency removes the learned prior but retains edge-content attention (log(1+epsilon) is a constant softmax-invariant shift). Ordinary two-block MixHop retains the same learned graph. All three still end in Candidate MoE.')
    section('Temporal Ablations',table(temporal)+'\nNo RPE keeps the causal mask and Transformer. Uniform routing retains three different generators and has KL=0. Shared transitions retain adaptive p and native KL but use G0 for each candidate. No microstate nulls the post-normalization readout, not recurrence. No balancing bypasses only the two LayerNorms. All five retain Candidate MoE.')
    section('MoE Fusion Ablations',table(fusion)+'\nWhole-MoE control: h=Linear128→64([W_s h_s || W_t h_t]) into the unchanged shared head; experts/router absent. Router control: alpha=softmax(a), a initially [0,0], h=alpha_T e_T+alpha_ST e_ST; one global vector(2,), no sample input or router normalization/MLP. Eligible completed Global control is audited and freshly reproduced without retraining.')
    section('Full Multi-Horizon Results',table([row for row in allh if row['Variant']==FULL])+'\nAll variants and all splits/horizons are saved in multi_horizon_ablation.csv. Metrics pool all origins×24 commodities; Hit includes zero targets. MSE is directly pooled and RMSE=sqrt(MSE). Raw Hit is a fraction; main table shows percent.')
    section('Main TEST5 Ablation Table',table(rows)+'\nDelta = ablation−Full Candidate MoE; relative=100×delta/Full. Positive delta means removing the mechanism worsens MAE. Params counts autograd-connected prediction parameter tensors; total instantiated counts and historical/new runtime are below. Inactive original modules retained for initialization control are not counted as active capacity.\n\n'+table(runtime))
    section('Router / Regime / Readout Diagnostics','```json\n'+json.dumps(mechanism,ensure_ascii=False,indent=2)+'\n```\n\nRouting variation is descriptive, not a success criterion or causal attribution. Uniform-regime argmax occupancy [1,0,0] is deterministic tie breaking, not learned collapse. Shared-G candidate disagreement should vanish; readout norms verify normalization intervention.')
    section('Required Answers','\n\n'.join(answers))
    section('Limitations','Seed42 single-run evidence; no statistical-significance or multi-seed robustness claim. Near tie is a descriptive convention of absolute relative MAE delta <0.1%, with exact deltas always reported. The whole-MoE ablation changes both mechanism and model capacity; it evaluates the contribution of the complete MoE fusion subsystem rather than an exactly parameter-matched alternative. No dummy layers are added. Historical source hashes and any drift are explicit in checkpoint_provenance.json; native-state/initialization and fresh evaluation are required for reference reuse. Historical artifacts without a completion flag require an audited COMPLETE report and matching full terminated history; checkpoints remain read-only. Historical Adaptive-Gate ablations cannot establish effects under the new architecture. No supplementary reruns, model optimization or extra ablations follow. STOP.')
    text='# Candidate-Aware 2-Expert MoE — Formal Main-Innovation Ablation\n\nStatus: '+r['status']+'\n\n'+''.join(sections)
    for file in ('FINAL_REPORT.md','REPORT.md'):(out/file).write_text(text,encoding='utf-8')
