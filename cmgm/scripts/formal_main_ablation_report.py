"""Fixed-order revised main table, hierarchy, and data-derived contribution statements."""
import csv
import json
from cmgm.models.formal_d0b_main_ablation import NAMES,DEFINITIONS
from cmgm.scripts.formal_v2_protocol import atomic_json
LEVELS=dict(zip(NAMES,('Branch','Branch','Spatial','Spatial','Spatial','Temporal','Temporal','Temporal','Temporal','Temporal','Fusion','Full')))
QUESTIONS=(
'Does the Spatial branch provide a positive net contribution?',
'Does the Temporal branch provide a positive net contribution?',
'Does spatial temporal attention outperform temporal mean pooling?',
'Does AdaptiveGraph add value beyond content-based edge attention?',
'Does EdgeAttnMixHop outperform ordinary MixHop on the same learned graph?',
'Does Base Relative Position Encoding contribute?',
'Does adaptive regime routing outperform uniform routing?',
'Do three regime-specific transitions outperform one shared transition?',
'Does microstate Z contribute beyond long-memory H?',
'Does Balanced Readout contribute?',
'Does adaptive gated fusion outperform fixed equal-weight fusion?')


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
    results=r['results'];full=results.get('Full D0B',{}).get('metrics',{}).get('test',{}).get('5')
    rows=[];contribution=[];runtime=[];allh=[];commodity=[];status=[]
    for n in NAMES:
        entry=results.get(n,{});metric=entry.get('metrics',{}).get('test',{}).get('5',{})
        delta=metric['MAE']-full['MAE'] if metric and full else None
        rel=100*delta/full['MAE'] if delta is not None else None
        rows.append(dict(Variant=n,Level=LEVELS[n],MAE=metric.get('MAE'),DeltaMAE=delta,RelativeDeltaMAE_percent=rel,
                         MSE=metric.get('MSE'),RMSE=metric.get('RMSE'),Hit_percent=100*metric['Hit'] if metric else None))
        if n!='Full D0B':
            supported,meaning=evidence(delta,rel)
            contribution.append(dict(Mechanism=n.removeprefix('w/o '),Ablation=n,Supported=supported,Evidence=meaning,DeltaMAE=delta))
        reuse=r['reuse'][n];check=r['sanity'].get(n,{})
        status.append(dict(Variant=n,OldCheckpointFound=reuse['old_checkpoint_found'],DefinitionExactMatch=reuse['definition_exact_match'],
            Reuse=reuse['reused'],NeedNewTraining=not reuse['reused'] and not bool(entry),SanityPASS=check.get('PASS'),Completed=bool(entry)))
        rt=entry.get('runtime',{})
        counts=entry.get('sanity',check).get('parameter_counts',{})
        runtime.append(dict(Variant=n,TotalInstantiated=counts.get('total_instantiated'),ActivePredictionParameters=counts.get('active_prediction_path_tensor_elements'),
            NonzeroGradientElements=counts.get('nonzero_prediction_gradient_elements'),BestEpoch=rt.get('best_epoch'),TrainSeconds=rt.get('train_seconds'),
            SecondsPerEpoch=rt.get('seconds_per_epoch'),Reused=entry.get('reused'),RuntimeScope='historical training' if entry.get('reused') else 'new run'))
        for split,hm in entry.get('metrics',{}).items():
            for h,m in hm.items():allh.append(dict(Variant=n,Split=split,Horizon=int(h),MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit']))
        for c in entry.get('per_commodity',[]):commodity.append(dict(Variant=n,**c))
    csv_file(out/'main_architecture_ablation.csv',rows)
    indexed={v['Variant']:v for v in rows}
    branch=[dict(Representation=label,**indexed[n]) for label,n in [('SpatialOnly',NAMES[1]),('TemporalOnly',NAMES[0]),('Spatial + Temporal',NAMES[-1])]]
    spatial=[dict(Replacement=replacement,**indexed[n]) for n,replacement in zip((*NAMES[2:5],NAMES[-1]),('Uniform temporal mean','All-ones neutral prior','Standard MixHop','Full Spatial'))]
    temporal=[dict(Replacement=replacement,**indexed[n]) for n,replacement in zip((*NAMES[5:10],NAMES[-1]),('No RPE bias','Uniform p','Shared G','Zero effective micro readout','No long/micro LayerNorm','Full Temporal'))]
    for file,data in [('branch_ablation',branch),('spatial_innovation_ablation',spatial),('temporal_innovation_ablation',temporal),('multi_horizon_ablation',allh),('per_commodity_ablation',commodity),('runtime_summary',runtime),('contribution_evidence',contribution)]:csv_file(out/(file+'.csv'),data)
    for file,data in [('protocol',r['protocol']),('variant_definitions',DEFINITIONS),('checkpoint_provenance',dict(main=r['reuse'],supplementary=r['supplementary_reuse'],completed={n:{k:e.get(k) for k in ('path','sha256','reused','source_experiment')} for n,e in results.items()})),
                      ('initialization_audit',r['initialization']),('structural_sanity',dict(initial=r['sanity'],best={n:e['sanity'] for n,e in results.items()},commodity_order=r['data']['mapping'])),
                      ('causality_audit',{n:{k:c.get(k) for k in ('causal_prefix','causality_scope','prediction_observed_prefix','PASS')} for n,c in r['sanity'].items()}),
                      ('mechanism_summary',{n:e.get('mechanism',{}) for n,e in {**results,**r['supplementary']}.items()})]:atomic_json(out/(file+'.json'),data)
    (out/'RUN_STATUS.md').write_text('# Revised main-innovation run status\n\n'+r['status']+'\n\n'+table(status)+'\nOnly pending non-reusable main configurations may train. Supplementary artifacts are reuse-only.\n',encoding='utf-8')
    text='# D0B FORMAL MAIN-INNOVATION ABLATION STUDY — REVISED\n\nStatus: '+r['status']+'\n\n'
    text+='## 1. Protocol\n\nSingle-run, seed42; same full historical input, chronological split, preprocessing, targets and pooled evaluator. No statistical-significance or multi-seed robustness claim.\n\n'
    text+='Training: Adam lr=1e-4, WD=1e-5, batch64, max200 epochs, patience10; sum of four Huber losses (delta=.02), native Switch KL where structurally active. Formal selection, early stopping and scheduler all use multi-horizon VAL Huber; VAL5 is secondary logging only. TRAIN shuffle=False/drop_last=True.\n\n'
    text+='Full reference is the verified same-protocol seed42 FullD0B-Control checkpoint, not an unverified historical number. Every reused checkpoint is reevaluated with the shared evaluator. Hit is unmasked sign agreement including zero targets, displayed in percent; MSE is directly pooled and RMSE its square root.\n\n'
    text+='Data/order audit: `'+json.dumps(r['data'],ensure_ascii=False)+'`\n\n'+table(status)
    text+='\n## 2. MAIN ARCHITECTURE ABLATION — TEST 5D\n\n'+table(rows)
    def better(n):
        d=indexed[n]['DeltaMAE']
        return 'PENDING' if d is None else ('YES' if d>0 else 'NO')
    complement='PENDING' if any(better(n)=='PENDING' for n in NAMES[:2]) else ('YES' if all(better(n)=='YES' for n in NAMES[:2]) else 'NO')
    text+='\n## 3. Branch-level evidence\n\n'+table(branch)
    text+=f"\nFull outperforms SpatialOnly: {better(NAMES[1])}. Full outperforms TemporalOnly: {better(NAMES[0])}. Spatial-temporal complementarity supported by both branch comparisons: {complement}.\n"
    for n in NAMES[:2]:
        d=indexed[n]['DeltaMAE'];text+=f"\nRemoving {n.removeprefix('w/o ')}: "+('PENDING' if d is None else ('degrades' if d>0 else 'improves' if d<0 else 'ties'))+' TEST5 MAE.\n'
    text+='\n## 4. Spatial innovation\n\n'+table(spatial)+'\nWHEN: temporal attention; WHERE: adaptive graph structural prior; HOW: edge-attentive propagation. Evidence is assigned separately below. A4 retains EdgeAttnMixHop with ones; native log(1+epsilon) is a constant softmax-invariant shift. A5 retains the learned graph and replaces only propagation with ordinary MixHop.\n'
    text+='\n## 5. Temporal innovation\n\n'+table(temporal)+'\nTemporal Position → Regime Identification → Regime-Specific Dynamics → Microstate → Long/Micro Integration. Uniform routing retains three distinct generators. Shared transitions retain adaptive routing and KL but use G0 for all candidates. These are different experiments.\n'
    text+='\n## 6. Fusion control\n\n'+table([indexed[NAMES[10]],indexed[NAMES[-1]]])+'\nAdaptive-gate vs fixed-fusion evidence is distinct from net benefit of combining two branches.\n'
    text+='\n## 7. Contribution evidence matrix\n\n'+table(contribution)
    text+='\n## 8. Initialization, structural sanity and efficiency\n\n'+table(runtime)+'\nShared initialization and old-wrapper equivalence are recorded in initialization_audit.json. Structural, gradient-connectivity and causality checks are in structural_sanity.json and causality_audit.json. Active parameter counts measure autograd-connected tensors; nonzero elements are batch-specific, not theoretical capacity. Full-window spatial pooling only uses observed inputs; prefix invariance is tested on causal temporal states, not the complete-window forecast.\n'
    text+='\n## 9. Minimal mechanisms and supplementary reuse\n\n```json\n'+json.dumps({n:e.get('mechanism',{}) for n,e in results.items() if e.get('mechanism')},ensure_ascii=False,indent=2)+'\n```\n'
    supplementary=r['supplementary']
    klrows=[];modrows=[]
    for n,e in {**({'Full D0B':results['Full D0B']} if 'Full D0B' in results else {}),**supplementary}.items():
        metric=e['metrics']['test']['5']
        if n in ('Full D0B','w/o Switch KL'):
            reg=e.get('mechanism',{}).get('regime',{})
            klrows.append(dict(Variant=n,TEST5_MAE=metric['MAE'],Entropy=reg.get('entropy'),HardOccupancy=reg.get('hard_occupancy')))
        if n!='w/o Switch KL':modrows.append(dict(Variant=n,TEST5_MAE=metric['MAE'],TEST5_MSE=metric['MSE']))
    csv_file(out/'regime_regularization_mechanism.csv',klrows);csv_file(out/'multimodal_input_sensitivity.csv',modrows)
    text+='\nREGIME REGULARIZATION MECHANISM (reuse-only; excluded from main table)\n\n'+table(klrows)
    text+='\nMULTIMODAL INPUT SENSITIVITY (reuse-only; excluded from main table)\n\n'+table(modrows)
    text+='\n## 10. Required answers\n\n'
    for i,(question,c) in enumerate(zip(QUESTIONS,contribution),1):text+=f"{i}. {question} **{c['Supported']}**. {c['Evidence']}\n\n"
    complete=all(row['DeltaMAE'] is not None for row in rows)
    if complete:
        worst=max(rows[:-1],key=lambda v:v['DeltaMAE']);weakest=min(rows[:-1],key=lambda v:v['DeltaMAE'])
        text+=f"12. Largest degradation: {worst['Variant']} (delta {worst['DeltaMAE']:.10g}).\n\n13. Weakest empirical support by signed delta: {weakest['Variant']}. Near ties receive no strong conclusion.\n\n"
    else:text+='12. PENDING: all 11 controls required to rank degradation.\n\n13. PENDING: all 11 controls required to rank weakest support.\n\n'
    better_rows=[v['Variant'] for v in rows[:-1] if v['DeltaMAE'] is not None and v['DeltaMAE']<0]
    text+='14. Completed ablations better than Full: '+(', '.join(better_rows) if better_rows else 'none among completed results')+('.\n\n' if complete else '; remaining comparisons pending.\n\n')
    text+='15. A1/A2 assess net branch contribution; A3–A10 assess local subsystem mechanisms; A11 assesses adaptive vs fixed fusion. Local benefit does not establish net branch benefit. No second seed, tuning, extra ablation or model optimization follows.\n\nSTOP\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
