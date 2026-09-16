"""Two fixed ablation tables, with deltas exclusively against the new Full control."""
import json
from cmgm.models.formal_d0b_ablation import NAMES
from cmgm.scripts.fixed_baseline_report import write,csv_file,table,fmt
from cmgm.scripts.formal_v2_protocol import atomic_json


def report(r,out):
    full=r['results'].get(NAMES[0],{}).get('metrics',{}).get('test',{}).get('5');rows=[];multi=[];per=[];runtime=[];mechanism=[]
    for n in NAMES:
        result=r['results'].get(n,{});m=result.get('metrics',{}).get('test',{}).get('5',{})
        row=dict(Variant=n,Category='Full' if n==NAMES[0] else 'Information' if n in NAMES[1:4] else 'Branch' if n in NAMES[4:6] else 'Spatial' if n in NAMES[6:8] else 'Temporal' if n in NAMES[8:12] else 'Fusion',
                 **{k:m.get(k)*100 if k=='Hit' and k in m else m.get(k) for k in ('MAE','MSE','RMSE','Hit')})
        row.update(DeltaMAE=m['MAE']-full['MAE'] if full and m else None,DeltaMSE=m['MSE']-full['MSE'] if full and m else None,
            RelativeDeltaMAE=100*(m['MAE']-full['MAE'])/full['MAE'] if full and m else None)
        rows.append(row)
        for split,hs in result.get('metrics',{}).items():
            for h,v in hs.items():multi.append(dict(Variant=n,Split=split,Horizon=h,**{k:(100*x if k=='Hit' else x) for k,x in v.items()}))
        per.extend(dict(Variant=n,**v) for v in result.get('per_commodity',[]))
        runtime.append(dict(Variant=n,**result.get('runtime',{})))
        for k,v in result.get('mechanism',{}).items():mechanism.append(dict(Variant=n,Scope=k,Values=json.dumps(v)))
    by={v['Variant']:v for v in rows};a=[by[n] for n in (*NAMES[1:4],NAMES[0])];b=[by[n] for n in (*NAMES[4:],NAMES[0])]
    for row in a:
        row.update(Stock=row['Variant'] not in ('CommodityOnly','w/o Stock'),Bond=row['Variant'] not in ('CommodityOnly','w/o Bond'),Commodity=True)
    csv_file(out,'multimodal_ablation.csv',['Variant','Stock','Bond','Commodity','MAE','DeltaMAE','RelativeDeltaMAE','MSE','DeltaMSE','RMSE','Hit'],a)
    csv_file(out,'architecture_ablation.csv',['Variant','Category','MAE','DeltaMAE','RelativeDeltaMAE','MSE','DeltaMSE','RMSE','Hit'],b)
    mapping=[('SpatialOnly','s',NAMES[5]),('TemporalOnly','t',NAMES[4]),('Fixed Fusion','0.5s+0.5t',NAMES[12]),('Adaptive Fusion','g*t+(1-g)*s',NAMES[0])]
    complements=[dict(Representation=label,Formula=formula,MAE=by[n]['MAE'],MSE=by[n]['MSE']) for label,formula,n in mapping]
    csv_file(out,'spatial_temporal_complementarity.csv',['Representation','Formula','MAE','MSE'],complements)
    csv_file(out,'multi_horizon_ablation.csv',['Variant','Split','Horizon','MAE','MSE','RMSE','Hit'],multi)
    csv_file(out,'per_commodity_ablation.csv',['Variant','commodity','MAE','MSE'],per)
    csv_file(out,'runtime_summary.csv',['Variant','best_epoch','train_seconds','seconds_per_epoch','val5_at_best'],runtime)
    csv_file(out,'mechanism_summary.csv',['Variant','Scope','Values'],mechanism)
    atomic_json(out/'initialization_audit.json',r['initialization']);atomic_json(out/'sanity_checks.json',dict(initial=r['sanity'],best={n:v['sanity'] for n,v in r['results'].items()},data=r['data']))
    atomic_json(out/'masking_audit.json',dict(definition=r['protocol']['neutral'],variants=r['masking']))
    status_rows=[[n,r['existing_audit'][n]['existing_checkpoint'],r['existing_audit'][n]['exact_protocol_match'],r['existing_audit'][n]['reuse'],r['existing_audit'][n]['need_training'],r['sanity'].get(n,{}).get('PASS','PENDING'),r['jobs'].get(n,{}).get('status','PENDING')] for n in NAMES]
    write(out/'RUN_STATUS.md','# Formal ablation status\n\n'+r['status']+'\n\n'+table(['Variant','Checkpoint inventory exists?','Exact match','Reuse','Need training','Sanity','Status'],status_rows)+'\n\n'+r['reuse_policy']+'\n')
    def main_table(items,information=False):
        headers=['Variant']+(['Stock','Bond','Commodity'] if information else ['Category'])+['MAE ↓','ΔMAE','Rel.ΔMAE%','MSE ↓','RMSE ↓','Hit% ↑']
        return table(headers,[[v['Variant']]+([v['Stock'],v['Bond'],v['Commodity']] if information else [v['Category']])+[fmt(v[k]) for k in ('MAE','DeltaMAE','RelativeDeltaMAE','MSE','RMSE','Hit')] for v in items])
    def support(n):
        v=by[n]
        if v['RelativeDeltaMAE'] is None:return 'PENDING'
        if abs(v['RelativeDeltaMAE'])<.1:return 'MIXED — near-tie under the single-run protocol'
        return 'YES — consistent with a beneficial contribution' if v['DeltaMAE']>0 else 'NO — this ablation outperforms FullD0B-Control under the single-run protocol'
    def complement():
        values=[by[n]['MAE'] for n in (NAMES[0],NAMES[4],NAMES[5])]
        if any(v is None for v in values):return 'PENDING'
        return 'YES' if values[0]<min(values[1:]) else 'NO'
    sections=['# D0B FORMAL ABLATION STUDY','Status: '+r['status'],
        '## 1. Protocol',json.dumps(r['protocol'],indent=2,ensure_ascii=False),
        'Original data/splits, node/feature preprocessing and clipped targets are unchanged. All configurations instantiate full native D0B in identical order; inactive modules are bypassed, not replaced or resized. Every stochastic fit is seed42. Full control is trained once and shared across both tables. No grid, extra seed, automatic model development or performance-based retries.',
        'The neutral mask is zero because main_ablation.build_data explicitly subtracts per-node/channel TRAIN means and divides by TRAIN standard deviations. No VAL/TEST statistics enter masking. Commodity order and split fingerprints are in sanity_checks.json.',
        '## 2. Full reference',
        'Historical D0B (evaluator sanity only): '+json.dumps(r.get('historical',{}).get('metrics',{}).get('test',{}).get('5',{})),
        '**All formal ablation deltas use FullD0B-Control(seed42), not the historical checkpoint.** Missing Full-control metrics leave all deltas pending. Hit in tables is percent; JSON is fraction. MAE/MSE pool all origins ×24 commodities; RMSE=sqrt(MSE); unmasked sign Hit includes zero targets.',
        '## 3. MULTIMODAL INFORMATION ABLATION — TEST 5D',main_table(a,True),
        *[f'{n}: {support(n)}.' for n in NAMES[1:4]],
        '## 4. Branch complementarity',table(['Representation','Formula','TEST5 MAE','MSE'],[[v['Representation'],v['Formula'],fmt(v['MAE']),fmt(v['MSE'])] for v in complements]),
        'Does Full outperform both single branches? '+complement(),
        '## 5. ARCHITECTURE ABLATION — TEST 5D',main_table(b),
        '## 6. Temporal/switching objective exceptions',
        'NoTemporal: Switch KL disabled because the entire temporal/switching branch is ablated. NoSwitchKL: contribution is an independent zero while p/Z/readout stay active. NoMarkov: p=prior=uniform, generators and recurrence retained, original KL is structurally zero without routing gradients. NoMicro: post-normalization micro channel zeroed while original switching KL stays active.',
        '## 7. Fusion', 'SpatialOnly=s; TemporalOnly=t; Fixed=0.5s+0.5t; Adaptive retains original gate and projections. No additional experiment is trained for this subtable.',
        '## 8. Minimal mechanism evidence',table(['Variant','Scope','Values'],[[v['Variant'],v['Scope'],v['Values']] for v in mechanism]),
        'Only Full/NoKL TEST mean p, entropy and hard occupancy, and Full/NoBalanced long/micro norms are reported. Higher entropy alone does not establish anti-collapse forecasting value. No numerical threshold for collapse is invented.',
        '## 9. Contribution summary',table(['Claimed component','Ablation','Supported?','TEST5 ΔMAE'],[[n,n,support(n),fmt(by[n]['DeltaMAE'])] for n in NAMES[1:]]),
        'Single-run differences do not prove causality or statistical significance. Relative absolute MAE differences below0.1% are near-ties, not a selection criterion.',
        '## 10. Parameter and structural audit',table(['Variant','Instantiated','Active prediction path tensor elements','Defined gradient elements','Nonzero gradient elements','Shared init diff','PASS'],[[n,r['initialization'].get(n,{}).get('total_instantiated'),*[r['sanity'].get(n,{}).get('parameter_counts',{}).get(k) for k in ('active_prediction_path_tensor_elements','defined_prediction_gradient_elements','nonzero_prediction_gradient_elements')],r['initialization'].get(n,{}).get('max_abs_diff'),r['sanity'].get(n,{}).get('PASS','PENDING')] for n in NAMES]),
        'Active-path counts use autograd connectivity at parameter-tensor granularity. Fixed-batch nonzero gradient counts can be smaller than active capacity. Prefix causality applies to temporal internal states/readouts; the spatial/final forecast represents the complete legally observed window and is not invariant to perturbing its observed timesteps.',
        '## 11. Required answers']
    questions=[('Does multimodal stock/bond information improve Full D0B?',support(NAMES[1])),('Does the spatial branch contribute?',support(NAMES[4])),('Does the temporal branch contribute?',support(NAMES[5])),('Does Full outperform SpatialOnly and TemporalOnly?',complement()),('Does graph propagation contribute?',support(NAMES[6])),('Does TempWeighted contribute?',support(NAMES[7])),('Does adaptive Markov switching contribute beyond uniform routing?',support(NAMES[8])),('Does the microstate prediction channel contribute?',support(NAMES[9])),('Does Switch KL prevent collapse and improve forecasting?',support(NAMES[10])+'; regime statistics above must be assessed separately'),('Does Balanced Readout improve forecasting and scale balance?',support(NAMES[11])+'; norm ratios above describe balance'),('Does adaptive gate outperform fixed fusion?',support(NAMES[12]))]
    valid=[v for v in rows[1:] if v['DeltaMAE'] is not None]
    questions.extend([('Largest TEST5 degradation?',max(valid,key=lambda v:v['DeltaMAE'])['Variant'] if len(valid)==12 else 'PENDING'),('Weakest empirical support?',min(valid,key=lambda v:v['DeltaMAE'])['Variant'] if len(valid)==12 else 'PENDING'),('Any ablations better than Full?',', '.join(v['Variant'] for v in valid if v['DeltaMAE']<0) or ('NO' if len(valid)==12 else 'PENDING'))])
    sections.append('\n\n'.join(f'{i}. {q} {answer}' for i,(q,answer) in enumerate(questions,1)))
    sections.extend(['## Runtime',table(['Variant','Best epoch','Train seconds','Seconds/epoch'],[[n,*[r['results'].get(n,{}).get('runtime',{}).get(k,'PENDING') for k in ('best_epoch','train_seconds','seconds_per_epoch')]] for n in NAMES]),'STOP. No additional seeds, tuning, ablations or model optimization.'])
    write(out/'REPORT.md','\n\n'.join(sections)+'\n')
