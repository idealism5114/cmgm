"""Partial-safe reports for the predefined, single-run benchmark."""
import csv
import io
import json
import os
from pathlib import Path
from cmgm.models.formal_baselines_v2 import ORDER
from cmgm.scripts.fixed_baseline_protocol import FIXED,TRAINABLE_NEURAL
from cmgm.scripts.formal_v2_protocol import atomic_json

CATEGORY={n:('Traditional ML' if n in ORDER[:3] else 'Deep temporal' if n in ORDER[3:6] else 'Graph' if n in ORDER[6:8] else 'Proposed') for n in ORDER}
ARCHITECTURE={
 'Ridge Regression':'sklearn.linear_model.Ridge; native multi-target regression',
 'Random Forest':'sklearn.ensemble.RandomForestRegressor; native multi-output regression',
 'XGBoost':'sklearn.multioutput.MultiOutputRegressor(xgboost.XGBRegressor); 96 outputs',
 'LSTM':'PyTorch nn.LSTM: input=N*21, hidden128, layers2, dropout.1, unidirectional; Linear128→96',
 'TCN':'Conv1d(N*21,128,1); three residual blocks, two causal Conv1d each, k3, dilation1/2/4, dropout.1, ReLU; Linear128→96',
 'Vanilla Transformer':'Linear(N*21,128), sinusoidal absolute PE, PyTorch TransformerEncoder layers2 heads4 FF256 ReLU dropout.1, causal mask; Linear128→96',
 'Graph WaveNet':'Vendored official Graph WaveNet; adaptive adjacency only; full-node task wrapper (ADAPTATION_NOTES.md)',
 'MTGNN':'Vendored official MTGNN; directed graph learning, mix-hop, dilated inception, residual/skip; full-node task wrapper (ADAPTATION_NOTES.md)',
 'D0B':'Existing frozen switching_latent_balanced_readout; original formal architecture',
}


def write(path,text):
    tmp=Path(str(path)+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)


def csv_file(out,name,fields,rows):
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);write(out/name,f.getvalue())


def table(headers,rows):
    def cell(v):return str(v).replace('|',';').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])


def fmt(v):return 'Pending / N/A' if v is None else f'{v:.10g}'


def report(r,out):
    out=Path(out);complete=len(r['models'])==9;rows=[];multi=[];commodity=[];runtime=[];config_rows=[]
    for n in ORDER:
        entry=r['models'].get(n,{});summary=entry.get('summary',{});status=entry.get('status',r.get('jobs',{}).get(n,{}).get('status','PENDING'))
        row=dict(Model=n,Category=CATEGORY[n],Configuration=json.dumps(FIXED[n],ensure_ascii=False),Status=status,
                 Runtime=summary.get('train_seconds'),BestEpoch=summary.get('best_epoch'),Seed='historical / unknown' if n=='D0B' else 42)
        for s in ('val','test'):
            for k in ('MAE','MSE','RMSE','Hit'):
                v=entry.get('metrics',{}).get(s,{}).get('5',{}).get(k)
                row[f'{s.upper()}5_{k}']=v*100 if k=='Hit' and v is not None else v
        rows.append(row)
        for s,hs in entry.get('metrics',{}).items():
            for h,vals in hs.items():multi.append(dict(Model=n,Split=s,Horizon=h,**{k:(v*100 if k=='Hit' else v) for k,v in vals.items()}))
        commodity.extend(dict(Model=n,**v) for v in entry.get('per_commodity',[]))
        runtime.append(dict(Model=n,Seconds=summary.get('train_seconds'),BestEpoch=summary.get('best_epoch'),Device=summary.get('device','CPU' if n in ORDER[:3] else None),
            Complexity=summary.get('complexity',300 if n=='Random Forest' and entry else None),Unit=summary.get('complexity_unit','trees' if n=='Random Forest' and entry else None),
            PeakGPUBytes=summary.get('peak_gpu_bytes'),PeakRSSKiB=summary.get('peak_cpu_RSS_KiB'),Source=entry.get('path'),SHA256=entry.get('sha256')))
        config_rows.append([n,ARCHITECTURE[n],json.dumps(FIXED[n],ensure_ascii=False),row['Seed'],'(B,20,N,21)','(B,4,24)'])
    csv_file(out,'single_run_results.csv',list(rows[0]),rows)
    csv_file(out,'multi_horizon_results.csv',['Model','Split','Horizon','MAE','MSE','RMSE','Hit'],multi)
    csv_file(out,'per_commodity_results.csv',['Model','commodity','MAE','MSE'],commodity)
    csv_file(out,'runtime_summary.csv',list(runtime[0]),runtime)
    atomic_json(out/'sanity_checks.json',dict(models=r['sanity'],prediction_audits={n:e['prediction_audit'] for n,e in r['models'].items()},input_equality=r['input_audit']['PASS']))
    statuses=[]
    for n in ORDER:
        p=r['plan'][n]
        statuses.append([n,p['old_run_exists'],p['architecture_valid'],p['full_input_valid'],p['seed42'],p['fixed_config_matches'],p['sanity_PASS'],p['action'],r['models'].get(n,{}).get('status',r.get('jobs',{}).get(n,{}).get('status','PENDING')),p['reason']])
    write(out/'RUN_STATUS.md','# Fixed single-run status\n\n'+str(r['status'])+'\n\n'+table(['Model','Old run?','Architecture','Full input','Seed42','Fixed config','Sanity','Decision','Status','Reason'],statuses)+'\n\nOld experiment preserved: '+str(r.get('old_experiment'))+'\n\nOld process stop evidence: '+json.dumps(r.get('old_process_stop',{}),ensure_ascii=False)+'\n\n'+json.dumps(r['computation'],indent=2)+'\n\nOnly missing fixed configurations may be fitted. Reused artifacts are identified by SHA256 in results.json and runtime_summary.csv.\n')
    sections=['# FORMAL BASELINE BENCHMARK V2 — FIXED CONFIG SINGLE RUN',
       '**Status: '+r['status']+'**. '+('All nine rows are available.' if complete else 'Incomplete: pending metrics are not valid results and are not ranked.'),
       '## 1. Protocol',r['protocol']['statement'],
       'D0B is the existing frozen formal checkpoint. New baselines are seed-42 single runs. All nine models receive exactly the same 20 × N × 21 historical information set. No node/feature deletion, market pooling, additional standardization or target transformation is permitted.',
       'Classical models flatten history losslessly (time, node, feature); sequence models flatten node/feature per timestep; graph models transpose to (B,F,N,T). Targets flatten horizon-major, commodity-minor. Actual dimensions, source rows, chronological split fingerprints and commodity node/target/output mapping are recorded in input_equality_audit.json.',
       'Neural training: sum of four Huber losses, delta=.02; Adam lr=1e-4, wd=1e-5, batch64, epochs≤200, patience10. Scheduler, checkpoint selection and early stopping use the same original batch-mean multi-horizon VAL Huber objective. VAL5 MAE/MSE are secondary logs only. Historical chronological loader uses shuffle=False and train drop_last=True; evaluation includes all origins.',
       'Evaluation: pooled origin ×24 commodities; MAE=mean(abs(error)), MSE=mean(error²), RMSE=sqrt(MSE), unmasked sign Hit includes zero targets. CSV/main tables display Hit in percent; results.json preserves Hit as a fraction. Metrics accumulate in float64; RMSE/MSE round-trip uses floating-point relative tolerance.',
       '## 2. Fixed definitions and provenance',table(['Model','Implementation / architecture','Fixed configuration','Seed','Source input','Output'],config_rows),
       'Graph source commits, verified file hashes and package versions are in baseline_provenance.json. Exact official wrapper configurations and compatibility changes are in ADAPTATION_NOTES.md. No graph blocks were replaced. Official MTGNN normalizes across the observed window; a strict streaming-prefix hidden-state invariance claim is not made for that normalization. Tests audit official causal convolution and legal observed-window access separately.',
       '## 3. Reuse and computation',json.dumps(r['computation'],indent=2),
       'See RUN_STATUS.md for all old-artifact eligibility decisions. Old neural fits selected/stopped by VAL5 MAE cannot establish equivalence to this multi-horizon selection protocol, even at the same LR. Only exact fixed-config artifacts are reused; TEST performance never determines eligibility.',
       'Parallelism: '+r['parallelism']+f"; cpu_jobs={r['cpu_jobs']}. One neural model at a time. Classical models remain standard CPU implementations.",
       '## 4. Sanity',table(['Model','Input audit','Model sanity','Status'],[[n,r['input_audit']['PASS'],r['sanity'].get(n,{}).get('PASS','PENDING'),r['models'].get(n,{}).get('status','PENDING')] for n in ORDER]),
       'Full numeric errors, output shapes, permutation/single-sample/causality checks and prediction mean/std/min/max/P1/P50/P99/finite fraction are saved in sanity_checks.json. A failed core check prevents a valid metric row.',
       '## 5. FORMAL SINGLE-RUN BASELINE BENCHMARK — TEST 5D',
       table(['Model','Category','MAE ↓','MSE ↓','RMSE ↓','Hit% ↑','Runtime seconds','Status'],[[v['Model'],v['Category'],*[fmt(v['TEST5_'+k]) for k in ('MAE','MSE','RMSE','Hit')],fmt(v['Runtime']),v['Status']] for v in rows]),
       '## 6. Validation',table(['Model','VAL5 MAE','VAL5 MSE','VAL5 RMSE','VAL5 Hit%'],[[v['Model'],*[fmt(v['VAL5_'+k]) for k in ('MAE','MSE','RMSE','Hit')]] for v in rows]),
       '## 7. MULTI-HORIZON MAE',table(['Model','1d','5d','10d','20d'],[[n,*[fmt(r['models'].get(n,{}).get('metrics',{}).get('test',{}).get(str(h),{}).get('MAE')) for h in (1,5,10,20)]] for n in ORDER]),
       '## 8. Efficiency',table(['Model','Seconds','Best epoch','Device','Complexity','Unit'],[[v['Model'],fmt(v['Seconds']),v['BestEpoch'],v['Device'],v['Complexity'],v['Unit']] for v in runtime]),
       'D0B runtime is historical / N/A and is not fabricated or compared as a new fit. Reused RF runtime is from its original fit. Peak RSS where available is a process high-water mark, not isolated model memory.',
       '## 9. Result interpretation']
    if complete:
        def err(n,h=5):return r['models'][n]['metrics']['test'][str(h)]['MAE']
        winners=[min(group,key=err) for group in (ORDER[:3],ORDER[3:6],ORDER[6:8],ORDER[:8])]
        sections.append('\n\n'.join(f'{label}: {n} (TEST5 MAE {err(n):.10g}).' for label,n in zip(['Best traditional ML baseline','Best deep temporal baseline','Best graph baseline','Best overall baseline'],winners)))
        rank=1+sum(err(n)<err('D0B') for n in ORDER if n!='D0B');sections.append(f'D0B TEST5 rank: {rank}/9.')
        for n in winners[:3]:
            delta=err('D0B')-err(n);sections.append(f'D0B − {n}: MAE delta={delta:.10g}, relative={100*delta/err(n):.6g}%. '+(f'{n} outperforms D0B on TEST5 MAE under the fixed protocol.' if delta>0 else ''))
        sections.append('Multi-horizon ranking: '+ '; '.join(f'{h}d: '+', '.join(sorted(ORDER,key=lambda n:err(n,h))) for h in (1,5,10,20)))
    else:sections.append('Best models, D0B rank and multi-horizon conclusions remain pending until all nine valid results are available.')
    sections.extend(['## 10. Fairness declaration',table(['Question','Answer'],[
        ['Did every evaluated model receive the full same historical information set?','YES' if r['input_audit']['PASS'] else 'NO'],
        ['Was any node removed for any baseline?','NO'],['Was any feature removed?','NO'],['Was any 284→28 compression used?','NO'],
        ['Were GraphWaveNet and MTGNN official/faithful implementations?','YES — vendored source hashes verified; adaptations disclosed'],
        ['Was TEST used for model/hyperparameter selection?','NO'],
        ['Was each new model fitted once under its fixed configuration?', 'YES (RF reused, D0B frozen)' if complete else 'PENDING — missing fits have not run'],
        ['Were any results rerun because performance was poor?','NO'],['Was D0B modified or retrained?','NO']]),
        'Validity caveat: single-run predefined-configuration benchmark; no multi-seed statistical claim. STOP after completing this table.'])
    write(out/'REPORT.md','\n\n'.join(sections)+'\n')
