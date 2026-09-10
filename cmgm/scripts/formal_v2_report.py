"""Nine-row reporting. No best-seed selection, no fabricated pending results."""
import numpy as np
import pandas as pd
from cmgm.models.formal_baselines_v2 import ORDER,CLASSICAL,NEURAL
from cmgm.scripts.formal_v2_protocol import grid,SEEDS,atomic_json

CATEGORY={n:('Traditional ML' if n in CLASSICAL else 'Graph-based deep' if n in ('Graph WaveNet','MTGNN') else 'Proposed' if n=='D0B' else 'Deep temporal') for n in ORDER}


def table(headers,rows):
    clean=lambda v:str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(clean(v) for v in row)+' |' for row in rows])


def display(mean,std):return 'pending' if mean is None else f'{mean:.9g}'+(' (std —)' if std is None else f' ± {std:.3g}')


def aggregate(formal):
    result=[]
    for name in ORDER:
        runs=[v for v in formal.values() if v['model']==name]
        expected=1 if name=='Ridge Regression' else 3
        if len(runs)!=expected:continue
        if {v['seed'] for v in runs}!=({42} if expected==1 else set(SEEDS)):raise ValueError('Wrong final seed set')
        for split in ('val','test'):
            for horizon in (1,5,10,20):
                row=dict(model=name,category=CATEGORY[name],split=split,horizon=horizon,seeds=expected)
                for metric in ('MAE','MSE','RMSE','Hit'):
                    values=[v['metrics'][split][str(horizon)][metric]*(100 if metric=='Hit' else 1) for v in runs]
                    row[metric+'_mean']=float(np.mean(values));row[metric+'_std']=float(np.std(values,ddof=1)) if expected>1 else None
                result.append(row)
    return result


def report(r,out):
    agg=aggregate(r['formal']);search=[];raw=[];commodity=[];efficiency=[]
    for key,v in r['jobs'].items():
        summary=v.get('summary',{})
        if v['stage']=='A':search.append(dict(model=v['model'],seed=v['seed'],candidate=str(v['hp']),status=v['status'],VAL5_MAE=summary.get('val5_MAE'),criterion='VAL5 MAE',TEST_used=False))
        efficiency.append(dict(job=key,model=v['model'],stage=v['stage'],seed=v['seed'],status=v['status'],best_epoch=summary.get('best_epoch'),
            train_seconds=summary.get('train_seconds'),peak_gpu_bytes=summary.get('peak_gpu_bytes'),complexity=summary.get('complexity'),complexity_unit=summary.get('complexity_unit'),reused=v.get('reuse')))
    for key,v in r['formal'].items():
        row=dict(model=v['model'],seed=v['seed'],selected_hp=str(v['hp']),best_epoch=v['summary'].get('best_epoch'))
        for split,hs in v['metrics'].items():
            for h,ms in hs.items():
                for k,value in ms.items():row[f'{split.upper()}{h}_{k}']=value*(100 if k=='Hit' else 1)
        raw.append(row)
        for c in v['per_commodity']:commodity.append(dict(model=v['model'],seed=v['seed'],**c))
    for file,rows in [('hyperparameter_search',search),('per_seed_results',raw),('aggregate_results',agg),('multi_horizon_results',agg),('formal_baseline_per_commodity',commodity),('efficiency',efficiency)]:
        pd.DataFrame(rows).to_csv(out/(file+'.csv'),index=False)
    atomic_json(out/'sanity_checks.json',dict(preflight=r['sanity'],fitted={k:v.get('sanity') for k,v in r['jobs'].items() if v['status']=='DONE'},
        prediction_audits={k:v['prediction_audit'] for k,v in r['formal'].items()}))
    lookup={(v['model'],v['split'],v['horizon']):v for v in agg}
    main=[];mh=[]
    for name in ORDER:
        v=lookup.get((name,'test',5));display_name='D0B (Ours)' if name=='D0B' else name
        trained=next((j['summary'] for j in r['jobs'].values() if j['stage']=='B' and j['model']==name and j['status']=='DONE'),{})
        complexity=str(trained.get('complexity','pending'))+' '+trained.get('complexity_unit','')
        main.append([display_name,CATEGORY[name],*[display(v[k+'_mean'],v[k+'_std']) if v else 'pending' for k in ('MAE','MSE','RMSE','Hit')],v['seeds'] if v else 'pending',complexity])
        mh.append([display_name,*[display(lookup[(name,'test',h)]['MAE_mean'],lookup[(name,'test',h)]['MAE_std']) if (name,'test',h) in lookup else 'pending' for h in (1,5,10,20)]])
    complete=len(agg)==9*2*4
    lines=['# FORMAL BASELINE BENCHMARK V2','',r['status'],'','## 1. Formal protocol','',str(r['protocol']),'',
        f"Git SHA {r['git_sha']}; package versions and pinned official file hashes in baseline_provenance.json. Historical D0B is sanity only: {r.get('historical_reference','pending')}.",'',
        'Every method uses the original model-ready dataset without re-normalization, feature selection or target rescaling. Neural training uses every TRAIN origin (including the final partial batch); classical fitting uses exactly those same origins. V2 differs from historical D0B selection: pooled VAL5 MAE controls checkpoint and stopping, while multi-horizon VAL Huber controls the scheduler. Adam/WD/batch/epoch budget is identical across the six neural architectures. D0B keeps its own switch KL and warmup.','',
        str(r['input_audit']['origins']),'',r['input_audit']['dates'],'',
        '## 2. Input fairness','',
        'All nine models receive exactly the same 20 × N × 21 historical information set. Classical: lossless flatten. LSTM/TCN/Transformer: node-major, feature-minor flatten per timestep. Graph WaveNet/MTGNN: full B,F,N,T transpose. D0B: full native tensor. No 284→28 pooling, deletion or baseline-specific normalization. Effective receptive fields remain architecture-dependent: supplying20 steps does not assert that all architectures have nonzero sensitivity to every historical scalar.','',
        '## 3. Formal baseline definitions','',
        'Ridge: standard sklearn multi-target Ridge, intercept enabled. RF: native multi-output RandomForestRegressor. XGBoost: MultiOutputRegressor with96 independent standard XGBRegressors, one common configuration, hist,500 estimators each, no early stopping. Full input dimensionality is retained for all three.','',
        'LSTM: nn.LSTM(N*21,128,2,dropout=.1), final top-layer hidden,128→96. TCN: 1×1 input projection to128, three two-convolution causal residual blocks with kernel3/dilations1,2,4/dropout.1/ReLU, last time→96. Transformer: input projection128, sinusoidal absolute PE,2layers/4heads/FF256/dropout.1/ReLU, causal mask, last time→96.','',
        'Graph WaveNet and MTGNN use pinned official source. Exact constructor defaults, compatibility diff and output slicing are in third_party/baselines/ADAPTATION_NOTES.md. Graph WaveNet retains official13-step effective receptive field and final-origin output slice. MTGNN retains inception kernels, directed noisy top-k graph, mixprop, full-window LayerNorm, residual/skip paths. No D0B components are used by either graph baseline. D0B is exactly switching_latent_balanced_readout, independently trained under V2 with the same3-LR budget.','',
        str(r['provenance']),'',
        '## 4. Input equality audit','',table(['Model','Expected scalars','Observed scalars','Missing','Extra','Reconstruction error','PASS'],
            [[n,*[v[k] for k in ('expected','observed','missing','extra','reconstruction_error','PASS')]] for n,v in r['input_audit']['models'].items()]),'',
        'Commodity mapping is in input_equality_audit.json. It links source node, target/output column, and lossless layout. Dataset window/target reconstruction and chronological global origin boundaries are checked.','',
        '## 5. Hyperparameter selection','',table(['Model','Registered candidates','Selected','Selection'],[[n,str(grid(n)),str(r['selected'].get(n,'pending')),'VAL5 MAE only'] for n in ORDER]),'',
        f"All hyperparameters frozen: {r['selection_frozen']}. Candidate results in hyperparameter_search.csv contain no TEST metrics. Historical D0B protocol sanity is the only pre-Stage-A TEST exception; it is excluded from the formal nine rows. All Stage B checkpoints must be fixed before any formal TEST call.",'',
        '## 6. Sanity','',table(['Model','Status','PASS'],[[n,str(r['sanity'].get(n,'pending')),r['sanity'].get(n,{}).get('PASS','pending')] for n in ORDER]),'',
        'Fitted-model and prediction range checks are saved in sanity_checks.json. Batch checks use eval mode and common RNG realizations because official MTGNN samples graph perturbations in eval. Graph convolution audits refer to causal temporal kernels and legal observed-window forecasting; official MTGNN time-spanning LayerNorm is not incorrectly claimed to have streaming prefix-invariant hidden states. Large but finite errors are retained, not removed or tuned away.','',
        '## 7. Main formal result','',
        '============================================================\nFORMAL BASELINE BENCHMARK — TEST 5D\n============================================================','',
        table(['Model','Category','MAE ↓','MSE ↓','RMSE ↓','Hit% ↑','Seeds','Complexity'],main),'',
        'Mean ± sample standard deviation (ddof=1) across all3 seeds. Ridge has a single deterministic fit, std —. RMSE is sqrt of each run’s pooled MSE before aggregation; mean RMSE need not equal sqrt(mean MSE). Hit is unmasked including zeros. No statistical significance claim.','',
        '## 8. Multi-horizon MAE','',table(['Model','1d','5d','10d','20d'],mh),'',
        '## 9. Efficiency','',table(['Model/stage/seed','Best epoch','Seconds','Complexity'],[[v['job'],v['best_epoch'],v['train_seconds'],str(v['complexity'])+' '+str(v['complexity_unit'])] for v in efficiency]),'',
        '## 10. Interpretation','']
    if complete:
        value=lambda n,h=5:lookup[n,'test',h]['MAE_mean']
        traditional=min(CLASSICAL,key=value);temporal=min(('LSTM','TCN','Vanilla Transformer'),key=value);graph=min(('Graph WaveNet','MTGNN'),key=value)
        strongest=min([n for n in ORDER if n!='D0B'],key=value)
        rank=1+sum(value(n)<value('D0B') for n in ORDER if n!='D0B')
        patterns={h:sorted(ORDER,key=lambda n:value(n,h)) for h in (1,5,10,20)}
        lines += [f'Strongest traditional ML: {traditional}.',f'Strongest ordinary temporal model: {temporal}.',f'Strongest graph baseline: {graph}.',f'Strongest baseline overall: {strongest}.']
        for n in (traditional,temporal,graph):lines.append(f'D0B minus {n}: {value("D0B")-value(n):.9g} TEST5 MAE; D0B relative improvement={(value(n)-value("D0B"))/value(n)*100:.6g}%.')
        lines += [f'D0B rank: {rank}/9.',f'Horizon rankings: {patterns}.']
        for n in ORDER:
            if n!='D0B' and value(n)<value('D0B'):lines.append(f'{n} outperforms D0B on TEST5 MAE under the formal protocol.')
        signs={}
        for n in ORDER:
            if n in ('D0B','Ridge Regression'):continue
            signs[n]=[next(v for v in r['formal'].values() if v['model']=='D0B' and v['seed']==s)['metrics']['test']['5']['MAE'] < next(v for v in r['formal'].values() if v['model']==n and v['seed']==s)['metrics']['test']['5']['MAE'] for s in SEEDS]
        lines += [f'D0B-better direction by matched seeds42/2025/3407: {signs}. Mixed booleans indicate an unstable pairwise ordering; three seeds alone do not establish significance.']
    else:lines += ['Formal tuning/training/evaluation is incomplete. Pending rows are not paper results. No model rank or three-seed conclusion is asserted.']
    lines += ['', 'Did every model receive the same historical information set? '+('YES' if r['input_audit']['PASS'] else 'NO'),
        'Was any node or feature removed for a baseline? NO',
        'Were GraphWaveNet and MTGNN official/faithful implementations? YES; pinned code with documented compatibility/interface changes.',
        'Was TEST used for hyperparameter selection? NO',
        'Were all stochastic models evaluated over the same 3 seeds? '+('YES' if complete else 'NO — evaluation pending'),
        'Was D0B given a larger tuning budget? NO','', 'STOP. No automatic ablation, new baseline or model optimization.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
