"""Paper-style tables with explicit input/fidelity and single-run caveats."""
import pandas as pd
from cmgm.models.comparison_baselines import HORIZONS,ORDER,TRAINABLE
from cmgm.scripts.d0b_risk_probe_analysis import table

DEFINITIONS='''ZeroReturn: zero tensor (B,4,24), no training.
Linear: deterministic sequence flatten (20×588) → Linear(11760,96), no hidden layer.
GRU: nn.GRU(588,128,2,dropout=.1,batch_first=True), final hidden → Linear(128,96), unidirectional.
VanillaTransformer: Linear(588,128), fixed sinusoidal absolute positions, two nn.TransformerEncoder layers (4 heads, FFN256, dropout.1), explicit causal temporal mask, last timestep → Linear(128,96).
iTransformer: the specified simplified inverted baseline, Linear(20,128) over each of 588 variate histories, two TransformerEncoder layers (4 heads, FFN256, dropout.1), reshape to 28×21 feature tokens, mean the 21 features of each commodity token4..27, shared Linear(128,4). No temporal mask across variates, no embeddings, no future histories. This is not a verbatim official implementation.
MTGNN: simplified MTGNN-style independent graph forecasting baseline, shared Linear(21,64), two causal gated temporal convolution blocks (kernel3, dilations1/2, hidden64), bidirectional MixHop propagation (K2,beta.05), residual ReLU update, final-time commodity nodes4..27 → shared Linear(64,4). The temporal receptive field is seven steps within the same 20-step input. Reuses only cmgm.graph.adaptive_graph.AdaptiveGraphLearner and cmgm.models.model.MixHopPropagation. The reused graph has embedding10, learnable alpha initialized .5, soft top-k10, antisymmetric adjacency; these repository choices differ from a verbatim canonical MTGNN. No EdgeAttention, TempWeighted, D0B gate or switching module is used.
All new architectures are in cmgm/models/comparison_baselines.py; no D0B source was modified for this suite.'''


def write_report(r,out):
    models=r['models'];valid={k:v for k,v in models.items() if 'metrics' in v and (k=='D0B' or r['sanity'].get(k,{}).get('PASS',False))}
    counts=[];training=[];overall=[];allh=[];sanity_rows=[];commodity=[]
    for name,v in models.items():
        p=v['parameters'];t=v.get('training',{})
        counts.append(dict(Model=name,TrainableParams=p['trainable'],NontrainableParams=p['nontrainable'],BufferElements=p['buffer_elements'],InputView=v['input_view']))
        training.append(dict(Model=name,Params=p['trainable']+p['nontrainable'],BestEpoch=t.get('best_epoch'),TrainSeconds=t.get('train_seconds'),SecondsPerEpochMean=t.get('seconds_per_epoch_mean'),
            FormalValObjective=t.get('best_val_objective'),SecondaryBestVAL5Epoch=t.get('secondary_best_val5_epoch'),Status='FROZEN_REFERENCE' if name=='D0B' else r['model_status'].get(name)))
        if name in r['sanity']:
            checks=r['sanity'][name]
            for stage in ('initial','best'):
                if stage not in checks:continue
                a=checks[stage];sanity_rows.append([name,stage,a['OutputShape'],a['Finite'],a['BatchPerm'],a['SingleSample'],a['Causal'],a['PASS']])
        if name not in valid:continue
        row=dict(Model='D0B (Ours)' if name=='D0B' else name,InputView=v['input_view'],Params=p['trainable']+p['nontrainable'],BestEpoch=t.get('best_epoch'),TrainSeconds=t.get('train_seconds'))
        for split,label in [('val','VAL5'),('test','TEST5')]:
            for k,value in v['metrics'][split]['5'].items():row[label+'_'+k]=value*100 if k=='Hit' else value
        d0b=models['D0B']['metrics']['test']['5']['MAE'];mae=v['metrics']['test']['5']['MAE']
        row.update(DeltaMAE_vs_D0B=mae-d0b,RelativeDeltaMAE_percent=100*(mae-d0b)/d0b,D0B_improvement_percent=100*(mae-d0b)/mae)
        overall.append(row)
        for split,horizons in v['metrics'].items():
            for h,m in horizons.items():allh.append(dict(Model=name,Split=split,Horizon=int(h),MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit'],RMSE_squared_error=abs(m['RMSE']**2-m['MSE'])))
    if valid:
        names=next(iter(valid.values()))['per_commodity']
        for i,c in enumerate(names):
            for metric in ('MAE','MSE'):
                commodity.append(dict(commodity=c['commodity'],metric=metric,**{name:v['per_commodity'][i][metric] for name,v in valid.items()}))
    overall.sort(key=lambda row:row['TEST5_MAE'])
    for filename,rows in [('baseline_overall',overall),('baseline_all_horizons',allh),('training_summary',training),('parameter_counts',counts),('per_commodity_baselines',commodity)]:
        pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
    r['tables']=dict(overall=overall,all_horizons=allh,training_summary=training,parameter_counts=counts)
    lines=['# Baseline comparison','',r['status'],'','## 1. Experimental protocol','',str(r['protocol']),'',
        f"Git SHA: {r.get('git_sha')}. Per-model source hashes are retained. D0B checksum: {r.get('baseline_checksum')}. Data fingerprints/order mapping: sanity_checks.json.",'',
        'The original dataset/build_data pipeline is reused, including chronological split boundaries, 21 features, target clipping/construction, 20-step windows and horizons [1,5,10,20]. No target renormalization. Training keeps chronological ordering and drops only the incomplete TRAIN batch as in current D0B. All reporting uses full loaders including that tail. Formal VAL Huber is an unweighted mean of batch means, matching the existing D0B selection protocol; final metrics are pooled over every origin ×24 commodities.','',
        'Every new trainable baseline uses Adam lr1e-4, WD1e-5, batch64, at most200 epochs, patience10 and original ReduceLROnPlateau protocol. No clipping, AMP, gradient accumulation or per-model tuning. D0B is never trained; its switch KL is a D0B mechanism and is not added to baselines.','',
        '## 2. Baseline definitions','',DEFINITIONS,'',
        'Neutral adapter: token0 stock_mean, token1 stock_std, token2 bond_mean, token3 bond_std, tokens4..27 commodity features. Population std (correction=0), no parameters. Output (B,20,28,21). Flatten is token-major then feature-major, variate index=21*token+feature. Commodity i occupies variates [(4+i)*21,(5+i)*21).','',
        '## 3. Fairness statement','',
        'Same dataset, chronological splits, feature families, target construction, historical input window, information timing, horizons, training budget and evaluator. D0B operates on its native full-node architecture; generic baselines use an architecture-appropriate deterministic neutral representation. These are NOT exactly identical architectural input representations. Stock/bond compression and differing receptive fields are explicit limitations. Simplified iTransformer/MTGNN should not be described as official-paper reproductions.','',
        'Single-run controlled comparison, one seed42. No statistical significance or multi-seed robustness claim. D0B is a historical checkpoint: unknown training metadata/time remain missing rather than fabricated. Its parameter count is architectural; all D0B parameters are frozen during this suite. Fixed sinusoidal position values are buffers, reported separately from parameters.','',
        '## 4. Sanity','',f"D0B reference reproduction: {r.get('reference_check')}",'',
        table(['Model','Stage','OutputShape','Finite','BatchPerm','SingleSample','Causal','PASS'],sanity_rows),'',
        'Only evaluation-mode forwards are used for these checks, with fixed VAL samples; no attention/gradient/risk probe. All raw FP32 errors and any scale-aware independent FP64 precision audit are retained in sanity_checks.json. Failed models do not produce valid comparison metrics.','',
        '## 5. Main result','',
        '============================================================\nBASELINE COMPARISON — STANDARDIZED TEST 5D\n============================================================','',
        table(['Model','Params','MAE↓','MSE↓','RMSE↓','Hit↑','ΔMAE vs D0B','Relative ΔMAE %','Train Seconds'],[[v[k] for k in ('Model','Params','TEST5_MAE','TEST5_MSE','TEST5_RMSE','TEST5_Hit','DeltaMAE_vs_D0B','RelativeDeltaMAE_percent','TrainSeconds')] for v in overall]),'',
        'Relative ΔMAE above divides New−D0B by D0B. D0B improvement below divides Baseline−D0B by the baseline. Hit is percent, zero targets included. Tables are sorted by TEST5 MAE, no automatic bold winner. D0B training time is unavailable, not zero.','',
        table(['Model','Params','MAE↓','MSE↓','RMSE↓','Hit↑'],[[v[k] for k in ('Model','Params','TEST5_MAE','TEST5_MSE','TEST5_RMSE','TEST5_Hit')] for v in overall]),'',
        table(['Model','VAL5 MAE','VAL5 MSE','VAL5 RMSE','VAL5 Hit%'],[[v[k] for k in ('Model','VAL5_MAE','VAL5_MSE','VAL5_RMSE','VAL5_Hit')] for v in overall]),'',
        '## 6. Multi-horizon result','',
        '============================================================\nMULTI-HORIZON MAE\n============================================================','',
        table(['Model','1d','5d','10d','20d'],[[('D0B (Ours)' if name=='D0B' else name),*[v['metrics']['test'][str(h)]['MAE'] for h in HORIZONS]] for name,v in valid.items()]),'',
        'All four metrics for TRAIN/VAL/TEST and all horizons are in baseline_all_horizons.csv. Per-commodity TEST5 MAE/MSE are in per_commodity_baselines.csv (metric column distinguishes MAE/MSE).','',
        '## 7. Efficiency','',table(['Model','Params','Best epoch','Train seconds','Mean seconds/epoch'],[[v[k] for k in ('Model','Params','BestEpoch','TrainSeconds','SecondsPerEpochMean')] for v in training]),'',
        '## 8. Result interpretation','']
    complete=all(name in valid and r['model_status'].get(name)=='COMPLETE' for name in ORDER)
    if not complete:
        lines+=['Five formal baseline trainings/evaluations are not all complete. Available ZeroReturn/D0B metrics are real; missing trained results are not fabricated. No strongest-baseline or D0B-rank conclusion yet.','',
            'Pending models: '+', '.join(name for name in ORDER if r['model_status'].get(name)!='COMPLETE'),'','STOP; user runs the fixed suite.']
    else:
        db=valid['D0B']['metrics']['test'];baseline_names=list(ORDER)
        strongest=min(baseline_names,key=lambda n:valid[n]['metrics']['test']['5']['MAE'])
        sb=valid[strongest]['metrics']['test']['5']['MAE'];rank=1+sum(valid[n]['metrics']['test']['5']['MAE']<db['5']['MAE'] for n in baseline_names)
        answers=[]
        for name in ORDER:
            mae=valid[name]['metrics']['test']['5']['MAE'];better=db['5']['MAE']<mae
            answers.append(f'D0B vs {name}: D0B better={better}; MAE improvement={100*(mae-db["5"]["MAE"])/mae:.6g}%. '+('The baseline outperforms D0B on this metric/horizon.' if mae<db['5']['MAE'] else ''))
        patterns={str(h):[name for name in ORDER if valid[name]['metrics']['test'][str(h)]['MAE']<db[str(h)]['MAE']] for h in HORIZONS}
        answers += [f'Strongest baseline TEST5 MAE: {strongest} ({sb:.12g}).',f'D0B improvement vs strongest baseline: {100*(sb-db["5"]["MAE"])/sb:.6g}%.',
            f'D0B beats every baseline at all four horizons: {not any(patterns.values())}.',f'Baselines outperforming D0B by horizon: {patterns}.']
        lines += [f'{i}. {text}' for i,text in enumerate(answers,1)]
        lines += ['',f'Strongest baseline: {strongest}',f'D0B rank: {rank} of 7 (strictly smaller TEST5 MAE ranks ahead; ties share rank).',
            f'D0B vs strongest baseline: {100*(sb-db["5"]["MAE"])/sb:.6g}% MAE improvement.',f'Multi-horizon pattern: {patterns}',
            'Validity caveats: one seed, historical frozen D0B, neutral compression vs native full nodes, simplified inverted/MTGNN implementations and distinct receptive fields. No statistical significance claim. STOP.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
