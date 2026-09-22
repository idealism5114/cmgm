"""Deep V3 paper tables: fixed scope, explicit fidelity and honest partial results."""
import json
from pathlib import Path
from cmgm.models.comparison_baselines import HORIZONS,ORDER,MODEL_CONFIGS,INPUT_VIEWS,CATEGORIES
from cmgm.scripts.formal_main_ablation_report import table,csv_file
from cmgm.scripts.formal_v2_protocol import atomic_json
OURS='D0B-Candidate-Aware 2-Expert MoE'
DISPLAY={**{n:n for n in ORDER},'VanillaTransformer':'Vanilla Transformer','GraphWaveNet':'Graph WaveNet','MTGNN':'MTGNN (adapted)',OURS:OURS+' (Ours)'}
NOTES={
 'RNN':'Standard PyTorch nn.RNN(tanh), last top-layer hidden; no attention.',
 'GRU':'Unchanged standard PyTorch nn.GRU, last top-layer hidden.',
 'LSTM':'Standard PyTorch nn.LSTM, h_n[-1]; no attention or bidirectionality.',
 'VanillaTransformer':'Unchanged causal vanilla Transformer encoder with sinusoidal absolute positions.',
 'GraphWaveNet':'Adapted Graph WaveNet: adaptive adjacency only (no physical graph); gated dilated temporal convolutions, diffusion, residual/skip paths. Left padding and no BatchNorm keep causal, sample-independent states.',
 'MTGNN':'Existing project MTGNN-style implementation, unchanged; not an official reproduction.',
 'MSGNet':'Core adaptation retaining FFT multiscale identification, independent adaptive graph/MixHop paths, intra-scale attention and frequency-weighted aggregation. Shared learned21→1 interface, per-sample FFT, parallel graph/attention paths, no input-level output restoration.',
 'CrossGNN':'Core adaptation retaining per-sample FFT scale identification, cross-scale temporal GNN and signed cross-variable GNN. Shared21→1 interface, device-independent parameters, explicit adjacent-time indexing. anti_ood=False because targets are future returns rather than input variable levels.'}


def write_report(r,out):
    models=r.get('models',{});status=r.get('model_status',{})
    valid={n:v for n,v in models.items() if 'metrics' in v and (n==OURS or (status.get(n)=='COMPLETE' and r['sanity'].get(n,{}).get('PASS')))}
    ours=valid.get(OURS,{}).get('metrics',{}).get('test',{}).get('5',{}).get('MAE')
    overall=[];allh=[];commodity=[];efficiency=[];counts=[];multi=[];checks=[]
    for n in (*ORDER,OURS):
        entry=models.get(n,{});parameters=entry.get('parameters',{});training=entry.get('training',{})
        runtime=entry.get('runtime',{})
        metric=valid.get(n,{}).get('metrics',{}).get('test',{}).get('5',{})
        delta=metric['MAE']-ours if metric and ours is not None else None
        overall.append(dict(Model=DISPLAY[n],Category=CATEGORIES.get(n,'Heterogeneous Graph + Switching Temporal + Dual-Expert MoE'),
            Params=parameters.get('trainable'),TEST5_MAE=metric.get('MAE'),MSE=metric.get('MSE'),RMSE=metric.get('RMSE'),
            Hit_percent=100*metric['Hit'] if metric else None,DeltaMAE_vs_Ours=delta,
            RelativeDeltaMAE_percent=100*delta/ours if delta is not None else None,
            TrainSeconds=training.get('train_seconds'),InferenceMsPerOrigin=runtime.get('inference_ms_per_origin'),
            Status=status.get(n,'PENDING')))
        counts.append(dict(Model=DISPLAY[n],**parameters))
        efficiency.append(dict(Model=DISPLAY[n],BestEpoch=training.get('best_epoch'),EpochsCompleted=training.get('epochs_completed'),
            TrainSeconds=training.get('train_seconds'),SecondsPerEpochMean=training.get('seconds_per_epoch_mean'),
            TestForwardSecondsMean=runtime.get('test_forward_seconds_mean'),TestForwardSecondsStd=runtime.get('test_forward_seconds_std'),
            InferenceMsPerOrigin=runtime.get('inference_ms_per_origin'),OriginsPerSecond=runtime.get('origins_per_second'),
            InferenceDevice=runtime.get('environment',{}).get('device_name'),
            TrainingDevice='N/A historical' if n==OURS else r.get('last_launch_device'),Status=status.get(n,'PENDING')))
        mh=dict(Model=DISPLAY[n])
        for h in HORIZONS:mh[f'{h}d_MAE']=valid.get(n,{}).get('metrics',{}).get('test',{}).get(str(h),{}).get('MAE')
        multi.append(mh)
        for split,hm in valid.get(n,{}).get('metrics',{}).items():
            for h,m in hm.items():allh.append(dict(Model=DISPLAY[n],Split=split,Horizon=int(h),**m))
        for c in valid.get(n,{}).get('per_commodity',[]):commodity.append(dict(Model=DISPLAY[n],commodity=c['commodity'],MAE=c['MAE'],MSE=c['MSE']))
        for stage in ('initial','best'):
            a=r['sanity'].get(n,{}).get(stage)
            if a:checks.append(dict(Model=DISPLAY[n],Stage=stage,OutputShape=a['OutputShape'],Finite=a['Finite'],BatchPerm=a['BatchPerm'],SingleSample=a['SingleSample'],Causal=a['Causal'],Mechanism=a.get('Mechanism',{}),PASS=a['PASS']))
    for name,rows in [('baseline_comparison',overall),('multi_horizon_metrics',allh),('per_commodity_5d',commodity),('training_efficiency',efficiency),('parameter_counts',counts)]:csv_file(out/(name+'.csv'),rows)
    for n in ORDER:
        key={'VanillaTransformer':'transformer','GraphWaveNet':'graph_wavenet'}.get(n,n.lower())
        path=out/f'history_{key}.json'
        if not path.exists():atomic_json(path,[])
    r['tables']=dict(overall=overall,all_horizons=allh,multi_horizon_mae=multi,training_summary=efficiency,parameter_counts=counts)
    finished=all(n in valid for n in ORDER) and OURS in valid
    answers=[]
    ranked=sorted((n for n in ORDER if n in valid),key=lambda n:valid[n]['metrics']['test']['5']['MAE'])
    answers.append(('Full ranking: ' if finished else 'PENDING complete suite; available ranking only: ')+', '.join(f"{n}={valid[n]['metrics']['test']['5']['MAE']:.10g}" for n in ranked))
    answers.append('Baseline−Ours (positive = baseline worse); relative=100×(Baseline−Ours)/Ours. '+str([{k:row[k] for k in ('Model','DeltaMAE_vs_Ours','RelativeDeltaMAE_percent')} for row in overall[:-1]]))
    for group in (('RNN','GRU','LSTM'),('VanillaTransformer',),('GraphWaveNet','MTGNN'),('MSGNet','CrossGNN')):
        answers.append('; '.join(n+': '+(str(valid[n]['metrics']['test']['5']) if n in valid else 'PENDING') for n in group))
    exceptions=[];metric_wins=[]
    if OURS in valid:
        for n in ORDER:
            if n not in valid:continue
            for h in HORIZONS:
                b,o=valid[n]['metrics']['test'][str(h)],valid[OURS]['metrics']['test'][str(h)]
                if b['MAE']<=o['MAE']:exceptions.append(dict(Model=n,Horizon=h,MAE=b['MAE'],Ours=o['MAE']))
                for k in ('MAE','MSE','RMSE','Hit'):
                    if (b[k]>o[k] if k=='Hit' else b[k]<o[k]):metric_wins.append(dict(Model=n,Horizon=h,Metric=k,Baseline=b[k],Ours=o[k]))
    answers.append(('YES: Ours has strictly lower MAE at all horizons than all eight baselines.' if finished and not exceptions else
                    'NO; exceptions: '+str(exceptions) if finished else 'PENDING; currently observed MAE exceptions/ties: '+str(exceptions)))
    answers.append(('Baseline wins on TEST metrics: ' if finished else 'PENDING complete suite; currently observed baseline wins: ')+str(metric_wins))
    sections=[]
    def section(title,body):sections.append(f'## {len(sections)+1}. {title}\n\n{body}\n\n')
    section('Experiment Objective','Representative controlled deep-learning benchmark. Eight seed42 baseline runs under predefined architectures and one frozen audited Candidate-Aware 2-Expert MoE reference. Ours is never trained or modified by this suite.')
    section('Baseline Scope','The present benchmark focuses on representative deep temporal and graph-based forecasting architectures.\n\nWe compare against representative deep temporal, Transformer-based, and graph-based multivariate forecasting architectures, including recurrent models (RNN, GRU, LSTM), a vanilla Transformer, classical spatio-temporal graph models (Graph WaveNet and MTGNN), and recent multi-scale graph forecasting approaches (MSGNet and CrossGNN).\n\nHistorical experiments are preserved outside this scope; no claim of exhaustive or state-of-the-art coverage.')
    section('Data / Target Protocol','Existing build_data/Dataset preprocessing and chronological splits, 20 observed timesteps, 21 features, original return target clipping and commodity order; no pipeline refit or target renormalization. Four slots explicitly mean [1d,5d,10d,20d], never four consecutive days.\n\n'+table([dict(Split=s,**v) for s,v in r.get('data_audit',{}).get('split_fingerprint',{}).items()])+'\nCommodity mapping and exact target reconstruction checks are in data_audit.json. Dataset split calendar dates are unavailable in the Dataset object; timeline counts and content hashes identify the original boundaries.')
    section('Input Views','Neutral adapter is unchanged and has zero parameters: stock mean/population std, bond mean/population std, then commodity nodes in target order. Output B,20,28,21. Sequence flatten is token-major then feature-major. MSGNet/CrossGNN each use their own shared trainable Linear(21,1), part of the model.\n\n'+table([dict(Model=DISPLAY[n],InputView=INPUT_VIEWS[n]) for n in ORDER])+'\n\nOurs retains its native full-node representation. These models share data, feature families, targets, splits and information timing; they do **not** use identical architectural input representations or identical uncompressed stock/bond information. Compression and learned scalarization are explicit fairness limitations.')
    section('Fixed Training Protocol','```json\n'+json.dumps(r.get('protocol',{}),indent=2)+'\n```\n\nAll new baselines reuse baseline_protocol.train_one/prediction_loss: sum four Huber(delta=.02). Scheduler, early stopping and checkpoint selection monitor only multi-horizon VAL Huber; VAL5 is secondary. No baseline auxiliary loss, clipping, tuning, retry for poor performance, AMP or accumulation. OOM/nonfinite/core sanity failure stops review without smaller batches.')
    section('Model Configurations','```json\n'+json.dumps(MODEL_CONFIGS,indent=2)+'\n```')
    section('MSGNet Adaptation',NOTES['MSGNet']+'\n\nBased on [official MSGNet](https://github.com/YoZhibo/MSGNet), AAAI2024; pinned model/layer/configuration file hashes are in baseline_provenance.json. Each graph retains the reference latent32→node28 mapping, MixHop beta=.3/depth2, graph residual and independent scale attention. No input-window restandardization/output de-normalization; learned node states and return targets differ in meaning. Per-sample FFT replaces upstream batch-averaged frequency selection to satisfy single-sample invariance. Parallel independent scale paths implement this protocol rather than upstream sequential graph overwrites/shared attention. Full details are in cmgm/models/baselines/ADAPTATION_NOTES.md.')
    section('CrossGNN Adaptation',NOTES['CrossGNN']+'\n\nBased on [official CrossGNN](https://github.com/hqh0728/CrossGNN), NeurIPS2023. Retains original scale plus four identified periods, stride-period moving averages, pad/crop to40, cross-scale top-k/local adjacency, signed positive/negative variable relations and residual graph refinement. Per-sample FFT avoids batch dependence. Four outputs are non-contiguous return slots. No input last-value restoration; no hard-coded CUDA device. No D0B graph substitution.')
    section('Graph WaveNet Adaptation',NOTES['GraphWaveNet']+'\n\nBased on [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet), whose source commit was already recorded by this project. Four blocks × two layers, dilation1/2, graph order2, node embedding10, channels32/32/256/512, dropout.1. One learned A support; last legal temporal position and commodity rows4:28. Left padding retains aligned causal states. BatchNorm is omitted to avoid training-time batch/time mixing; this is explicitly an adaptation, not exact reproduction. Receptive field13 lies inside the common20-day window.')
    section('Sanity Checks',table(checks)+'\nObserved-window-only interface applies to all models. Prefix10 tests apply to causal recurrent/Transformer/GraphWaveNet/MTGNN states; whole-window FFT models are not claimed prefix causal. Raw float32 discrepancies and any independent float64 roundoff audit remain visible. All initial architectures must pass before a fit; restored checkpoints pass again before TEST.')
    section('Parameter Counts',table(counts)+'\nArchitectural parameter counts, no capacity matching; Ours effective trainable parameters in this suite are zero. Buffers are reported separately.')
    section('TEST 5d Main Comparison','FORMAL DEEP BASELINE COMPARISON — STANDARDIZED TEST 5D\n\n'+table(overall)+'\nPositive ΔMAE means baseline worse than Ours. Relative delta denominator is Ours MAE. Pending/invalid metrics are not fabricated or ranked. Hit shown as percent here; raw JSON/metric CSV uses fractions. Primary ranking is TEST5 MAE.')
    section('Multi-Horizon Comparison',table(multi)+'\nFull TRAIN/VAL/TEST MAE/MSE/RMSE/Hit for all four horizons are saved in multi_horizon_metrics.csv. All metrics pool every origin ×24 commodities, including zero targets. RMSE is the square root of directly computed MSE.')
    section('Per-Commodity 5d Results','per_commodity_5d.csv contains commodity name, model, MAE and MSE for all24 targets. No extra risk/tail/regime diagnostics are run.')
    section('Training and Inference Efficiency',table(efficiency)+'\nTrainSeconds measures the completed early-stopped training loop, including validation/checkpoint/history overhead; SecondsPerEpochMean averages recorded epoch times. Different stopping epochs affect training cost. Ours training time is N/A (historical/unavailable), never zero.\n\nInference timing uses every TEST origin, batch64 including the final partial batch, eval + inference_mode, three warmup batches and ten complete forward-only passes on the same device/settings. CUDA synchronizes before/after each pass. Inputs are preloaded on device outside the timer; each model adapter is inside forward. Preprocessing, DataLoader, host-to-device transfer, checkpoint loading, metric computation and report I/O are excluded. InferenceMsPerOrigin is amortized batch throughput cost, NOT batch1 request latency. Raw repeat times, hardware/software settings and timing scope are saved in runtime_measurements.json and runtime_protocol.json. Repeated timing forwards never consume targets, select checkpoints or retrain. Ours receives exactly the same inference timing procedure.\n\nRuntime is descriptive, subject to device load, and never changes model selection/training budget; inference throughput and forecast error are reported separately. Each new model persists checkpoint/history/result immediately and releases memory before the next.')
    section('Implementation Fidelity Notes',table([dict(Model=DISPLAY[n],Note=NOTES[n]) for n in ORDER])+'\nPinned upstream references, local core source hashes and exact adaptation notes are retained. Official repositories are not runtime dependencies. Existing GRU/Transformer/MTGNN definitions and neutral adapter are unchanged. No exact-official-reproduction claim is made for adapted graph models.')
    section('Ours Reference Verification','```json\n'+json.dumps(r.get('reference_check',{}),indent=2)+'\n```\n\nCheckpoint/report hashes, COMPLETE status, seed/variant metadata, data fingerprints and best sanity are checked before evaluation. Metrics are freshly computed from the checkpoint on full TRAIN/VAL/TEST and compared to audited results; no prompt numbers substitute for forwards. Source metadata correspond to the historical reference, not a demand that current baseline source hashes equal historical ones.\n\nCheckpoint unchanged: '+str(r.get('ours_checkpoint_unchanged','verification in progress')))
    section('Limitations','Single seed42, no significance or multi-seed claim. Fixed training budget without per-model search. MSGNet/CrossGNN require task interfaces and per-sample FFT adaptation; Graph WaveNet uses no physical adjacency and has stated causal normalization adaptation. MTGNN is the project\'s adapted implementation. Ours is a frozen historical single run with its native auxiliary SwitchKL, whereas baselines only optimize prediction loss. No baseline is hidden for unfavorable results.\n\nRequired answers:\n\n'+'\n\n'.join(f'Q{i}. {a}' for i,a in enumerate(answers,1))+'\n\nSTOP after this suite; no automatic tuning or follow-up model.')
    text='# Deep Baseline Comparison V3\n\nStatus: '+r.get('status','PENDING')+'\n\n'+''.join(sections)
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
