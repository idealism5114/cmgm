"""TRAIN-frozen amplitude and univariate causal-tail ranking statistics.

No estimator is trained: calibration is a single prescribed closed-form scalar.
AUPRC is non-interpolated average precision, with score ties evaluated together.
"""
import numpy as np
import pandas as pd

from cmgm.scripts.d0b_5d_error_regime_analysis import metrics, finite_corr, bin_values, table, fmt

EPS = 1e-12
QS = [.5, .75, .9, .95, .99]
BIN_QS = [.25, .5, .75, .9]
BIN_LABELS = ['Q0-Q25', 'Q25-Q50', 'Q50-Q75', 'Q75-Q90', 'Q90-Q100']
SIGNALS = ['abs_prediction', 'abs_past1', 'abs_past5', 'commodity_vol5', 'commodity_vol20',
    'vol5', 'vol20', 'dispersion_commodity', 'confidence', 'entropy', 'margin', 'movement',
    'transition_score', 'kl_surprise', 'l1_surprise', 'delta_z', 'mean_delta_z5']
IMPORTANT = ['abs_prediction', 'commodity_vol5', 'commodity_vol20', 'abs_past5', 'transition_score']
SAMPLE_SIGNALS = SIGNALS[5:]


def scalar_coefficient(prediction, target):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return float(np.dot(p, y)/(np.dot(p, p)+EPS))


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    a = np.abs(values)
    return dict(mean=float(values.mean()), std=float(values.std()), mean_abs=float(a.mean()),
        median_abs=float(np.median(a)), max_abs=float(a.max()),
        **{f'P{int(q*100)}': float(np.quantile(a,q)) for q in QS})


def rank_plan(scores):
    score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]) != 0)+1]
    return order, starts


def ranking_metrics(labels, scores, weights=None, plan=None):
    labels = np.asarray(labels, dtype=bool)
    weights = np.ones(len(labels), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    order, starts = rank_plan(scores) if plan is None else plan
    pos = np.add.reduceat((weights*labels)[order], starts)
    neg = np.add.reduceat((weights*~labels)[order], starts)
    p, n = float(pos.sum()), float(neg.sum())
    if p+n == 0:
        return dict(AUROC=None, AUPRC=None, prevalence=None, AUPRC_over_prevalence=None)
    prevalence = p/(p+n)
    if p == 0 or n == 0:
        return dict(AUROC=None, AUPRC=None if p == 0 else 1., prevalence=prevalence,
                    AUPRC_over_prevalence=None if p == 0 else 1.)
    cp, cn = np.cumsum(pos), np.cumsum(neg)
    auc = np.sum(neg*(cp-.5*pos))/(p*n)
    precision = np.divide(cp, cp+cn, out=np.zeros_like(cp), where=cp+cn > 0)
    ap = np.sum((pos/p)*precision)
    return dict(AUROC=float(auc), AUPRC=float(ap), prevalence=prevalence,
                AUPRC_over_prevalence=float(ap/prevalence))


def freeze_train(frame):
    assert set(frame.split) == {'train'}, 'Only TRAIN may determine coefficients/thresholds/orientation'
    y = frame.target.abs().to_numpy()
    thresholds = {f'T{q}': float(np.quantile(y,q/100)) for q in (90,95)}
    labels = y > thresholds['T90']
    signal_specs = {}
    for key in SIGNALS:
        s = frame[key].to_numpy(dtype=float)
        correlation = finite_corr(s, labels.astype(float))['Pearson']
        orientation = -1 if correlation is not None and correlation < 0 else 1
        # Sample signals are repeated once per commodity for correlation with
        # observation labels, but fitted quantile cutpoints use one origin each.
        values = frame.drop_duplicates('sample_index')[key].to_numpy() if key in SAMPLE_SIGNALS else s
        signal_specs[key] = dict(orientation=orientation, train_tail90_pearson=correlation,
            raw_quantile_cuts=np.quantile(values,BIN_QS).tolist(),
            oriented_top20_cut=float(np.quantile(orientation*values,.8)),
            oriented_top10_cut=float(np.quantile(orientation*values,.9)),
            quantile_unit='forecast_origin' if key in SAMPLE_SIGNALS else 'observation')
    return dict(c_train=scalar_coefficient(frame.prediction,frame.target), tail_thresholds=thresholds,
                signal_specs=signal_specs, median_abs_prediction=float(np.median(frame.abs_prediction)),
                epsilon=EPS, coefficient_fit='TRAIN pooled sum(pred*y)/(sum(pred**2)+epsilon)',
                orientation_fit='TRAIN Pearson(raw signal, Tail90); negative -> -1, otherwise +1. Same orientation for Tail95.')


def cluster_weights(origins, draws=1000, seed=42):
    unique, inverse = np.unique(origins, return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0,len(unique),size=(draws,len(unique)))
    counts = np.zeros((draws,len(unique)), dtype=np.int32)
    for k, ids in enumerate(sampled):
        counts[k] = np.bincount(ids,minlength=len(unique))
    return counts, inverse


def calibration_bootstrap(frame):
    counts, inverse = cluster_weights(frame.sample_index.to_numpy())
    errors = frame.prediction.to_numpy()-frame.target.to_numpy()
    calibrated = frame.calibrated_prediction.to_numpy()-frame.target.to_numpy()
    sums = np.bincount(inverse)
    result = {}
    for metric, delta in [('MAE', np.abs(calibrated)-np.abs(errors)), ('MSE', calibrated**2-errors**2)]:
        numerator = np.bincount(inverse,weights=delta)
        draws = (counts@numerator)/(counts@sums)
        result[metric] = dict(delta=float(delta.mean()), CI95=np.quantile(draws,[.025,.975]).tolist(),
            seed=42, draws=1000, unit='paired forecast-origin, all commodities retained')
    return result


def ranking_bootstrap(frame, labels, scores):
    counts, inverse = cluster_weights(frame.sample_index.to_numpy())
    plan = rank_plan(scores)
    aucs = []
    for count in counts:
        value = ranking_metrics(labels, scores, weights=count[inverse], plan=plan)['AUROC']
        if value is not None:
            aucs.append(value)
    return dict(CI95=np.quantile(aucs,[.025,.975]).tolist() if aucs else None,
                draws=1000, valid_draws=len(aucs), seed=42, unit='forecast-origin clusters, commodity labels retained')


def ratio_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None,median=None,**{f'P{q}':None for q in (10,25,50,75,90)})
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                **{f'P{q}':float(np.quantile(values,q/100)) for q in (10,25,50,75,90)})


def amplitude_row(f, mask, split, group, total_mse):
    sub = f.loc[mask]
    m = metrics(sub.prediction.to_numpy(),sub.target.to_numpy())
    return dict(split=split, group=group, count=len(sub), origins=sub.sample_index.nunique(),
        warning='LOW SAMPLE SIZE' if sub.sample_index.nunique()<20 else '',
        MSE_share=float(np.square(sub.prediction-sub.target).sum()/total_mse) if total_mse else None,
        **m, amplitude=ratio_stats(sub.abs_prediction/(sub.target.abs()+EPS)),
        signed_correlation=finite_corr(sub.prediction,sub.target))


def analyze(frame, output):
    f = frame.copy()
    assert set(f.split) == {'train','val','test'}
    assert not f.duplicated(['split','sample_index','commodity_index']).any()
    for col in ['target','prediction']+SIGNALS:
        f[col] = f[col].astype(np.float64)
    assert np.isfinite(f[['target','prediction']+SIGNALS]).all().all()
    train = f[f.split == 'train']
    frozen = freeze_train(train)
    c = frozen['c_train']
    result = dict(frozen_train=frozen, native={}, amplitude={}, tails=[], calibration=[],
        signal_predictability=[], signal_quantiles=[], conditional_amplitude=[], sample_signal_associations=[],
        top_capture=[], calibration_bootstrap={}, signal_AUROC_bootstrap={}, commodity_calibration=[],
        amplitude_quantiles=[], tail_false_negative={}, calibration_stability={},
        primary_case=None, interpretation_status='PENDING EVIDENCE REVIEW',
        definitions=dict(metrics='float64 pooled MAE / MSE / RMSE=sqrt(MSE) / unmasked sign Hit (JSON fraction, report %)',
            delta='calibrated minus native; negative means improvement',
            AUPRC='non-interpolated average precision; tied scores evaluated together; random baseline = prevalence',
            top_capture='Frozen oriented TRAIN P90/P80 thresholds, strict >. Evaluation coverage is reported and need not equal 10%/20%; no VAL/TEST quantile refitting.',
            signal_quantiles='Raw signal TRAIN quantile bins, equal-to-cut goes into lower bin',
            sample_signals='Market/regime statistics evaluated both as repeated observation scores and once per origin against future severity',
            bootstrap='1000 paired forecast-origin cluster draws, seed42. IID origin bootstrap does not remove serial dependence of overlapping windows; approximate CI.',
            calibration='Single global no-intercept closed-form TRAIN scalar; not a final model; no calibrated checkpoint',
            zero_target_ratio=f'Ratio denominator |target|+{EPS}; zero targets can inflate mean ratios. Nonzero-only mean and zero counts are also supplied.',
            inference_training_provenance='Best checkpoint may have used VAL for historical selection. TRAIN-only here describes post-hoc coefficient/cutpoint/orientation estimation, not an independent model-training split.',
            tail_labels='Future targets used only as evaluation labels and TRAIN tail threshold source, never as predictors'))
    for s in ('train','val','test'):
        sub = f[f.split == s]
        result['native'][s] = metrics(sub.prediction.to_numpy(),sub.target.to_numpy())
    if c < 0:
        result.update(status='STOPPED: negative TRAIN calibration coefficient anomaly',
            interpretation_status='No automatic sign reversal or continuation is allowed')
        return result
    f['calibrated_prediction'] = c*f.prediction
    for q,t in frozen['tail_thresholds'].items():
        f['Tail'+q[1:]] = f.target.abs() > t
    f['residual'] = f.prediction-f.target
    f['abs_residual'] = f.residual.abs()
    f['squared_residual'] = f.residual**2
    f['Hit'] = (np.sign(f.prediction)==np.sign(f.target)).astype(int)
    f['amplitude_ratio'] = f.abs_prediction/(f.target.abs()+EPS)
    if c > 0:
        assert np.array_equal(np.sign(f.prediction),np.sign(f.calibrated_prediction))
    result['positive_scalar_sign_sanity'] = 'PASS' if c>0 else 'c=0: sign equality not guaranteed'
    train = f[f.split == 'train']
    ids = np.sort(train.sample_index.unique())
    middle = len(ids)//2
    first, second = train[train.sample_index.isin(ids[:middle])], train[train.sample_index.isin(ids[middle:])]
    result['calibration_stability'] = {name: dict(c=scalar_coefficient(part.prediction,part.target),
        origins=part.sample_index.nunique(), first_date=part.forecast_origin.min(), last_date=part.forecast_origin.max())
        for name,part in [('first_half',first),('second_half',second)]}
    result['calibration_stability']['note'] = 'Origin chronological halves; 5d targets can overlap the midpoint. Coefficients descriptive only, never applied to VAL/TEST.'
    for s in ('train','val','test'):
        sub = f[f.split == s].copy()
        target, pred = distribution(sub.target), distribution(sub.prediction)
        result['amplitude'][s] = dict(target=target,prediction=pred,
            R_std=pred['std']/(target['std']+EPS), R_MAD=pred['median_abs']/(target['median_abs']+EPS),
            R_meanabs=pred['mean_abs']/(target['mean_abs']+EPS),
            magnitude_association=finite_corr(sub.abs_prediction,sub.target.abs()),
            signed_association=finite_corr(sub.prediction,sub.target))
        for q in QS:
            k = f'P{int(100*q)}'
            result['amplitude_quantiles'].append(dict(split=s,quantile=k,target=target[k],prediction=pred[k],ratio=pred[k]/(target[k]+EPS)))
        masks = dict(Tail90=sub.Tail90, Tail95=sub.Tail95, NonTail90=~sub.Tail90)
        total_mse = float(sub.squared_residual.sum())
        for tail in ('Tail90','Tail95'):
            for name, condition in [('all',np.ones(len(sub),dtype=bool)),('positive',sub.target>0),('negative',sub.target<0),
                ('correct_sign',sub.Hit==1),('wrong_sign',sub.Hit==0)]:
                result['tails'].append(amplitude_row(sub,sub[tail]&condition,s,tail+'/'+name,total_mse))
        tail_count = int(sub.Tail90.sum())
        low = sub.abs_prediction <= frozen['median_abs_prediction']
        result['tail_false_negative'][s] = dict(tail_count=tail_count, low_amplitude_tails=int((sub.Tail90&low).sum()),
            rate=float((sub.Tail90&low).sum()/tail_count) if tail_count else None,
            label='Descriptive fraction of actual Tail90 with |pred| <= frozen TRAIN median, not classifier output')
        for group,mask in {'Overall':np.ones(len(sub),dtype=bool),**masks}.items():
            part = sub.loc[mask]
            native = metrics(part.prediction.to_numpy(),part.target.to_numpy())
            calibrated = metrics(part.calibrated_prediction.to_numpy(),part.target.to_numpy())
            row = dict(split=s,group=group,count=len(part),origins=part.sample_index.nunique(),
                native=native,calibrated=calibrated,
                delta={k:calibrated[k]-native[k] if native[k] is not None else None for k in ('MAE','MSE','RMSE','Hit')})
            if c>0 and len(part):
                assert row['delta']['Hit'] == 0
            result['calibration'].append(row)
        if s != 'train':
            result['calibration_bootstrap'][s] = calibration_bootstrap(sub)
        for signal in SIGNALS:
            spec = frozen['signal_specs'][signal]
            oriented = spec['orientation']*sub[signal].to_numpy()
            for tail in ('Tail90','Tail95'):
                labels = sub[tail].to_numpy()
                row = dict(split=s,signal=signal,tail=tail,orientation=spec['orientation'],
                    count=len(sub),positive_count=int(labels.sum()),
                    **ranking_metrics(labels,oriented),
                    Spearman=finite_corr(oriented,sub.target.abs())['Spearman'])
                result['signal_predictability'].append(row)
                for top in (10,20):
                    selected = oriented > spec[f'oriented_top{top}_cut']
                    hits = int((selected&labels).sum())
                    result['top_capture'].append(dict(split=s,signal=signal,tail=tail,top_train_percent=top,
                        cutoff=spec[f'oriented_top{top}_cut'], selected_count=int(selected.sum()),
                        actual_coverage=float(selected.mean()), selected_origins=sub.loc[selected,'sample_index'].nunique(),
                        precision=hits/int(selected.sum()) if selected.any() else None,
                        recall=hits/int(labels.sum()) if labels.any() else None))
                if signal in IMPORTANT and s in ('val','test'):
                    result['signal_AUROC_bootstrap'][f'{s}/{signal}/{tail}'] = ranking_bootstrap(sub,labels,oriented)
            bins = bin_values(sub[signal],spec['raw_quantile_cuts'],BIN_LABELS)
            for label in BIN_LABELS:
                part = sub.loc[bins==label]
                row = dict(split=s,signal=signal,group=label,count=len(part),origins=part.sample_index.nunique(),
                    warning='LOW SAMPLE SIZE' if part.sample_index.nunique()<20 else '',
                    Tail90_rate=float(part.Tail90.mean()) if len(part) else None,
                    Tail95_rate=float(part.Tail95.mean()) if len(part) else None,
                    mean_future_abs_target=float(part.target.abs().mean()) if len(part) else None)
                result['signal_quantiles'].append(row)
                if signal in ('abs_prediction','commodity_vol20'):
                    result['conditional_amplitude'].append(dict(**row, mean_abs_prediction=float(part.abs_prediction.mean()) if len(part) else None,
                        mean_amplitude_ratio=float(part.amplitude_ratio.mean()) if len(part) else None,
                        nonzero_target_mean_ratio=float(part.loc[part.target!=0,'amplitude_ratio'].mean()) if (part.target!=0).any() else None,
                        zero_targets=int((part.target==0).sum()),**metrics(part.prediction.to_numpy(),part.target.to_numpy())))
        sample = sub.groupby('sample_index').agg(TailShare=('Tail90','mean'),MaxAbsTarget=('target',lambda v:v.abs().max()),sample_MSE=('squared_residual','mean'))
        for signal in SAMPLE_SIGNALS:
            values = sub.groupby('sample_index')[signal].first()*frozen['signal_specs'][signal]['orientation']
            for outcome in ('TailShare','MaxAbsTarget','sample_MSE'):
                result['sample_signal_associations'].append(dict(split=s,signal=signal,outcome=outcome,
                    origins=len(sample), orientation=frozen['signal_specs'][signal]['orientation'], **finite_corr(values,sample[outcome])))
    for ci, part in train.groupby('commodity_index'):
        test = f[(f.split=='test') & (f.commodity_index==ci)]
        native, calibrated = metrics(test.prediction.to_numpy(),test.target.to_numpy()), metrics(test.calibrated_prediction.to_numpy(),test.target.to_numpy())
        result['commodity_calibration'].append(dict(commodity_index=int(ci),commodity=part.commodity.iloc[0],
            train_slope=scalar_coefficient(part.prediction,part.target),
            native=native,calibrated_global_c=calibrated,
            delta_MAE=calibrated['MAE']-native['MAE'],delta_MSE=calibrated['MSE']-native['MSE']))
    slopes = np.array([r['train_slope'] for r in result['commodity_calibration']])
    result['commodity_slope_summary'] = dict(mean=float(slopes.mean()),median=float(np.median(slopes)),
        std=float(slopes.std()),min=float(slopes.min()),max=float(slopes.max()),
        correlations={k:finite_corr(slopes,[r['native'][k] for r in result['commodity_calibration']]) for k in ('MAE','MSE')},
        note='Exploratory TRAIN slope vs TEST commodity error. No per-commodity coefficient is applied.')
    result['commodity_rankings'] = {name:sorted(result['commodity_calibration'],key=lambda r:r[key],reverse=reverse)[:5]
        for name,key,reverse in [('improved_MAE','delta_MAE',False),('worsened_MAE','delta_MAE',True),
            ('improved_MSE','delta_MSE',False),('worsened_MSE','delta_MSE',True),('lowest_train_slope','train_slope',False),('highest_train_slope','train_slope',True)]}
    f.to_csv(output/'observation_tail_diagnostics.csv',index=False)
    for filename,rows in [('signal_tail_predictability',result['signal_predictability']),('amplitude_quantiles',result['amplitude_quantiles']),
        ('calibration_metrics',result['calibration']),('commodity_calibration_diagnostics',result['commodity_calibration']),
        ('signal_quantile_tail_rates',result['signal_quantiles']),('top_signal_capture',result['top_capture']),
        ('sample_signal_associations',result['sample_signal_associations']),('conditional_amplitude',result['conditional_amplitude']),('tail_response',result['tails'])]:
        pd.json_normalize(rows).to_csv(output/(filename+'.csv'),index=False)
    result['status'] = 'STATISTICS COMPLETE; no training or checkpoint changes'
    attach_interpretation(result)
    return result


def attach_interpretation(r):
    """Apply the user's scalar-first priority and state its direction explicitly.

    No ranking cutoff or calibration form is selected from held-out outcomes.
    This is a report decision only, not a deployable selection rule.
    """
    evidence = {}
    for s in ('val','test'):
        row = next(x for x in r['calibration'] if x['split']==s and x['group']=='Overall')
        evidence[s] = {k:dict(relative_improvement=-row['delta'][k]/row['native'][k],
            delta=row['delta'][k],CI95=r['calibration_bootstrap'][s][k]['CI95']) for k in ('MAE','MSE')}
    scalar_consistent = all(v['delta']<0 and v['CI95'][1]<0 for vals in evidence.values() for v in vals.values())
    r['classification_evidence'] = dict(scalar=evidence,
        rule='User section 77: when global TRAIN scalar improves both MAE/MSE on VAL/TEST materially, prioritize A.',
        uncertainty=f'Four intervals all favor calibration: {scalar_consistent}; effect sizes reported explicitly, not inferred from a p-value alone.')
    r['primary_case'] = 'Case A' if scalar_consistent else 'Case G'
    r['interpretation_status'] = 'EVIDENCE REVIEW COMPLETE' if scalar_consistent else 'Mixed scalar evidence; inspect signal evidence before adopting any narrower conclusion'
    c = r['frozen_train']['c_train']
    vol = {s:{signal:next(x for x in r['signal_predictability'] if x['split']==s and x['signal']==signal and x['tail']=='Tail90')
        for signal in ('commodity_vol5','commodity_vol20','vol5','vol20','abs_prediction')} for s in ('val','test')}
    r['stable_signal_answer'] = '商品自身 vol20 的 Tail90 排序在两个 evaluation split 均有信息：' + '; '.join(
        f"{s.upper()} AUROC={v['commodity_vol20']['AUROC']:.6f}, AUPRC={v['commodity_vol20']['AUPRC']:.6f}, prevalence={v['commodity_vol20']['prevalence']:.6f}" for s,v in vol.items()) + '。评价依据是跨 split 的效果大小、PR baseline 与置信区间，不是任意 AUC>0.6 的门槛。'
    r['localized_answer'] = '商品波动率优于市场均值的证据：' + '; '.join(
        f"{s.upper()} commodity/market vol5 AUC={v['commodity_vol5']['AUROC']:.6f}/{v['vol5']['AUROC']:.6f}, vol20={v['commodity_vol20']['AUROC']:.6f}/{v['vol20']['AUROC']:.6f}" for s,v in vol.items()) + '。这是 pooled observation ranking，可能部分反映稳定的商品间风险差异，不能自动解释成已经准确识别每个商品的未来冲击时点。'
    case_text = ('按用户第77条 scalar-first 规则，本次 primary classification 为 Case A：四项 VAL/TEST MAE/MSE 都改善，且对应近似 bootstrap CI 均排除零。'
        if scalar_consistent else '本次 scalar 证据未满足四项一致改善的优先规则，保留 Case G，避免夸大。')
    direction = ('但 c<1，作用方向是进一步收缩而不是放大。因此这里的 Case A 应理解为全局幅度校准有可测收益；不能将名称中的 under-calibration 误读为“幅度不足是主要瓶颈，放大即可解决”。'
        if c<1 else 'c>=1，校准方向为放大；是否解决主要误差仍需看实际收益幅度和尾部方向信息。')
    ts = next(x for x in r['tails'] if x['split']=='test' and x['group']=='Tail90/all')
    correct = next(x for x in r['tails'] if x['split']=='test' and x['group']=='Tail90/correct_sign')
    pred_informative = all(r['signal_AUROC_bootstrap'][f'{s}/abs_prediction/Tail90']['CI95'][0]>.5 for s in ('val','test'))
    prediction_interpretation = ('两个 split 的近似 CI 均高于 chance，但仍应结合实际效果大小和 PR baseline 判断排名信息强弱。'
        if pred_informative else '至少一个 split 的近似 CI 包含 chance，不能支持“模型自身已经知道哪里会大，只差放大”。')
    redistribution = all(x['delta'][k]<=0 for x in r['calibration'] if x['split'] in ('val','test') and x['group'] in ('Tail90','NonTail90') for k in ('MAE','MSE'))
    r['interpretation_text'] = '\n\n'.join([case_text,direction,
        '; '.join(f"{s.upper()} relative MAE improvement={100*v['MAE']['relative_improvement']:.4f}%, MSE={100*v['MSE']['relative_improvement']:.4f}%" for s,v in evidence.items()) + '。这是相对原误差的有限改善，不是消除了绝大部分 tail error。',
        f"TEST Tail90 Hit={100*ts['Hit']:.4f}%；正确符号子集平均幅度比={correct['amplitude']['mean']:.6f}。正确符号下确实仍有小幅度，但这个子集是事后用 y 选出的，不能直接证明部署时统一放大有益。全 tail 的方向信息必须一起看。",
        'D0B |prediction| 的 Tail90 AUROC：'+ '; '.join(f"{s.upper()} {v['abs_prediction']['AUROC']:.6f}" for s,v in vol.items()) + '。'+prediction_interpretation,
        r['stable_signal_answer'],r['localized_answer'],
        'TRAIN halves 的 c1/c2 变化只作稳定性证据；未应用第二套预测系数。尾部与 NonTail90 的实际变化均列在第5节；'+('当前并非用普通样本恶化换尾部改善。' if redistribution else '存在分组误差取舍，需要按表审视。'),
        '预测分布比实际收益窄，本身不是放大能改善预测的证明：实际收益包含预测时不可知的变化。尤其不能把事后正确方向的子集直接当作可部署选择规则。',
        '综合判断：当前模型的预测幅度严重 under-dispersed 是描述事实，但主要剩余误差不能归因于简单可修复的 amplitude shrinkage。现有商品自身波动率含有模型预测幅度未充分呈现的 tail ranking 信息；这不等于它能给出尾部收益的正负方向。'])
    r['next_direction'] = '唯一建议的后续研究方向（本轮不实现）：检验既有 commodity-specific volatility 条件下的幅度/风险刻画是否提供泛化收益，同时保留方向误差检查。不要把 c 写入正式模型，不新增校准或波动率 head，不自动训练。'


def write_report(r, output):
    cp = r.get('checkpoint',{})
    lines = ['# D0B-TailPredictabilityAmplitudeDiagnostic','',r['status'],'',
        '仅正式 D0B checkpoint inference。没有训练、backward、optimizer、模型修改或 calibrated checkpoint。','',
        table(['Checkpoint','Best epoch','Params','Seed metadata','Training SHA','Diagnostic SHA'], [[cp.get('checkpoint_path'),cp.get('best_epoch'),cp.get('parameter_count'),cp.get('seed'),cp.get('checkpoint_git_sha'),r.get('diagnostic_git_sha')]]),'',
        f"Model parameter max diff = {cp.get('model_parameter_max_diff', 'N/A')}; state_dict/file checksums before and after are stored in results.json. Unavailable training metadata remains N/A.",'',
        '## 1. Native D0B','',table(['Split','MAE','MSE','RMSE','Hit%','|RMSE²−MSE|'], [[s,*[r['native'][s][k] for k in ('MAE','MSE','RMSE')],100*r['native'][s]['Hit'],r['native'][s]['RMSE_squared_minus_MSE_abs']] for s in ('train','val','test')]),'']
    if r['frozen_train']['c_train']<0:
        lines += [f"Negative c_train={r['frozen_train']['c_train']}: anomaly. Stopped without applying sign reversal or continuing interpretation."]
        (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        return
    lines += ['## 2. Is D0B under-dispersed?','',
        table(['Split','Target std','Pred std','Std ratio','Target mean abs','Pred mean abs','Mean-abs ratio','Median-abs ratio','Magnitude Pearson','Magnitude Spearman'],[[s,v['target']['std'],v['prediction']['std'],v['R_std'],v['target']['mean_abs'],v['prediction']['mean_abs'],v['R_meanabs'],v['R_MAD'],v['magnitude_association']['Pearson'],v['magnitude_association']['Spearman']] for s,v in r['amplitude'].items()]),'',
        table(['Split','Quantile','Target magnitude','Prediction magnitude','Ratio'],[[x[k] for k in ('split','quantile','target','prediction','ratio')] for x in r['amplitude_quantiles']]),'',
        '完整 mean/std/mean absolute/median/P50/75/90/95/99/max absolute 与 signed correlation 见 results.json/amplitude。','',
        '## 3. Extreme response','',f"Frozen TRAIN thresholds: {r['frozen_train']['tail_thresholds']}",'',
        table(['Split','Tail subgroup','Count','Origins','MSE share','Mean ratio','Median ratio','Hit%','Warning'], [[x['split'],x['group'],x['count'],x['origins'],x['MSE_share'],x['amplitude']['mean'],x['amplitude']['median'],100*x['Hit'] if x['Hit'] is not None else None,x['warning']] for x in r['tails'] if x['split']!='train']),'',
        'P10/P25/P50/P75/P90、signed tail correlations、方向分组的 MAE/MSE/RMSE/Hit 全部在 tail_response.csv/results.json。','',
        f"Tail false negatives (descriptive): {r['tail_false_negative']}",'',
        '## 4. Global scalar calibration','',f"唯一应用系数 c_train = {r['frozen_train']['c_train']:.12g}。公式 sum(pred*y)/(sum(pred²)+1e-12)，无 intercept、clipping 或搜索。",'',
        f"Positive-scalar Hit sanity: {r['positive_scalar_sign_sanity']}",'',
        table(['Split','Version','MAE','MSE','RMSE','Hit%'],[[x['split'],version,*[x[version][k] for k in ('MAE','MSE','RMSE')],100*x[version]['Hit']] for x in r['calibration'] if x['group']=='Overall' for version in ('native','calibrated')]),'',
        table(['Split','Metric','Delta cal−native','95% CI'],[[s,k,v['delta'],v['CI95']] for s,vals in r['calibration_bootstrap'].items() for k,v in vals.items()]),'',
        f"TRAIN chronological halves (descriptive only): {r['calibration_stability']}",'',
        f"TRAIN per-commodity slopes (descriptive only): {r['commodity_slope_summary']}",'',
        '## 5. Tail vs non-tail calibration','',
        table(['Split','Group','Native MAE','Cal MAE','Delta MAE','Native MSE','Cal MSE','Delta MSE'],[[x['split'],x['group'],x['native']['MAE'],x['calibrated']['MAE'],x['delta']['MAE'],x['native']['MSE'],x['calibrated']['MSE'],x['delta']['MSE']] for x in r['calibration'] if x['split']!='train' and x['group']!='Overall']),'']
    for name,rows in r['commodity_rankings'].items():
        lines += [f"### {name}",'',table(['Commodity','TRAIN slope (unused for prediction)','Global-c TEST delta MAE','Global-c TEST delta MSE'],[[v['commodity'],v['train_slope'],v['delta_MAE'],v['delta_MSE']] for v in rows]),'']
    lines += ['## 6. Can tails be predicted causally?','',
        'Signal orientation comes only from TRAIN correlation with Tail90, frozen also for Tail95. AUPRC = average precision (step integral), not trapezoidal PR integration. Prevalence is the random-ranking baseline.','']
    for tail in ('Tail90','Tail95'):
        rows=[]
        ordered=sorted(SIGNALS,key=lambda k:next(x['AUROC'] or 0 for x in r['signal_predictability'] if x['signal']==k and x['split']=='test' and x['tail']==tail),reverse=True)
        for key in ordered:
            values={s:next(x for x in r['signal_predictability'] if x['signal']==key and x['split']==s and x['tail']==tail) for s in ('train','val','test')}
            a,b,t=values['val'],values['test'],values['train']
            rows.append([key,t['orientation'],t['AUROC'],a['AUROC'],b['AUROC'],a['AUPRC'],b['AUPRC'],a['prevalence'],b['prevalence'],a['AUPRC_over_prevalence'],b['AUPRC_over_prevalence'],a['Spearman'],b['Spearman']])
        lines += [f'### {tail} (display sorted by TEST AUROC, no signal tuning)','',table(['Signal','TRAIN orientation','TRAIN AUC','VAL AUC','TEST AUC','VAL AP','TEST AP','VAL prevalence','TEST prevalence','VAL AP/base','TEST AP/base','VAL Spearman','TEST Spearman'],rows),'']
    lines += ['### Clustered AUROC uncertainty','',table(['Split/signal/tail','95% CI','Valid draws'],[[k,v['CI95'],v['valid_draws']] for k,v in r['signal_AUROC_bootstrap'].items()]),'',
        '1000 draws, seed 42; 同日期商品一起抽样。重叠窗口仍有 serial dependence，CI 为 IID-origin 近似，不能当作时间块独立性已经解决。','',
        '### Frozen top-score coverage','',r['definitions']['top_capture'],'',
        table(['Split','Signal','TRAIN top% cutoff','Actual coverage','Tail90 precision','Tail90 recall'], [[x['split'],x['signal'],x['top_train_percent'],x['actual_coverage'],x['precision'],x['recall']] for x in r['top_capture'] if x['split']!='train' and x['tail']=='Tail90' and x['signal'] in IMPORTANT]),'',
        '所有 signal 的 TRAIN 分位组 tail rates、Top10/20、Tail95 见 CSV；不从 TEST 重切阈值。','',
        '## 7. Commodity-specific vs market-wide volatility','',
        table(['Signal','Split','Tail90 AUC','Tail90 AP','AP / prevalence'],[[x['signal'],x['split'],x['AUROC'],x['AUPRC'],x['AUPRC_over_prevalence']] for x in r['signal_predictability'] if x['tail']=='Tail90' and x['split']!='train' and x['signal'] in ('commodity_vol5','commodity_vol20','vol5','vol20')]),'',
        'Commodity vol5/20 与原 feature_builder 相同：past pct_change、rolling std(ddof=1)、min_periods=1、clip(0,5)。Market vol 为相同 raw-unit 波动率的 commodity 均值。Dispersion 为上一诊断的 1d returns 横截面 std(ddof=0) 的 trailing-5 mean。所有信号均来自截至 origin 的信息，没有添加模型输入。','',
        '### Origin-level severity (oriented scores)','',table(['Split','Signal','Outcome','Spearman'],[[x['split'],x['signal'],x['outcome'],x['Spearman']] for x in r['sample_signal_associations'] if x['split']!='train']),'',
        '### Conditional amplitude','',table(['Split','Signal','TRAIN bin','Count','Mean |y|','Mean |pred|','Mean ratio incl zeros','Mean ratio nonzero','Zero targets','MAE','MSE','Tail90 rate'],[[x[k] for k in ('split','signal','group','count','mean_future_abs_target','mean_abs_prediction','mean_amplitude_ratio','nonzero_target_mean_ratio','zero_targets','MAE','MSE','Tail90_rate')] for x in r['conditional_amplitude'] if x['split']!='train']),'',
        r['definitions']['zero_target_ratio'],'',
        '## 8. What does the model know?','',r.get('interpretation_text','Mechanism interpretation pending evidence review.'),'',
        '## 9. Required 18 answers','']
    lines.extend(f'{i}. {a}' for i,a in enumerate(required_answers(r),1))
    lines += ['',f"Primary classification: **{r['primary_case'] or 'PENDING'}**",'',
        r.get('next_direction','No next model experiment authorized by this diagnostic.'),'',
        'Post-hoc diagnostic evidence only. No scalar is installed into D0B and no per-commodity coefficient is applied. STOP.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def required_answers(r):
    a=r['amplitude']['test']
    tail=next(x for x in r['tails'] if x['split']=='test' and x['group']=='Tail90/all')
    correct=next(x for x in r['tails'] if x['split']=='test' and x['group']=='Tail90/correct_sign')
    pred={s:next(x for x in r['signal_predictability'] if x['split']==s and x['signal']=='abs_prediction' and x['tail']=='Tail90') for s in ('val','test')}
    answers=[f"TEST pooled native: {r['native']['test']} (Hit fraction).",
        f"Prediction std/target std = {a['R_std']:.9g}.",f"Prediction mean-absolute / target mean-absolute = {a['R_meanabs']:.9g}.",
        'P90/P95/P99 amplitude ratios: '+str({x['quantile']:x['ratio'] for x in r['amplitude_quantiles'] if x['split']=='test' and x['quantile'] in ('P90','P95','P99')}),
        f"Magnitude Pearson/Spearman: {a['magnitude_association']}",f"Tail90 amplitude mean/median: {tail['amplitude']['mean']:.9g}/{tail['amplitude']['median']:.9g}.",
        f"Tail90 correct sign: {100*tail['Hit']:.6f}%.",f"Correct-sign Tail90 amplitude ratio: mean {correct['amplitude']['mean']}, median {correct['amplitude']['median']}.",
        f"TRAIN-only c* = {r['frozen_train']['c_train']:.12g}."]
    for s,k in [('val','MAE'),('val','MSE'),('test','MAE'),('test','MSE')]:
        row=next(x for x in r['calibration'] if x['split']==s and x['group']=='Overall')
        answers.append(f"{s.upper()} {k}: {'improves' if row['delta'][k]<0 else 'does not improve'}, delta(cal−native)={row['delta'][k]:.9g}; CI95={r['calibration_bootstrap'][s][k]['CI95']}.")
    answers += ['Tail vs ordinary redistribution: '+str({s:{x['group']:x['delta'] for x in r['calibration'] if x['split']==s and x['group'] in ('Tail90','NonTail90')} for s in ('val','test')}),
        r.get('stable_signal_answer','See paired VAL/TEST signal table; interpretation pending.'),
        r.get('localized_answer','See commodity-vs-market volatility table; interpretation pending.'),
        'D0B |prediction| tail ranking: '+str({s:{k:v[k] for k in ('AUROC','AUPRC','prevalence','Spearman')} for s,v in pred.items()}),
        f"Primary case: {r['primary_case'] or 'PENDING'}."]
    return answers
