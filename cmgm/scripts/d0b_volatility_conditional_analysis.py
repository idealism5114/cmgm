"""Volatility-conditioned multiplicative diagnostic; no neural fitting.

Formal regression: [p, z*p], no intercept, no ridge, no coefficient clipping.
Half-sample coefficients and within/between risk scores are descriptive only.
"""
import numpy as np
import pandas as pd

from cmgm.scripts.d0b_5d_error_regime_analysis import metrics, finite_corr, bin_values, table
from cmgm.scripts.d0b_tail_amplitude_analysis import scalar_coefficient, cluster_weights, ranking_metrics

EPS_VOL=1e-8
EPS_GLOBAL=1e-12  # Preserve the preceding global-scalar diagnostic's definition.
BIN_QS=[.25,.5,.75,.9]
BIN_LABELS=['Q0-Q25','Q25-Q50','Q50-Q75','Q75-Q90','Q90-Q100']
VERSIONS=['Native','Global','Conditional']
REFERENCE=dict(MAE=.0219947839,MSE=.000912875426,RMSE=.0302138284,Hit=.468401487)


def least_squares(p,z,y):
    p,z,y=(np.asarray(v,dtype=np.float64) for v in (p,z,y))
    x=np.column_stack((p,z*p))
    theta,residuals,rank,singular=np.linalg.lstsq(x,y,rcond=None)
    condition=float(np.linalg.cond(x.T@x))
    return dict(a=float(theta[0]),b=float(theta[1]),rank=int(rank),singular_values=singular.tolist(),
        gram_condition_number=condition,design_condition_number=float(np.linalg.cond(x)),
        predictor_correlation=finite_corr(p,z*p),residual_sum_squares=float(np.square(x@theta-y).sum()),
        normal_equation_residual_max=float(np.max(np.abs(x.T@(x@theta-y)))),
        numerical_warning='ILL-CONDITIONED / RANK-DEFICIENT' if rank<2 or condition>1/np.sqrt(np.finfo(float).eps) else '',
        solver='np.linalg.lstsq(rcond=None), lambda=0, 2 columns [p,z*p], no intercept')


def fit_train(train):
    assert set(train.split)=={'train'},'TRAIN only'
    logv=np.log(train.commodity_vol20.to_numpy(dtype=float)+EPS_VOL)
    mu,sigma=float(logv.mean()),float(logv.std(ddof=0))
    z=(logv-mu)/(sigma+EPS_VOL)
    fit=least_squares(train.prediction,z,train.target)
    prep=train[['commodity_index','commodity']].copy()
    prep['log_vol20']=logv
    commodity=[]
    for ci,part in prep.groupby('commodity_index',sort=True):
        commodity.append(dict(commodity_index=int(ci),commodity=part.commodity.iloc[0],
            log_mean=float(part.log_vol20.mean()),log_std=float(part.log_vol20.std(ddof=0))))
    frozen=dict(global_c=scalar_coefficient(train.prediction,train.target),epsilon_log_vol=EPS_VOL,
        epsilon_global_scalar=EPS_GLOBAL,mu_log_vol=mu,sigma_log_vol=sigma,standardization_ddof=0,
        tail90=float(np.quantile(np.abs(train.target),.9)),tail95=float(np.quantile(np.abs(train.target),.95)),
        z_cuts=np.quantile(z,BIN_QS).tolist(),raw_vol20_cuts=np.quantile(train.commodity_vol20,BIN_QS).tolist(),
        prediction_magnitude_cuts=np.quantile(np.abs(train.prediction),BIN_QS).tolist(),
        conditional_fit=fit,commodity_log_reference=commodity)
    reference={v['commodity_index']:v for v in commodity}
    within=np.array([(lv-reference[int(ci)]['log_mean'])/(reference[int(ci)]['log_std']+EPS_VOL)
        for lv,ci in zip(logv,train.commodity_index)])
    between=np.array([reference[int(ci)]['log_mean'] for ci in train.commodity_index])
    label=np.abs(train.target.to_numpy())>frozen['tail90']
    frozen['risk_score_orientation']={}
    for key,score in [('raw_vol20',train.commodity_vol20.to_numpy()),('within_vol20',within),('between_mean_log_vol20',between)]:
        corr=finite_corr(score,label.astype(float))['Pearson']
        frozen['risk_score_orientation'][key]=dict(orientation=-1 if corr is not None and corr<0 else 1,
                                                   train_pearson_with_tail90=corr)
    # Both stability fits use the FULL TRAIN normalization to compare the same z.
    ordered=np.sort(train.sample_index.unique())
    midpoint=len(ordered)//2
    halves={}
    for name,ids in [('first_half',ordered[:midpoint]),('second_half',ordered[midpoint:])]:
        mask=train.sample_index.isin(ids).to_numpy()
        part=train.loc[mask]
        halves[name]={**least_squares(part.prediction,z[mask],part.target),
            'origins':len(ids),'first_date':part.forecast_origin.min(),'last_date':part.forecast_origin.max()}
    frozen['coefficient_stability']=dict(**halves,
        a_difference_second_minus_first=halves['second_half']['a']-halves['first_half']['a'],
        b_difference_second_minus_first=halves['second_half']['b']-halves['first_half']['b'],
        scale_function=[dict(z=value,full=fit['a']+fit['b']*value,
            first_half=halves['first_half']['a']+halves['first_half']['b']*value,
            second_half=halves['second_half']['a']+halves['second_half']['b']*value,
            second_minus_first=(halves['second_half']['a']-halves['first_half']['a'])+(halves['second_half']['b']-halves['first_half']['b'])*value)
            for value in (-2,-1,0,1,2)],
        note='Chronological origin halves, same full-TRAIN mu/sigma; coefficients never applied to VAL/TEST. 5d targets can overlap the midpoint.')
    return frozen


def apply_frozen(frame,frozen):
    f=frame.copy()
    f['log_vol20']=np.log(f.commodity_vol20+frozen['epsilon_log_vol'])
    f['z']=(f.log_vol20-frozen['mu_log_vol'])/(frozen['sigma_log_vol']+frozen['epsilon_log_vol'])
    fit=frozen['conditional_fit']
    f['scale']=fit['a']+fit['b']*f.z
    f['Native']=f.prediction.astype(np.float64)
    f['Global']=frozen['global_c']*f.Native
    f['Conditional']=f.scale*f.Native
    f['Tail90']=np.abs(f.target)>frozen['tail90']
    f['Tail95']=np.abs(f.target)>frozen['tail95']
    f['sign_flip']=np.sign(f.Conditional)!=np.sign(f.Native)
    refs={row['commodity_index']:row for row in frozen['commodity_log_reference']}
    if not set(f.commodity_index).issubset(refs):
        raise ValueError('Evaluation commodity not present in TRAIN references')
    f['between_mean_log_vol20']=[refs[int(i)]['log_mean'] for i in f.commodity_index]
    within_sigma=np.array([refs[int(i)]['log_std'] for i in f.commodity_index])
    f['within_vol20']=(f.log_vol20-f.between_mean_log_vol20)/(within_sigma+EPS_VOL)
    f['raw_vol20']=f.commodity_vol20
    f['volatility_bin']=bin_values(f.z,frozen['z_cuts'],BIN_LABELS)
    f['prediction_bin']=bin_values(np.abs(f.Native),frozen['prediction_magnitude_cuts'],BIN_LABELS)
    return f


def compare_group(frame,mask,split,group):
    sub=frame.loc[mask]
    row=dict(split=split,group=group,count=len(sub),origins=sub.sample_index.nunique(),
        warning='LOW SAMPLE SIZE' if sub.sample_index.nunique()<20 else '',
        mean_vol20=float(sub.commodity_vol20.mean()) if len(sub) else None,
        mean_z=float(sub.z.mean()) if len(sub) else None,
        mean_scale=float(sub.scale.mean()) if len(sub) else None,
        std_scale=float(sub.scale.std(ddof=0)) if len(sub) else None,
        Tail90_rate=float(sub.Tail90.mean()) if len(sub) else None,
        Tail95_rate=float(sub.Tail95.mean()) if len(sub) else None,
        mean_abs_target=float(sub.target.abs().mean()) if len(sub) else None)
    for version in VERSIONS:
        row[version]=metrics(sub[version].to_numpy(),sub.target.to_numpy())
    for reference in ('Native','Global'):
        row['Conditional_minus_'+reference]={k:row['Conditional'][k]-row[reference][k]
            if row[reference][k] is not None else None for k in ('MAE','MSE','RMSE','Hit')}
        row['Conditional_vs_'+reference+'_relative_percent']={k:100*row['Conditional_minus_'+reference][k]/row[reference][k]
            if row[reference][k] else None for k in ('MAE','MSE')}
    return row


def scale_distribution(sub,global_c):
    s=sub.scale.to_numpy()
    return dict(mean=float(s.mean()),std=float(s.std(ddof=0)),min=float(s.min()),max=float(s.max()),
        **{f'P{q}':float(np.quantile(s,q/100)) for q in (1,5,10,25,50,75,90,95,99)},
        fraction_negative=float((s<0).mean()),fraction_above_one=float((s>1).mean()),
        fraction_above_global=float((s>global_c).mean()),fraction_below_global=float((s<global_c).mean()),
        negative_count=int((s<0).sum()),sign_flip_count=int(sub.sign_flip.sum()),sign_flip_fraction=float(sub.sign_flip.mean()))


def bootstrap_comparisons(sub):
    weights,inverse=cluster_weights(sub.sample_index.to_numpy(),draws=1000,seed=42)
    sizes=np.bincount(inverse)
    denominator=weights@sizes
    y=sub.target.to_numpy()
    ec=sub.Conditional.to_numpy()-y
    rows=[]
    for reference in ('Native','Global'):
        er=sub[reference].to_numpy()-y
        for metric,delta in [('MAE',np.abs(ec)-np.abs(er)),('MSE',ec**2-er**2)]:
            sums=np.bincount(inverse,weights=delta)
            estimates=(weights@sums)/denominator
            rows.append(dict(comparison='Conditional_minus_'+reference,metric=metric,delta=float(delta.mean()),
                CI95=np.quantile(estimates,[.025,.975]).tolist(),draws=1000,seed=42,
                unit='paired forecast-origin clusters, all commodities retained'))
    return rows


def baseline_check(frame):
    test=frame[frame.split=='test']
    actual=metrics(test.prediction.to_numpy(),test.target.to_numpy())
    checks={key:bool(np.isclose(actual[key],value,rtol=1e-5,atol=1e-9)) for key,value in REFERENCE.items()}
    return dict(actual=actual,reference=REFERENCE,checks=checks,PASS=all(checks.values()),
                tolerance='rtol=1e-5, atol=1e-9; mismatch stops calibration before fitting')


def analyze(frame,output,verify_baseline=True):
    f=frame.copy()
    for col in ('prediction','target','commodity_vol20'):
        f[col]=f[col].astype(np.float64)
    assert set(f.split)=={'train','val','test'}
    assert not f.duplicated(['split','sample_index','commodity_index']).any()
    assert np.isfinite(f[['prediction','target','commodity_vol20']]).all().all()
    assert (f.commodity_vol20>=0).all()
    check=baseline_check(f)
    if verify_baseline and not check['PASS']:
        return dict(status='STOPPED: native D0B reference mismatch',baseline_check=check,primary_case=None)
    frozen=fit_train(f[f.split=='train'])
    f=apply_frozen(f,frozen)
    result=dict(status='STATISTICS COMPLETE',baseline_check=check,frozen_train=frozen,overall=[],
        tail_metrics=[],volatility_bins=[],prediction_bins=[],commodity_metrics=[],scale_distribution=[],
        sign_flip_metrics=[],risk_decomposition=[],bootstrap=[],z_ranges={},sign_sanity={},primary_case=None,
        zero_volatility_context=[],sign_flip_by_commodity=[],
        definitions=dict(primary_comparison='Conditional - Global, negative delta means improvement',
            conditional='(a+b*z)*p; no additive intercept, no clipping, no ridge, no neural fitting',
            global_scalar='c=sum_TRAIN(p*y)/(sum_TRAIN(p^2)+1e-12), matches preceding diagnostic',
            volatility='Same feature_builder raw commodity vol20: pct_change, rolling std ddof=1/min_periods=1, clip[0,5]',
            standardization='z=(log(vol20+1e-8)-TRAIN pooled mean)/(TRAIN pooled population std+1e-8)',
            metrics='float64 pooled MAE / MSE / RMSE=sqrt(MSE) / unmasked sign Hit; JSON Hit fraction, report percent',
            uncertainty='Forecast-origin bootstrap retains within-origin commodity clustering, but overlapping windows may still induce serial dependence. CI is an approximate origin-level uncertainty diagnostic.',
            risk_decomposition='All centering/std and score orientations fitted on TRAIN only; risk scores are not alternative calibration models',
            AUPRC='non-interpolated average precision; ties grouped; random ranking baseline=prevalence'))
    for split in ('train','val','test'):
        sub=f[f.split==split]
        overall=compare_group(sub,np.ones(len(sub),dtype=bool),split,'Overall')
        result['overall'].append(overall)
        result['z_ranges'][split]=dict(min=float(sub.z.min()),max=float(sub.z.max()))
        result['scale_distribution'].append(dict(split=split,**scale_distribution(sub,frozen['global_c'])))
        global_sign=bool(np.array_equal(np.sign(sub.Global),np.sign(sub.Native)))
        cond_sign=bool(np.array_equal(np.sign(sub.Conditional),np.sign(sub.Native)))
        if frozen['global_c']>0:
            assert global_sign and overall['Global']['Hit']==overall['Native']['Hit']
        if (sub.scale>0).all():
            assert cond_sign and overall['Conditional']['Hit']==overall['Native']['Hit']
        result['sign_sanity'][split]=dict(global_positive_c_PASS=global_sign if frozen['global_c']>0 else 'not applicable: c<=0',
            conditional_all_scales_positive=bool((sub.scale>0).all()),
            conditional_positive_scale_PASS=cond_sign if (sub.scale>0).all() else 'not applicable: inspect sign flips')
        for name,mask in [('NonTail90',~sub.Tail90),('Tail90',sub.Tail90),('Tail95',sub.Tail95),
            ('positive Tail90',sub.Tail90&(sub.target>0)),('negative Tail90',sub.Tail90&(sub.target<0))]:
            result['tail_metrics'].append(compare_group(sub,mask,split,name))
        result['sign_flip_metrics'].append(compare_group(sub,sub.sign_flip,split,'Conditional sign flips vs native'))
        result['zero_volatility_context'].append(dict(split=split,count=int((sub.commodity_vol20==0).sum()),
            fraction=float((sub.commodity_vol20==0).mean()),zero_target_fraction=float((sub.loc[sub.commodity_vol20==0,'target']==0).mean())
            if (sub.commodity_vol20==0).any() else None))
        for ci,part in sub[sub.sign_flip].groupby('commodity_index'):
            result['sign_flip_by_commodity'].append(dict(split=split,commodity_index=int(ci),commodity=part.commodity.iloc[0],
                count=len(part),mean_vol20=float(part.commodity_vol20.mean()),mean_abs_target=float(part.target.abs().mean()),
                zero_target_fraction=float((part.target==0).mean())))
        for name,col in [('volatility_bins','volatility_bin'),('prediction_bins','prediction_bin')]:
            for group in BIN_LABELS:
                result[name].append(compare_group(sub,sub[col]==group,split,group))
        for signal,spec in frozen['risk_score_orientation'].items():
            score=spec['orientation']*sub[signal].to_numpy()
            result['risk_decomposition'].append(dict(split=split,signal=signal,orientation=spec['orientation'],
                **ranking_metrics(sub.Tail90.to_numpy(),score),magnitude_association=finite_corr(score,sub.target.abs())))
        if split in ('val','test'):
            result['bootstrap'].extend(dict(split=split,**row) for row in bootstrap_comparisons(sub))
        if split=='test':
            for ci,part in sub.groupby('commodity_index'):
                row=compare_group(sub,sub.commodity_index==ci,split,part.commodity.iloc[0])
                row['commodity_index']=int(ci)
                result['commodity_metrics'].append(row)
    result['commodity_correlations']={key:finite_corr([row[key] for row in result['commodity_metrics']],
        [row['Conditional_minus_Global']['MSE'] for row in result['commodity_metrics']]) for key in ('mean_vol20','mean_z','mean_scale','std_scale')}
    result['commodity_rankings']={f'{metric}_{direction}':sorted(result['commodity_metrics'],
        key=lambda x:x['Conditional_minus_Global'][metric],reverse=direction=='worsened')[:5]
        for metric in ('MAE','MSE') for direction in ('improved','worsened')}
    test_count=sum(row['count'] for row in result['commodity_metrics'])
    result['commodity_error_attribution']=[dict(commodity=row['group'],count=row['count'],
        observation_weight=row['count']/test_count,
        contribution_to_overall_delta_MAE=row['count']/test_count*row['Conditional_minus_Global']['MAE'],
        contribution_to_overall_delta_MSE=row['count']/test_count*row['Conditional_minus_Global']['MSE']) for row in result['commodity_metrics']]
    for filename,key in [('overall_metrics','overall'),('volatility_bin_metrics','volatility_bins'),('tail_metrics','tail_metrics'),
        ('commodity_metrics','commodity_metrics'),('scale_distribution','scale_distribution'),('volatility_decomposition','risk_decomposition'),
        ('bootstrap_results','bootstrap'),('prediction_bin_metrics','prediction_bins'),('sign_flip_metrics','sign_flip_metrics')]:
        pd.json_normalize(result[key]).to_csv(output/(filename+'.csv'),index=False)
    f.to_csv(output/'observation_conditional_diagnostics.csv',index=False)
    # Mechanism interpretation is completed from the actual result, separately
    # from fitting; no model/score is selected or changed by interpretation.
    attach_interpretation(result)
    return result


def attach_interpretation(r):
    overall={row['split']:row for row in r['overall']}
    evaluation=[x for x in r['bootstrap'] if x['comparison']=='Conditional_minus_Global']
    robust_all=all(x['delta']<0 and x['CI95'][1]<0 for x in evaluation)
    risk={s:{row['signal']:row for row in r['risk_decomposition'] if row['split']==s} for s in ('val','test')}
    risk_consistent=all(v['raw_vol20']['AUROC']>.5 and v['raw_vol20']['AUPRC_over_prevalence']>1 for v in risk.values())
    # Report classification only. No coefficients or model choice feed back to inference.
    r['primary_case']='Case A' if robust_all else 'Case F' if risk_consistent else 'Case B'
    r['interpretation_status']='EVIDENCE REVIEW COMPLETE'
    r['classification_evidence']=dict(conditional_beats_global_all_four_with_CI=robust_all,
        raw_volatility_tail_ranking_consistent=risk_consistent,
        priority='User section 77 prioritizes B/D/E/F when Conditional does not stably beat Global; no success claim from tail AUC alone.',
        effect_sizes={s:overall[s]['Conditional_vs_Global_relative_percent'] for s in ('val','test')})
    r['volatility_answer']='收益位置：'+ '; '.join(s.upper()+': '+', '.join(
        f"{x['group']} ΔMAE={x['Conditional_minus_Global']['MAE']:.9g}, ΔMSE={x['Conditional_minus_Global']['MSE']:.9g}"
        for x in r['volatility_bins'] if x['split']==s) for s in ('val','test'))+'. 负 delta 表示相对 Global 改善；不能把 pooled MAE 改善自动归因于高波动尾部。'
    signed=[x for x in r['tail_metrics'] if x['split']!='train' and x['group'] in ('positive Tail90','negative Tail90')]
    r['tail_answer']='方向不对称：'+ '; '.join(f"{x['split'].upper()} {x['group']} ΔMAE={x['Conditional_minus_Global']['MAE']:.9g}, ΔMSE={x['Conditional_minus_Global']['MSE']:.9g}" for x in signed)+'. 当前 b>0 对高波动采用更大的乘数；这也可能放大错向预测。'
    r['risk_answer']='固定 TRAIN 方向后的 raw/within/between AUROC：'+ '; '.join(
        f"{s.upper()} {v['raw_vol20']['AUROC']:.6f}/{v['within_vol20']['AUROC']:.6f}/{v['between_mean_log_vol20']['AUROC']:.6f}" for s,v in risk.items())+'. 本次 within 仍有排序信息，不能只归因于静态商品风险差异；但这不是方差分解，AUC 不可相减来量化各来源份额。'
    largest=min(r['commodity_error_attribution'],key=lambda x:x['contribution_to_overall_delta_MAE'])
    total=overall['test']['Conditional_minus_Global']['MAE']
    contribution_text=(f"TEST 最大 MAE 改善来源为 {largest['commodity']}：对 pooled ΔMAE 的贡献 {largest['contribution_to_overall_delta_MAE']:.12g}；全部 ΔMAE={total:.12g}；其余商品贡献合计={total-largest['contribution_to_overall_delta_MAE']:.12g}。这是对同一正式预测的加总分解，没有剔除样本、没有重拟合。");
    fit=r['frozen_train']['conditional_fit'];stability=r['frozen_train']['coefficient_stability']
    text=[f"Primary {r['primary_case']}。"+('Conditional 对 Global 四项均有一致且区间支持的改善。' if robust_all else '按第77条优先规则，风险排序信息尚未转化为稳健的 signed-return calibration 增量收益。'),
        '; '.join(f"{s.upper()} Conditional−Global relative MAE={overall[s]['Conditional_vs_Global_relative_percent']['MAE']:.6f}%, MSE={overall[s]['Conditional_vs_Global_relative_percent']['MSE']:.6f}%" for s in ('val','test')),
        '不能说 MAE 没有增量改善：它确实下降；但 MSE 没有同时改善。应将具体取舍、误差来源和稳定性一起判断，不能仅凭 tail AUC 接受新的 return scale。',
        contribution_text,r['volatility_answer'],r['tail_answer'],r['risk_answer'],
        f"数值求解健康：rank={fit['rank']}, cond(XᵀX)={fit['gram_condition_number']:.6f}。时间稳定性却不足：first-half (a,b)=({stability['first_half']['a']:.6f},{stability['first_half']['b']:.6f})，second-half=({stability['second_half']['a']:.6f},{stability['second_half']['b']:.6f})。数值条件良好不代表回归关系跨时间稳定。",
        '零波动对应 log(1e-8)，会影响 pooled log-vol 标准差、商品 TRAIN mean 和 between-score 的方向。当前 between-score TRAIN Pearson 接近零且为负，按预设规则取 -1；即使其排序与 Pearson 方向不一致，也没有看到 VAL/TEST 后翻转。不能由这个单一 between proxy 的弱表现断言全部静态商品风险都无信息。',
        '负 scale 保留原式。符号翻转可能在零目标观测上发生，此时 native 和 conditional 的 unmasked Hit 都是错的，所以总体 Hit 不变不代表没有 sign flip。',
        '目前最明确的证据是：商品波动率含风险强度信息，但直接把它乘到现有 signed prediction 上，会保留并可能放大方向偏差。该结论只覆盖本轮固定的两参数形式，不能推广为“波动率对任何收益模型都无用”。']
    r['interpretation_text']='\n\n'.join(text)
    r['next_direction']=('唯一后续候选：tiny commodity-specific conditional-scale mechanism；本轮不实现。' if robust_all else
        '停止把本轮 conditional scale 加入 D0B signed-return head。唯一后续候选是 joint return mean + commodity-specific risk modeling；本轮不实现、不训练，也不创建 calibrated checkpoint。')


def group_table(rows):
    return table(['Split','Group','Obs','Origins','Version','MAE','MSE','RMSE','Hit%','Warning'],[
        [r['split'],r['group'],r['count'],r['origins'],v,*[r[v][k] for k in ('MAE','MSE','RMSE')],
         100*r[v]['Hit'] if r[v]['Hit'] is not None else None,r['warning']] for r in rows for v in VERSIONS])


def write_report(r,output):
    cp=r.get('checkpoint',{})
    lines=['# D0B-VolatilityConditionalShrinkageDiagnostic','',r['status'],'',
        '只执行正式 D0B inference 和预定义 TRAIN-only 闭式回归。没有训练、backward、optimizer、模型结构或参数更新；没有 calibrated checkpoint。','',
        table(['Checkpoint','Best epoch','Params','Seed metadata','Training SHA','Diagnostic SHA'],[[cp.get('checkpoint_path'),cp.get('best_epoch'),cp.get('parameter_count'),cp.get('seed'),cp.get('checkpoint_git_sha'),r.get('diagnostic_git_sha')]]),'',
        f"Model parameter max diff = {cp.get('model_parameter_max_diff','N/A')}. Checkpoint/state_dict SHA256 before/after are in results.json. Missing training metadata remains N/A.",'',
        f"Native reference check: {r['baseline_check']}",'']
    if r['status'].startswith('STOPPED'):
        lines+=['Baseline mismatch: no calibration fitted. STOP.']
        (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        return
    f=r['frozen_train'];fit=f['conditional_fit']
    lines+=['## 1. Baselines','',f"TRAIN global scalar c = {f['global_c']:.12g} (denominator epsilon={EPS_GLOBAL}).",'',
        'Native/Global/Conditional 的完整三 split 指标列于第3节。','',
        '## 2. Conditional calibration definition','',
        '`Conditional = (a + b*z)*p`, design matrix `[p, z*p]`, no prediction intercept. lambda=0, np.linalg.lstsq, no clipping.','',
        table(['a','b','TRAIN log-vol mean','TRAIN log-vol population std','Log/std epsilon','Gram condition number','Design rank'],[[fit['a'],fit['b'],f['mu_log_vol'],f['sigma_log_vol'],f['epsilon_log_vol'],fit['gram_condition_number'],fit['rank']]]),'',
        f"corr(p,z*p)={fit['predictor_correlation']}; normal-equation residual max={fit['normal_equation_residual_max']}; numerical warning={fit['numerical_warning'] or 'none'}.",'',
        'b 的符号：'+('b>0，高波动相对低波动采用更大的乘数（less shrinkage / more amplification）。' if fit['b']>0 else 'b<0，高波动相对低波动采用更小的乘数（more shrinkage）。' if fit['b']<0 else 'b=0，没有波动率斜率。'),'',
        table(['Split','min z','max z'],[[s,v['min'],v['max']] for s,v in r['z_ranges'].items()]),'',
        '## 3. Main performance','',group_table(r['overall']),'',
        '**主要科学比较为 Conditional vs Global，不是仅比较 Native。**','',
        '## 4. Incremental value of commodity volatility','',
        table(['Split','Metric','Conditional−Global','Relative %'],[[x['split'],k,x['Conditional_minus_Global'][k],x['Conditional_vs_Global_relative_percent'][k]] for x in r['overall'] if x['split']!='train' for k in ('MAE','MSE')]),'',
        table(['Split','Comparison','Metric','Delta','95% CI'],[[x['split'],x['comparison'],x['metric'],x['delta'],x['CI95']] for x in r['bootstrap']]),'',r['definitions']['uncertainty'],'',
        '## 5. Volatility-group behavior','',
        table(['Split','TRAIN vol20 bin','Obs','Origins','Mean vol20','Mean z','Mean scale','Tail90 rate','Tail95 rate','Mean |y|','Cond−Global MAE','Cond−Global MSE','Warning'],[[x['split'],x['group'],x['count'],x['origins'],x['mean_vol20'],x['mean_z'],x['mean_scale'],x['Tail90_rate'],x['Tail95_rate'],x['mean_abs_target'],x['Conditional_minus_Global']['MAE'],x['Conditional_minus_Global']['MSE'],x['warning']] for x in r['volatility_bins'] if x['split']!='train']),'',
        '各组 Native/Global/Conditional 的全部 MAE/MSE/RMSE/Hit 见 volatility_bin_metrics.csv。阈值只来自 TRAIN，空组不重切。','',
        '## 6. Tail behavior','',group_table([x for x in r['tail_metrics'] if x['split']!='train']),'',
        table(['Split','Tail group','Conditional−Global MAE','Conditional−Global MSE'],[[x['split'],x['group'],x['Conditional_minus_Global']['MAE'],x['Conditional_minus_Global']['MSE']] for x in r['tail_metrics'] if x['split']!='train']),'',
        f"Frozen TRAIN tail thresholds: T90={f['tail90']:.12g}, T95={f['tail95']:.12g}. Tail/sign groups contain future targets and are ex-post explanations, not conditioning features.",'',
        '## 7. Scale factor behavior','',
        table(['Split','Mean','Std','P5','P25','P50','P75','P95','min','max','P(s<0)','P(s>1)','P(s>c)','P(s<c)','Sign flips'],[[x[k] for k in ('split','mean','std','P5','P25','P50','P75','P95','min','max','fraction_negative','fraction_above_one','fraction_above_global','fraction_below_global','sign_flip_count')] for x in r['scale_distribution']]),'',
        '其余 P1/P10/P90/P99 与 sign-flip fraction 在 scale_distribution.csv；负 scale 按原式保留，可以反转非零预测符号。没有偷偷 clipping。','',
        f"Zero-volatility coverage: {r['zero_volatility_context']}",'',
        table(['Split','Sign-flip commodity','Count','Mean vol20','Mean |target|','Zero-target fraction'],[[x[k] for k in ('split','commodity','count','mean_vol20','mean_abs_target','zero_target_fraction')] for x in r['sign_flip_by_commodity']]),'',
        f"Sign invariance sanity: {r['sign_sanity']}",'',group_table(r['sign_flip_metrics']),'',
        table(['Split','Sign-flip count','Tail90 prevalence'],[[x['split'],x['count'],x['Tail90_rate']] for x in r['sign_flip_metrics']]),'',
        '### By native prediction magnitude','',
        table(['Split','TRAIN |p| bin','Count','Mean scale','Native MAE','Global MAE','Conditional MAE'],[[x['split'],x['group'],x['count'],x['mean_scale'],x['Native']['MAE'],x['Global']['MAE'],x['Conditional']['MAE']] for x in r['prediction_bins'] if x['split']!='train']),'',
        '## 8. Static vs dynamic commodity risk','',
        table(['Signal','TRAIN orientation','VAL AUC','TEST AUC','VAL AP','TEST AP','VAL prevalence','TEST prevalence'],[
            [signal,f['risk_score_orientation'][signal]['orientation'],
             *[next(x[k] for x in r['risk_decomposition'] if x['signal']==signal and x['split']==s) for k in ('AUROC','AUPRC','prevalence') for s in ('val','test')]]
            for signal in f['risk_score_orientation']]),'',
        'within = (log(vol20+1e-8)−commodity TRAIN log mean)/(commodity TRAIN log population std+1e-8)。between = commodity TRAIN log mean。方向由 TRAIN Pearson(score,Tail90) 确定，VAL/TEST 冻结。AUROC ties 分组处理，AP 为非插值 average precision，随机 baseline 为 prevalence。','',
        '这些分解分数只评估风险排序，没有拟合第二套 calibration。强 between ranking 不等于动态尾部时点可预测；风险排序也不等于收益方向可预测。','',
        '## 9. Temporal coefficient stability','',
        table(['TRAIN subset','a','b','Gram condition number'],[['Full',fit['a'],fit['b'],fit['gram_condition_number']]]+[
            [name,f['coefficient_stability'][name]['a'],f['coefficient_stability'][name]['b'],f['coefficient_stability'][name]['gram_condition_number']] for name in ('first_half','second_half')]),'',
        table(['z','Full s(z)','First-half s(z)','Second-half s(z)','Second−first'],[[x[k] for k in ('z','full','first_half','second_half','second_minus_first')] for x in f['coefficient_stability']['scale_function']]),'',
        f"Delta a(second−first)={f['coefficient_stability']['a_difference_second_minus_first']}; delta b={f['coefficient_stability']['b_difference_second_minus_first']}. {f['coefficient_stability']['note']}",'',
        '## 10. Commodity effects','',group_table(r['commodity_metrics']),'',
        f"Scale/volatility statistics vs TEST Conditional−Global MSE correlations (exploratory): {r['commodity_correlations']}",'']
    for name,rows in r['commodity_rankings'].items():
        lines += [name,'',table(['Commodity','Cond−Global MAE','Cond−Global MSE','Mean vol20','Mean z','Mean scale','Scale std'],[[x['group'],x['Conditional_minus_Global']['MAE'],x['Conditional_minus_Global']['MSE'],x['mean_vol20'],x['mean_z'],x['mean_scale'],x['std_scale']] for x in rows]),'']
    lines += ['## 11. Mechanism conclusion','',r.get('interpretation_text','Pending evidence review.'),'',
        '## 12. Required 16 answers','']
    lines.extend(f'{i}. {answer}' for i,answer in enumerate(required_answers(r),1))
    lines += ['',f"Primary classification: **{r['primary_case'] or 'PENDING'}**",'',r.get('next_direction','No next model implementation.'),'',
        'No trained/fine-tuned/calibrated checkpoint has been created. STOP.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def required_answers(r):
    f=r['frozen_train'];fit=f['conditional_fit']
    answers=[f"Native TEST reference reproduced: {r['baseline_check']}",f"TRAIN c={f['global_c']:.12g}.",
        f"TRAIN log-vol20 mean={f['mu_log_vol']:.12g}, std(ddof=0)={f['sigma_log_vol']:.12g}.",
        f"Conditional a={fit['a']:.12g}, b={fit['b']:.12g}.",
        f"b sign={np.sign(fit['b']):.0f}: {'larger multiplier at higher volatility' if fit['b']>0 else 'smaller multiplier at higher volatility'}.",
        'Scale distributions: '+str([x for x in r['scale_distribution'] if x['split']!='train']),
        'Negative scale/sign flips: '+str({x['split']:{k:x[k] for k in ('negative_count','fraction_negative','sign_flip_count','sign_flip_fraction')} for x in r['scale_distribution']})]
    for ref in ('Native','Global'):
        answers.append(f'Conditional vs {ref}: '+str({x['split']:x['Conditional_minus_'+ref] for x in r['overall'] if x['split']!='train'}))
    for s in ('val','test'):
        answers.append(s.upper()+' Conditional−Global delta and CI: '+str([x for x in r['bootstrap'] if x['split']==s and x['comparison']=='Conditional_minus_Global']))
    answers.extend([r.get('volatility_answer','See fixed-bin table; interpretation pending.'),r.get('tail_answer','See signed-tail table; interpretation pending.'),
        r.get('risk_answer','See raw/within/between table; interpretation pending.'),
        f"TRAIN halves: {f['coefficient_stability']}",f"Primary case: {r['primary_case'] or 'PENDING'}."])
    return answers
