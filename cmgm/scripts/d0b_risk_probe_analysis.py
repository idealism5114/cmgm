"""Fixed ordinary least-squares probes of frozen D0B representations.

Only |5d return| is fitted. No ridge, nonlinear features, clipping or selection.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew

from cmgm.scripts.d0b_5d_error_regime_analysis import finite_corr as _finite_corr,table as _table
from cmgm.scripts.d0b_tail_amplitude_analysis import ranking_metrics,ranking_bootstrap,cluster_weights

EPS_X=1e-8
EPS_VOL=1e-8
EPS_RISK=1e-6
PROBES=['MeanRisk','Vol20Only','SimpleRiskFeatures','GlobalFused','TemporalOnly','SpatialGlobal','LongMicro',
        'CommodityNode','CommodityNodePlusGlobal','CommodityNodePlusVol20','GlobalPlusVol20']
STATIC='CommodityTrainMeanRisk'
MAIN=['Vol20Only','GlobalFused','CommodityNode','CommodityNodePlusGlobal','CommodityNodePlusVol20','GlobalPlusVol20']
STABILITY=['Vol20Only','CommodityNode','CommodityNodePlusVol20','GlobalPlusVol20']
PAIRS=[('CommodityNode','GlobalFused'),('Vol20Only','CommodityNode'),
       ('CommodityNodePlusVol20','CommodityNode'),('GlobalPlusVol20','GlobalFused'),
       ('CommodityNodePlusGlobal','CommodityNode')]
INPUTS=dict(MeanRisk='TRAIN pooled mean |y|',Vol20Only='log(commodity vol20 + 1e-8)',
    SimpleRiskFeatures='log(vol5+1e-8), log(vol20+1e-8), abs(past1), abs(past5)',
    GlobalFused='h_fused',TemporalOnly='h_temporal',SpatialGlobal='h_spatial',LongMicro='[h_long,h_micro]',
    CommodityNode='h_comm_i',CommodityNodePlusGlobal='[h_comm_i,h_fused]',
    CommodityNodePlusVol20='[h_comm_i,log(vol20+1e-8)]',GlobalPlusVol20='[h_fused,log(vol20+1e-8)]')


def table(headers,rows):
    def escape(value):
        return value.replace('|','\\|') if isinstance(value,str) else value
    return _table([escape(h) for h in headers],[[escape(v) for v in row] for row in rows])


def finite_corr(a,b):
    a,b=np.asarray(a),np.asarray(b)
    # Exact constant coordinates can acquire tiny nonzero std from summation
    # rounding; correlation remains undefined, not an informative zero.
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:
        return dict(Pearson=None,Spearman=None)
    return _finite_corr(a,b)


def fit_linear(x,y,normalization=None):
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or len(y)!=len(x):
        raise ValueError('Expected observation × feature design and unnormalized |y| target')
    if x.shape[1]>len(x):
        raise ValueError('STOP: feature dimension exceeds observation count')
    if normalization is None:
        mean,std=x.mean(0),x.std(0,ddof=0)
    else:
        mean,std=np.asarray(normalization['mean']),np.asarray(normalization['std'])
    constant=(std<1e-12)|(x.std(0,ddof=0)<1e-12)
    active=~constant
    scaled=(x[:,active]-mean[active])/(std[active]+EPS_X)
    design=np.column_stack([scaled,np.ones(len(x))])
    theta,_,rank,singular=np.linalg.lstsq(design,y,rcond=None)
    weights=np.zeros(x.shape[1]);weights[active]=theta[:-1]
    cond=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    return dict(mean=mean.tolist(),std=std.tolist(),active=active.tolist(),weights=weights.tolist(),
        intercept=float(theta[-1]),dimension=x.shape[1],coefficient_count=int(active.sum())+1,
        constant_dimensions=int(constant.sum()),constant_threshold=1e-12,standardization_epsilon=EPS_X,
        rank=int(rank),singular_values=singular.tolist(),condition_number=cond,
        numerical_warning='RANK DEFICIENT / ILL CONDITIONED' if rank<design.shape[1] or cond>1/np.sqrt(np.finfo(float).eps) else '',
        solver='np.linalg.lstsq(rcond=None), lambda=0; standardized features plus intercept',
        target='raw |5d return|, not normalized')


def predict_linear(x,fit):
    x=np.asarray(x,dtype=np.float64)
    active=np.asarray(fit['active'],dtype=bool)
    # Dropped constant coordinates never react to held-out changes.
    return ((x[:,active]-np.asarray(fit['mean'])[active])/(np.asarray(fit['std'])[active]+EPS_X))@np.asarray(fit['weights'])[active]+fit['intercept']


def audit_linear(x,y,train,fit,splits):
    """Verify the SAME fitted coefficients; no second fit or model selection."""
    active=np.asarray(fit['active'],dtype=bool)
    z=(x[:,active]-np.asarray(fit['mean'])[active])/(np.asarray(fit['std'])[active]+EPS_X)
    design=np.column_stack([z,np.ones(len(z))])
    theta=np.r_[np.asarray(fit['weights'])[active],fit['intercept']]
    pred=design@theta;residual=pred[train]-y[train]
    return dict(normal_equation_mean_residual_max=float(np.max(np.abs(design[train].T@residual))/train.sum()),
        train_RiskMSE=float(np.mean(residual**2)),
        equivalent_design_max_diff=float(np.max(np.abs(pred-predict_linear(x,fit)))),
        weight_norm=float(np.linalg.norm(theta[:-1])),
        **{f'{split}_standardized_abs_max':float(np.max(np.abs(z[splits==split]))) if z.shape[1] else 0.
           for split in ('train','val','test')})


def make_features(frame,reps):
    keys=pd.MultiIndex.from_frame(frame[['split','sample_index','forecast_origin']])
    origin_index,unique=pd.factorize(keys,sort=False)
    commodity=frame.commodity_index.to_numpy(dtype=int)
    assert len(unique)==len(reps['h_fused'])
    assert reps['h_comm'].ndim==3
    assert np.all(commodity<reps['h_comm'].shape[1])
    local=reps['h_comm'][origin_index,commodity].astype(float)
    global_fused=reps['h_fused'][origin_index].astype(float)
    v20=np.log(frame.commodity_vol20.to_numpy(dtype=float)+EPS_VOL)[:,None]
    v5=np.log(frame.commodity_vol5.to_numpy(dtype=float)+EPS_VOL)[:,None]
    return dict(Vol20Only=v20,SimpleRiskFeatures=np.column_stack([v5,v20,frame.abs_past1,frame.abs_past5]),
        GlobalFused=global_fused,TemporalOnly=reps['h_temporal'][origin_index].astype(float),
        SpatialGlobal=reps['h_spatial'][origin_index].astype(float),
        LongMicro=np.column_stack([reps['h_long'][origin_index],reps['h_micro'][origin_index]]).astype(float),
        CommodityNode=local,CommodityNodePlusGlobal=np.column_stack([local,global_fused]),
        CommodityNodePlusVol20=np.column_stack([local,v20]),GlobalPlusVol20=np.column_stack([global_fused,v20]))


def regression_metrics(pred,target,train_mean):
    pred,target=np.asarray(pred,dtype=float),np.asarray(target,dtype=float)
    error=pred-target
    mse=float(np.mean(error**2))
    denom=float(np.sum((target-train_mean)**2))
    return dict(RiskMAE=float(np.mean(np.abs(error))),RiskMSE=mse,RiskRMSE=float(np.sqrt(mse)),
        R2=1-float(np.sum(error**2))/denom if denom else None,
        negative_prediction_fraction=float(np.mean(pred<0)),**finite_corr(pred,target),
        RMSE_squared_minus_MSE_abs=abs(float(np.sqrt(mse))**2-mse))


def target_distributions(frame,tail_cut):
    rows=[]
    for split in ('train','val','test'):
        y=frame.loc[frame.split==split,'target'].to_numpy(dtype=float)
        for name,value in [('absolute',np.abs(y)),('squared',y**2),('log_absolute',np.log(np.abs(y)+EPS_RISK))]:
            mask=np.abs(y)>tail_cut
            # Split-local order statistics are distribution descriptions only;
            # no evaluation-derived threshold is used by a fitted probe.
            top_count=max(1,int(np.ceil(.05*len(value))))
            top_values=np.sort(value)[-top_count:]
            total=float(value.sum())
            rows.append(dict(split=split,target=name,mean=float(value.mean()),std=float(value.std(ddof=0)),
                median=float(np.median(value)),max=float(value.max()),
                **{f'P{q}':float(np.quantile(value,q/100)) for q in (50,75,90,95,99)},
                skewness=float(skew(value,bias=False)) if np.std(value)>0 else None,
                split_top5_count=top_count,
                split_top5_signed_sum_share=float(top_values.sum()/total) if total else None,
                split_top5_target_mass_share=float(top_values.sum()/total) if total and name!='log_absolute' else None,
                top5_train_threshold_mass_share=float(value[mask].sum()/total) if total and name!='log_absolute' else None,
                top5_signed_sum_share=float(value[mask].sum()/total) if total else None,
                top5_train_threshold_count=int(mask.sum()),top5_train_threshold_coverage=float(mask.mean()),
                log_target_top5_absolute_mass_share=float(np.abs(value[mask]).sum()/np.abs(value).sum()) if name=='log_absolute' and np.abs(value).sum() else None,
                mass_note='Frozen TRAIN P95(|y|); evaluation coverage need not be 5%. Log-risk is signed, so a positive target-mass contribution is not well-defined; absolute log mass supplied separately.'))
    return rows


def representation_statistics(frame,reps):
    origins=frame[['split','sample_index','forecast_origin']].drop_duplicates().reset_index(drop=True)
    rows=[]
    for split in ('train','val','test'):
        mask=(origins.split==split).to_numpy()
        for key,array in reps.items():
            value=array[mask].astype(float)
            norm=np.linalg.norm(value,axis=-1)
            row=dict(split=split,representation=key,norm_mean=float(norm.mean()),norm_std=float(norm.std(ddof=0)),
                norm_unit='origin × commodity' if key=='h_comm' else 'origin')
            if key=='h_comm':
                normalized=np.divide(value,norm[...,None],out=np.zeros_like(value),where=norm[...,None]>0)
                cos=np.einsum('bid,bjd->bij',normalized,normalized)
                i,j=np.triu_indices(value.shape[1],k=1)
                pairs=cos[:,i,j]
                row.update(pairwise_cosine_mean=float(pairs.mean()),pairwise_cosine_std=float(pairs.std(ddof=0)),
                    cross_commodity_variance=float(value.var(axis=1,ddof=0).mean()),
                    within_commodity_temporal_variance=float(value.var(axis=0,ddof=0).mean()),
                    zero_norm_count=int((norm==0).sum()))
            rows.append(row)
    return rows


def stability_fit(x,y,full,origin_ids):
    ordered=np.unique(origin_ids);half=len(ordered)//2
    fits=[]
    for ids in (ordered[:half],ordered[half:]):
        mask=np.isin(origin_ids,ids)
        fits.append(fit_linear(x[mask],y[mask],normalization=full))
    a,b=(np.asarray(f['weights']) for f in fits)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    out=dict(first_weight_norm=float(np.linalg.norm(a)),second_weight_norm=float(np.linalg.norm(b)),
        weight_cosine=float(a@b/denom) if denom else None,
        first_intercept=fits[0]['intercept'],second_intercept=fits[1]['intercept'],
        first_fit=fits[0],second_fit=fits[1],
        note='Chronological TRAIN origin halves, same FULL TRAIN feature normalization for comparable coefficient coordinates; no held-out predictions use these coefficients. Half-constant dimensions dropped, intercept absorbs their constant contribution.')
    if x.shape[1]==1:
        for label,fit in zip(('first','second'),fits):
            w=fit['weights'][0]/(fit['std'][0]+EPS_X)
            out[label+'_raw_log_vol_slope']=w
            out[label+'_raw_log_vol_intercept']=fit['intercept']-w*fit['mean'][0]
    return out


def bootstrap_regression(frame,predictions,target,train_mean):
    counts,inverse=cluster_weights(frame.sample_index.to_numpy(),draws=1000,seed=42)
    sizes=np.bincount(inverse)
    denominator=counts@sizes
    errors={p:predictions[p]-target for p in predictions}
    rows=[]
    for probe in MAIN:
        for metric,values in [('RiskMAE',np.abs(errors[probe])),('RiskMSE',errors[probe]**2)]:
            sums=np.bincount(inverse,weights=values)
            values_boot=(counts@sums)/denominator
            rows.append(dict(kind='absolute',probe=probe,comparison=probe,metric=metric,estimate=float(values.mean()),
                CI95=np.quantile(values_boot,[.025,.975]).tolist(),draws=1000,seed=42))
    for first,second in PAIRS:
        for metric,delta in [('RiskMAE',np.abs(errors[first])-np.abs(errors[second])),
                             ('RiskMSE',errors[first]**2-errors[second]**2)]:
            sums=np.bincount(inverse,weights=delta)
            values=(counts@sums)/denominator
            rows.append(dict(kind='paired_delta',comparison=f'{first} - {second}',metric=metric,
                estimate=float(delta.mean()),CI95=np.quantile(values,[.025,.975]).tolist(),draws=1000,seed=42))
    return rows


def within_commodity(frame,predictions,threshold):
    rows=[]
    for ci,part in frame.groupby('commodity_index',sort=True):
        positions=frame.index.get_indexer(part.index)
        y=np.abs(part.target.to_numpy(dtype=float));label=y>threshold
        positive,negative=int(label.sum()),int((~label).sum())
        for probe,pred in predictions.items():
            pred=pred[positions]
            corr=finite_corr(pred,y)
            auc=ranking_metrics(label,pred)['AUROC'] if positive>=10 and negative>=10 else None
            rows.append(dict(commodity_index=int(ci),commodity=part.commodity.iloc[0],probe=probe,
                count=len(part),positives=positive,negatives=negative,Spearman=corr['Spearman'],Pearson=corr['Pearson'],
                Tail90_AUROC=auc,AUC_status='VALID' if auc is not None else 'INSUFFICIENT: require >=10 positives and >=10 negatives',
                correlation_status='VALID' if corr['Spearman'] is not None else 'UNDEFINED: constant target or score'))
    summaries=[]
    for probe in predictions:
        sub=[r for r in rows if r['probe']==probe]
        correlations=[r['Spearman'] for r in sub if r['Spearman'] is not None]
        aucs=[r['Tail90_AUROC'] for r in sub if r['Tail90_AUROC'] is not None]
        summaries.append(dict(probe=probe,valid_correlations=len(correlations),total_commodities=len(sub),
            Spearman_mean=float(np.mean(correlations)) if correlations else None,
            Spearman_median=float(np.median(correlations)) if correlations else None,
            Spearman_std=float(np.std(correlations,ddof=0)) if correlations else None,
            positive_fraction=float(np.mean(np.asarray(correlations)>0)) if correlations else None,
            valid_AUC_commodities=len(aucs),AUC_mean=float(np.mean(aucs)) if aucs else None,
            AUC_median=float(np.median(aucs)) if aucs else None,AUC_above_chance_count=int(np.sum(np.asarray(aucs)>.5)),
            denominator_note='Correlation summaries use valid/nonconstant commodities only; constants are UNDEFINED, not zero. AUC requires >=10 positives/negatives.'))
    return rows,summaries


def analyze(frame,reps,output):
    frame=frame.reset_index(drop=True).copy()
    assert set(frame.split)=={'train','val','test'}
    train=(frame.split=='train').to_numpy()
    risk=np.abs(frame.target.to_numpy(dtype=float))
    mean=float(risk[train].mean());t90,t95=np.quantile(risk[train],[.9,.95])
    inputs=make_features(frame,reps)
    assert set(inputs)==set(PROBES)-{'MeanRisk'}
    fits={};audits={};predictions={'MeanRisk':np.full(len(frame),mean)}
    definitions=[dict(probe='MeanRisk',input=INPUTS['MeanRisk'],dimension=0,coefficient_count=0,constant_dimensions=0,kind='TRAIN pooled baseline')]
    stability=[]
    for name in PROBES[1:]:
        x=inputs[name]
        assert np.isfinite(x).all()
        fit=fit_linear(x[train],risk[train])
        fits[name]=fit
        audits[name]=audit_linear(x,risk,train,fit,frame.split.to_numpy())
        predictions[name]=predict_linear(x,fit)
        definitions.append(dict(probe=name,input=INPUTS[name],dimension=x.shape[1],coefficient_count=fit['coefficient_count'],
            constant_dimensions=fit['constant_dimensions'],kind='TRAIN-only linear probe',rank=fit['rank'],condition_number=fit['condition_number'],numerical_warning=fit['numerical_warning']))
        if name in STABILITY:
            stability.append(dict(probe=name,**stability_fit(x[train],risk[train],fit,frame.loc[train,'sample_index'].to_numpy())))
        print(f"[D0B risk probe] {name}: d={x.shape[1]}, dropped={fit['constant_dimensions']}, rank={fit['rank']}, condition={fit['condition_number']:.4g}",flush=True)
    mean_by_commodity=frame.loc[train].assign(risk=risk[train]).groupby('commodity_index').risk.mean().to_dict()
    assert set(frame.commodity_index).issubset(mean_by_commodity)
    predictions[STATIC]=frame.commodity_index.map(mean_by_commodity).to_numpy(dtype=float)
    definitions.append(dict(probe=STATIC,input='TRAIN per-commodity mean |y|',dimension=0,coefficient_count=0,constant_dimensions=0,kind='static commodity baseline, not a learned probe'))
    result=dict(probe_definitions=definitions,probe_fits=fits,numerical_audit=audits,probe_metrics=[],tail_probe_metrics=[],probe_bootstrap=[],
        within_commodity_metrics=[],within_commodity_summary=[],commodity_risk_metrics=[],probe_stability=stability,
        representation_stats=representation_statistics(frame,reps),risk_target_distribution=target_distributions(frame,t95),
        frozen_train=dict(mean_risk=mean,T90=float(t90),T95=float(t95),commodity_mean_risk=mean_by_commodity),
        definitions=dict(target='|5d signed target|, raw return units',R2='1 - SSE/sum((risk - TRAIN pooled mean risk)^2), NOT centered on evaluation mean; may be negative',
            fitting='11 fixed probes including MeanRisk; one separate static CommodityTrainMeanRisk baseline. All coefficients and standardization TRAIN-only. No ridge or negative prediction clipping.',
            bootstrap='1000 draws seed42, forecast-origin clusters retaining all commodities. Overlapping origins still induce serial dependence; approximate IID-origin CI.',
            AUPRC='non-interpolated average precision; ties processed together; baseline=prevalence. No post-fit orientation flip.',
            attribution='Linear-probe coefficient magnitude is scale-dependent and not causal attribution, even after standardization.',
            interpretation='Linear decodability under these frozen probes only; failure does not imply absence of nonlinear risk information.',
            log_risk_epsilon=EPS_RISK,log_volatility_epsilon=EPS_VOL),primary_case=None,status='STATISTICS COMPLETE')
    for split in ('train','val','test'):
        mask=(frame.split==split).to_numpy();sub=frame.loc[mask].reset_index(drop=True)
        y=risk[mask];pred={key:value[mask] for key,value in predictions.items()}
        for name,value in pred.items():
            assert np.isfinite(value).all()
            result['probe_metrics'].append(dict(split=split,probe=name,**regression_metrics(value,y,mean)))
            for tail,threshold in [('Tail90',t90),('Tail95',t95)]:
                result['tail_probe_metrics'].append(dict(split=split,probe=name,tail=tail,**ranking_metrics(y>threshold,value)))
            if name in ('GlobalFused','TemporalOnly','SpatialGlobal','LongMicro'):
                # A shared origin vector MUST produce identical commodity risk scores.
                spread=pd.DataFrame(dict(origin=sub.sample_index,value=value)).groupby('origin').value.agg(lambda x:float(x.max()-x.min()))
                assert np.max(spread)<1e-9
        rows,summary=within_commodity(sub,pred,t90)
        result['within_commodity_metrics'].extend(dict(split=split,**row) for row in rows)
        result['within_commodity_summary'].extend(dict(split=split,**row) for row in summary)
        for ci,part in sub.groupby('commodity_index',sort=True):
            positions=sub.index.get_indexer(part.index);yr=y[positions]
            row=dict(split=split,commodity_index=int(ci),commodity=part.commodity.iloc[0],mean_future_risk=float(yr.mean()),
                std_future_risk=float(yr.std(ddof=0)),Tail90_rate=float((yr>t90).mean()),count=len(part))
            row['probes']={name:regression_metrics(value[positions],yr,mean) for name,value in pred.items()}
            result['commodity_risk_metrics'].append(row)
        if split!='train':
            result['probe_bootstrap'].extend(dict(split=split,**row) for row in bootstrap_regression(sub,pred,y,mean))
            for name in MAIN:
                result['probe_bootstrap'].append(dict(split=split,kind='Tail90_AUROC',probe=name,comparison=name,metric='AUROC',
                    estimate=ranking_metrics(y>t90,pred[name])['AUROC'],**ranking_bootstrap(sub,y>t90,pred[name])))
        print(f"[D0B risk probe] {split.upper()} regression/ranking/within-commodity evaluation complete",flush=True)
    longmicro=np.asarray(fits['LongMicro']['weights'])
    norms=[float(np.linalg.norm(longmicro[:64])),float(np.linalg.norm(longmicro[64:]))]
    result['LongMicro_coefficient_attribution']=dict(long_norm=norms[0],micro_norm=norms[1],micro_long_ratio=norms[1]/norms[0] if norms[0] else None,
        basis='Full TRAIN standardized features; no causal attribution')
    test_comm=[x for x in result['commodity_risk_metrics'] if x['split']=='test']
    result['commodity_rankings']=dict(highest_risk=sorted(test_comm,key=lambda x:x['mean_future_risk'],reverse=True)[:5],lowest_risk=sorted(test_comm,key=lambda x:x['mean_future_risk'])[:5])
    for filename,key in [('probe_metrics','probe_metrics'),('probe_bootstrap','probe_bootstrap'),('tail_probe_metrics','tail_probe_metrics'),
        ('commodity_risk_metrics','commodity_risk_metrics'),('within_commodity_metrics','within_commodity_metrics'),
        ('within_commodity_summary','within_commodity_summary'),('representation_stats','representation_stats'),
        ('probe_stability','probe_stability'),('risk_target_distribution','risk_target_distribution')]:
        pd.json_normalize(result[key]).to_csv(output/(filename+'.csv'),index=False)
    output_records=frame[['split','sample_index','forecast_origin','commodity_index','commodity','target']].copy()
    output_records['risk_target']=risk
    for name,value in predictions.items():
        output_records[name]=value
    output_records.to_csv(output/'probe_observation_predictions.csv',index=False)
    result['interpretation_status']='PENDING EVIDENCE REVIEW'
    return result


def metric_table(rows):
    return table(['Split','Probe','RiskMAE','RiskMSE','RiskRMSE','R² vs TRAIN mean','Pearson','Spearman','Negative pred fraction'],[
        [r[k] for k in ('split','probe','RiskMAE','RiskMSE','RiskRMSE','R2','Pearson','Spearman','negative_prediction_fraction')] for r in rows])


def write_report(r,output):
    cp=r.get('checkpoint',{})
    native=r.get('native_reference_check',{}).get('actual',{})
    lines=['# D0B-RiskRepresentationProbeDiagnostic','',r['status'],'',
        'D0B 全程冻结；一个全量原始 forward extraction，固定 TRAIN-only linear probes，raw |5d return| 为唯一拟合目标。没有改 signed return，没有集成 risk head。','',
        '## 1. Data and checkpoint sanity','',table(['Checkpoint','Best epoch','Params','Parameter max diff','Diagnostic SHA'],[[cp.get('checkpoint_path'),cp.get('best_epoch'),cp.get('parameter_count'),cp.get('model_parameter_max_diff'),r.get('diagnostic_git_sha')]]),'',
        f"Split origins: {r.get('split_origins')}; commodities: {r.get('commodity_count')}; native TEST baseline: {r.get('native_reference_check')}",'',
        table(['Native TEST 5d MAE','MSE','RMSE','Hit%','|RMSE²-MSE|'],[[native.get('MAE'),native.get('MSE'),native.get('RMSE'),100*native['Hit'] if 'Hit' in native else None,native.get('RMSE_squared_minus_MSE_abs')]]),'',
        f"Training seed metadata: {cp.get('seed')}; training SHA metadata: {cp.get('checkpoint_git_sha')}. null 表示 checkpoint 未提供，不推断训练元数据。",'',
        'Checkpoint file/state_dict 前后 SHA256、完整商品名称/node index、特征源码指纹及 tensor shapes 保存在 results.json。商品 target 与节点顺序已逐项核验。','',
        '## 2. Risk target','',
        table(['Split','Target','Mean','Std','Median','P75','P90','P95','P99','Max','Skewness','Split top5 mass share','TRAIN top5 threshold mass share'],[[x[k] for k in ('split','target','mean','std','median','P75','P90','P95','P99','max','skewness','split_top5_target_mass_share','top5_train_threshold_mass_share')] for x in r['risk_target_distribution']]),'',
        '正式 probe target 固定 |y_5d|，不 normalization。y²/log(|y|+1e-6) 仅作分布描述。上表 Top5 使用 TRAIN P95(|y|)，evaluation 覆盖率不必为5%；CSV/JSON 另存各 split 自身最大的 ceil(5% × count) 个观测之质量占比，仅作分布描述，不用于 fit。log-risk为有符号量，target mass share 不作正质量解释，另存 signed sum share 和 absolute log mass share。','',
        '## 3. Fixed probe definitions','',table(['Probe','Input','Dimension','Fitted coefficients','Constant dims','Rank','Condition','Kind'],[[x.get(k) for k in ('probe','input','dimension','coefficient_count','constant_dimensions','rank','condition_number','kind')] for x in r['probe_definitions']]),'',
        '每个线性 probe 使用 TRAIN mean/std，std<1e-12 的维度丢弃；标准化分母 std+1e-8，带 intercept，lambda=0，np.linalg.lstsq(rcond=None)。即使 rank/conditioning 不理想，也没有调 ridge 或看 evaluation 后改 cutoff。','',
        '## 4. Main risk regression table','',metric_table([x for x in r['probe_metrics'] if x['split']!='train']),'',r['definitions']['R2'],'',
        '## 5. Tail ranking','',
        table(['Split','Probe','Tail','AUROC','AUPRC','Prevalence','AP/baseline'],[[x[k] for k in ('split','probe','tail','AUROC','AUPRC','prevalence','AUPRC_over_prevalence')] for x in r['tail_probe_metrics'] if x['split']!='train']),'',r['definitions']['AUPRC'],'',
        '## 6. Key pairwise comparisons','',
        table(['Split','Comparison','Metric','Delta','95% CI'],[[x['split'],x['comparison'],x['metric'],x['estimate'],x['CI95']] for x in r['probe_bootstrap'] if x['kind']=='paired_delta']),'',
        'Delta = first probe − second probe；负数为前者误差更小。'+r['definitions']['bootstrap'],'',
        table(['Split','Probe','Tail90 AUC','95% CI'],[[x['split'],x['probe'],x['estimate'],x['CI95']] for x in r['probe_bootstrap'] if x['kind']=='Tail90_AUROC']),'',
        '## 7. Cross-sectional vs dynamic risk','',
        metric_table([x for x in r['probe_metrics'] if x['split']!='train' and x['probe'] in (STATIC,'Vol20Only','CommodityNode','CommodityNodePlusVol20')]),'',
        table(['Split','Probe','Within Spearman mean','Median','Std','Positive fraction','Valid correlations','Valid commodity AUCs','Mean AUC','Median AUC','AUC > .5 count'],[[x[k] for k in ('split','probe','Spearman_mean','Spearman_median','Spearman_std','positive_fraction','valid_correlations','valid_AUC_commodities','AUC_mean','AUC_median','AUC_above_chance_count')] for x in r['within_commodity_summary'] if x['split']!='train']),'',
        'Constant score/target 的 correlation 标 UNDEFINED，不强行当0；within summary 只使用有效商品，分母已报告。每商品 AUC 要求至少10正例和10负例，否则 INSUFFICIENT。','',
        '### TEST commodity risk difficulty','',
        table(['Commodity','Mean future |y|','Std','Tail90 rate','Vol20Only MAE','CommodityNode MAE','NodePlusVol20 MAE'],[[x['commodity'],x['mean_future_risk'],x['std_future_risk'],x['Tail90_rate'],x['probes']['Vol20Only']['RiskMAE'],x['probes']['CommodityNode']['RiskMAE'],x['probes']['CommodityNodePlusVol20']['RiskMAE']] for x in r['commodity_risk_metrics'] if x['split']=='test']),'',
        'Highest future-risk commodities: '+', '.join(x['commodity'] for x in r['commodity_rankings']['highest_risk']),
        'Lowest future-risk commodities: '+', '.join(x['commodity'] for x in r['commodity_rankings']['lowest_risk']),'',
        '## 8. Representation location','',r.get('interpretation_text','Pending evidence review.'),'',
        table(['Representation','Linear risk information','Evidence'],[[x.get('representation'),x.get('verdict'),x.get('reason')] for x in r.get('representation_verdicts',[])]),'',
        table(['Split','Representation','Norm mean','Norm std','Pairwise cosine mean','Pairwise cosine std','Cross-commodity variance','Within-commodity temporal variance'],[[x.get(k) for k in ('split','representation','norm_mean','norm_std','pairwise_cosine_mean','pairwise_cosine_std','cross_commodity_variance','within_commodity_temporal_variance')] for x in r['representation_stats']]),'',
        f"LongMicro coefficient attribution: {r['LongMicro_coefficient_attribution']}",'',r['definitions']['attribution'],'',
        '## 9. Stability','',table(['Probe','First weight norm','Second weight norm','Cosine','First intercept','Second intercept','First raw-log-vol slope','Second raw-log-vol slope'],[[x.get(k) for k in ('probe','first_weight_norm','second_weight_norm','weight_cosine','first_intercept','second_intercept','first_raw_log_vol_slope','second_raw_log_vol_slope')] for x in r['probe_stability']]),'',
        '两个 TRAIN chronological halves 的系数统一放在 full-TRAIN 标准化坐标中比较；只作描述，未应用到 VAL/TEST。Vol20Only 的原始 log-vol slope/intercept 另外保存在 results.json。','',
        r.get('numerical_audit_note','数值细节见秩、条件数和系数稳定性。'),'',
        '## 10. Required 18 answers','']
    lines.extend(f'{i}. {answer}' for i,answer in enumerate(required_answers(r),1))
    lines += ['',f"Primary classification: **{r['primary_case'] or 'PENDING'}**",'',r.get('next_direction','No risk head integration or further training.'),'',
        r['definitions']['interpretation'],'',
        'Cross-sectional commodity discrimination、same-commodity time-series discrimination、future-risk magnitude regression 必须分别解释。成功不作 causal attribution；失败不等于 representation 完全没有风险信息。STOP.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def required_answers(r):
    def m(name):
        return str([{k:x[k] for k in ('split','RiskMAE','RiskMSE','R2','Spearman')} for x in r['probe_metrics'] if x['probe']==name and x['split']!='train'])
    def pair(a,b):
        return str([x for x in r['probe_bootstrap'] if x['kind']=='paired_delta' and x['comparison']==f'{a} - {b}'])
    return [f"Baseline/checkpoint reproduction: {r.get('native_reference_check')}; parameter max diff={r.get('checkpoint',{}).get('model_parameter_max_diff')}",
        'Risk distributions: '+str([x for x in r['risk_target_distribution'] if x['target']=='absolute']),
        'MeanRisk: '+m('MeanRisk'),'Vol20Only: '+m('Vol20Only'),'GlobalFused: '+m('GlobalFused'),
        'TemporalOnly: '+m('TemporalOnly'),'SpatialGlobal: '+m('SpatialGlobal'),'LongMicro: '+m('LongMicro'),
        r.get('comparison_verdicts',{}).get('node_global','')+' CommodityNode vs GlobalFused: '+pair('CommodityNode','GlobalFused'),
        r.get('comparison_verdicts',{}).get('node_plus_global','')+' NodePlusGlobal increment: '+pair('CommodityNodePlusGlobal','CommodityNode'),
        r.get('comparison_verdicts',{}).get('node_plus_vol','')+' NodePlusVol20 increment: '+pair('CommodityNodePlusVol20','CommodityNode'),
        r.get('comparison_verdicts',{}).get('global_plus_vol','')+' GlobalPlusVol20 increment: '+pair('GlobalPlusVol20','GlobalFused'),
        r.get('comparison_verdicts',{}).get('simple','')+' SimpleRiskFeatures: '+m('SimpleRiskFeatures'),
        'Static commodity baseline: '+m(STATIC)+'; constant within-commodity scores have undefined temporal correlations. '+str(r.get('static_explained_fraction_of_simple_risk_MSE_gain',{})),
        r.get('within_answer','See within-commodity summary; pending interpretation.'),
        r.get('heterogeneity_answer','See static baseline and within-commodity results; pending interpretation.'),
        r.get('location_answer','Pending representation-location interpretation.'),f"Primary case: {r['primary_case'] or 'PENDING'}."]
