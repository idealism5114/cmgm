"""Render only measured TargetScaleHuber results; never launch another experiment."""
import json
from cmgm.training.target_scale_huber import HORIZONS
from cmgm.scripts.d0b_huber_horizon_scale_diagnostic import markdown_table
from cmgm.scripts.d0e_diagnostics import functional_values


def write_report(r,path):
    models=r['models'];meta=r['metadata'];initial=meta['initialization'];comparison=r['comparison']
    text=['# D0B-TargetScaleHuber\n\n']
    def para(s):text.append(str(s)+'\n\n')
    def heading(s):para('## '+s)
    def table(h,rows):text.append(markdown_table(h,rows))
    def metric(m,s,h='5'):
        a=m['splits'][s]['native_metrics'][h]
        return [a['MAE'],a['MSE'],a['RMSE'],100*a['Hit']]
    def jsonblock(v):para('```json\n'+json.dumps(v,ensure_ascii=False,indent=2)+'\n```')
    heading('来源、冻结 scale 与初始化')
    para('唯一变化是 prediction loss：Σ_h (s5/sh) Huber(e_h; .02 sh/s5)。5d loss 保持原样；单 residual 的 absolute gradient cap 都是 .02。mean loss 的 prediction gradient 另除以 batch×commodity 数。')
    jsonblock({k:r[k] for k in ('checkpoint_paths','checkpoint_sha256','fixed_TEST_shape','integrity')})
    table(['Variant','Checkpoint seed','Checkpoint Git SHA','Params','BestEpoch','TrainTime seconds'],
          [[name,m['metadata'].get('seed','unknown'),m['metadata'].get('git_sha','unknown'),m['params'],m['best_epoch'],m['train_time_seconds']] for name,m in models.items()])
    para('历史 D0B 未记录训练时间时标为 None，不从含诊断的总耗时推断。TRAIN target std 使用全 1396 窗口×24 commodities、float64、ddof=0；不从 VAL/TEST、residual 或 checkpoint 误差估计 scale。')
    jsonblock(meta['scale_sanity'])
    table(['Horizon','TRAIN target std','delta_h','w_h','w_h*delta_h'],[[h,meta[f'target_scale_{h}'],meta[f'delta_{h}'],meta[f'weight_{h}'],meta[f'delta_{h}']*meta[f'weight_{h}']] for h in HORIZONS])
    table(['D0B params','New params','difference','parameter max diff','mismatch count','PASS'],[[initial[k] for k in ('D0B_params','TargetScaleHuber_params','difference','max_abs_diff','mismatch_count','PASS')]])
    table(['Trace','mean diff','max diff'],[[k,*v.values()] for k,v in initial['forward_differences'].items()])
    table(['Horizon','D0B fixed-.02 raw Huber','new delta_h raw Huber','weighted new Huber'],[[h,initial['D0B_raw_huber'][str(h)],initial['raw_huber_delta_h'][str(h)],initial['weighted_huber'][str(h)]] for h in HORIZONS])
    para(f"D0B sum={initial['D0B_sum']:.12g}; new weighted sum={initial['weighted_sum']:.12g}; 5d loss abs diff={initial['five_day_loss_abs_diff']:.12g}.")
    para('训练前同一 initialized D0B batch 的 counterfactual coverage（只换 loss threshold，没有换预测）：')
    table(['Horizon','native Q','counterfactual Q'],[[h,initial['native_initial_batch']['horizons'][str(h)]['quadratic_fraction'],initial['counterfactual_initial_batch']['horizons'][str(h)]['quadratic_fraction']] for h in HORIZONS])
    if 'pre_training_D0B_counterfactual' in meta:
        para('正式训练前，在已有 trained D0B 的完整 TRAIN residual 上，仅替换 threshold 的 counterfactual（不用于拟合 scale）：')
        jsonblock(meta['pre_training_D0B_counterfactual'])
    heading('Performance：统一 MAE / MSE / RMSE / Hit%')
    para(r['metric_standard']+'. MSE 直接计算，RMSE=sqrt(MSE)，Hit 不掩码；不使用 legacy RMSE/Hit 作主比较。')
    table(['Variant','Params','BestEpoch','TrainTime','VAL5 MAE','VAL5 MSE','VAL5 RMSE','VAL5 Hit%','TEST5 MAE','TEST5 MSE','TEST5 RMSE','TEST5 Hit%'],
          [[name,m['params'],m['best_epoch'],m['train_time_seconds'],*metric(m,'VAL'),*metric(m,'TEST')] for name,m in models.items()])
    for s in ('TRAIN','VAL','TEST'):
        para(s)
        table(['Horizon','D0B MAE/MSE/RMSE/Hit%','New MAE/MSE/RMSE/Hit%','ΔMAE','relative ΔMAE','ΔMSE','relative ΔMSE'],
              [[h,metric(models['D0B'],s,h),metric(models['TargetScaleHuber'],s,h),*[comparison[s][h][k] for k in ('delta_MAE','relative_MAE','delta_MSE','relative_MSE')]] for h in map(str,HORIZONS)])
    heading('Calibration 与 gradient-cap verification')
    para('D0B counterfactual 列在完全相同的正式 D0B residual 上仅应用新的 delta_h，区分公式作用与训练后预测变化。Coverage range 只是描述性 max−min，不以其单独判断 forecasting improvement。')
    table(['Split','Horizon','D0B Q fixed .02','same D0B residual Q new delta','new checkpoint Q new delta','new linear fraction'],
          [[s,h,models['D0B']['splits'][s]['calibration']['horizons'][h]['quadratic_fraction'],models['D0B']['splits'][s]['counterfactual_calibration']['horizons'][h]['quadratic_fraction'],row['quadratic_fraction'],row['linear_fraction']]
           for s,sp in models['TargetScaleHuber']['splits'].items() for h,row in sp['calibration']['horizons'].items()])
    table(['Variant','Split','Q range'],[[name,s,sp['calibration']['quadratic_fraction_range']] for name,m in models.items() for s,sp in m['splits'].items()])
    table(['Variant','Split','Horizon','mean |g|','median |g|','P90','P95','RMS','max |g|','saturation fraction','saturated mean |g|','cap','cap error'],
          [[name,s,h,*[row['weighted_gradient'][k] for k in ('mean_abs','median_abs','P90_abs','P95_abs','RMS','max_abs','saturation_fraction','saturated_mean_abs','cap','cap_error')]] for name,m in models.items() for s,sp in m['splits'].items() for h,row in sp['calibration']['horizons'].items()])
    heading('Network parameter gradients：固定 TEST batch')
    para('D0B 使用 raw Huber；新模型使用该 horizon 的 w_h Huber(delta_h)。均 eval、autograd.grad，无 optimizer step、不加 switch KL。Combined groups 与单模块有重叠，不可直接相加。ratio 不要求等于 1；单 batch 的 Jacobian/方向差异仍会影响范数，不能据此证明训练全过程的优化改善。')
    table(['Module','D0B 1/5','D0B 10/5','D0B 20/5','New 1/5','New 10/5','New 20/5'],
          [[k,*[models[name]['gradients']['ratios_to_5d'][str(h)][k] for name in models for h in (1,10,20)]] for k in models['D0B']['gradients']['norms']['5']])
    table(['Variant','Module','1d norm','5d norm','10d norm','20d norm'],[[name,k,*[m['gradients']['norms'][str(h)][k] for h in HORIZONS]] for name,m in models.items() for k in m['gradients']['norms']['5']])
    heading('Regime / candidate / micro-state / BalancedReadout / Base RPE')
    table(['Variant','Split','p entropy','temporal p L1','KL(p||prior)','candidate pairwise cosine','Z norm','delta Z','zero-micro5','zero-long5','RoutingFraction5','BaseRPE/QK'],
          [[name,s,sp['normal_regime']['entropy'],sp['normal_regime']['temporal_L1'],sp['normal_regime']['posterior_prior_KL'],sp['candidate_specialization']['pairwise'],m['fixed']['micro']['mean_Z_norm'],m['fixed']['micro']['mean_delta_Z_norm'],functional_values(m,s,5)[0],functional_values(m,s,5)[1],functional_values(m,s,5)[4],m['fixed']['base_rpe']['base_QK_ratio']]
           for name,m in models.items() for s,sp in m['splits'].items() if s!='TRAIN'])
    for name,m in models.items():
        para(name+' transition / fixed Z、readout、Base RPE：')
        jsonblock({'transition':m['transition'],'micro':m['fixed']['micro'],'readout_weights':m['fixed']['readout_weights'],'base_rpe':m['fixed']['base_rpe']})
        for s,sp in m['splits'].items():
            para(name+' '+s+' full regime and candidate diagnostics：')
            jsonblock({'regime':sp['normal_regime'],'candidate_specialization':sp['candidate_specialization']})
    para('regime over-specialization warning：'+json.dumps(r['regime_over_specialization'],ensure_ascii=False))
    heading('每 horizon zero-component / uniform routing utilization')
    para('Uniform 仅用于 checkpoint 反事实诊断，保持自身 p recursion，完整重算 Z；没有进入训练。RoutingFraction=I_uniform/(I_zeroMicro+1e-8)。')
    table(['Variant','Split','Horizon','zero-micro','zero-long','micro/long','uniform impact','RoutingFraction'],[[name,s,h,*functional_values(m,s,h)] for name,m in models.items() for s in ('VAL','TEST') for h in HORIZONS])
    heading('Commodity 5d error decomposition')
    for name,m in models.items():
        for s,sp in m['splits'].items():
            c=sp['commodity_error'];para(name+' '+s)
            table(['Commodity','MAE','MSE','RMSE','Hit%','residual std','median |e|','clipped fraction .02'],[[v['name'],v['MAE'],v['MSE'],v['RMSE'],100*v['Hit'],v['residual_std'],v['median_abs_residual'],v['linear_fraction']] for v in c['commodity_5d']])
            table(['5d commodity MSE mean','std','CV'],[[c['commodity_heterogeneity']['MSE'][k] for k in ('mean','std','CV')]])
            table(['Rank group','Commodity','MSE'],[[label,v['name'],v['MSE']] for label,key in [('highest','highest_MSE_commodities'),('lowest','lowest_MSE_commodities')] for v in c[key]])
    heading('Training history / checkpoint selection')
    para('Train 和 validation 共用同一个新 prediction helper。原 D0B validation 不加 switch KL，本变体保持此 policy；scheduler、early stopping 和 best checkpoint 都选择 scale-aware validation prediction loss。日志为原 mean-of-batch-means，performance 使用完整总体，不混用。')
    history=models['TargetScaleHuber']['history']
    table(['Epoch',*[f'raw H{h}' for h in HORIZONS],*[f'weighted H{h}' for h in HORIZONS],'prediction','switch','total','val prediction','LR','switch beta'],
          [[row['epoch'],*[row['train'][f'raw_huber_{h}'] for h in HORIZONS],*[row['train'][f'weighted_huber_{h}'] for h in HORIZONS],*[row['train'][k] for k in ('prediction_loss','switch_loss','total_loss')],row['val_prediction_loss'],row['lr'],row['switch_beta']] for row in history['objective_history']])
    heading('Causality / batch / within-market sanity')
    jsonblock({name:m['sanity'] for name,m in models.items()})
    heading('20 个问题与最终判定')
    answers={
        '1. TRAIN std 与 diagnostic 一致':meta['scale_sanity']['PASS'],
        '2. 仅 TRAIN target source':meta['target_scale_source'],
        '3. delta_h':{h:meta[f'delta_{h}'] for h in HORIZONS},
        '4. w_h':{h:meta[f'weight_{h}'] for h in HORIZONS},
        '5. w_h delta_h':meta['caps'],
        '6. 5d loss numerical difference':initial['five_day_loss_abs_diff'],
        '7. parameter difference':initial['difference'],
        '8. initial forward max diff':max(v['max'] for v in initial['forward_differences'].values()),
        '9. working regions more aligned across all splits':r['working_regions_more_aligned'],
        '10. maximum analytic gradient':max(row['weighted_gradient']['max_abs'] for sp in models['TargetScaleHuber']['splits'].values() for row in sp['calibration']['horizons'].values()),
        '11. 20d/5d norm ratios':{name:{k:m['gradients']['ratios_to_5d']['20'][k] for k in ('regime evidence','Market Encoder','LongMemory Transformer','spatial branch','prediction head')} for name,m in models.items()},
        '12. VAL5 error deltas':{k:comparison['VAL']['5'][k] for k in ('delta_MAE','delta_MSE','relative_MAE','relative_MSE')},
        '13. TEST5 error deltas':{k:comparison['TEST']['5'][k] for k in ('delta_MAE','delta_MSE','relative_MAE','relative_MSE')},
        '14. RMSE/MSE consistency':r['integrity']['max_RMSE_squared_minus_MSE'],
        '15. Hit percentage-point change':{s:100*(comparison[s]['5']['TargetScaleHuber']['Hit']-comparison[s]['5']['D0B']['Hit']) for s in ('VAL','TEST')},
        '16. regime concentration warning':r['regime_over_specialization'],
        '17. candidate specialization': 'See full norms, pairwise L1/cosine and weighted contribution above; occupancy alone is not collapse evidence.',
        '18. micro dynamics':{name:m['fixed']['micro'] for name,m in models.items()},
        '19. commodity MSE improvement':{s:{'count_improved':sum(a['MSE']<b['MSE'] for a,b in zip(models['TargetScaleHuber']['splits'][s]['commodity_error']['commodity_5d'],models['D0B']['splits'][s]['commodity_error']['commodity_5d'])),'total':24} for s in ('VAL','TEST')},
        '20. Case':r['assessment'],
    }
    for key,value in answers.items():para(key+'：'+json.dumps(value,ensure_ascii=False))
    para('Coverage 更一致本身不等于 forecasting improvement。只有 VAL/TEST 5d 改善、MSE 无实质恶化且机制健康，才值得把本变体作为候选。单 seed 的方向分类不代表统计显著性，不根据 Hit 单独接受/拒绝。')
    para('最终：'+str(r['assessment']['case'])+'。'+r['assessment']['reason'])
    para('STOP。不自动修改 delta、weight、loss、readout、routing 或架构；不自动替换正式 D0B。')
    path.write_text(''.join(text),encoding='utf-8')
