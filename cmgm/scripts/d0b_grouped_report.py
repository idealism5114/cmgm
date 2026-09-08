"""Render the three-way D0B objective comparison from measured diagnostics."""
import json

from cmgm import config
from cmgm.scripts.d0e_diagnostics import functional_values


def write_report(report,path):
    models=report['models'];comparison=report['performance_comparison'];assessment=report['assessment']
    lines=['# D0B-MidLongGroupedObjective','']
    def paragraph(text): lines.extend([text,''])
    def heading(text): paragraph('## '+text)
    def fmt(v):
        if isinstance(v,float):return f'{v:.9g}'
        if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
        return str(v)
    def table(headers,rows):
        lines.append('| '+' | '.join(headers)+' |')
        lines.append('| '+' | '.join('---' for _ in headers)+' |')
        lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows)
        lines.append('')
    def metric(m,s,h='5'):
        v=m['splits'][s]['native_metrics'][h]
        return [v['MAE'],v['MSE'],v['RMSE'],100*v['Hit']]
    def regime(m,s,key):return m['splits'][s]['normal_regime'][key]
    def functional(m,s,h=5):return functional_values(m,s,h)
    def between(key,source):
        values={k:source(m,key) for k,m in models.items()}
        return {'values':values,'Grouped_between_references':min(values['D0B'],values['5dOnly'])<=values['Grouped']<=max(values['D0B'],values['5dOnly'])}

    paragraph('唯一新训练目标为 **scale-matched grouped-objective diagnostic**：固定 (4/3)×(L5+L10+L20)，'
              '总损失加原 switch loss；validation / scheduler / early stopping / checkpoint selection 使用相同 grouped prediction objective。'
              'head 保留四 horizons；D0B architecture、固定 alpha=.5、routing、optimizer 与数据协议保持原样。')
    paragraph('D0B 和 5d-only 均重新加载已有 checkpoint，仅作对照。性能以收益率空间报告；RMSE 使用 pooled sqrt(MSE)，Hit 为不掩码 sign 一致率，'
              'Hit 以百分比显示。TRAIN 指 best checkpoint 的完整 TRAIN loader eval，包含尾 batch。')
    heading('Provenance / shared initialization / loss-scale sanity')
    table(['Field','Value'],[[k,report[k]] for k in ('checkpoint_paths','checkpoint_sha256','metadata','seed','fixed_TEST_batch_shape','initialization','integrity')])
    paragraph('初始化 scale ratio 使用同 seed、同固定 TRAIN batch、eval 模式；乘数始终4/3，不根据 ratio 调整。'
              '共享初始化检查涵盖所有参数与 E/H/prior/p/candidates/Z/h_long/h_micro/h_temporal/h_spatial/gate/prediction。')
    heading('Primary performance：5d')
    table(['Variant','Params','BestEpoch','Train time seconds','VAL 5d MAE/MSE/RMSE/Hit%','TEST 5d MAE/MSE/RMSE/Hit%'],
          [[k,m['params'],m['best_epoch'],m['train_time_seconds'],metric(m,'VAL'),metric(m,'TEST')] for k,m in models.items()])
    table(['Variant','Timing provenance'],[[k,m['train_time_note']] for k,m in models.items()])
    heading('完整 TRAIN / VAL / TEST 四 horizon performance')
    for s in ('TRAIN','VAL','TEST'):
        paragraph(s)
        table(['Horizon','D0B MAE/MSE/RMSE/Hit%','5dOnly MAE/MSE/RMSE/Hit%','Grouped MAE/MSE/RMSE/Hit%',
               'Grouped−D0B ΔMAE','relative change%','Grouped−5dOnly ΔMAE','relative change%'],
              [[h]+[metric(models[k],s,h) for k in ('D0B','5dOnly','Grouped')]
               +[v for k in ('D0B','5dOnly') for v in (row['grouped_vs'][k]['delta_MAE'],
                    100*row['grouped_vs'][k]['relative_change'] if row['grouped_vs'][k]['relative_change'] is not None else None)]
               for h,row in comparison[s].items()])
    paragraph('1d 不参与 prediction backward，退化是可能的 trade-off；10d/20d 保留监督，必须同时观察两者的泛化。'
              '所有参数仍使用原 weight decay，包括无直接1d prediction gradient 的 head 参数。')
    heading('Training history：loss terminology')
    history=models['Grouped']['history']
    table(['Epoch','raw L1 (diagnostic)','raw L5','raw L10','raw L20','group_raw','group_scaled',
           'switch_loss','total_loss','val_grouped','raw val L5','LR','switch beta','best/final'],
          [[r['epoch']]+[r['train'][k] for k in ('raw_L1','raw_L5','raw_L10','raw_L20','group_raw','group_scaled','switch_loss','total_loss')]
           +[r['val']['group_scaled'],r['val']['raw_L5'],r['lr'],r['switch_beta'],
             '/'.join(k for k,e in (('best',history.get('best_epoch')),('final',history.get('final_epoch'))) if e==r['epoch'])]
           for r in history.get('objective_history',[])])
    table(['Variant','Split','raw eval L1/L5/L10/L20','group raw','group scaled','sum multi (descriptive)'],
          [[k,s,m['splits'][s]['prediction_losses']['per_horizon']]
           +[m['splits'][s]['prediction_losses'][v] for v in ('group_raw','group_scaled','sum_multi')]
           for k,m in models.items() for s in ('TRAIN','VAL','TEST')])
    paragraph('History 使用原 mean-of-batch-means，TRAIN 优化过程包含 dropout 与参数更新；checkpoint TRAIN eval 与之分开。'
              'raw val L5 不能通过 grouped val loss 除以4/3得到，metadata 从 best epoch 的独立 raw L5 记录中读取。')
    heading('Regime behavior：三组完整对照')
    keys=('mean','entropy','prior_entropy','mean_max','margin','occupancy','posterior_prior_KL','posterior_prior_L1','temporal_L1')
    table(['Variant','Split']+list(keys),[[k,s]+[regime(m,s,key) for key in keys] for k,m in models.items() for s in ('TRAIN','VAL','TEST')])
    for k,m in models.items():
        paragraph(k+' transition / logits drift:\n```json\n'+json.dumps(m['transition'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('Entropy 用自然对数；滑窗内相同日期可重复计数；occupancy 仅描述。历史 checkpoint 无初始 L snapshot 时，'
              '明确区分 source-assumed zero-init drift。用户提供的 5d-only TEST entropy≈1.0689、temporal L1≈.0449 仅作核对，以上实际加载结果为准。')
    heading('Mechanism table：fixed dynamics + full split utilization')
    table(['Variant','Split','p entropy','p temporal L1','transition L drift','candidate cosine mean',
           'Z norm','delta Z norm','Z consecutive cosine','BaseRPE/QK','zero-micro 5d','zero-long 5d','RoutingFraction_5d'],
          [[k,s,regime(m,s,'entropy'),regime(m,s,'temporal_L1'),m['transition']['transition_logits_drift_L2'],
            sum(v['cosine'] for v in m['splits'][s]['candidate_specialization']['pairwise'].values())/3]
           +[m['fixed']['micro'][key] for key in ('mean_Z_norm','mean_delta_Z_norm','consecutive_cosine')]
           +[m['fixed']['base_rpe']['base_QK_ratio'],functional(m,s)[0],functional(m,s)[1],functional(m,s)[4]]
           for k,m in models.items() for s in ('VAL','TEST')])
    heading('Candidate specialization')
    for s in ('TRAIN','VAL','TEST'):
        paragraph(s)
        table(['Variant','candidate norms','p-weighted contributions','pairwise L1/cosine'],
              [[k]+[m['splits'][s]['candidate_specialization'][v] for v in ('candidate_L2_norm','weighted_contribution_L2_norm','pairwise')]
               for k,m in models.items()])
    heading('Z / Base RPE / BalancedReadout / representation norms')
    table(['Variant','Z/H raw ratio','Base RPE norm','mean |QK|','mean |base bias|','base/QK'],
          [[k,m['fixed']['micro']['Z_T_H_T_raw_ratio']]+[m['fixed']['base_rpe'][v] for v in ('norm','mean_abs_QK','mean_abs_base_bias','base_QK_ratio')]
           for k,m in models.items()])
    paragraph('Base/QK 仅统计 causal 下三角（含对角）；QK 是缩放后的内容 logits，未加入 base bias，按层等权平均。')
    table(['Variant','||W_long||','||W_micro||','W_micro/W_long','||h_long||','||h_micro||','post-balance micro/long'],
          [[k]+[m['fixed']['readout_weights'][v] for v in ('W_long_norm','W_micro_norm','W_micro_W_long')]
           +[m['fixed']['micro'][v] for v in ('h_long_norm','h_micro_norm','micro_long_norm_ratio')] for k,m in models.items()])
    paragraph('W_long/W_micro 指 readout projections W_H/W_Z 的 Frobenius norm；final state_readout 的两个输入块范数另存 results.json，避免混淆。')
    table(['Variant','||H||','||Z||','p entropy TEST','h_temporal norm','h_spatial norm','gate mean','gate std'],
          [[k,m['fixed']['representation_norms']['H'],m['fixed']['representation_norms']['Z'],regime(m,'TEST','entropy'),
            m['fixed']['representation_norms']['h_temporal'],m['fixed']['representation_norms']['h_spatial'],m['fixed']['gate']['mean'],m['fixed']['gate']['std']]
           for k,m in models.items()])
    heading('Per-horizon zero-components / native-vs-uniform / forced regimes')
    for s in ('VAL','TEST'):
        paragraph(s)
        table(['Variant','Horizon','zero-micro','zero-long','micro/long','uniform','RoutingFraction','state0','state1','state2'],
              [[k,h]+functional(m,s,h)+[m['splits'][s]['modes'][f'state{i}']['impact']['per_horizon'][str(h)]['mean'] for i in range(3)]
               for k,m in models.items() for h in config.MULTI_HORIZONS])
    paragraph('Uniform / forced controls 仅用于 checkpoint diagnostic，正常 p recursion 保留，Z 完整递推；'
              '没有在训练中引入 q。RoutingFraction 是响应比，不是可加性贡献。')
    heading('Raw per-horizon gradients：无4/3乘数')
    for h in map(str,config.MULTI_HORIZONS):
        paragraph(h+'d')
        table(['Module','D0B norm','5dOnly norm','Grouped norm'],
              [[module]+[models[k]['gradients']['norms'][h][module] for k in ('D0B','5dOnly','Grouped')]
               for module in models['Grouped']['gradients']['norms'][h]])
    heading('Conflict geometry：三种训练目标')
    table(['Module']+[f'{k} cos(5,{h})' for k in ('D0B','5dOnly','Grouped') for h in (1,10,20)],
          [[module]+[models[k]['gradients']['cosines'][module][f'5d-vs-{h}d'] for k in ('D0B','5dOnly','Grouped') for h in (1,10,20)]
           for module in models['Grouped']['gradients']['cosines']])
    paragraph('固定 TEST batch，eval raw single-horizon Huber、autograd.grad；无 optimizer step。'
              'Cosine 是局部且依赖当前 representation 的统计，不单独决定 Case A。')
    heading('Representation drift / prediction differences')
    table(['Comparison','Representation','mean abs diff','max abs diff'],
          [[k,key,v['mean'],v['max']] for k,values in report['representation_drift'].items() for key,v in values.items()])
    table(['Split','Grouped vs','Horizon','mean abs prediction diff'],
          [[s,k,h,v] for s,values in report['prediction_impact'].items() for k,hs in values.items() for h,v in hs.items()])
    paragraph('Checkpoint 间 drift 不要求为0；不同层尺度与表征坐标不相同，不能按数值大小直接归因 temporal/spatial/fusion 的收益。')
    heading('Causality / batch / within-market sanity')
    for k,m in models.items():
        paragraph(k+'\n```json\n'+json.dumps(m['sanity'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('Prefix 检查 E/H/p/Z 和逐时 readout；完整空间分支使用整个输入窗口。Within-market 完整图检查同步重排 asset embeddings。'
              '此处验证模型机制，不重新认证历史预处理；数据 split/normalization 沿用原实现。')
    heading('十七个问题与最终判定')
    init=report.get('initialization') or {}
    paragraph('1. 与 D0B 共享初始化一致：'+fmt(init.get('PASS','not recorded'))+'。')
    paragraph('2. 参数量差：'+fmt(models['Grouped']['params']-models['D0B']['params'])+'。')
    paragraph('3. 初始 group_scaled/sum_multi：'+fmt(init.get('loss_scale',{}).get('ratio','not recorded'))+'；固定4/3。')
    for number,s in ((4,'VAL'),(5,'TEST')):
        paragraph(f'{number}. {s} 5d 相对 D0B：'+fmt(comparison[s]['5']['grouped_vs']['D0B'])+'。')
    paragraph('6. 相对5d-only：'+fmt({s:comparison[s]['5']['grouped_vs']['5dOnly'] for s in ('VAL','TEST')})+'。')
    paragraph('7. 1d trade-off：'+fmt({s:comparison[s]['1']['grouped_vs']['D0B'] for s in ('VAL','TEST')})+'。')
    paragraph('8. 10d/20d 是否保持泛化：'+fmt({'healthy':assessment['auxiliary_10_20_healthy'],'definition':assessment['health_definition'],
              'relative_changes':{s:{h:comparison[s][h]['grouped_vs']['D0B']['relative_change'] for h in ('10','20')} for s in ('VAL','TEST')}})+'。有效正则化仍需结合对5d-only的优势判断。')
    paragraph('9. Posterior 是否介于两 reference 之间：'+fmt({s:{key:between(key,lambda m,k:regime(m,s,k)) for key in ('entropy','mean_max','margin','temporal_L1')}
              for s in ('VAL','TEST')})+'。')
    paragraph('10. Z dynamics：'+fmt({key:between(key,lambda m,k:m['fixed']['micro'][k]) for key in ('mean_Z_norm','mean_delta_Z_norm','consecutive_cosine')})+'。居中仅是描述，不自动等于更健康。')
    paragraph('11. Base RPE：'+fmt({key:between(key,lambda m,k:m['fixed']['base_rpe'][k]) for key in ('norm','base_QK_ratio')})+'。不能仅凭更小或更大判定“过激”。')
    paragraph('12. 5d micro utilization（impact、micro/long）：'+fmt({s:{k:[functional(m,s)[0],functional(m,s)[2]] for k,m in models.items()} for s in ('VAL','TEST')})+'。')
    paragraph('13. RoutingFraction_5d：'+fmt({s:{k:functional(m,s)[4] for k,m in models.items()} for s in ('VAL','TEST')})+'。')
    geometry=models['Grouped']['gradients']['cosines']
    mid_conflict={module:any(v[f'5d-vs-{h}d']<0 for h in (10,20)) for module,v in geometry.items()}
    one_conflict={module:v['5d-vs-1d']<0 for module,v in geometry.items()}
    paragraph('14. 5d 与10d/20d仍有局部冲突：'+fmt(mid_conflict)+'。')
    paragraph('15. 1d与5d仍冲突：'+fmt(one_conflict)+'。')
    if any(mid_conflict.values()):
        paragraph('Gradient geometry is representation-dependent and cannot directly justify static horizon grouping；不要仅据 cosine 推导结构。')
    elif all(one_conflict.values()) and all(comparison[s]['5']['grouped_vs']['D0B']['delta_MAE']<0 for s in ('VAL','TEST')):
        paragraph('5d-vs-10/20兼容、5d-vs-1d冲突且VAL/TEST改善，支持1d与中长期表征不一致的机制解释；仍需三组 performance 支持完整Case A。')
    paragraph('16. 最终判定：'+fmt(assessment)+'。')
    paragraph('17. Grouped objective 是否有资格成为正式优化方向：'+fmt(assessment['candidate_objective_justified'])+'。只有最强Case A可支持候选；C/D/E保留D0B，B不足以宣称主要瓶颈。')
    paragraph('0.1% 为预先写明的效应大小和10/20健康度报告约定，不参与调参或选模；1e-6 relative MAE 仅用于数值持平检查。'
              '若VAL/TEST均改善但并不明显优于5d-only或10/20退化，题设A–E未完全覆盖：报告未分类、需复核，绝不强行给Case A。')
    paragraph(assessment['case'] or '未分类：题设 Case 条件未覆盖；不提升候选模型')
    paragraph('STOP。不启动下一组合、loss-weight search 或新模型。')
    path.write_text('\n'.join(lines),encoding='utf-8')
