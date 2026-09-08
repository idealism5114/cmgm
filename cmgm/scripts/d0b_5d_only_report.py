"""Render measured 5d-only objective diagnostics; no model or training code."""
import json

from cmgm import config
from cmgm.scripts.d0e_diagnostics import functional_values


def write_report(report, path):
    models = report['models']
    comparisons = report['performance_comparison']
    assessment = report['case_assessment']
    lines = ['# D0B-5dOnlyObjective', '']
    def paragraph(text):
        lines.extend([text,''])
    def heading(text):
        paragraph('## '+text)
    def fmt(v):
        if isinstance(v,float): return f'{v:.9g}'
        if isinstance(v,(dict,list)): return json.dumps(v,ensure_ascii=False)
        return str(v)
    def table(headers,rows):
        lines.append('| '+' | '.join(headers)+' |')
        lines.append('| '+' | '.join('---' for _ in headers)+' |')
        lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows)
        lines.append('')
    def metric(m,s,h='5'):
        v=m['splits'][s]['native_metrics'][h]
        return [v['MAE'],v['RMSE'],100*v['Hit']]
    def functional(m,s,h=5):
        return functional_values(m,s,h)
    def pairs(m,s):
        return {k:v['cosine'] for k,v in m['splits'][s]['candidate_specialization']['pairwise'].items()}
    def gradient_ratio(name):
        a,b=[models[k]['gradients']['norms']['5'][name] for k in ('D0B','5dOnly')]
        return [a,b,b/a if a>0 else None]

    paragraph('唯一干预是 **scale-matched 5d-only diagnostic objective**：训练 prediction loss 为固定 4×L5；'
              '总损失为 4×L5 + 原 switch loss；验证、scheduler、early stopping 和 best checkpoint selection 使用 4×L5。'
              '四 horizon 输出和 D0B 模型结构完全保留，alpha 固定 .5。乘数不依据初始 ratio 调整。')
    paragraph('全部指标为原收益率空间。RMSE 沿用 mean(asset-wise RMSE)，Hit 以下以百分比表示；'
              'TRAIN 包括完整尾 batch。训练/选模 loss 沿用 mean of batch means；性能指标按完整样本计算。'
              'VAL/TEST 和全部 horizons 均保留，不自动替换 D0B。')
    heading('来源与初始化')
    table(['Field','Value'],[[k,v] for k,v in report.items() if k in
                            ('checkpoint_paths','checkpoint_sha256','metadata','seed','fixed_TEST_batch_shape','initialization','integrity')])
    paragraph('D0B reference 来自本次实际加载的 checkpoint。历史 TEST 5d MAE≈.0219948、RMSE≈.0284856、Hit≈49.05% 仅供核对。'
              '初始化 scale ratio 来自同 seed、同固定 TRAIN batch、eval 模式，不代表全训练期的梯度范数严格匹配。')
    heading('Performance：primary 5d')
    table(['Variant','Params','BestEpoch','VAL MAE/RMSE/Hit%','TEST MAE/RMSE/Hit%'],
          [[k,m['params'],m['best_epoch'],metric(m,'VAL'),metric(m,'TEST')] for k,m in models.items()])
    heading('各 horizon 的完整 TRAIN / VAL / TEST 指标与 relative delta')
    for s in ('TRAIN','VAL','TEST'):
        paragraph(s)
        table(['Horizon','D0B MAE/RMSE/Hit%','5dOnly MAE/RMSE/Hit%','ΔMAE (new−D0B)','RelativeChange%'],
              [[h,metric(models['D0B'],s,h),metric(models['5dOnly'],s,h),v['delta_MAE'],
                100*v['relative_change'] if v['relative_change'] is not None else None]
               for h,v in comparisons[s].items()])
    paragraph('1/10/20d 不受 prediction objective 直接监督，其性能下降用于衡量 trade-off。'
              'Adam 原 weight decay 仍作用于整个参数集合。')
    heading('Training history / objective terminology')
    history=models['5dOnly']['history']
    table(['Epoch','raw train L5','scaled train prediction','switch loss','total train loss',
           'raw val L5','scaled val 5d (selection)','aux val multi (descriptive)','LR','switch beta','best/final'],
          [[r['epoch'],r['train']['raw_L5'],r['train']['scaled_prediction_loss'],r['train']['switch_loss'],r['train']['total_loss'],
            r['val']['raw_L5'],r['val']['scaled_prediction_loss'],r['val']['aux_multi_horizon_loss'],r['lr'],r['switch_beta'],
            '/'.join(name for name,e in (('best',history.get('best_epoch')),('final',history.get('final_epoch'))) if r['epoch']==e)]
           for r in history.get('objective_history',[])])
    table(['Variant','Split','best checkpoint raw L5','4×L5','sum multi'],
          [[k,s]+[m['splits'][s]['prediction_losses'][v] for v in ('raw_L5','scaled_5d_loss','sum_multi')]
           for k,m in models.items() for s in ('TRAIN','VAL','TEST')])
    paragraph('训练 history 是优化过程的 batch loss；上表 TRAIN loss 是 best checkpoint 的 eval-mode 完整 TRAIN 集结果。'
              '两者不能直接混称为同一个 training loss。')
    heading('Regime behavior')
    keys=('mean','entropy','prior_entropy','mean_max','margin','occupancy','posterior_prior_KL','posterior_prior_L1','temporal_L1')
    table(['Variant','Split']+list(keys),[[k,s]+[m['splits'][s]['normal_regime'][v] for v in keys]
                                        for k,m in models.items() for s in ('TRAIN','VAL','TEST')])
    for k,m in models.items():
        paragraph(k+' transition matrix / logits drift:\n```json\n'+json.dumps(m['transition'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('熵用自然对数。统计涵盖滑窗全部相对时刻，重复日期重复计数。Occupancy 只作描述；'
              'posterior 更尖锐不自动等于预测更好。没有初始 transition snapshot 的历史 checkpoint 明确标注 source zero-init assumption。')
    heading('Mechanism comparison：5d')
    table(['Variant','Split','p entropy','p temporal L1','candidate pairwise cosine',
           'zero-micro 5d','zero-long 5d','micro/long impact','uniform 5d','RoutingFraction_5d'],
          [[k,s,m['splits'][s]['normal_regime']['entropy'],m['splits'][s]['normal_regime']['temporal_L1'],pairs(m,s)]
           +functional(m,s) for k,m in models.items() for s in ('VAL','TEST')])
    paragraph('RoutingFraction_5d 使用当前重新计算的 D0B 5d reference，不能混用跨 horizon 平均 RoutingFraction。'
              'Uniform / forced-state 仅用于 checkpoint 干预，保留正常 p recursion，完整重算 Z 递推。'
              'Impact ratio 是非线性干预响应比，不是可加性贡献。')
    heading('各 horizon 的 zero-component / routing / forced-state impact')
    for s in ('VAL','TEST'):
        paragraph(s)
        table(['Variant','Horizon','zero-micro','zero-long','micro/long','uniform','RoutingFraction','state0','state1','state2'],
              [[k,h]+functional(m,s,h)+[m['splits'][s]['modes'][f'state{i}']['impact']['per_horizon'][str(h)]['mean'] for i in range(3)]
               for k,m in models.items() for h in config.MULTI_HORIZONS])
    heading('Candidate specialization / micro state / balanced readout')
    for k,m in models.items():
        paragraph(k+' fixed TEST micro/readout:\n```json\n'+json.dumps(m['fixed']['micro'],indent=2)+'\n```')
        for s in ('TRAIN','VAL','TEST'):
            paragraph(k+' '+s+' candidates:\n```json\n'+json.dumps(m['splits'][s]['candidate_specialization'],indent=2,ensure_ascii=False)+'\n```')
    heading('5d prediction-only gradients：未乘4')
    table(['Module','D0B raw L5 grad norm','5dOnly raw L5 grad norm','ratio'],
          [[name]+gradient_ratio(name) for name in models['D0B']['gradients']['norms']['5']])
    paragraph('梯度由固定 TEST batch、eval、单 horizon Huber 计算；不包含 switch loss，不乘4，不执行 optimizer step。'
              'Market Encoder、LongMemory layers、Base RPE、evidence、L、G0/G1/G2、long/micro/state readout 分别列出。')
    heading('Horizon gradient cosine comparison')
    table(['Variant','Module','cos(5d,1d)','cos(5d,10d)','cos(5d,20d)'],
          [[k,name]+list(values.values()) for k,m in models.items() for name,values in m['gradients']['cosines'].items()])
    table(['Module','Δcos(5d,1d)','Δcos(5d,10d)','Δcos(5d,20d)'],
          [[name]+[models['5dOnly']['gradients']['cosines'][name][f'5d-vs-{h}d']-
                   models['D0B']['gradients']['cosines'][name][f'5d-vs-{h}d'] for h in (1,10,20)]
           for name in models['D0B']['gradients']['cosines']])
    paragraph('负 cosine 表示该固定 batch 的局部梯度方向冲突；Δcos 更负表示更强的局部冲突，不能单独证明 generalization bottleneck。')
    heading('Prediction specialization / representation drift / spatial and fusion')
    table(['Split','Horizon','mean |prediction_5only − prediction_D0B|'],
          [[s,h,v] for s,hs in report['prediction_specialization'].items() for h,v in hs.items()])
    table(['Representation (fixed TEST)','mean abs diff','max abs diff'],
          [[k,v['mean'],v['max']] for k,v in report['representation_drift'].items()])
    table(['Variant','Gate stats'],[[k,m['fixed']['gate']] for k,m in models.items()])
    paragraph('两个 independently trained checkpoints 的 representation drift 只作描述，不要求接近0；'
              '不同层的维度、尺度及内部表征坐标不同，不能按绝对 drift 大小断言收益主要来自 temporal/spatial/fusion。')
    heading('Causality / batch / within-market sanity')
    for k,m in models.items():
        paragraph(k+'\n```json\n'+json.dumps(m['sanity'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('未来扰动检查 E/H/p/Z 和各时刻 readout 的 prefix；完整空间分支使用完整输入窗口。'
              '市场内置换检查 temporal invariant；完整图模型同步重排 asset embeddings 保持节点身份。'
              '这些检验不重新认证历史数据预处理，本轮沿用原 data split/normalization。')
    heading('十四个问题')
    init=report.get('initialization') or {}
    paragraph('1. 初始化与 D0B 一致：'+fmt(init.get('PASS','unavailable; see training log'))+'。共享参数及全部 forward traces 见初始化表。')
    paragraph('2. 参数量差值：'+fmt(models['5dOnly']['params']-models['D0B']['params'])+'。')
    paragraph('3. 初始 (4L5)/sum_multi：'+fmt(init.get('loss_scale',{}).get('ratio','unavailable'))+'；multiplier 始终为4。')
    for number,s in ((4,'VAL'),(5,'TEST')):
        paragraph(f'{number}. {s} 5d：'+fmt(comparisons[s]['5'])+'。')
    paragraph('6. 改善幅度及解释：'+fmt(assessment)+'。0.1% 仅用于报告效应大小，不参与训练或调参。')
    paragraph('7. 1/10/20d trade-off（relative MAE change）：'+fmt({s:{h:comparisons[s][h]['relative_change'] for h in ('1','10','20')} for s in ('VAL','TEST')})+'。')
    paragraph('8. Posterior 变化（new−D0B）：'+fmt({s:{key:models['5dOnly']['splits'][s]['normal_regime'][key]-models['D0B']['splits'][s]['normal_regime'][key]
              for key in ('entropy','mean_max','margin','temporal_L1')} for s in ('VAL','TEST')})+'。是否明显应结合上述绝对基线，不仅看低 entropy。')
    paragraph('9. Candidate specialization：'+fmt({k:{'pairwise_cosine':pairs(m,'TEST'),
              'candidate_norm':m['splits']['TEST']['candidate_specialization']['candidate_L2_norm'],
              'pairwise_L1':{p:v['L1'] for p,v in m['splits']['TEST']['candidate_specialization']['pairwise'].items()}}
              for k,m in models.items()})+'。非零 L1 和不相同的 candidate 输出支持分化；语义 specialization 不由这些统计单独证明。')
    paragraph('10. 5d micro-state 利用（zero-micro impact / micro-long ratio）：'+fmt({s:{k:[functional(m,s)[0],functional(m,s)[2]] for k,m in models.items()} for s in ('VAL','TEST')})+'。')
    paragraph('11. 5d RoutingFraction：'+fmt({s:{k:functional(m,s)[4] for k,m in models.items()} for s in ('VAL','TEST')})+'。')
    paragraph('12. Gradient conflict 是否仍存在：'+fmt({k:{module:any(v<0 for v in cos.values()) for module,cos in m['gradients']['cosines'].items()} for k,m in models.items()})+'；强弱变化见 Δcos 表。')
    paragraph('13. '+assessment['case']+'：'+assessment['reason']+' Case C 是无稳健泛化证据；是否训练5d变好需另看完整 TRAIN 的 raw L5 对照，不自动假定。')
    all_better=all(comparisons[s][h]['delta_MAE']<0 for s in ('VAL','TEST') for h in ('1','5','10','20'))
    if all_better:
        paragraph('所有 horizons 在 VAL/TEST MAE 均改善：original multi-horizon gradient composition may be globally suboptimal；不据此推导新权重。')
    elif all(comparisons[s]['5']['delta_MAE']<0 for s in ('VAL','TEST')):
        if any(comparisons[s][h]['delta_MAE']>0 for s in ('VAL','TEST') for h in ('1','10','20')):
            paragraph('Specialization toward the primary horizon improves 5d forecasting at the expense of auxiliary horizons.')
    paragraph('14. 是否有资格进一步测试移除1d、保留5/10/20：'+fmt(assessment['next_5_10_20_eligible'])+'。仅 Case A 达到触发条件；本轮不实现、不运行。')
    paragraph(assessment['case'])
    paragraph('STOP。不自动继续 loss weighting、horizon-specific readout 或其他实验。')
    path.write_text('\n'.join(lines),encoding='utf-8')
