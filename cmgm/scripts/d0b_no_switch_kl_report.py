"""Report observed NoSwitchKL effects; never changes training or checkpoints."""
import json

from cmgm import config
from cmgm.scripts.d0e_diagnostics import functional_values


def write_report(report,path):
    models=report['models'];comparison=report['comparison'];assessment=report['assessment']
    lines=['# D0B-NoSwitchKL','']
    def paragraph(t):lines.extend([t,''])
    def heading(t):paragraph('## '+t)
    def fmt(v):
        if isinstance(v,float):return f'{v:.9g}'
        if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
        return str(v)
    def table(headers,rows):
        lines.append('| '+' | '.join(headers)+' |')
        lines.append('| '+' | '.join('---' for _ in headers)+' |')
        lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows);lines.append('')
    def metric(m,s,h='5'):
        v=m['splits'][s]['native_metrics'][h];return [v['MAE'],v['RMSE'],100*v['Hit']]
    def functional(m,s,h=None):return functional_values(m,s,h)
    def regime(m,s,k):return m['splits'][s]['normal_regime'][k]
    def change(key):return {s:regime(models['NoSwitchKL'],s,key)-regime(models['D0B'],s,key) for s in ('TRAIN','VAL','TEST')}

    paragraph('唯一消融是 **有效 switch KL 权重恒为0**。Prediction objective 仍为原始 L1+L5+L10+L20，'
              '每项 Huber delta=.02；validation、scheduler、early stopping 和 checkpoint selection 仍使用该四项总和。'
              '原 prior/posterior recursion、KL计算、alpha=.5、架构及 optimizer 不变。')
    paragraph('Filter 内部原 beta 日程保留为诊断 reference；训练从独立零值取得 switch loss，不把 KL 梯度图接入 backward。'
              '因此 reference_schedule_beta 可以非零，而 beta_effective 和 actual weighted switch loss 始终为0。')
    heading('来源 / 初始化 / loss sanity')
    table(['Field','Value'],[[k,report[k]] for k in ('checkpoint_paths','checkpoint_sha256','metadata','seed','fixed_TEST_batch_shape','initialization','integrity')])
    paragraph('初始 TRAIN/TEST forward、prediction loss 均独立比较。总损失差检查包括 epoch1（原 beta=0）与 epoch20（原 beta=5e-4），'
              '避免只验证一个平凡的零差。Epoch20只是初始化下的 loss sanity，不执行训练、不更改参数，随后恢复epoch。')
    heading('Performance：5d primary')
    table(['Variant','Params','BestEpoch','VAL MAE/RMSE/Hit%','TEST MAE/RMSE/Hit%'],
          [[k,m['params'],m['best_epoch'],metric(m,'VAL'),metric(m,'TEST')] for k,m in models.items()])
    paragraph('D0B 为同环境重新加载的实际 checkpoint。历史 TEST MAE≈.0219948、RMSE≈.0284856、Hit≈49.05%仅作核对。'
              'RMSE沿用mean(asset-wise RMSE)，Hit以百分比表示，误差在收益率空间计算。')
    for s in ('TRAIN','VAL','TEST'):
        heading(s+' full horizon metrics')
        table(['Horizon','D0B MAE/RMSE/Hit%','NoSwitchKL MAE/RMSE/Hit%','ΔMAE','RelativeChange%'],
              [[h,metric(models['D0B'],s,h),metric(models['NoSwitchKL'],s,h),v['delta_MAE'],
                100*v['relative_change'] if v['relative_change'] is not None else None] for h,v in comparison[s].items()])
    heading('训练 KL trajectory / actual loss')
    history=models['NoSwitchKL']['history']
    table(['Epoch','train prediction','val prediction','raw KL','posterior entropy','prior entropy','posterior-prior L1',
           'weighted switch','effective beta','reference beta','total loss','LR','best/final'],
          [[r['epoch'],r['train']['prediction_loss'],r['val_prediction_loss']]
           +[r['train'][v] for v in ('raw_KL','posterior_entropy','prior_entropy','posterior_prior_L1','weighted_switch_loss')]
           +[r['beta_effective'],r['reference_schedule_beta'],r['train']['total_loss'],r['lr'],
             '/'.join(k for k,e in (('best',history.get('best_epoch')),('final',history.get('final_epoch'))) if e==r['epoch'])]
           for r in history.get('objective_history',[])])
    paragraph('上表是整个训练 epoch 已有 batch forward 的均值；不额外改变训练随机序列。'
              '下表是固定TRAIN batch在epoch1/5/10/20/best/final权重上的eval probe，和优化过程均值明确区分。')
    table(['Stage','raw KL','posterior entropy','prior entropy','posterior-prior L1','weighted switch','effective beta'],
          [[stage]+[v[k] for k in ('raw_KL','posterior_entropy','prior_entropy','posterior_prior_L1','weighted_switch_loss','beta_effective')]
           for stage,v in history.get('epoch_diagnostics',{}).items()])
    heading('Regime diagnostics / raw KL full loaders')
    keys=('mean','entropy','prior_entropy','mean_max','margin','occupancy','min_probability','posterior_prior_KL','posterior_prior_L1','temporal_L1')
    table(['Variant','Split']+list(keys),[[k,s]+[regime(m,s,key) for key in keys] for k,m in models.items() for s in ('TRAIN','VAL','TEST')])
    paragraph('Raw KL继续按原KL(p||prior)定义计算。概率统计覆盖全部滑窗相对时刻，重复日期重复计数；entropy用自然对数。')
    heading('Transition / collapse checks')
    for k,m in models.items():
        paragraph(k+' transition:\n```json\n'+json.dumps(m['transition'],indent=2,ensure_ascii=False)+'\n```')
        paragraph(k+' concentration / candidate warnings:\n```json\n'+json.dumps(m['collapse_checks'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('Argmax occupancy>95%只标记 concentration warning，不能单凭它判collapse。'
              'Hard posterior warning 的描述约定为entropy<.1、mean max>.99、min p<1e-4；同时查看mean p和性能。')
    heading('Prediction vs switch-KL gradient contribution')
    table(['Variant','effective beta','counterfactual beta','prediction loss','raw KL','actual weighted switch','counterfactual weighted switch','actual total'],
          [[k]+[m['switch_gradients'][v] for v in ('beta_effective','counterfactual_beta','prediction_loss','raw_KL',
                                                'actual_weighted_switch_loss','counterfactual_weighted_switch_loss','actual_total_loss')]
           for k,m in models.items()])
    table(['Variant','Module','pred grad norm','actual total grad norm','counterfactual switch grad norm',
           'raw KL grad norm','cos(pred,switch)','switch/pred norm ratio','max |total grad−pred grad|'],
          [[k,name]+[v[key] for key in ('prediction_norm','total_norm','counterfactual_switch_norm','raw_KL_norm',
                                       'cos_prediction_switch','switch_prediction_ratio','total_prediction_max_diff')]
           for k,m in models.items() for name,v in m['switch_gradients']['modules'].items()])
    table(['Variant','evidence weight norm','evidence bias norm'],
          [[k,m['switch_gradients']['evidence_weight_norm'],m['switch_gradients']['evidence_bias_norm']] for k,m in models.items()])
    paragraph('Counterfactual KL使用每个checkpoint自身best epoch对应的原D0B beta日程；若best epoch=1，weighted梯度为0，cosine报告null。'
              '同时提供未加权raw-KL梯度。NoSwitchKL的实际total gradient应等于prediction gradient；假想梯度不参与任何更新。'
              'Cosine<0表示固定TEST batch上的局部冲突，>0表示局部对齐，不代替VAL/TEST证据。')
    heading('Candidate specialization / Z dynamics / balanced readout / Base RPE')
    for s in ('TRAIN','VAL','TEST'):
        table(['Variant',s+' candidate norms','pairwise L1/cosine','p-weighted contributions'],
              [[k]+[m['splits'][s]['candidate_specialization'][v] for v in ('candidate_L2_norm','pairwise','weighted_contribution_L2_norm')] for k,m in models.items()])
    for k,m in models.items():
        paragraph(k+' fixed TEST:\n```json\n'+json.dumps({key:m['fixed'][key] for key in ('micro','readout_weights','base_rpe','representation_norms')},indent=2,ensure_ascii=False)+'\n```')
    paragraph('Readout W_long/W_micro 是W_H/W_Z projection的Frobenius norm；final state_readout输入块另列。'
              'Base/QK统计causal下三角（含对角）、缩放后加bias前内容logits，按层等权平均。')
    heading('Functional utilization：overall and per horizon')
    for s in ('VAL','TEST'):
        paragraph(s)
        table(['Variant','Horizon','zero-micro','zero-long','micro/long','native-uniform','RoutingFraction'],
              [[k,h if h is not None else 'overall']+functional(m,s,h) for k,m in models.items() for h in (None,1,5,10,20)])
    paragraph('D0B历史TEST overall RoutingFraction约2%，实际对照以上方同环境计算为准，不与5d-specific比例混用。'
              'Uniform仅改generator weighting，p recursion保持原样，完整递推Z。RoutingFraction是干预响应比，不是可加性贡献。')
    heading('Forced-state impact by horizon')
    for s in ('VAL','TEST'):
        paragraph(s)
        table(['Variant','State','Horizon','shared Z_T mean/max','shared h_micro mean/max','prediction mean/max'],
              [[k,i,h,m['splits'][s]['modes'][f'state{i}']['impact']['Z_T'],m['splits'][s]['modes'][f'state{i}']['impact']['h_micro'],
                m['splits'][s]['modes'][f'state{i}']['impact']['per_horizon'][str(h)]]
               for k,m in models.items() for i in range(3) for h in config.MULTI_HORIZONS])
    paragraph('Z_T和h_micro是所有horizons共用的表示，因此相同state的latent impact重复列出；只有prediction impact按horizon区分。')
    heading('Raw single-horizon gradients / conflict geometry')
    for h in map(str,config.MULTI_HORIZONS):
        table([h+'d Module','D0B norm','NoSwitchKL norm'],
              [[name]+[models[k]['gradients']['norms'][h][name] for k in ('D0B','NoSwitchKL')] for name in models['D0B']['gradients']['norms'][h]])
    table(['Variant','Module','cos(5,1)','cos(5,10)','cos(5,20)'],
          [[k,name]+list(v.values()) for k,m in models.items() for name,v in m['gradients']['cosines'].items()])
    heading('Causality / batch / within-market sanity')
    for k,m in models.items():paragraph(k+'\n```json\n'+json.dumps(m['sanity'],indent=2,ensure_ascii=False)+'\n```')
    paragraph('本研究额外要求future perturbation的E/H/p/Z/readout prefix差异严格为0；batch与市场置换允许原数值容差。'
              '完整图置换同步重排asset embeddings，不改变节点身份。此检查不重新认证历史数据预处理。')
    heading('十八个问题')
    init=report.get('initialization') or {}
    paragraph('1–3. Shared initialization、参数量、initial forward：'+fmt({'PASS':init.get('PASS','not recorded'),
              'parameter_difference':models['NoSwitchKL']['params']-models['D0B']['params']})+'；TRAIN/TEST的完整差异见初始化表。')
    paragraph('4. 唯一训练差异为KL contribution：prediction objective仍为四项sum；有效beta恒0，rawKL计算保留；loss差检查和actual total−prediction梯度检查见上表。')
    for number,key in ((5,'posterior_prior_KL'),(6,'entropy'),(7,'temporal_L1')):
        paragraph(f'{number}. {key}变化（NoSwitch−D0B）：'+fmt(change(key))+'。')
    paragraph('8. Transition-logit drift：'+fmt({k:m['transition']['transition_logits_drift_L2'] for k,m in models.items()})+'。更大漂移不等于承担了更多有益正则化。')
    paragraph('9. Regime evidence gradients：'+fmt({k:m['switch_gradients']['modules']['regime evidence'] for k,m in models.items()})+'。')
    paragraph('10. Candidates分化：'+fmt({k:m['splits']['TEST']['candidate_specialization']['pairwise'] for k,m in models.items()})+'；结合candidate norms与warnings，不把更大分化自动当成更好。')
    paragraph('11. Z dynamics：'+fmt({k:{v:m['fixed']['micro'][v] for v in ('mean_Z_norm','mean_delta_Z_norm','consecutive_cosine')} for k,m in models.items()})+'。')
    paragraph('12. RoutingFraction_5d变化：'+fmt(assessment['routing_fraction_5d_delta'])+'；overall及其他horizons见上表。')
    paragraph('13. Forced-state 5d prediction effects：'+fmt({k:{f'state{i}':m['splits']['TEST']['modes'][f'state{i}']['impact']['per_horizon']['5']['mean'] for i in range(3)} for k,m in models.items()})+'。')
    for n,s in ((14,'VAL'),(15,'TEST')):paragraph(f'{n}. {s} 5d：'+fmt(comparison[s]['5'])+'。')
    paragraph('16. 实际意义：'+assessment.get('reason','')+' '+assessment['threshold_note'])
    directions={}
    for k,m in models.items():
        directions[k]={}
        for name in ('regime evidence','transition logits'):
            cosine=m['switch_gradients']['modules'][name]['cos_prediction_switch']
            directions[k][name]='undefined (zero gradient)' if cosine is None else ('conflicting' if cosine<0 else 'aligned' if cosine>0 else 'orthogonal')
    paragraph('17. Prediction vs switch gradient方向：'+fmt(directions)+'。')
    paragraph('18. '+fmt(assessment)+'。')
    entropy_down=all(v<0 for s,v in change('entropy').items() if s in ('VAL','TEST'))
    kl_up=all(v>0 for s,v in change('posterior_prior_KL').items() if s in ('VAL','TEST'))
    dynamics_up=models['NoSwitchKL']['fixed']['micro']['mean_delta_Z_norm']>models['D0B']['fixed']['micro']['mean_delta_Z_norm']
    if assessment['case']=='Case C' and (any(v['hard_posterior_warning'] for v in models['NoSwitchKL']['collapse_checks'].values()) or (entropy_down and kl_up and dynamics_up)):
        paragraph('**regime over-specialization**：entropy下降、KL/dynamics更强或hard-posterior warning伴随VAL/TEST恶化；支持KL有益，不能称机制改善。')
    if entropy_down and all(abs(v)<=1e-4 for v in assessment['routing_fraction_5d_delta'].values()):
        paragraph('Posterior更尖锐而5d RoutingFraction基本不变：浓度变化未转化为更强的功能性expert routing。')
    paragraph('单seed诊断不构成统计显著性证明。仅Case A具备保留NoSwitchKL为候选的机制证据；脚本不会替换D0B。'
              '若题设A–F未覆盖实际组合，报告未分类、不强行宣称成功。')
    paragraph(assessment['case'] or '未分类：题设条件未覆盖，保留D0B')
    paragraph('STOP。不自动进入Huber、routing、evidence、objective weighting或其他实验。')
    path.write_text('\n'.join(lines),encoding='utf-8')
