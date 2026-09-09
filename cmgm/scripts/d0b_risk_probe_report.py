"""Evidence-reviewed interpretation for this official checkpoint diagnostic.

The prose below describes the observed result, not a probe-selection algorithm.
If its supporting comparisons change, require another review instead of silently
reusing the conclusion. No probe, threshold or coefficient is changed here.
"""


def attach_reviewed_conclusion(r):
    metrics={(x['split'],x['probe']):x for x in r['probe_metrics']}
    within={(x['split'],x['probe']):x for x in r['within_commodity_summary']}
    static='CommodityTrainMeanRisk'
    # Guard the concrete comparisons underlying this run's manual Case G review.
    # These are not a general-purpose statistical classification rule.
    conditions=[]
    for split in ('val','test'):
        m=lambda p:metrics[split,p]
        conditions += [m(static)['RiskMSE']<m('Vol20Only')['RiskMSE'],
            m('Vol20Only')['R2']>0,m('SimpleRiskFeatures')['R2']>0,
            m('CommodityNode')['RiskMSE']>m('GlobalFused')['RiskMSE'],
            all(m(p)['R2']<0 for p in ('GlobalFused','TemporalOnly','SpatialGlobal','LongMicro','CommodityNode')),
            0<within[split,'Vol20Only']['Spearman_mean']<m('Vol20Only')['Spearman']]
    conditions += [metrics['val',static]['RiskMSE']<metrics['val','SimpleRiskFeatures']['RiskMSE'],
                   metrics['test','CommodityNodePlusVol20']['RiskMSE']>metrics['test','CommodityNode']['RiskMSE']]
    if not all(conditions):
        raise RuntimeError('Statistics saved. Observed comparisons differ from the reviewed run; review evidence before assigning a primary case. No extra probe may be fitted.')

    r['primary_case']='Case G'
    r['interpretation_status']='EVIDENCE REVIEW COMPLETE'
    r['status']='COMPLETE — frozen extraction, fixed OLS probes, evaluation and bootstrap; STOP'
    r['static_explained_fraction_of_simple_risk_MSE_gain']={
        split:(metrics[split,'MeanRisk']['RiskMSE']-metrics[split,static]['RiskMSE'])/
              (metrics[split,'MeanRisk']['RiskMSE']-metrics[split,'SimpleRiskFeatures']['RiskMSE'])
        for split in ('val','test')}
    shares=r['static_explained_fraction_of_simple_risk_MSE_gain']
    v=[within[s,'Vol20Only']['Spearman_mean'] for s in ('val','test')]
    a=[within[s,'Vol20Only']['AUC_mean'] for s in ('val','test')]
    r['interpretation_text']=(
        'Primary Case G：本轮 pooled risk predictability 主要由持续的商品风险差异解释，动态风险 timing 证据较弱。'
        f'静态商品均值相对 MeanRisk 的 MSE 收益，达到 SimpleRiskFeatures 收益的 VAL {shares["val"]:.2%}、TEST {shares["test"]:.2%}；'
        '这只是相对基线收益比，不是因果解释度或完整方差分解。静态均值在 VAL 的 MSE 甚至略好于四项原始风险特征。'
        f'Vol20Only 的 within-commodity Spearman 均值仅 VAL {v[0]:.6f} / TEST {v[1]:.6f}，'
        f'有效商品内 Tail90 AUC 均值为 {a[0]:.6f} / {a[1]:.6f}。'
        '原始 vol20 的 pooled tail 排序明显更强，但不能据此宣称已经能有效监测同一商品随时间变化的风险。'
        'Secondary finding：原始 causal risk features 比本轮 learned-representation OLS 更稳定，符合 Case C 的信息源倾向；'
        '按照用户要求的静态对照和 within-commodity 优先级，本轮只选 Case G，不另选第二个 primary case。')
    r['representation_verdicts']=[
        dict(representation='GlobalFused',verdict='NO',reason='本次 frozen OLS 未展示稳定可泛化的线性风险读出：两 split R²<0，within Spearman 均值均为负。NO 不表示完全没有风险信息。'),
        dict(representation='Temporal',verdict='WEAK',reason='VAL 有弱正向排序，TEST 反向；两 split 风险回归均差于 MeanRisk。'),
        dict(representation='SpatialGlobal',verdict='WEAK',reason='VAL within 排序为正，但 TEST 不稳定且绝对风险预测严重外推；不能据此推断 pooling 丢失风险信息。'),
        dict(representation='CommodityNode',verdict='WEAK',reason='VAL within 排序为正，TEST 反向；风险误差大于 global，系数跨 TRAIN halves 不稳定。当前 OLS 未稳定读出动态风险。'),
        dict(representation='Raw vol20 adds information',verdict='YES',reason='相对 GlobalFused，加入 vol20 在 VAL/TEST 的 MAE/MSE 改善且四个配对 CI 均低于0；但仍差于 Vol20Only/MeanRisk。相对 node 的增量不稳定，TEST MSE 恶化。不能宣称 node 已充分吸收 vol20。')]
    r['comparison_verdicts']=dict(
        node_global='否。CommodityNode 在 VAL/TEST 的 MAE/MSE 均显著更差；不支持风险信息被 pooling 压制的结论。',
        node_plus_global='否。加入 global 后两 split 的 MAE/MSE 均更差。',
        node_plus_vol='无稳定增益。VAL 改善；TEST MAE delta 的 CI 跨0，TEST MSE 明确恶化。不能从此推断 node 已吸收 raw vol20。',
        global_plus_vol='是。四项 MAE/MSE delta 的 CI 均低于0，但绝对表现仍不如 MeanRisk/Vol20Only，不能据此接受 global risk head。',
        simple='是。SimpleRiskFeatures 的风险回归稳定优于 learned-representation probes；不过商品内动态排序仅弱正相关，静态对照解释大部分收益。')
    r['within_answer']=(
        'Within-commodity Spearman：VAL 均值最高的是 SpatialGlobal '
        f'({within["val","SpatialGlobal"]["Spearman_mean"]:.6f})，但其 TEST 均值为 '
        f'{within["test","SpatialGlobal"]["Spearman_mean"]:.6f}，不稳定。'
        f'TEST 均值最高且两 split 保持正向的是 Vol20Only ({v[0]:.6f}/{v[1]:.6f})，力度仍弱。'
        'SimpleRiskFeatures 的 TEST median 高于 Vol20Only，不能把均值排序当成所有商品一致胜出。'
        'VAL 有效相关商品24个，TEST23个；TEST小麦 target 为常量，相关系数未定义而非0。')
    r['heterogeneity_answer']=(
        'Cross-sectional：静态商品均值已取得大部分 pooled MSE 收益。Time-series：Vol20Only 有弱但同方向的商品内排序信息，'
        '并非完全没有动态信息。Magnitude regression：SimpleRiskFeatures TEST 略优于静态均值，VAL MSE 未超过静态均值，'
        '不能把全部 pooled AUROC 当作风险 timing 能力；也不能断言全部收益都来自商品身份。')
    r['location_answer']=(
        '当前稳定可线性读取的风险信息主要体现在 raw commodity risk features 和持续的商品风险差异，'
        '没有证据支持把 commodity node 或 global D0B state 直接作为有效动态 risk head 的充分输入。'
        'H/Z 的 LongMicro probe 在 VAL/TEST R² 均为负，因此不能仅凭 long/micro coefficient norm 判定风险主要来自哪一支。')
    r['next_direction']='唯一后续研究方向（本轮未执行）：先验证扣除静态商品差异后、raw commodity risk features 的动态风险可预测性；当前不直接进入 risk-head 实现。STOP.'
    r['numerical_audit_note']=(
        '数值审计：同一组 OLS 系数的显式 design-matrix 预测与正式预测一致，TRAIN normal-equation residual 接近浮点误差。'
        'Spatial/Node 外推误差不是“负风险裁剪”或重新缩放后的数值；原值全部保留。'
        'SpatialGlobal 的 TRAIN 条件数约388，但 VAL/TEST 标准化坐标超出 TRAIN 范围，表明条件数本身不足以描述外推风险。'
        'Node/LongMicro 条件数约 10^6–10^7，且 Node 的 TRAIN halves 权重 cosine 接近 -1；'
        '因此大幅负 R² 反映当前无正则线性读出的泛化失败，不能证明表示缺少所有风险信息。'
        '缓存 h_fused 的 fusion 重建和现有 head 回放均与导出的原预测一致；没有重复整套 D0B extraction。')
    return r
