"""Commodity adapter reports: pooled metrics and direct residual interventions."""
from cmgm.scripts.d0b_huber_horizon_scale_diagnostic import markdown_table

HORIZONS=(1,5,10,20)
NEW='CommodityResidual'


def write_preflight(m,path):
    i=m['initialization'];g=i['gradients']['objectives']['prediction_sum']
    lines=['# D0B-CommodityResidualAdapter — preflight', '',
           '仅实现与代码/真实数据验证；没有正式训练。泛化表现、learned alpha 和 Case A–G 尚未判定。', '',
           '唯一新增：shared Linear(64→4,bias=False) + 一个 alpha_0=0。GNN+LN 后、type_pool 前取 batch-aware commodity nodes。原 D0B spatial、temporal、gate、fusion、head 均保留。', '',
           f'Git SHA: `{m["git_sha"]}`。源文件 hashes 和原始检查见 preflight.json。', '',
           markdown_table(['Check','Result'],[['D0B params',i['D0B_params']],['New params',i['new_params']],['Extra params',i['difference']],
            ['Shared tensors',i['shared_parameter_count']],['Shared max diff',i['max_abs_diff']],['Mismatches',i['mismatch_count']],
            ['Fixed TRAIN',i['fixed_TRAIN_shape']],['Fixed TEST',m['fixed_TEST_shape']],['h_nodes',i['h_nodes_shape']],['h_comm',i['h_comm_shape']],
            ['Alpha init',i['initial_alpha']],['Initial effective residual',i['initial_residual']['effective_mean_abs']],
            ['Initial signed alpha gradient',g['signed_alpha_gradient']],['Initial residual-head gradient norm',g['norms']['commodity_residual_head.weight']],
            ['Initialization PASS',i['PASS']],['Sanity PASS',m['fixed_TEST_sanity']['PASS']]]),
           markdown_table(['Trace','Mean difference','Max difference'],[[k,v['mean'],v['max']] for k,v in i['forward_differences'].items()]),
           markdown_table(['Variant','L1','L5','L10','L20','Prediction sum','Switch epoch20','Total epoch20'],
            [[k,*[v['raw_horizons'][str(h)] for h in HORIZONS],v['prediction'],v['switch_epoch20'],v['total_epoch20']] for k,v in i['losses'].items()]),
           'alpha=0 时 residual-head prediction-loss gradient=0 是数学预期；alpha 离开 0 后该梯度可激活。原 Adam weight decay 仍施加于全部参数，不能仅用 head 权重变化证明 prediction-gradient 已激活。', '',
           '未来扰动仅要求 temporal prefix 不变；spatial nodes/base prediction 是 window-end states，按截断窗口验证因果性。合法 commodity 重标记同时调整 graph embeddings 与原 head 输出行。FP32 数值误差如需 FP64 副本复核，两套数值均保留。', '',
           '运行命令与 checkpoint-only 说明见 docs/D0B_CommodityResidualAdapter.md。正式训练由用户执行；STOP。','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def write_report(r,path):
    models=r['models'];a=r['assessment'];initial=r['metadata'].get('initialization',{})
    lines=['# D0B-CommodityResidualAdapter','',f'**{a["case"]}** — {a["reason"]}','',
           'Shared D0B global forecast + shared lightweight node-conditioned commodity residual。仅添加 bias-free Linear(64,4) 和一个 unconstrained scalar alpha，初始化为 0。', '',
           '原 D0B encoder/dynamics/temporal readout/global spatial pooling/gate/head 全部保留；loss 为四 horizon Huber(delta=.02) 求和 + 原 switch KL。Validation selection 仍是 prediction-only sum。', '',
           f'Git SHA: `{r["metadata"].get("git_sha")}`；fixed TEST batch: `{r["fixed_TEST_shape"]}`。', '',
           f'Metrics: {r["metric_standard"]}。MSE 直接聚合 residual²；表内 Hit 转为百分比。', '',
           'alpha=0 control 来自新训练 checkpoint 的 base path，包含 backbone co-adaptation，不能当作原始 D0B checkpoint。', '']
    def table(headers,rows):lines.extend([markdown_table(headers,rows),''])
    def heading(name):lines.extend([f'## {name}',''])
    def metric_values(d):return [d[k]*(100 if k=='Hit' else 1) for k in ('MAE','MSE','RMSE','Hit')]
    table(['Variant','Checkpoint','SHA256'],[[k,v,r['checkpoint_sha256'][k]] for k,v in r['checkpoint_paths'].items()])
    lines += [f'Integrity: `{r["integrity"]}`', '',
              f'固定 residual commodity permutation: `{r["commodity_permutation"]}`。只作用于 residual branch 的 commodity 维，不交换 batch 样本、预测目标或未来日期。', '']
    heading('Performance')
    table(['Variant','Params','BestEpoch','TrainTime seconds','VAL5 MAE','MSE','RMSE','Hit%','TEST5 MAE','MSE','RMSE','Hit%'],
          [[k,m['params'],m['best_epoch'],m['train_time'],*metric_values(m['splits']['VAL']['native_metrics']['5']),
            *metric_values(m['splits']['TEST']['native_metrics']['5'])] for k,m in models.items()])
    lines += ['旧 checkpoint 未保存的 train time/best epoch 标记为 n/a；不推算。所有 comparison 使用实际重新加载的 D0B，而非 HSR 或历史 legacy evaluator。', '']
    for split in ('TRAIN','VAL','TEST'):
        heading(split+' all horizons')
        table(['Variant','Horizon','MAE','MSE','RMSE','Hit%'],[[k,h,*metric_values(m['splits'][split]['native_metrics'][str(h)])] for k,m in models.items() for h in HORIZONS])
        table(['Horizon','Δ MAE','Relative MAE %','Δ MSE','Relative MSE %'],
              [[h,r['comparison'][split][str(h)]['delta_MAE'],100*r['comparison'][split][str(h)]['relative_MAE'],
                r['comparison'][split][str(h)]['delta_MSE'],100*r['comparison'][split][str(h)]['relative_MSE']] for h in HORIZONS])
    heading('Residual functional utilization')
    for split in ('VAL','TEST'):
        s=models[NEW]['splits'][split];lines += [f'**{split}**','']
        table(['Mode','Horizon','MAE','MSE','RMSE','Hit%','Native prediction impact mean','max'],
              [[mode,h,*metric_values(s['native_metrics'][str(h)] if mode=='native' else s['control_metrics'][mode][str(h)]),
                0 if mode=='native' else s['impacts'][mode][str(h)]['mean'],0 if mode=='native' else s['impacts'][mode][str(h)]['max']]
               for mode in ('native','alpha=0','shuffled','mean-commodity') for h in HORIZONS])
        table(['Horizon','alpha','Raw residual mean abs','Effective mean abs','Effective/base','P50','P90','P95','max','Alpha0 impact','Shuffle impact','MeanCommodity impact'],
              [[h,models[NEW]['alpha'],*[s['residual']['horizons'][str(h)][k] for k in ('raw_mean_abs','effective_mean_abs','effective_base_ratio','P50','P90','P95','max')],
                *[s['impacts'][mode][str(h)]['mean'] for mode in ('alpha=0','shuffled','mean-commodity')]] for h in HORIZONS])
        lines += [f'Base max diff during residual interventions = {s["base_max_diff_during_residual_controls"]}.', '']
    lines += [f'Activity evidence and fixed descriptive conventions: `{r["activity"]}`', '',
              'native vs alpha=0 衡量直接 residual 效应；新 alpha=0 vs 原 D0B 反映 backbone co-adaptation。Shuffle 检查 commodity alignment；mean replacement 去掉 individual commodity differences，保留共同 spatial 信息。', '']
    heading('Regime, spatial and temporal health')
    for split in ('TRAIN','VAL','TEST'):
        lines += [f'**{split}**','']
        table(['Variant','p entropy','prior entropy','mean max p','margin','KL','posterior-prior L1','p temporal L1','transition drift'],
              [[k,*[m['splits'][split]['normal_regime'][v] for v in ('entropy','prior_entropy','mean_max','margin','posterior_prior_KL','posterior_prior_L1','temporal_L1')],m['transition']['transition_logits_drift_L2']] for k,m in models.items()])
        table(['Variant','Mean p','Occupancy','Candidate norms','Candidate pairwise L1/cosine','Weighted contribution'],
              [[k,m['splits'][split]['normal_regime']['mean'],m['splits'][split]['normal_regime']['occupancy'],
                m['splits'][split]['candidate_specialization']['candidate_L2_norm'],m['splits'][split]['candidate_specialization']['pairwise'],
                m['splits'][split]['candidate_specialization']['weighted_contribution_L2_norm']] for k,m in models.items()])
        table(['Variant','Z norm','delta Z','Z cosine','raw Z/H','long norm','micro norm','post-balance micro/long','state Wmicro/Wlong','projection Wmicro/Wlong','BaseRPE norm','mean abs QK','mean abs RPE','RPE/QK'],
              [[k,*[m['splits'][split]['micro'][v] for v in ('mean_Z_norm','mean_delta_Z_norm','consecutive_cosine','Z_T_H_T_raw_ratio','h_long_norm','h_micro_norm','micro_long_norm_ratio','W_micro_W_long')],
                m['splits'][split]['projection_weights']['W_micro_W_long'],*[m['splits'][split]['base_rpe'][v] for v in ('norm','mean_abs_QK','mean_abs_base_bias','base_QK_ratio')]] for k,m in models.items()])
        table(['Variant','Horizon','zero-micro impact','zero-long impact','micro/long','uniform routing impact','RoutingFraction'],
              [[k,h,*[m['splits'][split]['impacts'][mode][str(h)]['mean'] for mode in ('zero-micro','zero-long')],m['splits'][split]['micro_long_ratio'][str(h)],
                m['splits'][split]['impacts']['uniform'][str(h)]['mean'],m['splits'][split]['RoutingFraction'][str(h)]] for k,m in models.items() for h in HORIZONS])
        table(['Variant','Gate mean','std','min','max','Commodity MSE mean','std','CV','min','max'],
              [[k,*[m['splits'][split]['gate'][v] for v in ('mean','std','min','max')],*[m['splits'][split]['commodity_MSE']['native'][v] for v in ('mean','std','CV','min','max')]] for k,m in models.items()])
    table(['Variant','alpha sticky','A','Mean diagonal','Row entropy','Transition logits drift','Effective A drift'],
          [[k,m['transition']['alpha'],m['transition']['A'],m['transition']['mean_diagonal'],m['transition']['row_entropy'],
            m['transition']['transition_logits_drift_L2'],m['transition']['effective_A_change_L2']] for k,m in models.items()])
    lines += [f'Regime over-specialization warning: `{r["regime_over_specialization_warning"]}`。仅当 occupancy≥.99、mean max≥.99、entropy≤.1 同时成立时触发；occupancy 单独不作为 collapse 判据。', '']
    table(['Variant','TEST commodity node statistics'],[[k,m['splits']['TEST']['commodity_nodes']] for k,m in models.items()])
    heading('Commodity performance and error redistribution')
    for split in ('VAL','TEST'):
        comparison=r['commodity_comparison'][split];lines += [f'**{split} 5d**','']
        for k,m in models.items():
            modes=('native','alpha=0') if k==NEW else ('native',)
            table(['Variant','Mode','Commodity','MAE','MSE','RMSE','Hit%'],[[k,mode,v['name'],*metric_values(v)] for mode in modes for v in m['splits'][split]['commodity_5d'][mode]])
        table(['Commodity','D0B MAE','New MAE','Δ MAE','D0B MSE','New MSE','Δ MSE','Effective residual mean abs'],
              [[v['name'],v['D0B']['MAE'],v[NEW]['MAE'],v['delta_MAE'],v['D0B']['MSE'],v[NEW]['MSE'],v['delta_MSE'],v['effective_mean_abs']] for v in comparison['rows_sorted_by_delta_MSE']])
        table(['Group','Commodity','Δ MAE','Δ MSE'],[[group,v['name'],v['delta_MAE'],v['delta_MSE']] for group in ('most_improved','most_worsened') for v in comparison[group]])
        table(['Pearson baseline MSE vs improvement','Spearman','Baseline error halves'],[[comparison['Pearson_baseline_MSE_vs_improvement'],comparison['Spearman_baseline_MSE_vs_improvement'],comparison['baseline_error_halves']]])
    corrections=models[NEW]['splits']['TEST']
    table(['TEST5 commodity','Effective residual mean abs','std'],[[v['name'],v['mean_abs_effective'],v['std_effective']] for v in corrections['commodity_corrections']])
    table(['Correction rank','Commodity','Effective mean abs','std'],[[group,v['name'],v['mean_abs_effective'],v['std_effective']] for group in ('strongest_corrections','weakest_corrections') for v in corrections[group]])
    heading('Adapter and shared-branch gradients')
    for k,m in models.items():
        g=m['gradients']['objectives'];lines += [f'**{k}** — eval raw Huber per horizon; autograd.grad; no optimizer step.','']
        table(['Module','L1','L5','L10','L20','Prediction sum','Total'],[[module,*[g[h]['norms'][module] for h in ('1','5','10','20','prediction_sum','total')]] for module in g['5']['norms']])
        table(['Objective','Signed grad(alpha)'],[[h,v['signed_alpha_gradient']] for h,v in g.items()])
    lines += ['梯度用于描述，不作为 accept/reject 条件。alpha=0 时 head 的 prediction-loss gradient 为零是预期；weight decay 仍按原协议作用于所有参数。', '']
    heading('Training history and activation')
    history=models[NEW]['history']
    table(['Epoch','Best','Train total','Prediction','Switch','Val prediction','L1','L5','L10','L20','Alpha','Raw mean abs','Effective mean abs','Effective/base','Head norm','Head init drift','LR'],
          [[v['epoch'],v['epoch']==models[NEW]['best_epoch'],v['train']['total_loss'],v['train']['prediction_loss'],v['train']['switch_loss'],v['val']['prediction_loss'],
            *[v['train'][f'raw_L{h}'] for h in HORIZONS],v['alpha'],v['train']['raw_residual_mean_abs'],v['train']['effective_residual_mean_abs'],
            v['train']['effective_residual_base_ratio'],v['residual_head_weight_norm'],v['residual_head_distance_from_init'],v['lr']] for v in history.get('objective_history',[])])
    table(['Stage','Alpha','Head norm','Init drift','Raw mean abs','Effective mean abs','Effective/base','Prediction grad(alpha)','Prediction head grad','Total grad(alpha)','Total head grad'],
          [[stage,v['alpha'],v['residual_head_weight_norm'],v['residual_head_distance_from_init'],v['raw_mean_abs'],v['effective_mean_abs'],v['effective_base_ratio'],
            v['gradients']['objectives']['prediction_sum']['signed_alpha_gradient'],v['gradients']['objectives']['prediction_sum']['norms']['commodity_residual_head.weight'],
            v['gradients']['objectives']['total']['signed_alpha_gradient'],v['gradients']['objectives']['total']['norms']['commodity_residual_head.weight']]
           for stage,v in history.get('epoch_diagnostics',{}).items()])
    lines += ['每 epoch residual 数值是训练 batch 均值；alpha/head norm 是该 epoch 结束值。1/5/10/20（若达到）、best/final 的固定 TRAIN probes 用同一个 batch、eval 模式。完整 validation raw horizon losses 也保存在 results.json/checkpoint history。', '']
    heading('Research answers')
    split_effect={s:{mode:models[NEW]['splits'][s]['impacts'][mode]['5'] for mode in ('alpha=0','shuffled','mean-commodity')} for s in ('VAL','TEST')}
    q=[('1. Only shared Linear + scalar?', 'Yes: Linear(64,4,bias=False) shared across commodities; scalar unconstrained alpha.'),
       ('2. Exactly +257 parameters?',str(initial.get('difference'))),
       ('3. Batch-aware node states?',str(initial.get('h_nodes_shape'))),
       ('4. Batch averaging avoided?', 'Yes; no reduction over batch axis; batch permutation and single-sample checks recorded.'),
       ('5. Node tap position?', 'After attn_mixhop2 and gcn_norm, before type_pool.'),
       ('6. Global spatial path retained?', 'Yes; existing type_pool still feeds original D0B gate.'),
       ('7. Temporal H/p/Z retained?', 'Yes; unchanged D0B branch, shared state_readout, fixed sticky=.5.'),
       ('8. Original gate/head retained?', 'Yes; same gate_fc, lstm_proj, gcn_proj, head parameters.'),
       ('9. Shared initialization equal?',str({k:initial.get(k) for k in ('max_abs_diff','mismatch_count','PASS')})),
       ('10. Initial base equal D0B?',str(initial.get('forward_differences',{}).get('base_pred'))),
       ('11. Initial alpha0 final equal D0B?',str(initial.get('forward_differences',{}).get('prediction'))),
       ('12. Initial head gradient zero expected?', 'Yes; see initialization gradients in metadata; alpha is the first loss-gradient entry point.'),
       ('13. Alpha leaves zero?',str(models[NEW]['alpha'])),
       ('14. Later head loss gradients active?',str({h:v['norms'].get('commodity_residual_head.weight') for h,v in models[NEW]['gradients']['objectives'].items()})),
       ('15. Best effective residual size?',str({s:models[NEW]['splits'][s]['residual']['effective_mean_abs'] for s in ('VAL','TEST')})),
       ('16. Native-alpha0 impact?',str({s:v['alpha=0'] for s,v in split_effect.items()})),
       ('17. Shuffle impact?',str({s:v['shuffled'] for s,v in split_effect.items()})),
       ('18. Mean-commodity impact?',str({s:v['mean-commodity'] for s,v in split_effect.items()})),
       ('19. VAL5 MAE/MSE improves?',str({k:v for k,v in r['comparison']['VAL']['5'].items() if k.startswith('relative')})),
       ('20. TEST5 MAE/MSE improves?',str({k:v for k,v in r['comparison']['TEST']['5'].items() if k.startswith('relative')})),
       ('21. Commodities most improved?',str([v['name'] for v in r['commodity_comparison']['TEST']['most_improved']])),
       ('22. Commodities most worsened?',str([v['name'] for v in r['commodity_comparison']['TEST']['most_worsened']])),
       ('23. MSE CV decreases?',str({k:m['splits']['TEST']['commodity_MSE']['native']['CV'] for k,m in models.items()})),
       ('24. Baseline difficulty vs improvement?',str({s:{k:v for k,v in c.items() if k.startswith(('Pearson','Spearman'))} for s,c in r['commodity_comparison'].items()})),
       ('25. Regime/Z healthy?',str(r['regime_over_specialization_warning'])+'; inspect continuous candidate/Z tables as well.'),
       ('26. Final Case?',a['case']+' — '+a['reason'])]
    table(['Question','Answer'],q)
    lines += [f'Case details: `{a}`','',
              'Sanity 数值见 results.json。Spatial 是截至当前 window end 的状态；不得将 full-window spatial output 宣称为每个 prefix 的状态。合法 commodity 重标记与 residual-only shuffle 分开。', '',
              '报告仅提供单 seed 机制与泛化证据，绝不以 gradient conflict 或单独 TEST 改善自动接受模型。所有 counterfactual 保持 checkpoint 和参数不变。', '', '**STOP**','']
    path.write_text('\n'.join(lines),encoding='utf-8')
