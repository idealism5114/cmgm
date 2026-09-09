"""Reviewable reports for shared dynamics plus horizon-specific state decoding."""
from cmgm.scripts.d0b_huber_horizon_scale_diagnostic import markdown_table

HORIZONS=(1,5,10,20)
NEW='HorizonSpecific'


def write_preflight(metadata,path):
    i=metadata['initialization']
    lines=['# D0B-HorizonSpecificStateReadout — preflight', '',
           '仅实现与验证；未进行正式训练。VAL/TEST 泛化、readout 学习后分化与 Case A–F 尚未判定。', '',
           '唯一变化：保留 5d 原始 state_readout，并为 1d/10d/20d 精确克隆三个 128→64 Linear。共享 E/H/p/G/Z、H/Z projections、gate、fusion、head 参数。', '',
           '目标保持四个 raw Huber(delta=.02) 求和 + 原 switch KL；validation selection 仍只用 prediction sum。', '',
           f'Git SHA: `{metadata["git_sha"]}`；源文件 hashes 与完整数值见 preflight.json。', '',
           markdown_table(['Check','Result'],[['D0B params',i['D0B_params']],['New params',i['new_params']],
             ['Extra params',i['difference']],['Shared tensors',i['shared_parameter_count']],['Shared parameter max diff',i['max_abs_diff']],
             ['Mismatch count',i['mismatch_count']],['TRAIN batch',i['fixed_TRAIN_shape']],['TEST batch',metadata['fixed_TEST_shape']],
             ['Initialization PASS',i['PASS']],['Causality/batch/market PASS',metadata['fixed_TEST_sanity']['PASS']],
             ['Readout gradient isolation',metadata['initial_gradients']['readout_isolation_PASS']]]),
           markdown_table(['Horizon','Clone weight max diff','Clone bias max diff'],[[h,*i['clone_differences'][str(h)].values()] for h in HORIZONS]),
           markdown_table(['Trace','Mean difference','Max difference'],[[k,v['mean'],v['max']] for k,v in i['forward_differences'].items()]),
           markdown_table(['Model','L1','L5','L10','L20','Prediction','Switch epoch20','Total epoch20'],
                          [[k,*v['raw_horizons'],v['prediction'],v['switch_epoch20'],v['total_epoch20']] for k,v in i['losses'].items()]),
           '四个 readout 的 off-horizon raw prediction loss gradients 必须为零；u-gradient 与各模块梯度见 preflight.json。', '',
           '完整执行命令与指标约定见 docs/D0B_HorizonSpecificStateReadout.md。等待人工运行唯一正式训练；STOP。', '']
    path.write_text('\n'.join(lines),encoding='utf-8')


def write_report(r,path):
    models=r['models'];assessment=r['assessment']
    lines=['# D0B-HorizonSpecificStateReadout', '',
           f'**{assessment["case"]}** — {assessment["reason"]}', '',
           '机制定义：shared market-state learning + task-specific decoding subspaces。四 horizons 共同监督同一套 E/H/p/G/Z。', '',
           '5d 保留原 state_readout；1/10/20 为其精确克隆初始化。共享 H/Z projections、LayerNorm、gate、fusion projections、head body 与原 output rows。', '',
           'Loss = Σ Huber_h(delta=.02) + 原 switch KL。Validation early stopping/scheduler = 原四-horizon prediction sum；val total 仅描述。', '',
           f'指标：{r["metric_standard"]}。Hit 表内为百分比；MSE 直接从 residual² 聚合；不使用 legacy asset-averaged RMSE 或 masked Hit。', '',
           f'Git SHA: `{r["metadata"].get("git_sha")}`；fixed TEST batch: `{r["fixed_TEST_shape"]}`。', '',
           'Checkpoint 路径和 SHA256：', '']
    def table(headers,rows):lines.extend([markdown_table(headers,rows),''])
    def heading(text):lines.extend([f'## {text}',''])
    table(['Model','Checkpoint','SHA256'],[[k,p,r['checkpoint_sha256'][k]] for k,p in r['checkpoint_paths'].items()])
    lines += [f'Integrity: `{r["integrity"]}`', '', '下列判定使用固定描述性阈值，不是统计显著性证据。单 seed 不足以估计实验方差；不自动替换正式 D0B。', '']
    heading('Performance')
    table(['Variant','Params','BestEpoch','TrainTime seconds','VAL5 MAE','MSE','RMSE','Hit%','TEST5 MAE','MSE','RMSE','Hit%'],
          [[k,m['params'],m['best_epoch'],m['train_time'],*[m['splits'][s]['native_metrics']['5'][v]*(100 if v=='Hit' else 1)
            for s in ('VAL','TEST') for v in ('MAE','MSE','RMSE','Hit')]] for k,m in models.items()])
    lines += ['TrainTime 若为 n/a，表示旧 checkpoint 未保存该字段，不推算训练时长。', '']
    for split in ('TRAIN','VAL','TEST'):
        heading(split+' all horizons')
        table(['Variant','Horizon','MAE','MSE','RMSE','Hit%'],
              [[k,h,*[m['splits'][split]['native_metrics'][str(h)][v]*(100 if v=='Hit' else 1) for v in ('MAE','MSE','RMSE','Hit')]]
               for k,m in models.items() for h in HORIZONS])
        table(['Horizon','Δ MAE','Relative MAE %','Δ MSE','Relative MSE %'],
              [[h,r['comparison'][split][str(h)]['delta_MAE'],100*r['comparison'][split][str(h)]['relative_MAE'],
                r['comparison'][split][str(h)]['delta_MSE'],100*r['comparison'][split][str(h)]['relative_MSE']] for h in HORIZONS])
    heading('Shared dynamics and H/Z utilization')
    for split in ('TRAIN','VAL','TEST'):
        lines += [f'**{split}**','']
        table(['Variant','p entropy','prior entropy','max p','margin','KL','posterior-prior L1','temporal L1','transition drift','candidate cosine','Z norm','delta Z','Z cosine','Z/H','BaseRPE/QK'],
              [[k,*[m['splits'][split]['normal_regime'][v] for v in ('entropy','prior_entropy','mean_max','margin','posterior_prior_KL','posterior_prior_L1','temporal_L1')],
                m['transition']['transition_logits_drift_L2'],m['splits'][split]['candidate_specialization']['pairwise'],
                *[m['splits'][split]['micro'][v] for v in ('mean_Z_norm','mean_delta_Z_norm','consecutive_cosine','Z_T_H_T_raw_ratio')],
                m['splits'][split]['base_rpe']['base_QK_ratio']] for k,m in models.items()])
        table(['Variant','p mean','argmax occupancy','candidate norms','weighted contribution norms'],
              [[k,m['splits'][split]['normal_regime']['mean'],m['splits'][split]['normal_regime']['occupancy'],
                m['splits'][split]['candidate_specialization']['candidate_L2_norm'],m['splits'][split]['candidate_specialization']['weighted_contribution_L2_norm']] for k,m in models.items()])
        table(['Variant','Horizon','zero-micro impact','zero-long impact','micro/long','gate mean','std','min','max'],
              [[k,h,m['splits'][split]['impacts']['zero-micro'][str(h)]['mean'],m['splits'][split]['impacts']['zero-long'][str(h)]['mean'],
                m['splits'][split]['micro_long_ratio'][str(h)],*[m['splits'][split]['temporal_specialization']['horizons'][str(h)]['gate'][v] for v in ('mean','std','min','max')]] for k,m in models.items() for h in HORIZONS])
        table(['Variant','Base RPE norm','mean abs QK','mean abs RPE','long norm','micro norm','post-balance micro/long','state Wmicro/Wlong'],
              [[k,*[m['splits'][split]['base_rpe'][v] for v in ('norm','mean_abs_QK','mean_abs_base_bias')],
                *[m['splits'][split]['micro'][v] for v in ('h_long_norm','h_micro_norm','micro_long_norm_ratio','W_micro_W_long')]] for k,m in models.items()])
    table(['Variant','alpha','A matrix','mean diagonal','row entropy','logits drift','effective A drift'],
          [[k,m['transition']['alpha'],m['transition']['A'],m['transition']['mean_diagonal'],m['transition']['row_entropy'],
            m['transition']['transition_logits_drift_L2'],m['transition']['effective_A_change_L2']] for k,m in models.items()])
    lines += ['Argmax occupancy 仅描述。regime health 使用 occupancy≥.99、mean max≥.99、entropy≤.1 三者同时成立的极端集中提示；candidate pairwise 数值用于人工审查 collapse。', '']
    heading('Readout specialization')
    for split in ('VAL','TEST'):
        lines += [f'**{split} — new model**','']
        rows=[]
        m=models[NEW]
        for h in HORIZONS:
            w=m['readouts']['horizons'][str(h)];t=m['splits'][split]['temporal_specialization']['horizons'][str(h)]
            rows.append([h,w['weight_norm'],w['bias_norm'],w['weight_distance_from_init'],w['bias_distance_from_init'],
                         w['distance_from_5d'],w['relative_distance_from_5d'],t['mean_norm'],t['vs_5d']['mean_abs'],t['vs_5d']['cosine'],
                         m['splits'][split]['impacts']['shared-readout'][str(h)]['mean']])
        table(['Horizon','Weight norm','Bias norm','Weight distance init','Bias distance init','Distance 5d','Relative distance 5d','Output norm','Output vs5 mean abs','Cosine vs5','Shared-readout pred impact'],rows)
        for k,m in models.items():
            table(['Variant','Temporal pair','cosine','L1','L2','mean abs'],[[k,p,v['cosine'],v['L1'],v['L2'],v['mean_abs']] for p,v in m['splits'][split]['temporal_specialization']['pairwise'].items()])
        table(['Horizon','Native MAE','Shared5 MAE','Shared5 MSE','Shared5 RMSE','Shared5 Hit%'],
              [[h,models[NEW]['splits'][split]['native_metrics'][str(h)]['MAE'],
                *[models[NEW]['splits'][split]['control_metrics']['shared-readout'][str(h)][v]*(100 if v=='Hit' else 1) for v in ('MAE','MSE','RMSE','Hit')]] for h in HORIZONS])
    table(['Variant','Weight pair','cosine'],[[k,p,v] for k,m in models.items() for p,v in m['readouts']['pairwise_weight_cosine'].items()])
    lines += ['高 weight cosine 不意味着无效；必须结合 temporal 差异及临时 shared-5d-readout intervention。5d 本身仍使用原路径，因此该 intervention 的 5d impact 应为零。', '',f'功能证据原始量与描述阈值：`{r["specialization_evidence"]}`','']
    heading('Raw-horizon gradients: where conflict remains')
    for k,m in models.items():
        g=m['gradients'];lines += [f'**{k}** — {g["method"]}','']
        table(['Module','1d norm','5d norm','10d norm','20d norm','cos(5,1)','cos(5,10)','cos(5,20)'],
              [[name,*[g['norms'][str(h)][name] for h in HORIZONS],*[g['cosines_vs_5d'][name][str(h)] for h in (1,10,20)]] for name in g['norms']['5']])
        table(['Loss','1d readout grad','5d readout grad','10d readout grad','20d readout grad'],
              [[h,*[row[str(v)] for v in HORIZONS]] for h,row in g['readout_loss_gradient_matrix'].items()])
        table(['Geometry','1d norm','5d norm','10d norm','20d norm','cos(5,1)','cos(5,10)','cos(5,20)'],
              [['u-gradient',*[g['u_gradient']['norms'][str(h)] for h in HORIZONS],*[g['u_gradient']['cosines_vs_5d'][str(h)] for h in (1,10,20)]]])
        table(['Readout-coordinate gradient comparison','cos(5,1)','cos(5,10)','cos(5,20)'],
              [['Own ∇W_h L_h',*[g['own_readout_weight_cosines_vs_5d'][str(h)] for h in (1,10,20)]]])
    lines += ['新模型不同 readout 是不同参数空间；上表 own-readout cosine 将相同形状的 weight 梯度按坐标比较，只作描述。LongMemory norm 含 Base RPE，Base RPE 另列；H/Z projection 梯度包含对应共享 LN。', '']
    table(['Shared module','D0B cos(5,1)','D0B cos(5,10)','D0B cos(5,20)','HSR cos(5,1)','HSR cos(5,10)','HSR cos(5,20)'],
          [[name,*[models[k]['gradients']['cosines_vs_5d'][name][str(h)] for k in ('D0B',NEW) for h in (1,10,20)]]
           for name in ('regime evidence','generators combined','Balanced H/Z projections','shared gate_fc','shared head body','final output layer')])
    table(['Shared state input','D0B cos(5,1)','D0B cos(5,10)','D0B cos(5,20)','HSR cos(5,1)','HSR cos(5,10)','HSR cos(5,20)'],
          [['u-gradient',*[models[k]['gradients']['u_gradient']['cosines_vs_5d'][str(h)] for k in ('D0B',NEW) for h in (1,10,20)]]])
    heading('TEST 5d commodity errors')
    for k,m in models.items():
        s=m['splits']['TEST'];lines += [f'**{k}** — commodity MSE CV={s["commodity_MSE_CV"]:.9g}','']
        table(['Commodity','MAE','MSE','RMSE','Hit%'],[[v['name'],v['MAE'],v['MSE'],v['RMSE'],100*v['Hit']] for v in s['commodity_5d']])
        table(['Rank group','Commodity','MSE'],[[group,v['name'],v['MSE']] for group,key in [('Highest 5','highest_MSE_commodities'),('Lowest 5','lowest_MSE_commodities')] for v in s[key]])
    heading('Training history')
    rows=models[NEW]['history'].get('objective_history',[])
    best=models[NEW]['best_epoch']
    table(['Epoch','Best','Train total','Prediction','Switch','Val total descriptive','Val prediction selection','L1','L5','L10','L20','LR'],
          [[v['epoch'],v['epoch']==best,v['train']['total_loss'],v['train']['prediction_loss'],v['train']['switch_loss'],
            v['val']['total_loss'],v['val']['prediction_loss'],*[v['train'][f'raw_L{h}'] for h in HORIZONS],v['lr']] for v in rows])
    heading('Research answers and decision')
    init=r['metadata'].get('initialization',{})
    q=[('1. Only final state readouts added?', 'Yes; extra trainable keys are the three cloned Linear weights/biases.'),
       ('2. Shared initialization equal?',f"PASS={init.get('PASS')}; max diff={init.get('max_abs_diff')}"),
       ('3. Clones exactly copied?',str(init.get('clone_differences'))),
       ('4. Exactly +24,768 parameters?',str(init.get('difference'))),
       ('5. Initial four temporal outputs equal?',str({k:v for k,v in init.get('forward_differences',{}).items() if k.startswith('h_temporal')})),
       ('6. Initial full prediction equal?',str(init.get('forward_differences',{}).get('prediction'))),
       ('7. 5d retains original state_readout?',str(r['metadata'].get('5d_uses_original_state_readout'))),
       ('8. E/H/p/G/Z shared?', 'Yes; dynamics computed once per input batch.'),
       ('9. Original four-horizon Huber?', 'Yes; delta=.02, unit horizon coefficients.'),
       ('10. Original switch KL?', 'Yes; beta_max=5e-4, warmup=20, fixed alpha=.5.'),
       ('11. Readout weights diverge?',str(r['specialization_evidence']['max_weight_relative_distance'])),
       ('12. Functional prediction impact?',str(r['specialization_evidence']['max_counterfactual_impact_over_native_MAE'])),
       ('13. Horizon gates differ?',str({s:{h:v['gate']['mean'] for h,v in models[NEW]['splits'][s]['temporal_specialization']['horizons'].items()} for s in ('VAL','TEST')})),
       ('14. H/Z utilization differentiated?',str({s:models[NEW]['splits'][s]['micro_long_ratio'] for s in ('VAL','TEST')})+'; more differentiation alone is not better.'),
       ('15. VAL5 MAE/MSE improves?',str({k:v for k,v in r['comparison']['VAL']['5'].items() if k.startswith('relative')})),
       ('16. TEST5 MAE/MSE improves?',str({k:v for k,v in r['comparison']['TEST']['5'].items() if k.startswith('relative')})),
       ('17. 10/20d tradeoffs healthy?',str({s:{h:{k:v for k,v in r['comparison'][s][h].items() if k.startswith('relative')} for h in ('10','20')} for s in ('VAL','TEST')})+'; inspect error magnitude, not just sign.'),
       ('18. Shared regime healthy?',str(r['regime_health_no_extreme_concentration'])+'; candidate pairwise table complements the extreme-concentration screen.'),
       ('19. Commodity heterogeneity remains?',str({k:m['splits']['TEST']['commodity_MSE_CV'] for k,m in models.items()})+' (MSE CV; descriptive).'),
       ('20. Final Case?',assessment['case']+' — '+assessment['reason'])]
    table(['Question','Answer'],q)
    lines += [f'Assessment details: `{assessment}`', '',
              'Future perturbation、batch permutation、single-sample 与 within-market relabeling sanity 详见 results.json 的各 model.sanity；包含所有 horizon temporal prefix。', '',
              '所有 counterfactual 仅为临时 forward，checkpoint hashes 与参数未变。没有额外训练 variant、loss 搜索、commodity decoder 或后续模型。', '', '**STOP**', '']
    path.write_text('\n'.join(lines),encoding='utf-8')
