# D0B-HybridGraphPriorHeads

本轮按确认恢复正式 D0B 的 SEQ_LEN=20。唯一结构改动是两个 EdgeAttnMixHop 内部，前4个 heads 使用 content logits，后4个 heads 使用原 `content + prior_scale * log(A.clamp_min(0)+1e-6)`。共8个 heads，不使用 cross-market mask，不增加 QKV 或可训练参数。D0B 与 Hybrid 均为520,549参数，同 seed 的全部 state_dict 张量完全一致。

`graph_prior_heads=None` 保留旧 variant 行为；0和8只用于代码级边界测试。正式实验始终4+4。AdaptiveGraphLearner、MixHop recurrence、TempWeighted、D0B temporal branch、fusion/head和所有训练超参数保持原样。Markov sticky_alpha仍为固定0.5，不能与 graph learner 自身的可训练 alpha 混淆。

## 运行

激活原有 GPU Python 环境，在 Commedities 目录执行一次：

```bash
python -m cmgm.scripts.d0b_hybrid_graph_prior
```

默认要求 CUDA，不会静默退回 CPU。原 main_ablation 入口也已注册：

```bash
python -m cmgm.scripts.main_ablation --variants switching_latent_balanced_hybrid_graph_prior
```

以上两个入口二选一，不要重复启动。checkpoint 默认保存到 `checkpoints`，文件名 `switching_latent_balanced_hybrid_graph_prior_best.pt`。若该文件已存在，正式入口停止，避免覆盖既有实验。原 D0B best checkpoint仅用于重新加载和对照，不用于 Hybrid 初始化。

仅运行实际数据的前置检查（包括原 D0B VAL/TEST 重新评估；不创建 optimizer、不训练）：

```bash
python -m cmgm.scripts.d0b_hybrid_graph_prior --sanity-only
```

训练仍使用 seed42、batch64、epochs200、patience10、Adam、lr1e-4、weight_decay1e-5、Huber .02、四 horizon 求和加原 switching KL。正式 scheduler、early stopping、checkpoint 均使用原 multi-horizon validation loss。每 epoch 额外记录 pooled VAL5 MAE/MSE，以及 secondary best-VAL5 epoch；不保存或评估第二个选择标准的 checkpoint。

## 验证和结果

```bash
python -m pytest tests -k hybrid_graph_prior -q
```

Sanity 检查 cutoff10 的 E/H/p/Z/long/micro/temporal prefix、batch permutation、single-sample，以及原 graph embeddings 和 commodity output rows 一起重标记后的 equivariance。固定数据的 batch 维保持独立，spatial representation仍只使用当前已观察窗口。

注意力统计只在 opt-in diagnostic 中捕获 pre-dropout logits/attention，包含每层、每个 MixHop hop 的 entropy、原图 top-k attention mass、free-free/graph-graph/free-graph cosine与correlation。Free heads无直接 A bias，但后续 hops/layers 的 content state 可以间接受此前 graph-head aggregation 影响。

真实数据可能产生很大的FP32 logits，直接用 `(content+bias)-content` 计算 bias 会有消去误差。代码同时保存原始残差、原公式的精确重组误差和同一操作数的FP64检查；不改 forward dtype、不降低 logits、不改 prior_scale。任一实际公式、free-bias、FP64审计、roundoff-bound 或核心因果/独立性 sanity 失败，停止解释性能。数值审计结果在 REPORT 中逐项展示，不把原始FP32减法错误伪装为小于1e-6。

结果目录：`experiments/d0b_hybrid_graph_prior/<timestamp>`，包含 REPORT.md、results.json、training_history.json、overall_metrics.csv、commodity_metrics.csv、target_magnitude_groups.csv、attention_head_diagnostics.json、gradient_diagnostics.json 和 sanity_checks.json。训练前、epoch1/5/10及正式best记录 graph/QKV prediction-only与total-loss gradients，不做额外 optimizer step。

报告提供原 D0B与Hybrid pooled MAE/MSE/RMSE/unmasked Hit，商品贡献和焦煤/焦炭/燃料油/原油分解、TRAIN-derived P75/P90/P95分组。贡献同时报告净变化、gross改善和gross恶化，避免把负净收益或抵消后的share误读为集中改善。

Case A–F只在训练完成后依据实际结果给出。报告使用显式、固定的描述性约定：TEST MSE相对恶化超过0.1%记为material；attention entropy或top-k mass组间差超过1e-3记为active。它们只影响解释，不参与训练或模型选择；所有原始数值保留。若结果不满足用户列出的任何case，标记需要审阅，不强行宣称成功。单seed实验不等于统计显著性证据。

本轮代码验证不运行正式训练；由用户自行运行。完成这一组4+4实验后STOP，不启动第二比例或其他模型。
