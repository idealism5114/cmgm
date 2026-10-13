# 冻结 D0B：独立预测头与短／长任务偏重监督

本实验先检验直接监督是否形成可验证的任务专长。没有新增 router，不修改上一轮 Ridge，也不修改 D0B、已有 MoE、数据处理或原训练器的默认行为。当前只完成代码、合成验证及真实 checkpoint 元数据审计；正式训练、真实 TRAIN/VAL 特征提取和预测评价均为 **NOT_RUN**，TEST 不存在运行入口。

## 1. 已核验基座和研究边界

当前 Git commit：`b96b90d4cec13b0579f9ca19999f1d52ad9a98a0`，工作区已有未提交修改保留。实际 Git 状态及源码哈希随每次 preflight/prepare/train/summary 保存，不以 commit 代替工作区版本。

通过六组实验的 `file_index.json` 定位独立训练的 D0B：

```text
variant: switching_latent_balanced_readout
seed: 42
best epoch: 85
checkpoint:
checkpoints/d0b_short_long_dynamic_moe/
  switching_latent_balanced_readout_seed42_20261012_204210_677171_776224/best.pt
SHA256: 14a62435709994ca97b2d8fcc1bef65b5625c2a1f9b0c38bb64fadb25577f600
```

配置、原 best epoch、训练历史 argmin、checkpoint 内元数据、文件 SHA256 和 `strict=True` 加载均核验。恢复原 Switch epoch85，对应 beta=.0005；基座的原 KL 代码没有删除。B-dynamic 共同训练后的 z_base 不用于本实验。

已有审计记录的 TRAIN/VAL origins 为1396/268，指纹分别为：

```text
TRAIN 13fb97ef23784a698ec0f263b6a77eae74b368475747d89f620554c3b87c7c6e
VAL   3a25e30575c88a473b624a1978fc6ac3e1d39735696f44f1bff59f71230d1ce5
```

preflight只读取 checkpoint、配置、历史与已有审计，不重新计算真实数据指标。正式 prepare 阶段必须重新核对 TRAIN/VAL-only 缓存的源码、参考配置、数值指纹、商品/资产顺序，并复现 D0B 正常预测；不一致即停止。

## 2. 一次原前向，冻结全部 D0B

`FrozenD0BFeatures` 对原 `gcn_proj`、`lstm_proj` 输出及原 head 的实际输入注册临时 hook，调用一次 **原 D0B forward**，结束后移除 hook。没有重写门控、重跑编码器或读取诊断缓存。

\[
s=gcn\_proj(h_s),\quad t=lstm\_proj(h_t),\quad
z_0=g\odot t+(1-g)\odot s,\quad y_0=Head(z_0),\quad
u=[s\Vert t\Vert z_0].
\]

gate 输入与所有运算顺序来自原 forward。实际投影维度从模块读出并断言，当前三个表示均64维，u为192维。

完整 D0B 的参数 `requires_grad=False`。即使外层 `FrozenD0BExpert.train()`，特征提取器及全部 D0B 子模块仍保持 eval。提取使用 no_grad，得到可供新 Linear 保存为输入的普通 detached tensor；不把 inference tensor直接交给需要反向的 Linear。

正式训练采用不可变的冻结特征缓存：prepare时每个完整 loader batch只运行一次基座；专家 optimizer step读取同一份 u/y0，不再运行主干。这是允许缓存的冻结读出适配，**不是端到端训练**。不会重新拟合 VAL 标准化，不增加原始特征。

## 3. 三个独立残差预测头

G、S、L均为：

```text
Linear(192,32) → GELU → Linear(32,64) → GELU → Linear(64,96)
```

最后 Linear 的 weight、bias均为0，其后无激活。没有 Dropout、LayerNorm、router、混合参数或新辅助loss。

\[
\hat Y_k=y_0+F_k(u),\qquad F_k(u)\rightarrow(B,4,24).
\]

输出按 horizon-major 顺序 `[1,5,10,20]` × 原24商品顺序 reshape。每个专家独立预测完整四周期；残差直接加到收益率上，不送回基座非线性 head。

构造一次 seed42 的专家初始 state_dict，显式复制到 G/S/L，检查逐张量差、状态哈希及独立参数对象。正式 prepare保存唯一 `expert_initial_state.pt`，每次独立训练 strict加载该初值。

| 参数 | 数量 |
|---|---:|
| 冻结 D0B | 520,549 |
| 每个新增专家可训练参数 | 14,528 |
| 单个在线适配器总参数 | 535,077 |
| 三专家可训练参数合计 | 43,584 |
| 一份基座与三专家合计 | 564,133 |

每专家三层分别为6,176、2,112、6,240参数。正式训练只创建包含当前专家参数的 Adam；没有基座梯度或 KL 优化。

## 4. 直接监督与选择标准

令 L_h为该周期 batch×商品平均 Huber，delta从原配置读取，当前.02。标签保持累计收益率小数：`clip(p[t+h]/max(abs(p[t]),1e-8)-1,-1,1)`。不使用log-return、周期尺度归一化或新标签。

\[
L_s=(L_1+L_5)/2,\quad L_l=(L_{10}+L_{20})/2,
\]
\[
L_G=\sum_hL_h,\quad
L_S=\frac4{1+\epsilon}(L_s+\epsilon L_l),\quad
L_L=\frac4{1+\epsilon}(\epsilon L_s+L_l),\quad\epsilon=.1.
\]

四周期系数：

```text
G: [1, 1, 1, 1]
S: [20/11, 20/11, 2/11, 2/11]
L: [2/11, 2/11, 20/11, 20/11]
```

各组系数和为4，全部正值；epsilon=1严格退化为原四周期损失求和。新训练/选择损失不加入冻结基座的 Switch KL常数，原模型KL代码保持原样。

沿用 seed42，Adam lr=1e-4、WD=1e-5，batch64，最多200epochs，patience10，ReduceLROnPlateau factor=.5、patience5，TRAIN shuffle=False/drop_last=True，完整评价drop_last=False。没有混合精度、梯度裁剪、搜索或新随机性策略。

三次训练均以 **global四周期 prediction-only VAL Huber的原batch-mean** 监控 scheduler和早停；反向分别使用G/S/L目标。每次沿同一轨迹同时保存 global、short-weighted、long-weighted 三个最佳 checkpoint。short/long选择器使用上述epsilon=.1的四周期加权目标，**不是只看两个周期，也不是MAE@5d选优**。

预定主 checkpoint：G-global、S-short、L-long。匹配对照固定为S-short vs G-short、L-long vs G-long。G-short/G-long来自同一次G训练，没有增加G训练次数。三个选择器只能在global早停的已观察轨迹内选，不能延长某个专家的预算。

原batch-mean继续用于选择。历史另外记录 sample-mean的float32 batch项聚合；最终能力矩阵则由完整预测以float64离线计算样本等权组内Huber。字段分别命名，不混用。

## 5. 运行命令

均在 `/home/yangxiaotong/projects/myresearch/Commedities` 仓库根目录执行。每行是单独一条命令。

默认只做合成preflight和checkpoint审计：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts
```

可选：先准备共享冻结输入与初值（只做真实TRAIN/VAL冻结推理，不训练）：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts --mode prepare --study experiments/d0b_supervised_task_experts/seed42_supervised_v1 --device cuda:0
```

三条独立正式训练命令；单GPU建议依次执行，等待前一个结束：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts --mode train --expert G --study experiments/d0b_supervised_task_experts/seed42_supervised_v1 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts --mode train --expert S --study experiments/d0b_supervised_task_experts/seed42_supervised_v1 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts --mode train --expert L --study experiments/d0b_supervised_task_experts/seed42_supervised_v1 --device cuda:0
```

首次train可自动完成prepare，其余两组复用相同READY缓存和初值。完成且审计一致的专家跳过；中断、失效、源码/配置漂移或checkpoint目录冲突停止复核，不自动重跑、不用权重热启动伪装续训。不要同时让多个进程初始化同一个未准备目录；需要并行时先完成prepare，本轮推荐顺序运行。

三组完成后汇总checkpoint与能力矩阵：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_supervised_task_experts --mode summarize --study experiments/d0b_supervised_task_experts/seed42_supervised_v1
```

可用 `--file-index` 明确指定原六组file_index，`--data-cache` 指定经过核验的TRAIN/VAL缓存。自动定位发生冲突即停止，不按修改时间猜测。不存在TEST开关。更换study路径用于独立实验，不覆盖旧study。

## 6. 输出和固定组合

共享目录：`experiments/d0b_supervised_task_experts/seed42_supervised_v1/`，保存config、Git/源码哈希、基座审计、初始化审计、冻结特征、初始state、准备状态与数据指纹。

每组结果：该目录下 `G/`、`S/`、`L/`，保存config、逐epoch完整training_history、三个选择器的best epoch/loss、完整TRAIN/VAL预测与指标、梯度诊断、训练时间、结果和文件索引。训练耗时与共享prepare耗时分开记录，不与原D0B完整训练预算混为一谈。

checkpoint：

```text
checkpoints/d0b_supervised_task_experts/seed42_supervised_v1/{G,S,L}/
    global_best.pt
    short_best.pt
    long_best.pt
```

这些只存新增专家state及原基座身份/数据/初值/源码元数据，不复制或覆盖原D0B checkpoint。所有选择的checkpoint均strict恢复并完整评价；只有历史、选优、有限指标和产物核验通过才标COMPLETE。

summary输出新timestamp目录，六种策略：D0B、G-global完整输出、S-short完整输出、L-long完整输出、G-short/G-long按组拼接、S-short/L-long按组拼接。拼接只选对应周期行，不用标签、不训练任何融合权重。

输出 `summary_report.md`、`expert_capability.csv`（所有九个checkpoint×两组能力）、`strategy_comparison.csv`、`metrics.json`、`checkpoint_index.json`（SHA/epoch/准则）、`strategy_predictions.npz`、`audit.json` 和 `file_index.json`。MAE、RMSE和Hit遵循原总体定义，Hit原值为比例，展示用百分数。每组改善率均相对同组G-global计算，不比较短长不同尺度的绝对loss。

## 7. 实际验证与未执行事项

已执行11项合成测试（含参数化loss检查）：真实原D0B单次空间/时间前向；冻结eval与零残差逐项等价；u顺序与形状；独立初值及无参数共享；原horizon-major顺序；epsilon=1原loss等价；跨组全部输出有梯度；第一步只输出层非零梯度，后续内部层可学习；optimizer范围和基座无更新；临时checkpoint恢复；六种label-free策略；合成三次独立短训练、九checkpoint导出和汇总接口。

默认preflight已核验真实独立D0B checkpoint、实际参数量和初值审计，但未运行真实TRAIN/VAL特征提取、真实专家训练、bootstrap、TEST或真实能力评价。合成随机数据的误差不是方案效果证据。

## 8. 解释规则

- S-short在短组优于匹配G-short、L-long在长组优于匹配G-long，才支持监督形成专长；名字/权重/梯度不算专长证据。
- 固定任务组合须超过通用分组对照，才能讨论超出普通新增预测头的任务收益。
- 固定组合不超过D0B时，不支持替换基线。若单个专家四周期均最好，解释为更强的通用候选，不要求其他周期故意变差。
- MAE@5d仍为重点，同时完整报告四周期，不临时切换checkpoint标准、不挑选最好辅助方案。
- 这是已训练基座上的额外预测读出适配，额外训练预算和新增参数不能被称为纯架构因果效应。VAL已参与此前研究，因此评价属于探索性；没有独立确认性数据的声明。
- 不承诺收益，不运行Oracle或router，不根据结果自动改模型、训练更多seed或访问TEST。
