# D0B 预测任务条件化三专家融合（待正式训练）

状态：`NOT_RUN`。当前只完成实现、静态检查及合成数据预检；未读取 TEST 指标，未运行真实训练。

## 模型与公平对照

四个新变体均使用原 D0B 的空间分支、Switching Latent 时间分支、`gcn_proj`、`lstm_proj` 和同一个预测头。每个样本的两个分支只计算一次。投影后 `s,t∈R⁶⁴` 进入共享的三个专家：`E_T(t)=t+MLP₆₄→₆₄→₆₄(t)`、`E_S(s)=s+MLP₆₄→₆₄→₆₄(s)`、`E_ST([s,t])=MLP₁₂₈→₆₄→₆₄([s,t])`。三个专家的隐藏激活为 ReLU，Dropout 为 0.1，ST 专家无残差。

当前工作树已无旧 `moe_fusion.py`，主模型中的旧三专家代码分派不能实例化。本实现核对了历史三专家 checkpoint 的参数形状：T/S 各为 `64→64→64`，ST 为 `128→64→64`，内容 router 为 `128→32→3`；新对照采用这些结构定义，旧 checkpoint 不参与初始化或匹配对照。新内容 router 保留 PyTorch 默认 Linear 初始化，静态 logits 初始化为零。
四个匹配 MoE 在构造完专家后消耗同一段 router 初始化随机流；静态对照随后删除不使用的内容 router，故专家初始化和进入训练前的 PyTorch RNG 状态保持一致，且不留下闲置可训练 router 参数。

| 变体 | 路由定义 |
| --- | --- |
| `switching_latent_balanced_task_moe_global` | 所有样本、四周期共用 `softmax(θ_global)` |
| `switching_latent_balanced_task_moe_content` | 原始 `[h_spatial,h_temporal]` 经 `128→32→3`，每样本一组权重，四周期共用 |
| `switching_latent_balanced_task_moe_short_long` | 短周期 `{1,5}` 用 `softmax(θ_short)`；长周期 `{10,20}` 用 `softmax(θ_long)` |
| `switching_latent_balanced_task_moe_per_horizon` | 每周期独立一组静态 3 维 logits |
| `switching_latent_balanced_readout` | 原 D0B 逐通道门控基线；不改动 |

后三个静态路由只依赖任务索引，不读取样本内容。它们是**任务条件化静态路由**，并不表示样本级动态专家选择。所有 MoE 对照共享完全相同的专家和预测头实现，没有旧三专家 MoE 的 warm-up 或均衡损失。每个周期的 `z_h=Σ_k π_{h,k}E_k` 经过同一 `head[0:3]`；原 `head[3]` 中第 `h×24:(h+1)×24` 行仅输出该周期的 24 商品。输出固定 `(B,4,24)`，周期顺序 `[1,5,10,20]`，商品顺序由数据审计保存。

实际实例化 `N=284` 时，可训练参数量依次为：D0B `520,549`，全局静态 `541,352`，内容路由 `545,576`，短/长任务 `541,355`，逐周期静态 `541,361`。各实验结果会再次记录实际数据节点数下的参数量。

训练入口使用项目现有 `build_data` 与 `train`：四周期 Huber(`δ=.02`) 求和，原生 Switch KL（`β_max=5e-4`、20 epoch warm-up）；Adam `lr=1e-4, weight_decay=1e-5`，batch 64，最多 200 epoch，早停 10，ReduceLROnPlateau factor 0.5、patience 5。TRAIN `shuffle=False,drop_last=True`；TRAIN/VAL 完整指标覆盖全部样本。仍用**四周期 prediction-only VAL Huber 的现有 batch-mean 口径**选择 checkpoint、调度和早停，VAL MAE@5d 只用于事先固定的跨模型判定。

## 事先固定的 VAL 判定

比较仅使用同一数据指纹、同一 seed 42、同一源码哈希和协议的五个完整结果。短/长任务 MoE 要进入后续探索性 TEST，必须同时满足：

1. `VAL MAE@5d` **严格小于** D0B 与全局静态 MoE；
2. 相对 D0B，10d 和 20d 的 VAL MAE **不能同时**各增加超过 **2%**。

这个 2% 阈值固定于正式运行前的 `VAL_GATE`；它不是统计显著性标准。若两组专家权重接近，即使预测改善，也不得声称学出了明显长短周期专家分工。最终在 VAL 记录短/长权重、路由熵、Jensen–Shannon 散度；冻结 best checkpoint 后仅作诊断地交换两组路由、逐一强制单专家推理。强制单专家可能产生分布偏移，不能替代重新训练的消融。

## 手动运行命令

在仓库根目录 `/home/yangxiaotong/projects/myresearch/Commedities` 执行。默认命令只用合成数据：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe
../.venv/bin/python -m pytest -q tests/test_task_conditioned_moe.py
```

正式 seed 42 的五个**独立训练命令**如下；单 GPU 请顺序运行，不要并发：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --run --variant switching_latent_balanced_readout --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --run --variant switching_latent_balanced_task_moe_global --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --run --variant switching_latent_balanced_task_moe_content --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --run --variant switching_latent_balanced_task_moe_short_long --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --run --variant switching_latent_balanced_task_moe_per_horizon --seed 42 --device cuda:0
```

每次运行生成独立 `experiments/d0b_task_conditioned_moe/<variant>_seed42_<timestamp>/`，含 `config.json`、`data_audit.json`、`initialization_audit.json`、`training_history.json`、`metrics.json`、`routing_diagnostics.json`、`val_frozen_interventions.json`、`result.json`。Best checkpoint 独立存于 `checkpoints/d0b_task_conditioned_moe/<同名>/best.pt`。`result.json` 的 `metrics.val.<h>` 包含 MAE/MSE/RMSE/Hit（Hit 存比例），并记录 best epoch、四周期 VAL Huber、参数量、耗时和 checkpoint SHA256；不含 TEST 指标，`test_evaluated=false`。初始化审计保护 D0B 分支、投影和头的逐张量一致性。

五个 `result.json` 都完成后，可用 `--compare-results` 后面依次传入五个实际文件路径，生成 `seed42_val_comparison.json`。该命令会验证 seed、数据、协议、源码和 checkpoint 哈希，执行上述 VAL 门槛，不读取 TEST。示例中的占位符须替换为真实路径：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_task_conditioned_moe --compare-results /path/D0B/result.json /path/global/result.json /path/content/result.json /path/short_long/result.json /path/per_horizon/result.json
```

若 VAL 门槛通过，后续才制定 seed 42/43/44 的配对 TEST 清单：三个 seed 对五个变体采用各自按 VAL Huber 选出的 best checkpoint，冻结 15 个 checkpoint 路径与哈希后，统一 TEST 推理；报告每 seed 的 TEST MAE@5d、四周期 MAE 及跨 seed 均值/样本标准差。原项目 TEST 已多次被查看，因此此计划只能是**探索性验证**。确认性结论需要未来未参与模型选择的时间区间。本次没有实现自动 TEST 评价，也没有启动该计划。

## 本次合成验证

`tests/test_task_conditioned_moe.py` 验证五变体形状与有限值、原 D0B 严格 state_dict 往返和预测不变、全部保留参数初始化一致、共同权重时短/长路由与全局路由预测相同、按周期取头输出行与完整头结果一致、路由归一化、两组 logits 与三专家梯度、四周期 Huber 加原生 KL、严格 checkpoint 恢复以及冻结模型路由干预。正式预测性能：`NOT_RUN`。
