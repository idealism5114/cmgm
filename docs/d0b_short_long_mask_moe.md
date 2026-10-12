# D0B 短／长周期任务掩码 MoE

**状态：`NOT_RUN`。** 已完成实现与合成验证。本文没有真实 TRAIN/VAL 结果，也没有读取 TEST。当前工作树存在此前未提交修改；每次正式运行会记录当时 Git commit、工作树状态和关键源码 SHA256。

## 已核对的数据流

原 D0B 输入 `(B,20,N,21)`，预测顺序 `[1,5,10,20]`、形状 `(B,4,24)`，目标是小数收益率。空间分支产生 `h_spatial(B,64)`；时间分支一次前向产生 balanced readout `h_temporal(B,64)`，并能通过现有 `return_live_sequences=True` 同时返回带梯度的 long-memory `H(B,20,128)` 与 latent states `Z(B,20,64)`。该接口不读取 `last_long_memory` 等已 `detach` 的诊断副本，不重算编码器。两个原投影保持 `s=gcn_proj(h_spatial)`、`t=lstm_proj(h_temporal)`，均为 `(B,64)`。

五个专家每次只计算一次，输出均为 64 维并各自使用相同形式的 `LayerNorm(64)`，以减轻表示尺度差异。瓶颈宽度固定为 16，低秩乘性交互秩固定为 8；本次没有宽度搜索。

| 专家 | 输入与公式（`LN` 为专家输出归一化） | 参数量 |
|---|---|---:|
| `E0` 共享时空 | `LN((s+t)/2 + W_o[(U_s s)⊙(U_t t)])`；两支 rank-8 投影 | 1,664 |
| `ES1` 短期记忆 | `LN(MLP16([Z_last, mean(Z_last5), Z_last−mean(Z_last5)]))` | 4,304 |
| `ES2` 短期冲击 | `LN(MLP16([s, ΔZ_5, abs(s−ΔZ_5)]))`，`ΔZ_5=Z_last−mean(Z_last5)` | 4,304 |
| `EL1` 长期记忆 | `LN(MLP16([H_last, mean(H_all20), H_last−H_first]))` | 7,376 |
| `EL2` 长期结构 | `q=W_h mean(H_all20)`；`LN((s+q)/2+W_o[(U_s s)⊙(U_h q)])`，rank 8 | 9,920 |

专家合计 **27,568** 参数。所有新五专家变体还共享一个融合后 `LayerNorm(64)`，共 128 参数；旧 `gate_fc` 从新变体中删除，且没有闲置可训练门控参数。

## 固定任务掩码与三种匹配路由

预先固定短组 `{1,5}`、长组 `{10,20}`。主模型仅学习 `theta_short(2)` 和 `theta_long(2)`，初始化全零，故初始组内权重均为 `[0.5,0.5]`：

```text
pi_short = softmax(theta_short)
pi_long  = softmax(theta_long)
z_short  = LN_f(E0 + pi_short[0]*ES1 + pi_short[1]*ES2)
z_long   = LN_f(E0 + pi_long[0]*EL1 + pi_long[1]*EL2)
```

短组不能访问 `EL1/EL2`；长组不能访问 `ES1/ES2`。路由只取决于任务组，**不是样本条件化路由**。组内固定等权对照使用同一表达式但无可训练 logits。新五专家全局对照用一组四维 logits，对 `ES1/ES2/EL1/EL2` 做 softmax，所有周期共用 `LN_f(E0 + Σπ_k E_k)`。这一个对照不施加任务掩码，用于区分“新专家本身”和“掩码／任务分组”。

`z_short` 分别给 1/5 日输出行、`z_long` 分别给 10/20 日输出行。原共享预测头 `Linear(64,64)→ReLU→Dropout(0.3)→Linear(64,96)` 不变；按 horizon-major 顺序选择最终层相应 24 行，不复制预测头。

## 固定对照与参数审计

| 模型变体 | 作用 | 当前 284 节点配置可训练参数 |
|---|---|---:|
| `switching_latent_balanced_readout` | D0B 原门控 | 520,549 |
| `switching_latent_balanced_task_moe_global` | 此前三专家＋全局静态路由，本轮需同协议运行 | 541,352 |
| `switching_latent_balanced_short_long_moe_global` | 新五专家＋全局静态路由 | 539,993 |
| `switching_latent_balanced_short_long_moe_mask_equal` | 新五专家＋任务掩码／组内固定等权 | 539,989 |
| `switching_latent_balanced_short_long_moe_mask_learned` | 新五专家＋任务掩码／组内可学习权重（主模型） | 539,993 |

新三组的五专家、融合归一化层及所有 D0B 公共参数在相同 seed 下逐张量初值一致。原 D0B 的 state_dict、前向和 checkpoint 行为不变。此前四任务 MoE 的 seed42 结果只能作为设计背景，不能当成本轮匹配对照。

## 训练协议和预设解释

沿用原 `build_data`、`train` 和 evaluator：`SEQ_LEN=20`，21 特征，return 标签，四周期等权 Huber(`δ=.02`) 加原生 Switch KL（`beta_max=5e-4`，warmup 20 epoch）；Adam `lr=1e-4, weight_decay=1e-5`，batch 64，最多 200 epoch，早停 patience 10，ReduceLROnPlateau factor 0.5、patience 5。TRAIN 不 shuffle 且 `drop_last=True`；完整 TRAIN/VAL 评价 `drop_last=False`。checkpoint、scheduler、早停使用既有 **prediction-only 四周期 VAL Huber 的 batch-mean 口径**。本模型无 MoE auxiliary loss、路由 warm-up 或任务差异正则。

跨模型主指标预先固定为 **VAL MAE@5d**。同时完整报告 1/10/20 日 VAL MAE、四周期 VAL Huber、best epoch、参数量与训练秒数。若主模型优于 D0B 而不优于五专家全局对照，应将可能收益归于专家设计，不能归于任务路由；若不优于三专家全局对照，也须如实报告。没有临时性能阈值或改动 checkpoint 选择规则。

冻结最佳 checkpoint 后，在 VAL 上记录各组路由权重与熵、五专家输出的**逐样本 L2 范数再跨样本取均值**，并执行两组权重互换和逐个专属专家置零的只读诊断。组权重互换仍保留任务掩码，即交换两组的二元权重而不交换专家集合。移除专家诊断不重新训练，且不能当作训练后结构消融。

## 可直接运行的命令

在仓库根目录执行。以下预检只使用合成输入，不读取真实数据：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe
../.venv/bin/python -m pytest -q tests/test_short_long_mask_moe.py
```

以后手动运行 seed42 五组配对训练；单张 GPU 应顺序执行，每条命令均独立创建结果与 checkpoint 目录：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe --run --variant switching_latent_balanced_readout --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe --run --variant switching_latent_balanced_task_moe_global --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe --run --variant switching_latent_balanced_short_long_moe_global --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe --run --variant switching_latent_balanced_short_long_moe_mask_equal --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_mask_moe --run --variant switching_latent_balanced_short_long_moe_mask_learned --seed 42 --device cuda:0
```

五组运行后，使用各组实际生成的 `result.json` 调用 `--compare-results`，并按路径依次传入 D0B、三专家全局、五专家全局、掩码等权、掩码可学习五个文件。汇总器拒绝数据指纹、源码哈希、协议、seed 或 checkpoint 哈希不一致的结果。不得仅按旧目录中的同名模型复用历史数据。

结果目录：`experiments/d0b_short_long_mask_moe/<variant>_seed42_<timestamp>/`。每组保存 `config.json`、`data_audit.json`、`initialization_audit.json`、`training_history.json`、`metrics.json`、`routing_diagnostics.json`、`expert_norms.json`、`val_frozen_interventions.json`、`result.json`；checkpoint 保存至 `checkpoints/d0b_short_long_mask_moe/<同名>/best.pt`。`result.json` 的 `metrics.train/val.<h>` 是 MAE/MSE/RMSE/Hit（Hit 存比例），并记录 VAL Huber、参数量、best epoch、训练秒数、路由及 VAL 诊断。全部标记 `test_evaluated=false`。

本次只交付实现与合成验证。未来若根据 VAL 选择模型与 checkpoint 后另行评估 TEST，由于此前项目已多次查看 TEST，应标记为**探索性**，不能称完全独立确认。
