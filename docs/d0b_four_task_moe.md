# D0B 四预测任务条件化 MoE 融合

**状态：`NOT_RUN`。** 已完成代码和合成验证；未启动真实训练，未读取 TEST 指标。实现时工作树的 Git commit 为 `3b0d13463d03a1d0fccf0107a06292b64dfd5018`，且存在先前未提交修改；每次正式运行会另存实际源码哈希与工作树状态。

## 审计后的数据流

正式 D0B 输入是 `(B,20,N,21)`，输出是四周期、24 商品的 `(B,4,24)`；周期顺序 `[1,5,10,20]`。空间分支给出 `h_spatial∈R⁶⁴`。时间分支在一次前向中已经计算长期 Transformer 状态 `H∈R^{B×20×128}`、递推 latent 状态 `Z∈R^{B×20×64}` 和原 balanced readout `h_temporal∈R⁶⁴`。原代码只缓存了断开梯度的 `H/Z` 诊断副本；新接口 `return_live_sequences=True` 直接返回本次前向的实时 `H/Z`，默认 D0B 路径不变。空间与时间分支各执行一次。

原 `gcn_proj`、`lstm_proj` 仍将两个全局分支表示投影到 `s,t∈R⁶⁴`。四个专家每个样本只计算一次；专家名称表示输入和结构假设，不预设最佳周期：

| 专家 | 固定公式及输入 | 参数 |
| --- | --- | ---: |
| 近期变化 | `ReLU(W[Z_last, Z_last − mean(Z_last5)])`，`W:128→64` | 8,256 |
| 长期记忆 | `ReLU(W[H_last, mean(H_all20)])`，`W:256→64` | 16,448 |
| 跨市场结构 | `s + ReLU(W_s s)`，`W_s:64→64` | 4,160 |
| 低秩时空交互 | `t + W_o[(U_s s) ⊙ (U_t t)]`，秩 16，三个矩阵均无 bias | 3,072 |

四专家合计 **31,936** 参数。全局静态对照只有一组 4 维 logits；主模型有四组独立的 4 维 logits，分别对应 1、5、10、20 天。对任一周期 `h`：

`π_h = softmax(θ_h), z_h = Σ_k π_{h,k} E_k`。

权重只依赖预测任务索引、在样本间共享；这属于**任务条件化静态路由**，不是样本级动态路由。无固定专家分配、无路由差异正则、无 warm-up、无专家均衡损失。

`z_h` 经原预测头的同一个 `Linear(64,64)→ReLU→Dropout(0.3)` 主体，随后只使用原最终 `Linear(64,96)` 中周期 `h` 对应的 24 个输出行。FiLM 消融在该共享主体之后、最终输出层之前加 `body_h×(1+γ_h)+β_h`，其中 `γ_h=β_h=0` 初始化，故初始与未加 FiLM 的主模型输出一致；未复制预测头。

## 固定六组对照与参数量

下表按当前 `N=284`、24 商品配置实例化得到；每组全部参数均参与训练。新四专家全局静态模型与四任务模型共享**完全相同**的专家、分支、投影和预测头，是控制专家容量的直接对照。

| 变体 | 路由／头 | 可训练参数 |
| --- | --- | ---: |
| `switching_latent_balanced_readout` | 原 D0B 门控 | 520,549 |
| `switching_latent_balanced_task_moe_global` | 原三专家，全局静态 | 541,352 |
| `switching_latent_balanced_task_moe_per_horizon` | 原三专家，四任务静态 | 541,361 |
| `switching_latent_balanced_four_task_moe_global` | 新四专家，全局静态 | 544,233 |
| `switching_latent_balanced_four_task_moe_per_horizon` | **主模型**：新四专家，四任务静态 | 544,245 |
| `switching_latent_balanced_four_task_moe_per_horizon_film` | 主模型＋恒等初始化 FiLM | 544,757 |

旧三专家 checkpoint 只作为历史背景，不参与这六组训练或模型选择。原旧 MoE 模块目前已不在工作树中；原三专家对照使用当前可实例化的匹配实现，结构和历史 checkpoint 的参数形状已核对。相同 seed 下，六组保留的 D0B 共享参数逐张量初始化一致；新四专家的三种对照也共享完全相同的专家初值。

## 训练与预先固定的 VAL 规则

沿用当前正式 `build_data` 和 `train`：20 日输入、21 特征、return 目标、四周期等权 Huber(`δ=.02`) 加原生 Switch KL；Adam `lr=1e-4, weight_decay=1e-5`，batch 64，最多 200 epoch，早停 patience 10，ReduceLROnPlateau factor 0.5、patience 5。TRAIN 不打乱、`drop_last=True`；完整 TRAIN/VAL 评价 `drop_last=False`。Checkpoint、scheduler、早停继续按**四周期 prediction-only VAL Huber 的既有 batch-mean 口径**选择，不用 5d MAE 选 epoch。

跨模型的唯一主筛选指标是 **VAL MAE@5d**。主模型只有严格优于本实验 seed42 的 D0B 和“新四专家＋全局静态路由”，才可安排后续 TEST；1、10、20 日 VAL MAE 全量报告，但不追加临时门槛，也不因其变化改模型。FiLM 是独立消融，不参与主模型的这项准入判定。运行后 `--compare-results` 会检查六组的 seed、TRAIN/VAL 数据指纹、协议、源码哈希、checkpoint SHA256，再执行固定规则。

对每个静态路由模型记录四周期的四专家／三专家权重、熵及六对周期路由的 Jensen–Shannon 散度（nat）。对四任务路由模型，在冻结 best checkpoint 后只在 VAL 上逐对交换周期路由，记录四周期指标及相对原预测的 MAE 变化。若四组权重接近，即使 MAE 下降，也不能声称学出了预测周期分工；交换诊断也不是重新训练的消融。

## 运行命令

在仓库根目录 `/home/yangxiaotong/projects/myresearch/Commedities` 执行。默认只跑合成预检：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe
../.venv/bin/python -m pytest -q tests/test_four_task_moe.py
```

以后手动启动 seed42 六组正式训练时，单张 GPU 请依次运行：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_readout --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_task_moe_global --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_task_moe_per_horizon --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_four_task_moe_global --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_four_task_moe_per_horizon --seed 42 --device cuda:0
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --run --variant switching_latent_balanced_four_task_moe_per_horizon_film --seed 42 --device cuda:0
```

完成六组后，将各自实际 `result.json` 路径依次传给：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_four_task_moe --compare-results /path/D0B/result.json /path/three_global/result.json /path/three_four_task/result.json /path/four_global/result.json /path/four_four_task/result.json /path/four_film/result.json
```

每组使用独立 `experiments/d0b_four_task_moe/<variant>_seed42_<timestamp>/`，保存 `config.json`、`data_audit.json`、`initialization_audit.json`、`training_history.json`、`metrics.json`、`routing_diagnostics.json`、`val_route_swaps.json`、`result.json`。对应 checkpoint 位于 `checkpoints/d0b_four_task_moe/<同名>/best.pt`。`result.json` 中 `metrics.train/val.<h>` 记录 MAE/MSE/RMSE/Hit；收益率及 MAE 使用小数尺度，Hit 存比例。还记录四周期 VAL Huber、best epoch、训练秒数、参数量及 checkpoint SHA256。全部文件标记 `test_evaluated=false`。

若 VAL 门槛通过，未来才冻结主模型和同 seed D0B 的 checkpoint 与评估清单，在统一 TEST 口径下比较 TEST MAE@5d。历史上 TEST 已被多次查看，故该比较必须标为**探索性**；确认性结论需未来未参与选择的时间区间。本次没有启动真实训练或执行 TEST。

## 本次验证

合成测试验证实时 H/Z 未断开梯度、时间及空间分支各执行一次、四组路由和四专家梯度、原 D0B 路径与共享参数初始化、周期输出行映射、全局／四任务初始等价、FiLM 初始恒等及其梯度、周期路由交换、原生 Switch KL，以及既有训练器的一轮合成训练、严格 checkpoint 恢复。正式预测表现仍为 `NOT_RUN`。
