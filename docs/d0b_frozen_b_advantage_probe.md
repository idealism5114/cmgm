# 冻结 B 专家的相对优势预测探针

入口：`cmgm.scripts.d0b_frozen_b_advantage_probe`。本工具只拟合两个小型 Ridge；不会训练、微调原模型，也没有 TEST 入口。不会改变默认模型。

## 数据与冻结模型

通过上一轮完成的 `diagnostic_file_index.json` → 审计 → 原六组 `file_index.json` 定位 B-dynamic seed42 epoch81 的实际 checkpoint；不按修改时间选模型。原配置、checkpoint 哈希、历史选优记录和恢复来源必须一致，使用 `strict=True`，恢复最佳 epoch 对应的 Switch 调度，冻结全部参数并使用 `eval/inference_mode`。

加载上一轮指纹审核过的 **TRAIN/VAL-only** 数组缓存，重新核对原参考配置、数据源码哈希、特征/价格数值指纹、商品顺序。使用完整 loader，`shuffle=False, drop_last=False`；不重新拟合 VAL 预处理。正常预测、三候选预测和总体指标必须先复现，再进入探针拟合。保留旧结果、checkpoint，检查其前后 SHA256 和模型 state_dict 哈希不变。

## 特征与监督

从真实 router 两个 MLP 的 `forward_pre_hook` 捕获它们实际接收的同一向量：

\[
u_i=[LN_s(s_i)\Vert LN_t(t_i)]\in\mathbb R^{128}.
\]

LN 参数来自冻结 checkpoint。不构造另一套归一化，不增加日期或市场特征。输出逐维 TRAIN/VAL 标准差以及近零方差维度数。预测、标签、日期和商品顺序逐项核对。

原 Huber delta 从配置读取（本次为 .02）。float64 离线计算：

\[
L_{i,g,k}=mean_{h\in g,c}Huber_\delta(\hat y_{i,h,c,k}-y_{i,h,c}),\qquad
d_{i,g,k}=L_{i,g,k}-L_{i,g,0},\quad k=1,2.
\]

负损失差表示相对共享专家更好。两组分别为 short=[1,5]、long=[10,20]；组内 Huber 使用样本等权，不能直接当作原历史四周期求和、batch-mean 选模损失。

## TRAIN 内时间选择

固定 alpha `[0.001,0.1,1,10,100]`，两个输出、两组独立选 alpha。三个扩展折边界为 TRAIN origin 数量的 50%、66%、83%、100%。根据实际 dataset 的交易行语义 `origin_row=i+seq_len-1` 和 `label_end=origin_row+h`，保留 `label_end < validation_first_origin_row` 的训练行，等号也清除；short 最大周期5，long 最大周期20。日期只是对应交易行的可追溯标记，不按自然日做加减。

每折输入/目标 StandardScaler 和标准差不超过 `1e-12` 的输入维度处理都只在清除后的折训练数据拟合。Ridge 使用确定性的 SVD 求解。CV 评分为原损失差尺度的双输出 MSE，按验证 origins 数量加权；相同分数取候选表中先出现的 alpha。完整 TRAIN 重拟合后将参数、scaler、掩码、温度和固定选择写入 `TRAIN_LOCKED.json`，随后才计算 VAL 探针评价。

注意：B 预测模型已经见过整个 TRAIN，因此这不是整个预测系统无泄漏的交叉验证；它只用于探针选择。VAL 也已经用于此前探索。

## 选择与表示融合

主策略为 `argmin([0,d_hat_state,d_hat_interaction])`；绝对差不超过 `1e-12` 时取较小编号。一个 origin 的同组两个周期共用选择。核验其结果与真实表示融合位置的一次 one-hot 注入一致。

辅助策略使用 `softmax(-scores/tau_g)`，其中 `tau_g=sqrt(mean_TRAIN(d_g**2))`，包括两种损失差。温度不搜索、不使用 VAL；近零温度用均匀权重并记录退化。

软权重注入原 B 模块：

\[
z_g=z_{base}+\pi_{g,1}\Delta_{g,1}+\pi_{g,2}\Delta_{g,2}.
\]

之后使用原非线性预测头和周期对应输出行。**不是三个最终预测的加权平均。** 主干 live forward 产生的融合输入在冻结 eval 下缓存并复用，减少重复编码。

比较原 B、原 B TRAIN平均路由、TRAIN最佳固定候选、常量优势硬选择、Ridge硬选择、Ridge软路由、Ridge软路由TRAIN平均权重、整对短长权重固定 seed42 打乱、标签事后理想选择，以及经过来源审核的 D0B/三专家全局静态。另列常量优势软路由作为辅助。不会按 VAL 选择最好的辅助策略替换主策略。

理想选择使用相应 split 的标签，不能部署，也不是所有软融合方法的理论性能上界。B 的共享候选是共同训练后的 z_base，不等于独立 D0B。

## 命令

在仓库根目录运行，默认只做合成检查：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_frozen_b_advantage_probe
```

已授权的真实冻结评估和小型线性拟合（每次生成新目录）：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_frozen_b_advantage_probe --run --device cuda:0 --diagnostics experiments/d0b_short_long_dynamic_moe/expert_diagnostics_20261013_144741_463264
```

`--diagnostics` 也可传 `diagnostic_file_index.json`。默认自动定位唯一完整诊断；有冲突则停止，显式提供索引，不猜最新运行。`--file-index` 可指定原六组索引。`--output` 必须为不存在的新目录。无训练或 TEST 开关。

## 输出与最小检查

默认输出 `experiments/d0b_short_long_dynamic_moe/advantage_probe_seed42_<timestamp>/`：

- `probe_summary.md`：结论、VAL完整策略表、优势预测能力、解释边界。
- `probe_config.json`、`temporal_folds.json`、`TRAIN_LOCKED.json`：预定设置、折行号/日期、全部候选 CV 分数和折 scaler 统计、锁定设置。
- `probe_fit_metrics.csv`：各目标 MAE/RMSE/MSE/R²、相对常量变化、符号识别率。近零方差 R² 留空，不置零。
- `strategy_comparison.csv`：TRAIN/VAL 四周期 MAE/MSE/RMSE/Hit、组内 Huber、与 B/D0B/三专家参考的差值。Hit 原值为比例，另有 Hit_percent。
- `routing_diagnostics.csv`：权重均值/标准差/分位数、逐样本熵、平均权重熵、选择比例、非并列一致率与硬选择 regret；这些硬指标不是软融合损失。
- `router_input_statistics.json`：TRAIN/VAL各维标准差。
- `per_origin_predictions.npz`：train_/val_ 前缀下的输入、日期、商品顺序、标签、正常/三候选/各策略预测、真实/预测优势、路由、硬/理想选择、并列标记和配对置换，可复算统计。
- `probe_short.npz`、`probe_long.npz`：不使用 pickle 的 scaler 均值/尺度、维度掩码、Ridge系数/截距/alpha，加载后预测逐项一致。
- `audit.json`、`provenance.json`、`file_index.json`：完成状态、来源和哈希、实际环境、耗时、不可变性核验。

合成检查覆盖：优势方向与固定并列；两个周期同选；交易行重叠清除；scaler 没有看到折验证数据；常量输入/零温度/负或不可定义 R²；样本置换对齐；真实 checkpoint LN 输入路径；one-hot 选择等价；非线性 head 下的真实软融合；冻结参数和临时checkpoint不变。合成数据不用于证明预测收益。

本次真实运行目录为 `advantage_probe_seed42_20261013_155809_707114`。TRAIN/VAL分别1396/268 origins，正常和三候选预测复现差为0，short/long选出的alpha均100。四个VAL优势目标的MSE均高于常量参考（更差），Ridge硬选择虽然小幅降低B的5d MAE，仍未超过D0B/三专家全局静态，10/20d误差变大。完整数字以该目录原始文件为准；没有执行TEST或原模型训练。
