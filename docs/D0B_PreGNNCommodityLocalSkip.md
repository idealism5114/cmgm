# D0B-PreGNNCommodityLocalSkip

本轮只增加一个共享的 pre-GNN commodity-local residual：

```text
H_pre = type projection + node-wise TempWeighted aggregation
local = H_pre[:, n_stock+n_bond:, :]
input = concat(broadcast(h_fused), local)
residual = Linear(128,32) → ReLU → Linear(32,4,bias=False)
prediction = original_D0B_head(h_fused) + residual
```

最后一层权重严格 zero-init，不增加 alpha、gate、embedding 或 commodity-specific parameters。所有原 D0B 模块先按原顺序创建，再创建 residual head。同 seed 的共享参数完全一致，eval 初始 residual=0、prediction=D0B。参数量为 **520,549 → 524,805，差值 +4,256**。

采用原始 D0B raw dot-product Q/K、8 个 graph-prior heads、固定 sticky alpha=.5、BalancedReadout。不会启用 Hybrid、QKNorm、post-GNN residual、learnable persistence。原 spatial/global temporal/fusion/base head 完整保留，新增输入不 detach。

## GPU 运行

在已激活项目环境的 **Commedities** 目录运行唯一一次正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_pregnn_local_skip
```

可先单独检查，不训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_pregnn_local_skip --sanity-only
```

默认要求 CUDA，可用性失败会停止，不会悄悄切换 CPU 训练。正式 checkpoint 已存在时拒绝自动覆盖/重跑。上述训练命令先重新加载原始 D0B best checkpoint 复现 baseline，核心 sanity 通过后才训练一次。用户负责执行正式训练，代码修改阶段只做检查。

协议保持 SEQ_LEN=20、seed=42、batch=64、epochs=200、patience=10、Adam lr=1e-4、weight_decay=1e-5、Huber delta=.02、四 horizon loss 求和、原 switch KL 与 warmup、原 ReduceLROnPlateau。所有参数使用同一个 optimizer group。

正式 best checkpoint 仍由原 multi-horizon validation objective 选择。逐 epoch VAL5 MAE/MSE 只是 secondary logging，不改变 scheduler、early stopping、checkpoint 或 TEST selection。

## 检查语义

- `H_pre` 与独立重算的 TempWeighted 聚合、第一层 GNN 的实际输入逐项比较。
- 商品映射报告 name、node index、target index、output index；从实际 raw price columns 独立重建全部 split 的每个 horizon target，检查与 Dataset 输出完全一致。
- temporal prefix cutoff=10 检查 E/H/p/Z/h_long/h_micro/h_temporal。
- local prefix forecast 只向模型提供 observed prefix；完整 forecast-window 检查只允许读取窗口内数据。**完整 20 步空间输出可以依赖这 20 步中的任何已观察输入**，不能把窗口内部后半段也当成“模型不许看的未来”。
- H_pre、local、base、residual、prediction 都检查 batch permutation/single sample，以及 stock/bond/commodity relabeling。重标记包含 graph embeddings 与原始 commodity output rows；共享 residual head 没有 commodity-specific rows。
- 如大的 raw representation 导致 FP32 单样本/重标记舍入误差，保留原始数值并在独立 FP64 副本复核，报告明确标注，正式 forward 精度不变；不会放宽 batch/prefix 检查。
- 初始最后 residual 层 prediction gradient 必须非零；前一层初始 gradient=0 是 zero-init 的预期。epoch1/5/10/best 记录第一层梯度/权重范数、最后层权重、residual magnitude。诊断使用 autograd.grad，不写 `.grad` 或执行额外 optimizer step，并保护 RNG。

## 训练后控制与报告

正式训练完成后，重新加载唯一正式 best checkpoint，在完整 TRAIN/VAL/TEST loader 上评估，保留末尾非满 batch。

1. Original D0B checkpoint。
2. Full trained PreGNNLocalSkip。
3. ZeroResidual：同一个 trained model 的 base prediction。
4. ShuffledLocal：固定 seed42，在每个 sample 内使用相同的 commodity permutation，只替换 local residual 输入。
5. MeanLocal：每个 sample 自己的 commodity mean local state，复制到该 sample 的各商品。

后三种 local controls 复用 native forward 的完全相同 global context/base prediction，base max diff 必须为0。不会重新训练 control models，不按 control 的 TEST 表现选择模型。

分别报告 Full−ZeroResidual 的直接输出效果和 ZeroResidual−OriginalD0B 的 shared representation 训练效果。后者不意味着某个唯一模块的因果效应。报告包括每 horizon/commodity residual 大小、方向、完整 commodity error delta、gross improvement/degradation、最大及 top3 贡献，以及移除 TEST 最大单一改善商品后的 VAL/TEST supplementary arithmetic。

误差分组只使用 TRAIN target magnitude 的 P75/P90/P95，`>P95` 与 `>P90` 重叠。主指标统一 pooled 5d MAE/MSE/RMSE/Hit%，MSE直接平均平方误差，Hit 包括零 target。5d index 动态查找。

成功需要 VAL 与 TEST 同方向改善及 local controls 的机制证据，不能只因 residual 非零就称成功。报告中 near-tie/material-MSE=0.1%、control impact=1e-6 等是明确标注的描述性约定，不用于调参或 checkpoint selection。Case G 优先，其次 tradeoff，再区分 B/C 机制细分与 A；用户给定 Case 不覆盖的混合结果会明确标记，不编造成功。未训练时不赋予 Case。

## 产物

- `checkpoints/switching_latent_balanced_pregnn_local_skip_best.pt`
- `experiments/d0b_pregnn_local_skip/<timestamp>/REPORT.md`
- `results.json`、`training_history.json`
- `overall_metrics.csv`、`commodity_metrics.csv`、`target_magnitude_groups.csv`
- `residual_diagnostics.json`、`ablation_metrics.json`、`gradient_diagnostics.json`、`sanity_checks.json`

sanity-only 只输出检查与待训练报告，不伪造正式结果 CSV。执行一次正式实验后 STOP，不继续更大 MLP、embedding、gate、GRU、fusion 或新模型。

## 本次验证结果

16 项新增测试和 173 项相关回归测试通过。实际 fixed TRAIN/TEST batch 均为 `(4,20,284,21)`。参数量、shared init、初始 eval prediction、初始 residual 均符合精确要求；初始最后层 prediction gradient norm=0.003909277313，第一层=0。

`H_pre=(4,284,64)`、`h_comm_pre=(4,24,64)`。全部 split 的 target 商品顺序重建误差为0。TempWeighted 重建/首层 GNN input 对比误差为0。Batch permutation 与因果 prefix/window 差值为0。单样本 H_pre 最大差值约2.38e-7，prediction约5.22e-8。Stock/bond relabeling 的 FP32 global-path 差值最高约1.50e-6；独立 FP64 复核最高约1.03e-15，报告保留两者，未改变正式数值路径。

原始正式 D0B checkpoint（epoch85）完整 pooled 5d 复现：

| Split | MAE | MSE | RMSE | Hit% |
|---|---:|---:|---:|---:|
| VAL | 0.023200619296 | 0.001030383502 | 0.032099587257 | 47.5435323 |
| TEST | 0.021994783944 | 0.000912875426 | 0.030213828387 | 46.8401487 |

冻结修改前 attention 的完整 D0B checkpoint prediction 与当前 baseline 差值为0。原始 checkpoint checksum 不变。检查产物位于 `experiments/d0b_pregnn_local_skip/20260909_155641`。

本次未执行正式训练或生成 trained PreGNN checkpoint；非零 residual 的梯度/控制测试使用明确标记的合成 fixture，不能当作训练后机制结果。正式 A–G 分类需用户运行唯一一次 GPU 训练后评估。
