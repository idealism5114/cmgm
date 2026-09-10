# D0B-QKNormGraphAttention

本实验仅改变 `EdgeAttnMixHop` 的 Q/K 几何。正式窗口仍为 20，输出仍为 1、5、10、20d。两层空间 attention、每个 MixHop hop 都在 Linear 与 head reshape 后执行：

```python
Q = F.normalize(Q, p=2, dim=-1, eps=1e-6) * math.sqrt(self.head_dim)
K = F.normalize(K, p=2, dim=-1, eps=1e-6) * math.sqrt(self.head_dim)
```

之后保留原 einsum 除以 `sqrt(head_dim)`。非退化向量的 content score 为 `sqrt(head_dim) * cosine(Q, K)`；`F.normalize` 的分母为 `max(norm, eps)`，小于 epsilon 的向量会单独记录，不强求范数达到 `sqrt(head_dim)`。

所有 8 个 heads 仍加入原 graph prior。没有 Hybrid 的 free heads，没有额外 temperature 参数。V、AdaptiveGraphLearner、MixHop、TempWeighted、D0B temporal、融合和 head 均保持不变。新旧模型参数量均为 **520,549**，初始化同名 state_dict tensors 完全一致；初始预测不要求相同。

## 运行

在已激活项目 Python 环境的 **Commedities** 目录执行。以下命令默认使用 GPU；CUDA 不可用时会停止，不会静默转为 CPU 训练。

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_qknorm_graph_attention
```

这只启动 QKNorm 的一次正式训练。先复现正式 D0B checkpoint 的 pooled baseline 并完成 sanity，再训练、重新加载原 multi-horizon validation objective 选出的 best checkpoint，最后输出统一评估和报告。如果 QKNorm 正式 checkpoint 已存在，会拒绝自动覆盖或重新训练。

可先单独进行 GPU 检查，不训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_qknorm_graph_attention --sanity-only
```

也注册到了原 ablation 入口，但推荐上述专用入口，避免误运行其他 variants。训练协议固定为 seed 42、batch 64、200 epochs、patience 10、Adam、lr 1e-4、weight decay 1e-5、Huber delta .02、原 switch auxiliary loss、ReduceLROnPlateau。不会使用 VAL5 指标改变正式 checkpoint selection；VAL5 MAE/MSE 仅作为逐 epoch 的 secondary diagnostic，不保存或选择第二个正式 checkpoint。

## 产物

- 正式 checkpoint：`checkpoints/switching_latent_balanced_qknorm_graph_attention_best.pt`
- 每次报告目录：`experiments/d0b_qknorm_graph_attention/<timestamp>`
- `REPORT.md`、`results.json`、`training_history.json`
- `overall_metrics.csv`、`commodity_metrics.csv`、`target_magnitude_groups.csv`
- `qk_norm_diagnostics.json`、`attention_diagnostics.json`、`gradient_diagnostics.json`、`graph_diagnostics.json`、`sanity_checks.json`

sanity-only 不生成虚构的 QKNorm forecasting 表或 Case；这三份结果 CSV 在正式评估后生成。

固定 TEST 4 个样本记录 initial、epoch1、epoch5、epoch10、best 的两层/两 hop Q/K norms、content/prior/final logits、attention entropy 与 top-k mass、head diversity。正式 D0B best checkpoint 记录相同指标。预测梯度用固定 TRAIN 4 个样本、eval mode、四个 Huber 之和、`autograd.grad`，不执行 optimizer step、不写入 `.grad`，也不影响正式训练的 RNG 状态。

sanity 包含初始化、参数量、归一化公式、所有 head prior、cutoff=10 因果前缀、batch permutation、single sample、stock/bond/commodity relabeling。沿用已有 relabeling 定义：Graph E1/E2 和 commodity 输出 rows 同步重标记；FP32 舍入敏感的重标记采用独立 FP64 副本复核，原误差和复核均保留，正式 forward 精度不变。没有 batch averaging 或未来输入泄漏。

TEST 主指标固定为 pooled 5d MAE/MSE/RMSE/Hit%，Hit 包含零 target。5d index 从 `MULTI_HORIZONS.index(5)` 获得。Magnitude group 阈值仅来自 TRAIN；`>P95` 是 `>P90` 的子集。Commodity contribution 同时报净改善、gross improvement/degradation、largest 和 top3 shares。

报告不会预设 raw Q/K norm collapse、低 entropy 或更大 gradient 就意味着预测成功。Case 的判断只在正式训练与评估后进行；报告中的 0.1% error near-tie/material-MSE 和 geometry-change 阈值只是透明的描述性约定，不用于训练、调参或模型选择。若结果不满足用户给定 A–F 中任一个定义，会明确标记 mixed evidence，不编造分类。单 seed 不能证明统计显著性。

实现与检查完成后，由用户运行正式训练；不会自动继续 temperature、其他 normalization、graph 或新分支实验。

## 本次代码验证结果

18 项新测试与 134 项相关回归测试通过。实际固定 batch 为 `(4,20,284,21)`；参数差值 0、shared init 最大差值 0。冻结的修改前完整 D0B 路径与当前 D0B checkpoint 的预测最大差值 0。因果前缀差值 0；batch permutation 差值 0；single sample 最大差值约 4.10e-8，三类市场 relabeling 最大差值约 8.20e-8。初始 QKNorm Q/K 范数最大偏差约 5.12e-7。

实际重新加载正式 D0B best checkpoint（epoch 85），完整 loader 的统一 pooled 5d 结果：

| Split | MAE | MSE | RMSE | Hit% |
|---|---:|---:|---:|---:|
| VAL | 0.023200619296 | 0.001030383502 | 0.032099587257 | 47.5435323 |
| TEST | 0.021994783944 | 0.000912875426 | 0.030213828387 | 46.8401487 |

检查使用 CPU 做 inference 和 diagnostic gradients；未运行正式训练，未产生 QKNorm trained checkpoint，原 D0B checkpoint 文件 checksum 不变。这不代表已经验证 GPU 上的训练结果。

本机检查产物：`experiments/d0b_qknorm_graph_attention/20260909_151349`。其中 `results.json` 保留执行当时的代码指纹，`code_validation.json` 记录额外 frozen-full-model 比较及测试数量。正式 Case A–F 和 QKNorm 泛化结论待用户完成唯一一次 GPU training 后由报告给出。
