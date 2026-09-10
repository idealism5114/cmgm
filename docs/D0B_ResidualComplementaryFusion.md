# D0B-ResidualComplementaryFusion

本轮只在原 D0B competitive gate 之后加入一个共享 global residual：

```text
combined = concat(h_spatial, h_temporal)
gate = sigmoid(original gate_fc(combined))
h_base = gate * original temporal_proj(h_temporal)
       + (1 - gate) * original spatial_proj(h_spatial)
r = Linear(128,32) → ReLU → Linear(32,64,bias=False)
h_fused = h_base + r(combined)
prediction = original head(h_fused)
```

新增 module 在全部原 D0B modules 创建之后初始化，最后一层权重严格为0。所有共享 state_dict tensors 同 seed 完全相等；初始 eval residual=0、h_fused=h_base、prediction=D0B。参数量 **520,549 → 526,725，差值 +6,176**。

原 gate 的输入、维度、投影、sigmoid 及 temporal/spatial 方向全部保留。Residual 输入是原始 global `[h_spatial,h_temporal]`，不 detach，不使用 projected states、gate values、h_base、H/Z、pre/post-GNN nodes。没有额外 alpha、gate、norm、dropout、embedding 或 commodity/horizon-specific parameters。Spatial attention 使用原始 raw Q/K，所有8 heads保留graph prior。未继承 QKNorm、Hybrid、PreGNNLocalSkip 或其它失败路径。

## GPU 运行命令

在已激活项目 Python 环境的 **Commedities** 目录执行一次：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_residual_complementary_fusion
```

只检查、不训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.d0b_residual_complementary_fusion --sanity-only
```

默认要求 GPU，CUDA 不可用会停止，不会自动回退 CPU 训练。若新实验的正式 checkpoint 已存在，会拒绝自动覆盖或重新训练。正式运行前会重新加载 D0B baseline，在相同数据/evaluator 上复现并验证初始化与结构 sanity，然后才进行唯一一次训练。

协议保持 SEQ_LEN=20、seed=42、batch64、epochs200、patience10、Adam lr=1e-4、weight_decay=1e-5、Huber delta=.02、四 horizon loss 求和及原 switch KL/warmup、ReduceLROnPlateau。新增参数进入同一个 optimizer group。只有原 multi-horizon validation objective 控制 scheduler、early stopping、formal best checkpoint；VAL5 MAE/MSE 仅 secondary logging，不创建第二个正式 checkpoint。

## 验证与控制

初始化检查 h_base、residual、h_fused、prediction；初始最后层梯度必须非零，第一层梯度为0是正常链式梯度结果。非零权重的合成测试验证两侧 global input 保持梯度路径，不能把合成测试当作训练结果。

Initial、epoch1、epoch5、epoch10、best 在固定 TRAIN batch 上记录两层权重/梯度范数、mean abs residual、mean residual L2、residual/base L2 ratio。诊断使用 eval + autograd.grad，不执行额外 optimizer step，不写 `.grad`，保护训练 RNG。

因果检查包含原 temporal prefix cutoff10 及 global fusion 的 observed-prefix/window 语义。完整已观察窗口可以依赖其任意时间点，不能要求最终 spatial/fusion 输出不受已观察 suffix 影响。所有 fusion components 检查 batch permutation/single sample、三类市场 relabeling（graph embeddings 与原 commodity head rows 同步重标记）。FP32 数值误差保留；需要时使用独立 FP64 副本审计舍入，不改变正式 forward 精度。

正式 best checkpoint 只评估以下三个对象：

- Original D0B checkpoint。
- Full ResidualComplementaryFusion。
- ZeroResidual：直接复用 trained native forward 的真实 h_base，再输入相同原 head，只有 fusion residual 被移除。

没有 shuffle、mean replacement、spatial-only/temporal-only 或其它 inference controls。ZeroResidual 不重新训练、不修改参数。

Full−ZeroResidual 是直接 residual 输出效果；ZeroResidual−OriginalD0B 是 shared model 的训练变化，包含原 head 的共同适应，不能唯一归因到某个 encoder。报告优先标记 Case F：只要某 split 的 zero-residual MAE 出现 material deterioration，就明确给出该 split 与另一 split 的实际差值；若 Full 的 VAL/TEST MAE 也同时恶化，输出 **STOP D0B incremental patching**。

主指标固定 pooled 5d MAE/MSE/RMSE/Hit%，MSE直接计算平方误差均值，Hit包含零 targets。5d index动态查找。完整TRAIN/VAL/TEST不丢尾batch。报告 residual L2 分布、cos(base,fused)、spatial/temporal norms与cosine、原gate分布；只作描述，不调参。

Commodity/error 分解限于完整24商品、top5变化、困难商品、gross improvement/degradation与largest/top3 shares、一次排除最大TEST MAE改善商品的arithmetic，以及TRAIN P75/P90/P95 target magnitude groups。`>P95`包含在`>P90`内，不能相加。没有新的risk/regime诊断。

Case的near-tie/material阈值为明确标注的0.1%，activation采用final weight >1e-6及residual/base L2 ratio >0.1%；只用于透明描述，不用于训练、调参或selection，不声称单seed统计显著。Case F优先，B/C细分A；给定Case不覆盖的mixed result明确标记，不强行判成功。未训练时不赋予Case。

## 产物

- `checkpoints/switching_latent_balanced_residual_complementary_fusion_best.pt`
- `experiments/d0b_residual_complementary_fusion/<timestamp>/REPORT.md`
- `results.json`、`training_history.json`
- `overall_metrics.csv`、`commodity_metrics.csv`、`target_magnitude_groups.csv`
- `fusion_residual_diagnostics.json`、`ablation_metrics.json`、`gradient_diagnostics.json`、`sanity_checks.json`

sanity-only 不伪造训练后 CSV 或 Case。代码修改/验证完成后由用户运行 GPU 正式训练；执行一次后 STOP，不自动开始第二种 fusion 或 backbone redesign。

## 本次代码验证

14 项新测试与 189 项相关回归测试通过。实际 fixed TRAIN/TEST batch 均为 `(4,20,284,21)`。Residual input/output 为 `(4,128)` / `(4,64)`。实际参数 delta=6176，shared init mismatch=0，初始 h_base/new prediction/residual/fused-base 四项差异均为0。原 gate 公式复核误差为0。

初始最后 residual 层梯度范数 **0.0032349524854**，第一层梯度为0。Batch permutation 所有分量差值为0，single-sample 最大分量差值约2.38e-7，prediction差值约5.22e-8，causal prefix/window均为0。FP32重标记的 spatial-state 最高差值约1.84e-5、prediction约1.50e-6；独立FP64复核分别约1.16e-14和1.03e-15，报告保留原始误差及精度审计。

正式 D0B checkpoint（epoch85）完整 pooled 5d 再次复现：

| Split | MAE | MSE | RMSE | Hit% |
|---|---:|---:|---:|---:|
| VAL | 0.023200619296 | 0.001030383502 | 0.032099587257 | 47.5435323 |
| TEST | 0.021994783944 | 0.000912875426 | 0.030213828387 | 46.8401487 |

冻结修改前 attention 的完整 D0B prediction 与当前 baseline 差值为0，原 checkpoint checksum 不变。检查产物：`experiments/d0b_residual_complementary_fusion/20260909_163604`。这里没有正式训练结果，非零 residual 单元测试使用合成权重 fixture，不代表训练后激活或泛化。
