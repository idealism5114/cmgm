# Controlled baseline comparison

本轮只比较 ZeroReturn、Linear、GRU、VanillaTransformer、iTransformer、MTGNN 和既有 D0B checkpoint。D0B 不训练、不改参数。五个可训练 baseline 各运行一次，seed=42；ZeroReturn 无需训练。

## 运行

在 `Commedities` 项目目录、已激活原 GPU Python 环境的终端中运行。以下命令均不含斜杠。

可先做数据、checkpoint 和模型 sanity 检查，不训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.baseline_comparison --sanity-only
```

正式训练入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.baseline_comparison
```

程序先复现正式 D0B，再按 ZeroReturn、Linear、GRU、VanillaTransformer、iTransformer、MTGNN 顺序处理。CPU 不会被用于正式训练的自动回退。

继续最近的正式 suite（自动排除 sanity-only 目录）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.baseline_comparison --resume latest
```

已完成模型通过 checkpoint checksum 验证后跳过，不重新训练或评估其 TEST。若训练已完成、评估中断，则加载已保存的正式 best 权重继续，不重新创建 optimizer。

代码或数值错误被记为 INVALID，修复原因后才可显式重试：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.baseline_comparison --resume latest --retry-invalid
```

该开关不允许重跑已完成但性能不好的模型。未完成的训练遇到外部强制中断、状态仍为 RUNNING 时会停止，需先人工检查，不能自动当成代码错误重新训练。OOM 不会自动减小 batch。新 suite 遇到已有 baseline checkpoint 会拒绝覆盖。

## 固定协议

复用现有 `main_ablation.build_data`，保持 SEQ_LEN=20、21 features、原 chronological splits、target construction/clipping 和四 horizons `[1,5,10,20]`。不重新标准化 target。

训练为四个 Huber(delta=.02) loss 之和，Adam(lr=1e-4, weight_decay=1e-5)，同一个参数组，batch=64，最多200 epochs，patience=10，ReduceLROnPlateau(factor=.5, patience=5)。不向 baseline 添加 D0B 的 switch KL。

保持当前 D0B loader 的 shuffle=False、TRAIN drop_last=True。正式 VAL selection/scheduler/early stopping 使用原来的“每批四 horizon loss 之和，再对 batch 等权平均”。VAL5 MAE/MSE 只记录。正式报告则使用完整 TRAIN/VAL/TEST loader，包含不足64的最后一批。

统一 evaluator 在每个 horizon 的所有 origin × commodity 上计算 MAE、MSE、sqrt(MSE)、unmasked sign Hit（包含 zero targets）。Hit 在 CSV/report 中以百分比展示，在底层 results 的 metrics 中为比例。

## 输入与实现透明性

无参数 adapter 每天构造 stock_mean、stock_std、bond_mean、bond_std、24 commodity tokens。std 使用 population correction=0。每 token 保留21 features，得到 `(B,20,28,21)`。flatten 为 token-major、feature-minor，channel=21*token+feature；commodity i 对应 token4+i。

D0B 使用原生284节点架构，其他模型使用同一数据来源的 deterministic neutral representation。二者不应被写成完全相同的 architectural input representation。

- Linear：11760→96。
- GRU：588→128，2层，dropout=.1，最终 hidden→96。
- VanillaTransformer：588→128、固定 sinusoidal absolute PE、2层4heads、FFN256、dropout=.1、causal mask，最后 timestep→96。
- iTransformer：本轮指定的简化 inverted 实现，588个长度20的 variate tokens，20→128，2层4heads，commodity 的21个 feature tokens mean pooling，共享128→4。不是官方代码复现。
- MTGNN：简化 MTGNN-style baseline，复用独立 AdaptiveGraphLearner 和 MixHopPropagation，不使用 D0B attention/fusion/switching。21→64，两层 gated causal temporal conv，kernel3、dilation1/2，双向 MixHop K2/beta=.05，共享64→4。有效 temporal receptive field 为7步，输入窗口仍为20步。仓库 graph 的 learnable alpha、soft top-k 与 canonical 实现的差异会明确列在 REPORT。

实现文件：`cmgm/models/comparison_baselines.py`。独立训练、sanity、evaluator 和报告在 `cmgm/scripts/baseline_protocol.py`、`baseline_comparison.py`、`baseline_report.py`。本轮没有修改 D0B 模型、通用训练或原 data pipeline。

## 输出

每次 suite 保存到 `experiments/baseline_comparison/<timestamp>`：REPORT.md、results.json、baseline_overall.csv、baseline_all_horizons.csv、training_summary.csv、parameter_counts.csv、sanity_checks.json、per_commodity_baselines.csv 和各训练模型 history。sanity-only 报告只包含真实 D0B/Zero 指标，其余为待训练状态。

正式权重分别保存到 `checkpoints/baselines`，文件名 linear_best.pt、gru_best.pt、transformer_best.pt、itransformer_best.pt、mtgnn_best.pt。不覆盖 D0B。

完整 suite 按 TEST5 MAE 排序，输出相对 D0B 的有符号 delta、D0B 对每个 baseline 的 improvement，以及四 horizon 结果。不隐藏优于 D0B 的 baseline，不做调参或统计显著性声明。完成后停止。
