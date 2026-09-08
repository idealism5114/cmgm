# D0B-5dOnlyObjective

唯一新增目标函数诊断变体：`switching_latent_balanced_readout_5d_only`。

模型与原 D0B 使用相同 construction 和 forward，参数量均为 **520,549**。
不增加参数，head 仍输出 `[B, 4, N_commodities]`，horizons 为 `[1,5,10,20]`。
BalancedReadout 保留；alpha 固定 `.5`；不使用 D0C/D0D/D0E 的新模块。

## 唯一 objective 改动

```text
prediction training loss = 4.0 × raw Huber L5
total training loss      = 4.0 × raw Huber L5 + original switch loss
validation selection    = 4.0 × raw Huber L5
```

这是 **scale-matched 5d-only diagnostic objective**，不是正式的 loss-weight
优化设计。动态使用 `MULTI_HORIZONS.index(TARGET_HORIZON)` 定位 5d，检查
TARGET_HORIZON=5 和四 horizon 输出，乘数固定为 4.0，绝不根据 ratio 调整。

`train_epoch` 与 `validate_epoch` 共用 `cmgm/training/train.py` 的
`_prediction_loss(model, prediction, target, criterion)`。原 variants 继续
使用四项之和。辅助 multi-horizon validation loss 从已有预测 detached
计算，每个 epoch 保存，不参与 scheduler / early stopping / checkpoint selection。

Adam、lr=1e-4、weight_decay=1e-5、Huber delta=.02、ReduceLROnPlateau、seed=42、
batch=64、seq_len=20、200 epochs、patience=10、数据划分和特征归一化保持原样。
所有参数仍在同一个 optimizer group。非5d head 输出不贡献 prediction gradient；
原 weight decay 仍作用于完整参数集合。

## 运行命令：仅这一个正式训练变体

先使用已经修复 PyTorch/CUDA 兼容性的 Python 环境。在仓库根目录执行：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup python -u -m cmgm.scripts.main_ablation \
  --variants D0B-5dOnlyObjective \
  --epochs 200 --patience 10 --seed 42 \
  --batch-size 64 --seq-len 20 \
  --checkpoint-dir checkpoints \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --five-day-report-dir experiments/d0b_5d_only \
  > d0b_5d_only.log 2>&1 &
```

```bash
tail -f d0b_5d_only.log
```

需要 CPU-only 时追加 `--no-cuda`。新诊断的 CPU fixed-batch setup 使用
`fork_rng(devices=[])`；GPU probes 只保存模型实际使用设备的 RNG。

现有 D0B checkpoint 只用于训练后的精确 reference，**不用它初始化新模型**，
也不重新训练 D0B。训练前 independently same-seed 构造 D0B 来验证共享初始化。

产物：

- `checkpoints/switching_latent_balanced_readout_5d_only_best.pt`
- `experiments/d0b_5d_only/<timestamp>/REPORT.md`
- 同目录 `results.json`；`D0B/` 和 `5dOnly/` 下固定批次张量、各 split 原始预测/target。

训练完成后自动严格加载刚保存的 best checkpoint，做本轮要求的 checkpoint
对照诊断，然后停止。不自动训练 grouped objective 或其他新变体。

## 仅重新生成 checkpoint 报告

```bash
python -u -m cmgm.scripts.d0b_5d_only_diagnostics \
  --checkpoint checkpoints/switching_latent_balanced_readout_5d_only_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0b_5d_only/checkpoint_recheck \
  > d0b_5d_only_checkpoint.log 2>&1
```

输出目录必须尚不存在。由于 D0B 和该 objective variant 的 state_dict 完全同构，
checkpoint-only CLI 会额外检查 metadata 的 `objective="4x_5d_only"`，避免误加载 D0B
并把它当成训练过的 5d-only checkpoint。

## 日志和报告的语义

- `[D0B-5dOnly shared init]`：所有同名参数逐项比较，以及 E/H/prior/p/candidates/Z/
  h_long/h_micro/h_temporal/h_spatial/gate/prediction 的 eval forward 等价性。
- 同一固定 TRAIN batch 的 L1/L5/L10/L20、sum_multi、4x_L5、ratio；只记录、不调参。
- 每 epoch 的 raw_L5、scaled_prediction_loss、switch_loss、total_loss、
  val scaled 5d、aux multi-horizon val、LR、switch beta；history 标记 best/final epoch。
- Checkpoint metadata 记录 objective、固定 multiplier、best epoch、best scaled/raw val L5、
  参数量、seed、Git SHA、关键源文件 SHA256 和训练协议。
- 原 latent diagnostics 保留。其新变体 total-loss probe 使用 `4×L5+switch`，
  明确标记 scaled；新 REPORT 的 horizon gradient diagnostic 则使用 **未乘4的 L5**，
  eval/autograd.grad，不写 `.grad`、不消耗训练 RNG、不更新权重。
- TRAIN/VAL/TEST 全部 horizons 的 MAE/RMSE/Hit 和相对 D0B 的绝对/相对 MAE delta。
  RMSE 使用原 mean(asset-wise RMSE)，Hit 排除 near-zero targets。
- 所有 split 的 posterior/prior/transition/candidate 统计；fixed TEST micro dynamics、
  balanced readout、spatial/gate 和跨 checkpoint representation drift。
- VAL/TEST 各 horizon 的 zero-micro/zero-long、uniform 和 state0/1/2 完整递推控制。
  RoutingFraction_5d 只使用 5d 分子/分母，和实际 D0B 5d reference 比较。
- 完整各 split 的 prediction specialization；固定 TEST 的 raw horizon gradients 和
  cosine 变化；prefix causality、batch independence 和 within-market permutation。

训练/验证 history 沿用旧代码的 batch-mean aggregation。完整 split 指标按样本计算；
报告的 TRAIN checkpoint eval 包含尾 batch，与优化过程 train loss 明确区分。
空间/时间层的 drift 尺度不直接可比，不能仅按 drift 大小归因预测收益。

## 解释和停止规则

报告给出十四个问题的数值对照以及一个 Case A/B/C/D/E。
只有 VAL/TEST 5d 同方向改善且通过机制 sanity，才支持 negative transfer。
约0.1%的相对 MAE 改善用于区分 A/B 的效应大小，不参与训练或选模，也不构成统计显著性阈值。
TEST 单独改善归为无稳健泛化证据，不能作为支持结论。Case C 不自动假定 TRAIN 已改善，
必须另看 raw L5 对照。只有 A 才标记后续 5/10/20 objective 具备研究资格，绝不执行它。

本轮只做代码和合成验证，没有运行真实训练或市场数据实验。因此实际 loss-scale ratio、
VAL/TEST 收益以及 Case 结论，都须等运行上述命令后由真实结果回答。

## 验证

```bash
python -m pytest -q tests/test_d0b_5d_only_objective.py
python -m pytest -q
```

专项验证包括：实际参数规模的初始化等价、动态 horizon index、非5d prediction gradient
严格为0、train/validate 使用同一 helper、switch loss 保留、scheduler 与 checkpoint
选择不受 aux multi-horizon loss 影响、旧变体 loss/gradient 回归、原始 L5 gradients、
报告和单变体入口的 synthetic smoke test。入口测试 mock 掉训练，未执行市场实验。
