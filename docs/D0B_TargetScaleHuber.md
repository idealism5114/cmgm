# D0B-TargetScaleHuber

唯一新增正式 variant：`switching_latent_balanced_readout_target_scale_huber`。模型构造复用 D0B；参数数目仍为 520,549，输出保留 1/5/10/20d，原 Markov alpha=.5、evidence、generators、BalancedReadout、spatial、fusion/head 全部保留。

唯一新 prediction objective：

```
s_h = std(TRAIN target_h), population ddof=0
s_ref = s_5
delta_h = .02 * s_h / s_5
w_h = .02 / delta_h
L_pred = sum_h w_h * Huber(pred_h, target_h; delta_h)
```

5d 的 delta 精确为 .02、weight 精确为 1；每个 horizon 单个 residual 的 unreduced gradient cap 都为 .02。mean loss 的 prediction gradient 还要除以 batch×commodity 数。变的是 auxiliary horizons 的 threshold 和 quadratic 区导数斜率，不能声称所有梯度或 loss scale 都不变。

Scales 在训练前从 `data['loaders']['train'].dataset` 的全部 **1396×4×24 targets** 重新计算（包含被训练 loader drop_last 的尾窗口），转 float64 后使用 std(ddof=0)。与已授权的 diagnostic TRAIN std 核对，rtol=1e-6、atol=1e-10；不一致立即停止。参考数字只用于 sanity，delta/weight 动态计算。使用 immutable dataclass 常量；不注册为 Parameter 或 buffer，不进入 optimizer，不从 VAL/TEST 或 residual 估计。

在 `Commedities` 目录执行以下唯一正式训练命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.main_ablation \
  --variants D0B-TargetScaleHuber \
  --seed 42 --batch-size 64 --seq-len 20 --epochs 200 --patience 10 \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --checkpoint-dir checkpoints \
  --target-scale-report-dir experiments/d0b_target_scale_huber
```

CPU 环境追加 `--no-cuda`；GPU 运行需使用与驱动兼容的 PyTorch。不要运行未限定 `--variants` 的全量 ablation，也不要另跑 std/MAD、anchor、delta 或 weight 对照。脚本拒绝不符合本轮固定 seed/batch/epochs/patience/config 的训练选项。若新 checkpoint 已存在，会拒绝覆盖和再次训练。

训练使用原 Adam lr=1e-4、weight_decay=1e-5、同一 parameter group 和原 ReduceLROnPlateau；switch KL 与 warmup 完整保留。train / validate 共用 `_prediction_loss`。原 D0B validation 是 prediction-only，本变体保持这个 policy；scheduler、early stopping、best checkpoint 均由新的 weighted validation prediction loss决定。记录每 epoch raw(delta_h) / weighted per-horizon Huber、prediction/switch/total、val objective、LR、switch beta。Validation MAE/MSE 仅用于报告，不参与选模。

训练后写入 `checkpoints/switching_latent_balanced_readout_target_scale_huber_best.pt`，包含冻结尺度、delta/weight、种子/SHA、参数数、best epoch、best scale-aware val loss、同一 best epoch 的 pooled VAL5 MAE/MSE。恢复权重后重新加载正式 D0B 作同环境对照，不将 D0B 权重用于初始化新模型。

诊断自动输出 `experiments/d0b_target_scale_huber/<timestamp>/checkpoint_diagnostics/REPORT.md` 与 results.json，包含完整四指标、相对 MAE/MSE 变化、actual/counterfactual coverage、cap sanity、加权单 horizon autograd、regime/candidates/Z/readout/BaseRPE、uniform/zero-component、商品 5d 分解、causality/batch/market sanity。只有 checkpoint inference 干预允许 uniform，不进入训练。

只做训练前验证，不执行 optimizer：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_target_scale_huber \
  --preflight --no-cuda --output experiments/d0b_target_scale_huber/preflight_manual
```

训练已完成但需要单独生成报告时：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_target_scale_huber \
  --checkpoint checkpoints/switching_latent_balanced_readout_target_scale_huber_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0b_target_scale_huber/checkpoint_review --no-cuda
```

输出目录必须不存在。诊断只比较 D0B 和本变体，不重新跑 D0E、NoSwitchKL、5d-only 或 grouped。自动 Case 用方向与数值误差界限组织结果；单 seed 不能建立统计显著性，特别需审阅微小收益、tail-error tradeoff 和机制健康。若 sanity 或必要机制条件失败，会报告无法强行归入成功 Case，不自动接受模型。无论结果如何均 STOP，不实现下一实验。

数值 sanity 的补充：初始真实 TEST batch 的 stock 重排在 float32 中出现 3.1665e-6 的舍入差异，略超旧 3e-6 判据。保留该原始失败记录和原阈值，仅在 prefix/batch 检查通过而 permutation 超限时，创建相同权重的 float64 副本复核，并要求副本所有差异≤1e-10。已测副本最大差异为 7.52e-15。该过程不会改变训练模型的参数、dtype 或 RNG；若高精度复核仍失败则停止。旧 variants 默认仍使用原检查行为。
