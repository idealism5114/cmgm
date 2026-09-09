# D0B-CommodityResidualAdapter

唯一新增正式变体：`switching_latent_balanced_commodity_residual`。

形式严格为 `prediction = D0B_base_prediction + commodity_residual_alpha * residual_raw`。只增加共享 `Linear(64,4,bias=False)` 和一个零初始化、无约束的 scalar alpha；额外参数 `64*4+1=257`，520,549 → 520,806。该 scalar 与固定 Markov sticky_alpha=.5 完全不同。

## 架构与梯度约束

`_temp_weighted_spatial(x, return_nodes=True)` 返回原 type-pool 的 `(B,64)` global spatial vector，以及原 GNN×2 + LayerNorm 后、type_pool 前的 `(B,N,64)` node states。原默认返回仍是 `(B,64)`；没有 batch-axis averaging。Commodity slice 使用 `n_stock+n_bond`，实际 shape `(B,24,64)`。

新 full-model forward 各执行一次 spatial、D0B temporal 和原 gate/head。取 commodity node slice 后，所有商品共享同一个 Linear，输出从 `(B,24,4)` 转为 `(B,4,24)`；乘一个 scalar 后加至完整 D0B prediction。

残差 head 和 alpha 在全部 D0B full-model shared modules 构造后创建。head 使用默认随机初始化，alpha 精确为零；所有原有参数的初始化 name/shape/value 保持一致。没有 per-commodity 参数、head bias、MLP、embedding 或 horizon-specific readout/gate。

初始 alpha gradient 通常非零，residual-head prediction-loss gradient 应为零。alpha 离开零后 head loss gradient 可以激活。原 Adam 的 weight decay 对所有参数保持 `1e-5`，因此不能单凭 head 的权重漂移声称 prediction-loss gradient 已激活；报告同时保存真实 loss gradients。

共享 encoder/H/p/G/Z、BalancedReadout、global spatial pooling、gate/fusion/head 均保留原 D0B 机制。HSR 等历史变体保留在代码中，不作为本轮 backbone 或 comparator。

## 协议

- seed=42，batch=64，SEQ_LEN=20，horizons=[1,5,10,20]，primary=5d。
- 原 four-horizon Huber(delta=.02) 求和 + switch KL；beta_max=5e-4，warmup=20；sticky=.5、tau=1、K=3、z_dim=64。
- 原 Adam(lr=1e-4,weight_decay=1e-5)、ReduceLROnPlateau、200 epochs、patience=10。
- Validation selection、scheduler、early stopping 继续使用原 prediction-only four-horizon sum；val total 含 KL 只作描述。保留现有 mean-of-batch-means selection 聚合，不调整 loss 或 split。
- 正式 performance 使用完整 loader，含尾 batch；pooled MAE/MSE/RMSE、全体样本 sign Hit。MSE 直接由 residual² 计算，再计算 RMSE，不使用 legacy asset-averaged RMSE 或 masked Hit。

## 运行

从仓库根目录运行，使用已有项目虚拟环境：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
source ../.venv/bin/activate
```

仅做真实数据初始化和 sanity 验证：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_commodity_residual \
  --preflight --no-cuda \
  --output experiments/d0b_commodity_residual/preflight_manual
```

唯一正式训练命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.main_ablation \
  --variants D0B-CommodityResidualAdapter \
  --seed 42 --batch-size 64 --seq-len 20 --epochs 200 --patience 10 \
  --no-cuda \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --checkpoint-dir checkpoints \
  --commodity-residual-report-dir experiments/d0b_commodity_residual
```

以上是 CPU 命令。有可用 CUDA 环境时移除 `--no-cuda`。不要同时选其他 variants。

训练从新模型 seed-42 初始化开始；D0B checkpoint 仅验证/对照，不用于新模型初始化。入口先执行初始化与因果/重标记检查，再调用一次原训练循环。已存在新 checkpoint 时拒绝覆盖或重复训练。

checkpoint 名称：

```text
checkpoints/switching_latent_balanced_commodity_residual_best.pt
```

训练后自动重新加载 D0B 与新 best checkpoint，保存：

```text
experiments/d0b_commodity_residual/<timestamp>/preflight.json
experiments/d0b_commodity_residual/<timestamp>/checkpoint_diagnostics/REPORT.md
experiments/d0b_commodity_residual/<timestamp>/checkpoint_diagnostics/results.json
```

只重新生成 checkpoint 报告，不训练：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_commodity_residual \
  --checkpoint checkpoints/switching_latent_balanced_commodity_residual_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0b_commodity_residual/checkpoint_review \
  --no-cuda
```

输出目录须不存在，避免覆盖此前报告。训练 history 保存全部 epoch；在达到的 1/5/10/20 epoch，以及 best/final，使用固定 TRAIN batch 执行 eval activation/gradient probes。Best checkpoint 对照梯度使用同一个固定 TEST batch。

代码回归：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

## 功能诊断

在新 checkpoint 中比较以下四条预测路径，均保持 checkpoint 参数不变：

1. Native：原 base + alpha × 正常 commodity residual。
2. Alpha=0：直接使用同一 checkpoint 的 base。该 base 包含新训练的 backbone co-adaptation，不是原 D0B checkpoint。
3. Commodity shuffled：只打乱 residual 输入的 commodity 维，base/target/output ordering 不变。全 VAL/TEST 使用同一 seed=42 permutation，不交换 batch 样本或日期。
4. Mean commodity：将每个样本的 commodity nodes 替换为该样本所有 commodity nodes 的均值；保留 common spatial 信息，移除 individual commodity differences。

每个 horizon 报告四项指标、干预 prediction mean/max difference、residual strength/base ratio、effective residual P50/P90/P95/max。VAL/TEST5 逐 commodity 比较 D0B/new/native alpha0；完整保存 ΔMAE/ΔMSE、最好/最差五项、residual 修正强弱，以及 baseline MSE 与 improvement 的 Pearson/Spearman（仅描述）。

正常 TRAIN/VAL/TEST regime、candidate、Z、Base RPE、gate 统计保持；TEST 四 horizon zero-micro/zero-long 和 uniform-generator-routing 保留正常 spatial residual，计算 H/Z utilization 与 RoutingFraction。Uniform routing 只在诊断中递归重算 Z，p recursion 继续使用原 posterior。

固定 batch 的 raw L1/L5/L10/L20、prediction sum 和 total-loss gradients 使用 autograd.grad，记录 alpha signed gradient、head norm 和共享模块 norms，不执行 optimizer.step，也不改写参数 .grad。Gradient conflict 不参与 accept/reject。

## 因果性与重标记解释

Temporal E/H/p/Z 与逐时刻 H/Z readouts 对 full-sequence future perturbation 必须保持 prefix 不变。

TempWeighted node states、h_comm、gate 和 base prediction 是 window-end states，会使用整个已观测窗口。因此 spatial prefix 检查通过截断输入到 prefix end 后重新计算，并比较未扰动/未来扰动输入的同一 prefix；同时检查截断 temporal states 与 full causal prefix 一致。**不要求 full-window spatial states 对窗口内较晚观察不敏感**，也不把它们误称为每个时间步的 prefix state。

合法 within-market node relabeling 会同时 relabel graph embeddings；commodity relabeling 还需按每个 horizon 重排原 head 输出行。此时 base、residual、final prediction 均应按 commodity slot 等变。该检查与故意保留输出顺序的 residual-only shuffle 是两种不同操作。

保留原 FP32 3e-6 invariance threshold。若原始 numerical error 超过阈值，在独立 FP64 model copy 上要求误差≤1e-10；完整保留两套结果，不改变训练精度或参数。如果 FP64 仍存在异常，sanity 硬失败。

## 决策与停止条件

报告逐项回答附件的 26 个研究问题，并给出一个 Case A–G。MAE/MSE 约 0.1% 的 negligible/materiality、|alpha|≤1e-4 的 near-zero、impact/native-MAE≈0.1% 的 functional-effect 界限是固定描述约定，不是调参或统计显著性阈值。Case G 使用按 D0B commodity MSE 固定分成高低两半的误差分解，报告原始差值供人工判断。

当前正式 pooled D0B TEST5 历史参考：MAE=.021994784、MSE=.000912875、RMSE=.030213828、Hit=46.8401%；正式报告以本次实际 checkpoint 重算值为准。任何候选接受必须结合 VAL+TEST 与 residual 功能证据，并检查 MSE tail-error tradeoff；不会自动替换 D0B。

本轮代理仅修改与验证代码；正式训练由用户执行。未训练时不能判定 alpha 是否被 optimizer 接受、是否改善泛化或最终 Case。完成本实验后 STOP，不自动增加 nonlinear adapter、commodity IDs、horizon alpha、其他 loss 或新架构。
