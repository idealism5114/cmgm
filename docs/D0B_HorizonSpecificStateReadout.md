# D0B-HorizonSpecificStateReadout

唯一新增正式变体：`switching_latent_balanced_horizon_readout`。

研究假设是共享市场状态是否需要不同预测尺度的 decoding subspace。仅拆分最后的 `Linear(128,64)`：5d 保留 `state_readout`；1d、10d、20d 为 `horizon_state_readouts` 中的独立参数。通过 `deepcopy` 精确复制权重和 bias，不消耗 RNG，因此整个 full model 的共享参数初始化顺序保持不变。

E/H/p/G/Z、Balanced H/Z projections 和 LayerNorm 完全共享；每 batch 只计算一次 temporal dynamics。新 temporal 输出为 `(B,4,64)`。共享 spatial vector 按 horizon 展开，同一套 gate、lstm_proj、gcn_proj、head body 分别处理四个 temporal vector，原 head 最后一层的对应 horizon output rows 产生 `(B,4,24)`。不增加 gate、head 或 horizon dynamics 参数。

旧 D0B 的 `readout()` 和 `_market_token_predict()` 语义不变。5d 原始参数路径保留。固定 persistence=.5，K=3，z_dim=64，d_model=128。仅增加 `3*(128*64+64)=24,768` 参数：520,549 → 545,317。

训练仍是 `sum_h Huber(delta=.02) + switch KL`，四 horizon 系数均为 1；switch beta_max=5e-4，warmup=20。Validation selection、scheduler 和 early stopping 仍为原四-horizon prediction loss sum；val switch/total loss 单独记录，不参与选择。继续原 mean-of-batch-means selection 协议，正式 performance 采用 pooled population evaluator。

seed=42、batch=64、seq_len=20、epochs=200、patience=10、Adam lr=1e-4/weight_decay=1e-5、原 ReduceLROnPlateau 全部保留。不使用任何旧 objective 变体的 loss。训练入口对这些控制量做检查。

## 验证与运行

以下命令从仓库根目录运行：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
source ../.venv/bin/activate
```

真实数据 preflight（只有 forward 和诊断 autograd.grad，不执行正式训练）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_horizon_readout \
  --preflight --no-cuda \
  --output experiments/d0b_horizon_readout/preflight_manual
```

唯一正式训练命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.main_ablation \
  --variants D0B-HorizonSpecificStateReadout \
  --seed 42 --batch-size 64 --seq-len 20 --epochs 200 --patience 10 \
  --no-cuda \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --checkpoint-dir checkpoints \
  --horizon-readout-report-dir experiments/d0b_horizon_readout
```

这里给出 CPU 命令，与此前 CUDA 驱动不匹配的环境兼容。已有可用 CUDA 环境时移除 `--no-cuda`；设备选择不改变训练配置。不要同时选其他 variants。

训练先验证同 seed 初始化与 prefix/batch/market sanity，再进行一次训练。新模型从 seed-42 initialization 开始；D0B checkpoint 仅作 reference，不用于初始化新模型。已有新变体 checkpoint 时拒绝覆盖或再次训练。

输出 checkpoint：

```text
checkpoints/switching_latent_balanced_horizon_readout_best.pt
```

训练后自动重新加载 D0B 和新 best checkpoint，生成：

```text
experiments/d0b_horizon_readout/<timestamp>/preflight.json
experiments/d0b_horizon_readout/<timestamp>/checkpoint_diagnostics/REPORT.md
experiments/d0b_horizon_readout/<timestamp>/checkpoint_diagnostics/results.json
```

仅重新生成 checkpoint 报告（不训练，输出目录须不存在）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u -m cmgm.scripts.d0b_horizon_readout \
  --checkpoint checkpoints/switching_latent_balanced_horizon_readout_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0b_horizon_readout/checkpoint_review \
  --no-cuda
```

代码回归测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

## 报告内容与解释

- 初始化：全部 shared 参数 name/shape/value、clone weight/bias、E/H/prior/p/candidates/Z/H-Z readouts/spatial/gate/prediction、raw per-horizon loss、warmup 前后 switch/total loss。eval dropout 关闭后比较；完整 prediction 只允许浮点容差。
- TRAIN/VAL/TEST 完整 loader 含尾 batch；四 horizons 均报告直接计算的 MAE/MSE/RMSE/Hit%，以及 RMSE²−MSE。Hit 使用全体样本 sign equality，不屏蔽小 target，不混用 legacy RMSE。
- 重新加载的正式基线是 D0B。历史 pooled TEST5 参考为 MAE=.021994784、MSE=.000912875、RMSE=.030213828、Hit=46.8401%；报告始终以实际 checkpoint pooled evaluation 为准。旧 checkpoint 未保存的训练时长/best epoch 标记为不可用，不编造。
- Readout weight/bias norm、初始化 drift、relative distance from 5d、六对 weight cosine；全 loader temporal pairwise cosine/L1/L2、每 horizon gate mean/std/min/max。
- Shared-readout counterfactual：临时让四个 horizon 都用当前 5d readout，checkpoint 与参数不变。5d prediction impact 必须为零；其余 horizon 的差异和反事实 metrics 衡量功能性分化。高 weight cosine 不等于没有功能。
- 每 horizon zero-balanced-micro/long impact 及比率；正常 regime/prior/KL/L1/occupancy、candidate specialization、Z dynamics、BaseRPE/QK。Checkpoint hash 与参数前后比对。
- 固定同一 TEST batch，raw L1/L5/L10/L20 的 shared 模块 gradients、cosines；各 readout 的 loss×module gradient norm 矩阵必须只在对应 horizon 非零。单独比较同形状 `∇W_h L_h` 和真正共享 readout input `∇u L_h`。使用 autograd.grad，不写入参数 .grad、不执行 optimizer step。
- TEST5 每 commodity 四项指标、最高/最低 MSE 各五项、MSE CV。
- Case A–F 与 20 项研究问题逐项报告，包含具体 relative changes。约 0.1% 的 MAE/MSE negligible convention、MSE materiality convention，以及 weight/functional-effect convention 均固定用于描述，不据此调参，不当作统计显著性或多 seed noise estimate。

Causality 检查逐个时间的 E/H/p/Z/H-Z balanced outputs 及所有 horizon temporal output prefix。全窗口 spatial pooling 本身不作为 prefix 预测。Within-market full-model permutation 同时 relabel asset graph embeddings；保持原输出 commodity rows 语义。若 FP32 permutation 超过原 3e-6 阈值，只在 FP64 独立副本上复核 roundoff，保留两套原始数值，不改变训练精度/阈值。

按用户确认，本轮代理只实现与验证，正式训练由用户执行。未训练不能判定泛化或 Case，也不能据初始化克隆等价性声称 readout specialization 有效。完成这一个实验后 STOP，不自动实施 commodity decoder、loss tuning 或其他模块。
