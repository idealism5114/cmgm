# D0B-MidLongGroupedObjective

唯一新增正式训练诊断变体：`switching_latent_balanced_readout_5_10_20`。

模型直接共享 D0B construction / forward，参数量同为 **520,549**。仍输出
`[B,4,N_commodities]`，保留 `[1,5,10,20]` 四个 horizons。固定 alpha=.5，
BalancedReadout、linear evidence、G0/G1/G2、native mixture、spatial、fusion/head 不变。
不启用 D0C/D0D/D0E 或其他结构。

## 唯一 objective 改动

```text
group_raw   = L5 + L10 + L20
group_scaled = (4.0 / 3.0) * group_raw
train total  = group_scaled + original switch loss
validation   = group_scaled
```

名称为 **scale-matched grouped-objective diagnostic**。固定4/3；通过
`MULTI_HORIZONS.index(h)` 分别查找5/10/20，不写死 index，不搜索乘数。
训练与验证调用同一个 `_prediction_loss` helper。Scheduler、early stopping、best
checkpoint 选择使用 grouped validation objective；raw val L5 与四项总 loss 仅报告。

D0B 仍使用四项求和，5d-only 仍使用4×L5；所有旧 variants 保留原行为。
Adam、lr=1e-4、weight_decay=1e-5、Huber delta=.02、ReduceLROnPlateau、seed42、
batch64、seq_len20、epochs200、patience10、split/normalization 完全沿用原配置。

## 仅运行这一个新实验

使用已验证 PyTorch/CUDA 兼容性的环境，从项目根目录执行：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup python -u -m cmgm.scripts.main_ablation \
  --variants D0B-MidLongGroupedObjective \
  --epochs 200 --patience 10 --seed 42 \
  --batch-size 64 --seq-len 20 \
  --checkpoint-dir checkpoints \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --five-day-checkpoint checkpoints/switching_latent_balanced_readout_5d_only_best.pt \
  --grouped-report-dir experiments/d0b_grouped \
  > d0b_grouped.log 2>&1 &
```

```bash
tail -f d0b_grouped.log
```

CPU-only 可追加 `--no-cuda`。固定 TRAIN batch 的 RNG context 明确为 `devices=[]`；
GPU diagnostics 只保存模型实际使用的 GPU RNG，避免意外初始化其他设备。

两个 reference checkpoint 必须已有，**不重新训练它们，不用于初始化 grouped**。
训练前用独立同 seed 的 D0B 验证共享初始化，报告同一固定 TRAIN batch 的
L1/L5/L10/L20、sum_multi、group_raw、group_scaled、scale ratio。

输出：

- `checkpoints/switching_latent_balanced_readout_5_10_20_best.pt`
- `experiments/d0b_grouped/<timestamp>/REPORT.md`
- 同目录 `results.json`，以及 `D0B/`、`5dOnly/`、`Grouped/` 的固定 batch 张量、各 split 预测/targets。

训练结束严格重新加载 grouped best checkpoint，在同一环境中重新评估三组 checkpoint。
完成本轮诊断后停止，不训练任何其他 horizon 组合。

## 已有 checkpoint 的独立诊断命令

```bash
python -u -m cmgm.scripts.d0b_grouped_diagnostics \
  --checkpoint checkpoints/switching_latent_balanced_readout_5_10_20_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --five-day-checkpoint checkpoints/switching_latent_balanced_readout_5d_only_best.pt \
  --output experiments/d0b_grouped/checkpoint_recheck \
  > d0b_grouped_checkpoint.log 2>&1
```

输出目录必须尚不存在。此命令只做 forward / diagnostic gradients，不训练，不改权重。
模型 state_dict 相同不足以区分 objective，因此 grouped/5d-only checkpoint 会核对
metadata 的 objective 与 variant，避免把 D0B 错当成另一个训练目标的结果。

## History / metadata

每轮保存 raw_L1（仅描述）、raw_L5、raw_L10、raw_L20、group_raw、group_scaled、
switch_loss、total_loss、val group_scaled、raw val L5、LR、switch beta，标记best/final。
Raw L1 通过已有 forward 的 detached prediction 计算，不产生 prediction gradient。

Checkpoint metadata 保存 variant、Git SHA、关键源文件哈希、seed、best epoch、
best grouped val loss、raw val L5、参数量、objective=`4/3*(5d+10d+20d)`、固定乘数、
初始 transition logits、完整训练协议。Raw val L5 从 best epoch 的独立记录读取，
不能把 grouped validation loss 除以4/3当成 raw L5。

Grouped 单独保存 `training_elapsed_seconds`（优化器准备与训练循环，不含后续报告）。
历史 D0B / 5d-only checkpoints 没有保存纯训练耗时，报告为未记录，不把包含诊断的
ablation wall time 冒充训练时间，不修改历史 checkpoint。

## 三组诊断

- 完整 TRAIN/VAL/TEST 四 horizons MAE/RMSE/Hit，Grouped 相对 D0B 和5d-only的ΔMAE/relative change。
- Posterior/prior、temporal L1、L drift、A matrix、candidate norms/pairwise L1/cosine/weighted contributions。
- 固定 TEST 的 Z/deltaZ/consecutive cosine/Z:H、H/Z/temporal/spatial norms、gate mean/std。
- Base RPE norm、causal lower-triangle mean |QK|、mean |base bias|、base/QK。QK 是缩放后、加bias前的内容logits，按层等权汇总。
- W_H/W_Z projection Frobenius norms 与比例、balanced h_long/h_micro；state_readout 两输入块范数另列，避免混淆。
- VAL/TEST 每 horizon 的 zero-micro/zero-long/micro-long impact ratio、uniform impact/RoutingFraction、state0/1/2 full-recursion impact。
- Fixed TEST 每 horizon 的 raw single-horizon Huber gradients：不乘4/3、不含switch、不更新参数。
- 三种目标的 module-level cos(5,1)、cos(5,10)、cos(5,20)，覆盖 evidence、L、generators、balanced readouts。
- Grouped 相对两个 references 的 representation/prediction drift，三组统一的 prefix causality、batch independence、within-market permutation sanity。

Uniform / forced controls 仅用于 checkpoint diagnostics，保持原 p recursion，完整重算Z。
不在训练中引入 q、不改变正常 forward。复用此前诊断函数仅共享数值检查，不使用 D0E backbone。

## 解释边界

报告明确回答附件的17个问题。Case A 必须同时在 VAL/TEST 的5d上明显优于 D0B 和5d-only，
10/20d仍健康，并通过机制sanity；不能只看cosine。

效应大小约定：0.1% relative MAE 用于“明显”与微小改善的区分；10/20健康度的保守约定是
VAL/TEST MAE、RMSE 不恶化超过0.1%。1e-6 relative MAE 仅作为数值持平容差。
这些是报告约定，不参与训练、选模或调参，不表示统计显著性检验。

题设Case未覆盖所有可能结果：例如Grouped明显优于D0B，但不优于5d-only，或10/20明显退化。
这类结果标为“未分类，需复核”，不强行判Case A、不提升candidate。
Case C/D/E保留D0B；B不夸大为主要瓶颈。即使A成立，也只报告grouped objective具备候选资格，随后STOP。

## 本轮验证范围

只运行合成单元测试、forward/gradient和临时checkpoint报告测试，没有真实市场训练或评估实验。
实际固定TRAIN batch的scale ratio、VAL/TEST性能、三组机制变化及最终Case，需运行上述命令后判断。

```bash
python -m pytest -q tests/test_d0b_grouped_objective.py
python -m pytest -q
```
