# D0B-NoSwitchKL

唯一新增正式训练消融变体：`switching_latent_balanced_readout_no_switch_kl`。

模型与D0B共用construction和forward，参数量均为 **520,549**。共享初始化、固定TRAIN和TEST
的E/H/prior/evidence/p/candidates/Z/readouts/spatial/gate/prediction完全一致。

## 唯一差异：关闭KL对训练loss的贡献

```text
prediction = L1 + L5 + L10 + L20   (原Huber，delta=.02)
D0B total = prediction + scheduled switch KL
NoSwitchKL total = prediction
validation = L1 + L5 + L10 + L20   (两者均不加KL)
```

新variant仅设置Python loss配置 `disable_switch_kl=True`。Training层的
`_effective_switch_loss`返回不连接KL梯度图的独立零值。MarkovRegimeFilter不变：
prior、posterior、evidence、KL计算、`switch_loss()`、原beta日程全部保留。
Filter的 `current_beta` 是原日程的**诊断reference**；实际训练 `beta_effective=0`。
这两者在日志和metadata中明确区分。

没有使用5d-only/grouped prediction objective、缩放、routing温度、learnable alpha、
slope、balanced transition input、latent memory或其他新结构。固定alpha=.5、tau=1、K=3。
Adam、单一参数组、lr=1e-4、weight_decay=1e-5、scheduler、seed42、batch64、seq20、
epochs200、patience10、split和预处理保持原样。

## 只运行NoSwitchKL

在已解决PyTorch/CUDA兼容性的环境中执行：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup python -u -m cmgm.scripts.main_ablation \
  --variants D0B-NoSwitchKL \
  --epochs 200 --patience 10 --seed 42 \
  --batch-size 64 --seq-len 20 \
  --checkpoint-dir checkpoints \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --no-switch-report-dir experiments/d0b_no_switch_kl \
  > d0b_no_switch_kl.log 2>&1 &
```

```bash
tail -f d0b_no_switch_kl.log
```

CPU-only可追加 `--no-cuda`。固定batch的CPU RNG context使用 `devices=[]`，GPU probes只保存
实际使用GPU的RNG，不意外初始化所有GPU。

D0B reference只在训练后重新评估，不用于初始化NoSwitchKL、不重新训练D0B。
初始化用独立same-seed D0B检查同名参数、forward、prediction loss和总损失差。
Total-loss sanity检查epoch1和epoch20两个reference beta，随后恢复epoch，不训练或更新权重。

产物：

- `checkpoints/switching_latent_balanced_readout_no_switch_kl_best.pt`
- `experiments/d0b_no_switch_kl/<timestamp>/REPORT.md`
- 同目录 `results.json` 及 `D0B/`、`NoSwitchKL/` 的固定张量、完整split预测和targets。

训练后严格加载best checkpoint，完成本轮对照诊断，然后STOP。

## 已训练checkpoint的独立诊断

```bash
python -u -m cmgm.scripts.d0b_no_switch_kl_diagnostics \
  --checkpoint checkpoints/switching_latent_balanced_readout_no_switch_kl_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0b_no_switch_kl/checkpoint_recheck \
  > d0b_no_switch_kl_checkpoint.log 2>&1
```

输出目录须尚不存在。新checkpoint必须有正确variant及 `switch_kl_enabled=false`、
`beta_effective=0` metadata，避免因与D0B state_dict同构而误识别。

## History和梯度语义

- 每epoch记录已有TRAIN forwards的prediction loss、raw KL、posterior/prior entropy、
  posterior-prior L1、weighted switch loss=0、total loss、VAL prediction loss、LR。
- 另外在固定TRAIN batch上记录epoch1/5/10/20/final/best的eval KL trajectory。
  Final在恢复best前记录，best在恢复后记录；probe恢复RNG和mode，不更改训练。
- Metadata保存variant、Git SHA、源文件哈希、seed、best epoch/loss、参数量、
  `switch_kl_enabled=false`、`beta_effective=0`、reference beta/warmup和训练协议。
- Best checkpoint固定TEST分别计算prediction、actual total、raw KL、假想weighted KL梯度。
  Prediction始终是四个raw Huber之和。
- 假想KL采用**该checkpoint best epoch对应的原D0B beta日程**，不是新增beta实验。
  Epoch1的weighted gradient为0，cosine为null；raw-KL gradient仍报告。
- 使用 `autograd.grad`，不写参数 `.grad`、不执行optimizer step；NoSwitchKL的actual total
  gradients逐项等于prediction gradients。Counterfactual梯度不参与任何更新。
- Per-horizon raw Huber gradients和cos(5,1)/(5,10)/(5,20)单独报告，不改prediction objective。

## 保留的机制检查

完整TRAIN/VAL/TEST regime统计、min p、transition matrix/diagonal/row entropy/logit drift、
evidence weight/bias norms、candidate norms/pairwise L1/cosine/weighted contribution。
固定TEST Z/deltaZ/consecutive cosine/Z:H、balanced norms/weights、Base RPE/QK。

完整VAL/TEST四个horizons与overall的zero-micro/zero-long、uniform impact、RoutingFraction，
以及state0/1/2完整递推的Z_T/h_micro/prediction impact。Z和h_micro是共享表征，
不是horizon-specific变量；报告按horizon区分prediction impact。

Uniform/forced routing仅是checkpoint控制；正常p recursion保持，完整重算Z。
原D0B约2%的overall RoutingFraction不与5d-specific比例混用。

NoSwitchKL研究对future perturbation的E/H/p/Z/readout prefix要求严格0差异。
Batch/single-sample及within-market tests沿用浮点容差；完整图置换同步asset embeddings。

## 结论边界

报告回答附件18个问题，结合VAL/TEST与功能性routing判Case A–F。
Argmax occupancy>95%只作concentration warning，不能单凭它判collapse。
Hard-posterior描述阈值为entropy<.1、mean max>.99、min p<1e-4；若伴随性能下降，
明确标记over-specialization。更低entropy不能自动解释为更好。

约0.1% relative MAE用于报告效应大小；数值持平、机制近似相等和candidate近同的容差
都写入报告，不参与训练、选模或超参数调整。单seed不是统计显著性证明。
题设Case未覆盖的组合标为未分类并保留D0B，不强行判成功。
只有VAL/TEST和RoutingFraction均改善、机制检查健康时才支持Case A；不自动替换模型。

## 本轮验证范围

只做代码和合成测试，没有运行真实训练或市场数据评估实验。
实际KL/entropy/routing变化、泛化收益与Case结论须等运行上述命令后判断。

```bash
python -m pytest -q tests/test_d0b_no_switch_kl.py
python -m pytest -q
```
