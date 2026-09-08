# D0E-LearnablePersistence

唯一新增正式 variant：`switching_latent_learnable_persistence`。

模型差异只有一个标量 `regime_filter.sticky_logit`，初始化为 0：

```text
alpha = sigmoid(sticky_logit)
A = alpha * I + (1 - alpha) * softmax(transition_logits)
```

其余为原 D0B：level+dispersion、LongMemoryTransformer、linear evidence、G0/G1/G2、完整 Z 递推、BalancedReadout、TempWeighted 空间分支、融合与 head。D0E 不启用 D0D 输入平衡、D0C slope、latent memory 或 routing temperature。

实际默认规模：D0B **520,549** 参数，D0E **520,550** 参数。旧变体默认 `learnable_sticky_alpha=False`，不增加 checkpoint state key。新参数的零初始化不消耗 RNG；旧共享参数初始化顺序保持不变。

## 运行一个 D0E 实验

先确认实际运行的 Python 环境。当前已验证的 PyTorch 版本是
`2.7.1+cu118`；不要在新环境中无版本约束地安装 `torch`，以免装入
现有驱动不支持的 CUDA build。若使用 `myresearch/.venv`：

```bash
source /home/yangxiaotong/projects/myresearch/.venv/bin/activate
python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-geometric numpy pandas scipy pytest
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda"))'
```

最后一行只检查 GPU 分配，不训练。CPU-only 运行可在训练命令中增加
`--no-cuda`。固定 batch 的 CPU RNG 保存显式使用 `devices=[]`，不初始化 GPU；
GPU 上的梯度诊断仅保存实际模型所在设备的 RNG。

在 `Commedities` 根目录执行。下列命令仅训练 D0E；现有 D0B checkpoint 只在训练后的对照中使用，不用于初始化 D0E。

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
nohup python -u -m cmgm.scripts.main_ablation \
  --variants D0E-LearnablePersistence \
  --epochs 200 --patience 10 --seed 42 \
  --batch-size 64 --seq-len 20 \
  --checkpoint-dir checkpoints \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --d0e-report-dir experiments/d0e \
  > d0e.log 2>&1 &
```

```bash
tail -f d0e.log
```

沿用原 `train()` 的 Adam、lr=1e-4、weight_decay=1e-5、Huber delta=.02、ReduceLROnPlateau、early stopping 及多 horizon loss。`sticky_logit` 在同一 optimizer group，使用相同 weight decay。未添加调参循环。

产物：

- `checkpoints/switching_latent_learnable_persistence_best.pt`
- `experiments/d0e/<timestamp>/REPORT.md`
- 同目录 `results.json`，以及 D0B / D0E 子目录中的逐 split 预测和固定批次张量。
- 保留原消融日志/summary，并复用旧 latent diagnostics。

## 仅重新检查已训练 checkpoint

这条命令不训练。`--output` 指定一个尚不存在的目录。

```bash
python -u -m cmgm.scripts.d0e_diagnostics \
  --checkpoint checkpoints/switching_latent_learnable_persistence_best.pt \
  --d0b-checkpoint checkpoints/switching_latent_balanced_readout_best.pt \
  --output experiments/d0e/checkpoint_recheck \
  > d0e_checkpoint_diagnostics.log 2>&1
```

## 记录和诊断

- 每个 epoch 保存 `alpha_history`、`sticky_logit_history`，以及原 train/val loss、LR、switch beta；单独标记 best / final epoch。
- Epoch 1/5/10 和恢复 best 后，在固定 TRAIN batch 记录 prediction-only / total-loss 梯度；best checkpoint 另在固定 TEST batch 给出完整逐 horizon 梯度。
- `total-loss` 指原 Huber 多 horizon 之和加原 regularizer，不把 Adam weight decay 当成额外模型 loss。
- 梯度 probe 使用 eval 模式和 `autograd.grad`，不写参数 `.grad`，恢复模型模式与 RNG；不会改动优化步骤或训练随机序列。
- Checkpoint metadata 包括 variant、Git SHA/dirty 标记、seed、best epoch/loss、参数量、alpha、sticky_logit、初始 transition logits 和训练协议。
- `final_learned_alpha` 属于保存的 **best checkpoint**；`last_training_epoch_alpha` 是训练停止时最后一轮，避免恢复 best 后混淆两者。
- 分别输出 `alpha`、`S=softmax(L)`、`A`、row entropy、sigmoid derivative、L drift 和 A change；用精确分解区分 alpha 和 learned logits 的作用。
- TRAIN/VAL/TEST posterior、prior、candidate 统计；固定批次 micro 动态、BalancedReadout 健康度、state0/1/2 完整递推响应。
- VAL/TEST 四个 horizons 的 MAE/RMSE/Hit、zero-micro、zero-long、uniform routing、RoutingFraction 和 forced-state impacts。
- 训练后的 uniform/forced generator controls 仅在独立诊断 helper 中执行；D0E 的正式 forward 始终用原始 p 加权 G_k，p 递推不接收 q。
- 逐时 E/H/p/Z 和 readout 的 future perturbation 检查；batch permutation/single-sample 检查；各市场内部 permutation 检查。时间分支直接检验不变性；完整图模型同时重排对应 asset embeddings，保留节点身份对应关系。

## 逻辑验证范围

仅使用合成输入运行单元测试、forward、diagnostic gradients 与临时 checkpoint 测试。训练循环的新增 history/metadata/早停恢复用 mocked epoch 验证，不执行 D0E 优化实验。

初始化全规模检查：共享参数 mismatch=0，A 差异=0，E/H/p/prior/evidence/Z/h_long/h_micro/h_temporal/prediction 差异=0。新增 alpha 的预测梯度非零，并通过有限差分检查。

对 S1/S1C/S2F/D0/D0B/D0C/D0D/D1A/D1A2/D1，与修改前快照的 state_dict、eval forward 和 RNG 逐位一致。测试命令（需要 pytest）：

```bash
python -m pytest -q tests
```

实际 alpha 走向、VAL/TEST 是否改善及 Case A–F 的判断，必须等待运行上述训练命令后依据报告回答。报告保留两套 split 和所有 horizons；不自动替换 D0B，不自动启动后续 variant。
