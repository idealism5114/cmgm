# 时序基准＋门控时空交互修正

独立 variant：`switching_latent_balanced_candidate_gated_interaction_residual`。
这是候选关系与交互路径的整体结构实验，不预设预测改善、动态专家选择成功或消融贡献恢复；不能把结果单独归因于其中某一个子改动。

原 Candidate64 的时序基准保持不变：

```
e_base = t + Linear64,64(Dropout(ReLU(Linear64,64(t))))
a_s = ReLU(Us(s)); a_t = ReLU(Ut(t))
delta_ST = Wo(Dropout(a_s * a_t))
e_T = e_base
e_ST = e_base + delta_ST
pi = 原 CandidateAwareRouter(e_T, e_ST)
h = pi_T * e_T + pi_ST * e_ST = e_base + pi_ST * delta_ST
prediction = 原 shared head(h)
```

TemporalResidualExpert 每次 forward 仅调用一次，两个候选复用同一张量。
Us/Ut/Wo 均为 Linear(64,64,bias=False)，普通 Linear 初始化，Wo **不**置零。
修正路径 dropout=.1，无其它 norm、残差、alpha、warmup 或正则。
两个 Router LayerNorm 只服务权重计算；实际混合 raw candidates。
Router 保持 256→64→2，最后一层零初始化，初始 pi=.5/.5。

投影后的 s=0 或 t=0 时 delta_ST=0。这不等于删除整个 backbone 分支。
初始化时通常存在非零修正，因此不声称初始模型严格等于 TemporalOnly。

| 参数（N=284） | Candidate64 | 新变体 |
|---|---:|---:|
| TemporalResidualExpert | 8,320 | 8,320 |
| ST Expert / 交互修正 | 12,416 | 12,288 |
| 两部分合计 | 20,736 | 20,608 |
| Candidate Router | 16,834 | 16,834 |
| 总参数 | 549,863 | 549,735 |

先完整构造原 Candidate 融合模块，再删除旧 ST Expert、创建修正模块。
Temporal Expert、router、两条 backbone、branch projections、shared head 的
共享初始化逐元素不变。不加载已训练参数初始化新模型，不保留无效 ST 参数。
旧 Candidate64/Bottleneck16/Global/T+S/Utility/旧融合类均不更改。

## 运行命令

在仓库根目录，使用已有虚拟环境：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
../.venv/bin/python -m cmgm.scripts.d0b_candidate_gated_interaction
```

默认只做 CPU 合成预检，不读取真实数据、不读取参照 checkpoint、不训练。

可选服务器 TRAIN/VAL 数据预检（仍不训练）：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_candidate_gated_interaction \
  --data-preflight --device cuda --seed 42
```

只有显式 `--run` 才启动一次正式训练：

```bash
nohup ../.venv/bin/python -u -m cmgm.scripts.d0b_candidate_gated_interaction \
  --run --device cuda --seed 42 \
  > candidate_gated_interaction_seed42.log 2>&1 < /dev/null &
```

日志：

```bash
tail -f candidate_gated_interaction_seed42.log
```

每次自动生成独立时间戳目录。不要把 preflight 目录用作训练输出目录。
可用 `--output PATH` 指定不存在的新目录；不覆盖旧文件。
每 seed 正式训练登记和文件锁防止重复启动；中断训练需人工检查，不自动重跑。
没有 TEST 选项、消融队列、宽度搜索或多 seed 循环。

冻结完成 checkpoint 的 TRAIN/VAL 续评（不训练）：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_candidate_gated_interaction \
  --evaluate-completed experiments/d0b_candidate_gated_interaction_residual/run_seed42_TIMESTAMP \
  --device cuda
```

## 数据和训练

直接复用已修复并审计的 Bottleneck16 TRAIN/VAL-only 入口，不复制数据管线。
按既有日期元信息、正式审计行数及 TRAIN 缓存恢复同一历史节点顺序；严格比较
TRAIN/VAL feature_matrix+raw_prices 的完整 SHA256 和商品顺序。
不构建 TEST Dataset、不计算 TEST 特征/目标、不评价 TEST。

默认数据审计：
`experiments/baseline_comparison_deep_v3/20260922_143029/data_audit.json`。
必要时使用已有 TRAIN schema cache：
`experiments/formal_baseline_benchmark_v2/20260910_131615/arrays/train_x.npy`。
缺失或指纹不一致则 STOP，不偷偷改资产集合或标准化。

原 `cmgm.training.train.train` 负责训练：Adam1e-4、WD1e-5、batch64、max200、
patience10、ReduceLROnPlateau(.5,patience5)，TRAIN shuffle=False/drop_last=True。
完整 TRAIN/VAL 评估 drop_last=False。
四周期等权 Huber(.02)+原 native Switch KL；beta=.0005*clamp((epoch-1)/19,0,1)。
选择/早停/scheduler 使用 prediction-only 四周期 VAL Huber **batch mean**。
VAL5 仅诊断；不增加辅助损失、gradient clipping 或 optimizer groups。

参照优先只读使用原 Candidate64 的指定正式 results/checkpoint。
核验精确 variant、seed、协议、数据指纹、商品顺序、checkpoint SHA256、
metadata、历史选择规则、KL 调度和 strict load；仅保留参照 TRAIN/VAL 字段。
缺失或不合法则记 PENDING，不用 TEST 数字补位、不自动重训原模型。

## 输出

结果：
`experiments/d0b_candidate_gated_interaction_residual/run_seed42_TIMESTAMP/`

Checkpoint：
`checkpoints/candidate_gated_interaction_residual/seed42/run_seed42_TIMESTAMP/switching_latent_balanced_candidate_gated_interaction_residual_best.pt`

正式运行保存：

- `config.json`、`source_hashes.json`、`data_audit.json`、`reference_provenance.json`。
- `initialization_audit.json`、`structural_sanity.json`。
- `training_history.json`、`routing_history.json`（逐 epoch）。
- `best_checkpoint_metadata.json`：best epoch、正式选择损失、SHA256、训练秒数。
- `formal_train_val_predictions.npz`：先保存正式预测，再执行干预。
- `train_val_metrics.json`、`diagnostics.json`：四周期 MAE/MSE/RMSE/Hit；正式
  prediction-only Huber batch mean 写入 results；routing 均值/std/分位数/范围。
- `diagnostics.json` 还包含 e_base、delta_ST、pi_ST*delta_ST、h 的 L2 范数分布，
  effective/base 比值（分母+1e-8），base/delta 的余弦分布及零范数未定义数量。
  两侧任一范数为0即排除并记录；全为0时余弦分布为 null，不伪造0。
- `zero_correction_predictions.npz`、`zero_correction.json`。
- `results.json`、`REPORT.md`：TRAIN/VAL 结果、参照、干预和限制，Hit 乘100显示。

ZeroCorrection 取**同一次**冻结 eval forward 缓存的 e_base，再通过同一个
已训练 shared head；不重算 Temporal Expert，不更新参数。

**ZeroCorrection 是冻结 checkpoint 的推理干预，不能替代重新训练后的消融，也不能证明 backbone 机制贡献恢复。**

默认合成预检保存参数/初始化/公式/梯度/causal-prefix 审计。
预检通过不等于正式收敛或预测改善；这些结论只能在用户启动正式训练后评估。
