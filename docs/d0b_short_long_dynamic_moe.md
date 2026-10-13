# D0B 短／长任务条件化动态 MoE

状态：首次交付只完成实现与合成验证；之后用户已经完成六组 seed42 TRAIN/VAL 训练和冻结干预，汇总见 `experiments/d0b_short_long_dynamic_moe/summary_seed42_20261013_131817_245598/summary_report.md`。后续代码复核只做源码、已有 checkpoint 和合成输入检查，没有新增正式训练或读取 TEST。不预设 MAE 改善。

## 代码审计及研究边界

起始 commit：`b96b90d4cec13b0579f9ca19999f1d52ad9a98a0`（10.12）。起始工作树干净；未发现适用的 AGENTS.md。每次运行自动记录当时 Git commit、工作树状态及模型、特征、训练、评估代码 SHA256。旧模型、旧实验结果和默认 D0B 均保留。

现有 `short_long_mask_moe.py` 的静态路由符合旧规格，不是运行 bug。本次核实的设计限制如下：

- `routing_weights()` 不读取样本；它只对两组常量 logits 做 softmax。
- 旧融合把 E0 固定加到专属专家混合上，无法通过共享候选的权重关闭全部专属修正。
- 旧 ES2 直接比较已投影的 s 与 latent 差分，缺少显式表征对齐；A 组故意保留这个函数以控制变量。
- Z 是递推状态，末 5 步摘要仍可能包含更早的信息。
- `H_last−H_first` 是状态差，不能自动称为经济长期趋势；20 步也不是无限长期记忆。
- 交换旧组内权重只干预混合系数，不能独自证明专家分工。

`_temp_weighted_spatial(x)` 产生 `h_s(B,64)`；`switching_latent_transformer(x, return_live_sequences=True)` 在同一次前向中返回 `h_t(B,64), H(B,20,128), Z(B,20,64)`。H/Z 保持梯度，日志缓存才 detach。无需修改 temporal branch 接口或内部计算。

## 调用链与固定维度

`HeteroMixHopCMGM.forward → _switching_latent_transformer_forward → short_long_dynamic_moe → predict_by_horizon`。

保留原投影，`s=gcn_proj(h_s)`、`t=lstm_proj(h_t)`。这里 router 的输入是这两个投影后全局向量；B 的原 gate 仍读取原始 `concat(h_s,h_t)`。

维度由主模型构造参数传入融合模块，并显式校验本研究固定 `d=64,H_dim=128,Z_dim=64,T=20`；其他配置拒绝运行，不静默 reshape。所有 variant 使用 `[1,5,10,20]`，输出 `(B,4,Nc)`，当前 Nc=24。

## 两组匹配设计

每次计算五个专家；固定任务访问掩码：short 只能用 `[E0,ES1,ES2]`，long 只能用 `[E0,EL1,EL2]`。1/5 日共用同一融合表示，10/20 日共用另一表示。掩码本身不学习。

static：两组零初始化三维 logits，`softmax(theta)`，样本间共享。

dynamic：`u=[LN_s(s),LN_t(t)]`；两个独立 `Linear(128,16) → GELU → Linear(16,3)`，输出 stack 后 softmax，**pi shape `(B,2,3)`**，温度为 1。动态 router 最后一层 `weight ~ Normal(0, 1e-3)`，bias=0；其余 Linear 默认初始化、LN 标准初始化。没有 top-k、负载均衡、路由 warmup、熵损失或任务差异正则。

A 组直接 import 旧五专家类，内部计算不变：

| 专家 | 函数 | 参数量 |
|---|---|---:|
| E0 | `LN((s+t)/2 + W[(U_s s)⊙(U_t t)])`, rank 8 | 1,664 |
| ES1 | `LN(MLP16([Z_last,mean(Z_last5),Z_last−mean(Z_last5)]))` | 4,304 |
| ES2 | `LN(MLP16([s,dZ,abs(s−dZ)]))` | 4,304 |
| EL1 | `LN(MLP16([H_last,mean(H),H_last−H_first]))` | 7,376 |
| EL2 | 旧 full-window low-rank spatial/H-mean interaction | 9,920 |

A 融合为 `LN_f(sum_k pi_g,k * E_g,k)`；这是三候选凸组合，**不同于**旧 `LN_f(E0+两专属专家加权和)`。A-dynamic vs A-static 才是本轮的路由匹配比较。

B 保留 D0B 的完整门控路径与计算顺序：

```text
g = sigmoid(gate_fc([h_s,h_t]))
t = lstm_proj(h_t)
s = gcn_proj(h_s)
z_base = g*t + (1-g)*s
```

四个残差如下。所有 MLP 均为 `Linear(input,16) → GELU → Linear(16,64)`；末层 weight/bias 全零，末层后没有其他变换。

| 残差 | 实际输入处理 | 参数量 |
|---|---|---:|
| ΔS1 | `MLP16(LN128([Z_last,mean(Z_last5)]))` | 3,408 |
| ΔS2 | `a=LN64(P_s(s)); b=LN64(P_d(dZ)); MLP16([a,b,a⊙b])` | 12,752 |
| ΔL1 | `MLP16(LN256([H_last,mean(H)]))` | 5,712 |
| ΔL2 | `a=LN64(P_s_long(s)); b=LN64(P_h(mean(H))); MLP16([a,b,a⊙b])` | 16,848 |

交互投影均输出 64 维；短/长专家的投影和 LN 完全独立。残差共 38,720 参数。`E0=z_base`，其余完整专家是 `z_base+Δ`。

```text
z_short = z_base + pi_short,1*ΔS1 + pi_short,2*ΔS2
z_long  = z_base + pi_long,1*ΔL1 + pi_long,2*ΔL2
```

共享候选 `pi_g,0` 参与三候选 softmax，接近 1 时抑制专属修正。B 的残差和之后没有新 LN。专家名称只是输入和结构偏置，不能直接推断经济含义或任务分工。

## 初始化与预测头

先构造一次新的 D0B，然后显式复制保留的 state_dict key 到各候选，`load_state_dict(strict=True)`；B 包含 gate，A 和旧三专家移除 gate。随后 static→dynamic 复制同组全部非-router key（包括专家）。审计记录复制清单、张量数、参数元素数、差值和 PASS。未加载任何训练 checkpoint，不冻结主干。

所有新模块在 D0B 共享模块完成后才构造。static 也消耗与 dynamic 相同的 router 构造 RNG，然后删除不用的 router 模块；没有未使用的注册 router 参数。审计保存/恢复 Python、NumPy、PyTorch、CUDA RNG；正式 fit 在构造之后统一重新 seed，不让审核影响 Dropout 的起始随机状态。该 RNG 策略明确记录，因此旧历史结果不能直接充当本轮六组匹配结果。

沿用共享 `Linear(64,64) → ReLU → Dropout(.3) → Linear(64,96)`。对四周期表示分别应用同一 head 前三层，再选择最后层对应周期的24行。eval 下验证与逐周期调用、以及 B 零残差的原 D0B 输出等价（浮点容差）。train 时 `(B,4,64)` 上的 Dropout 给周期各自采样 mask，而原 D0B 的 `(B,64)` mask 在四周期输出前共享，故**不声称训练轨迹完全等价**；A/B static-dynamic 配对使用相同 head 执行方式。

B 第一步 router 梯度为零是专家相同的预期结果；残差末层先学习后，router 和内部层能够得到梯度。没有为了测试取消零初始化。

## 实际参数量（284 节点，全参数训练）

| 配置别名 | 总参数=可训练参数 | 路由参数 |
|---|---:|---:|
| d0b | 520,549 | 原 gate 8,256 |
| three-global | 541,352 | 原三专家全局 logits 3 |
| a-static | 539,995 | 6 |
| a-dynamic | 544,475 | 4,486 |
| b-static | 559,275 | 6 |
| b-dynamic（主模型） | 563,755 | 4,486 |

每组 dynamic 比 static 多 4,480 个参数，来自 router MLP 和两个 router-only LN。A/B 间同时改变专家、共享路径、归一化及容量，不能视为单因素因果对照；动态/静态间也存在上述 router 参数量差异。

## 训练与数据隔离

保持原协议：seed42；batch64；Adam lr=1e-4、WD=1e-5；最多200 epoch，patience10；ReduceLROnPlateau factor .5、patience5；TRAIN shuffle=False/drop_last=True，完整 TRAIN/VAL 评估 drop_last=False。

损失为四周期 Huber(delta=.02) 等权和加 native Switch KL。实际 KL 日程为 `5e-4 * clamp((epoch-1)/19,0,1)`，第1 epoch=0、第20 epoch=5e-4。checkpoint/scheduler/early stopping 均沿用 **prediction-only 四周期 VAL Huber，batch-mean 聚合**；VAL5 MAE 仅用于跨配置比较。直接使用原 `train()`，无单独训练器。

新入口不调用会创建 TEST dataset 的完整旧数据路径。只读 CSV 的日期列元数据建立候选日历；原 pivot_table 会丢弃没有有效收盘价的日期，因此不能直接按候选日历重新计算 70%/15% 边界。加载器由审计中的 origins、seq_len 和最大 horizon 反推固定 TRAIN/VAL 行数，先读取相应数量候选日期，再按有效日期缺口逐步延长截断点。每次只增加缺失行数，保证不会越过所需有效 VAL 末日。价格行通过 parser skiprows 排除截断点后的记录，再复用原 loaders/feature builder 的 TRAIN 拟合归一化、特征和标签代码。日期列可能包含 TEST 日历元数据；**不解析 TEST 价格、特征、标签，不创建 TEST loader**。通过原始审计 config 固定资产/商品顺序及 TRAIN/VAL 指纹，任一差异在训练前 STOP。商品顺序来自原审计，不根据当前截断文件重新选择资产。

默认参考仅为先前 D0B seed42 的 `config.json`（没有读取其 TEST 指标或 checkpoint）：

```text
experiments/d0b_short_long_mask_moe/switching_latent_balanced_readout_seed42_20261012_173338_417491/config.json
```

路径变化用 `--reference-config /absolute/path/config.json`。缺失或不匹配时拒绝训练，不猜测另一个实验。合成 CSV 已验证原特征口径和未来行排除，并覆盖开头和截断点附近存在全缺失收盘价日期的情况。实际日历重建步骤记录为 config.json 中的 calendar_audit；缺失日期不会补值、改变划分或放宽指纹检查。

## 命令（本次未执行正式训练）

仓库根目录：`/home/yangxiaotong/projects/myresearch/Commedities`。每条命令是一行。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python -m cmgm.scripts.d0b_short_long_dynamic_moe
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python -m pytest -q tests/test_short_long_dynamic_moe.py
```

以后明确授权后顺序运行下面六条，每条创建独立时间戳目录：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant d0b --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant three-global --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant a-static --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant a-dynamic --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant b-static --seed 42 --device cuda:0
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --run --variant b-dynamic --seed 42 --device cuda:0
```

汇总六个明确路径，不按修改时间自动选 seed/checkpoint：

```bash
../.venv/bin/python -m cmgm.scripts.d0b_short_long_dynamic_moe --compare-results /path/d0b/result.json /path/three-global/result.json /path/a-static/result.json /path/a-dynamic/result.json /path/b-static/result.json /path/b-dynamic/result.json --comparison-output experiments/d0b_short_long_dynamic_moe/seed42_comparison
```

汇总器要求同 seed、协议、数据指纹、代码哈希、RNG 初始化协议和合法 checkpoint 哈希；输出 CSV/JSON。目录存在时拒绝覆盖。

## 输出与冻结诊断

结果：`experiments/d0b_short_long_dynamic_moe/<variant>_seed42_<timestamp>_<pid>/`；checkpoint：`checkpoints/d0b_short_long_dynamic_moe/<同名>/best.pt`。

保存 `config.json`、`data_audit.json`、`initialization_audit.json`、逐 epoch `training_history.json`、`best_checkpoint_metadata.json`、`metrics.json`、`result.json`、`RUN_STATUS.json`、`routing_diagnostics.json`、`val_frozen_interventions.json`，以及完整 TRAIN/VAL `*_predictions_routes.npz`。后者含预测、目标、预测窗口末端日期、商品顺序、逐样本 `(2,3)` 权重；所有日期/名称用非 object dtype 保存。

报告四周期 MAE/MSE/RMSE/Hit（JSON 的 Hit 为比例，CSV 展示为百分数）、VAL Huber、best epoch、参数量、训练秒数。路由统计包括均值、总体标准差、min/max、P10/P25/P50/P75/P90/P99，明确区分 `mean_sample_entropy_nats` 与 `entropy_of_mean_weights_nats`；另报告共享权重分布。

A 记录专家 L2；B 额外记录各样本四残差 L2、z_base L2、两任务加权残差 L2、逐样本残差/base 比值。汇总先算样本 L2/比值再求均值，eps=1e-12；不以 LN 后范数当作重要性。

冻结 best checkpoint 后，在 VAL 执行：正常路由、TRAIN 全部 origins 的平均路由、固定 seed42 的 VAL 整对短/长权重置换、共享候选 one-hot、四个专属 logit 分别置为 `-inf` 后重新 softmax。置换只移动整对路由，保持该样本专家、目标和日期顺序。独立 NumPy Generator 不扰动训练 RNG。逐周期报告 MAE/MSE/RMSE/Hit 差值（Hit 差值是比例；乘100后才是百分点）。不会调整 epoch、参数或目标。

这些是冻结模型依赖性诊断，不是重训消融，也不是经济因果识别。权重标准差非零本身不证明动态路由有预测价值。

## 预设解释与尚未验证事项

- A-dynamic vs A-static：相同旧专家下的动态路由比较。
- B-dynamic vs B-static：相同残差专家下的动态路由比较。
- B-dynamic vs D0B：整体改动，含新增容量和 head Dropout 执行差异。
- B-dynamic vs three-global：既有三专家的匹配参考。
- 主关注 VAL MAE@5d，同时报告全部周期、VAL Huber和干预差值；方向不一致或差异小则如实报告。没有根据结果追加模型或改阈值。
- 此阶段没有自动 TEST 门槛或 TEST 入口。后续必须另行授权且冻结模型/checkpoint；因项目历史反复观察过 TEST，应标为探索性。

初始实现交付时正式实验尚未运行；当前已有用户执行的 seed42 TRAIN/VAL 结果和冻结 VAL 干预，见上述汇总。TEST 性能、跨 seed 稳定性及任何市场/经济专家分工仍未验证。数据加载复核、初始合成验证和后续实验结果分别记录；不以合成数据 MAE 作为有效性证据。

## 已执行的最小验证

- 新功能测试：**15 passed**；新功能早期13项加旧 mask/four-task/task-conditioned MoE 回归：**31 passed**。两项新增入口/汇总器测试随后通过。
- 六配置合成预检通过。N=284 实例化参数量如表；只在小 N=30 的合成输入上做预测检查。
- 所有应共享参数差值为0，mismatch_count=0；保存了具体复制 key 的运行时审计接口。
- B-static/B-dynamic 零残差相对 D0B 的最大预测绝对差均为 `3.725290298461914e-08`，来自按周期输出行计算的浮点执行差异；通过 atol=3e-7、rtol=1e-6 的等价测试。
- 静态不依赖样本、动态对构造的不同样本响应、权重归一化、组掩码、周期行、batch permutation、single-sample consistency、解释/干预上下文恢复、H/Z live 梯度、编码器单次调用、KL 日程、严格 checkpoint roundtrip、原门控回归均通过。
- B 首步 router/内部残差层梯度为零；在3次合成更新后，各新增参数均有有限非零梯度。按短/长组分别计算损失时，另一组专属专家/router 梯度为零。
- 使用原 `train()` 做两次各仅2个合成样本、1个 optimizer step 的集成检查；确认新参数更新、checkpoint选优/恢复、结果写出。没有正式训练。
- 合成 CSV 限制读取测试校验了未来价格行的 parser 排除和 TRAIN/VAL 指纹一致；错误指纹会 STOP。汇总器的6组输入、Hit单位和协议冲突拒绝均通过。
- `compileall` 与 `git diff --check` 通过。默认 CLI 只做合成预检，`--help` 已检查。

机器可读配置见 `docs/d0b_short_long_dynamic_moe_protocol.json`；实际验证和源码哈希见 `docs/d0b_short_long_dynamic_moe_validation.json`。没有 commit/push。

### 后续日历加载修复验证

原日期扫描包含 2016-12-30 至 2017-01-16 的 11 个在价格透视后不存在的日期，导致 `.loc` 报错。修复仅调整限制读取日历的重建，未改变模型、特征公式或训练协议。原始实现阶段的合成验证记录保留；本次新增只读复核结果见 `docs/d0b_short_long_dynamic_moe_calendar_fix_validation.json`。

- 缺失日期及训练入口回归测试：3 passed，13 deselected；覆盖起始及截断点附近缺失日期，并检查未来价格行不被解析。
- 实际 TRAIN/VAL 完整 `data_audit` 与参考逐项相同，包括指纹、资产和商品顺序；样本数为 1396 / 268。
- 有效原始行数 1435 / 307，TRAIN 末日 2022-12-12，VAL 末日 2024-03-20。
- 两轮日历重建后复现原划分；没有创建模型或优化器，没有正式训练，没有解析 TEST 价格、特征或标签。
- 此错误发生在正式训练前；修复后重新执行失败配置的原命令即可，勿重复已完成的实验。

### 训练完成后的路由均值诊断修复与恢复

后续实际运行在 `frozen_diagnostics` 报 `Override must be normalized`，原因是对 1396 个 float32 路由求均值时累积舍入误差，而非模型路由失效。三个失败运行的均值组内和偏差为约 `2.62e-6 / 3.93e-6 / 2.80e-6`，超过严格 `1e-6` 检查。现在用 float64 累积 TRAIN 均值，保留原权重校验阈值，也不通过重新归一化掩盖非法输入。训练路径与模型参数完全不变。

此次错误发生在**训练已完成、最佳 checkpoint 已保存、正常 TRAIN/VAL 预测已保存之后**。不要重新用 `--run` 启动这些配置。仓库根目录执行下面一行，仅恢复已知错误的冻结模型后处理：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --finish-failed --device cuda:0
```

也可明确指定一个或多个原运行目录：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_dynamic_moe --finish-run /absolute/path/to/failed_run --device cuda:0
```

- `--finish-failed` 只筛选原状态中明确记录该路由归一化错误的运行；已完成的配置不重训、不重评。已恢复的运行再次调用时跳过。
- 核验 checkpoint SHA256、variant、seed、原协议、完整历史和最优 epoch；严格加载权重。除本脚本外的模型、数据、训练关键源码哈希必须一致。
- TRAIN/VAL 只重建一次并在多个待恢复运行间共享，必须复现原数据审计。新预测和路由与错误发生前保存的数组在绝对容差 `1e-6` 下核对；标签、商品和日期索引必须完全一致。
- 没有训练调用、优化器、反向传播或 TEST 入口。恢复执行的是普通 TRAIN/VAL 推理和七项冻结 VAL 干预，不能称为训练续跑。
- 新产物写在原运行目录的 `postprocess_<timestamp>_<pid>/` 中，包含 `result.json`、`metrics.json`、`routing_diagnostics.json`、`val_frozen_interventions.json`、预测 NPZ 和 `RUN_STATUS.json`。原 checkpoint、失败日志和其他已有结果保持原样。
- `config.json`、训练历史及 checkpoint 元数据在子目录中原样复制，表示原训练来源；新增 `postprocess_provenance.json` 单独记录本次评估源码。训练耗时来自原 `history.train_time`，不包含恢复耗时。
- 后续六组汇总时，三组恢复运行使用子目录的 `result.json`，其他已完成运行继续使用原 `result.json`。

修复验证：本模块 **18 项测试通过**，包括 1396 行路由均值回归、最后不完整 batch、非法概率拒绝和失败后仅补做诊断的端到端合成测试。恢复测试禁止调用 `train` 或创建 Adam，并逐字节验证原文件、checkpoint 不变。真实 A-static/A-dynamic/B-dynamic 的最佳 epoch 分别为 63/62/81；三者通过只读严格 checkpoint 加载和修复后路由替换校验。完整正式后处理尚未代为执行。详见 `d0b_short_long_dynamic_moe_route_mean_fix_validation.json`。未读取或评估 TEST。

### seed42 结果后的代码复核

用户完成后处理后，六组均已 COMPLETE。独立审计见 `experiments/d0b_short_long_dynamic_moe/code_audit_20261013_133542_974388/CODE_AUDIT.md` 及同目录 `code_audit.json`，不覆盖原结果。

- 主模型、A/B 专家、动态路由、任务掩码、H/Z 实时接口、初始化复制、预测头行顺序和原训练 objective 均符合本实验定义；不能因结果不佳自动改变它们。
- 找到并修复一个日志接口问题：新变体的 `get_gate_stats()` 原来落入旧 LSTM 诊断路径，触发不存在的 `self.temporal`，且未恢复模型模式。现在为四个新变体分派到实际 MoE 前向，保存并恢复所有 module 模式，成功和异常两条路径都验证不修改参数或 RNG。原 D0B 的该接口保持原样。
- 此接口由原训练器在验证后调用；原异常被忽略。修复仅涉及诊断，不改变正常 forward、损失、优化器或 checkpoint 选择，现有结果不需要重训。去掉新增诊断分派后，模型源码 SHA256 与原训练记录完全相同。
- 冻结干预新增完整 origin 数检查：逐样本覆盖权重必须与 loader 长度一致，禁止悄悄截断或多传样本。此前完成的七项干预输入均合法，指标不受这一校验影响。
- 当前验证为 **42 passed**（动态 MoE 24 项，现有 masked/four-task/task-conditioned MoE 18 项）；增加 native train_epoch 的实际 objective 验证、所有 optimizer 参数覆盖、诊断接口及异常恢复、temporal prefix causality 和 detach 缓存污染检查。
- 四个已训练新变体的所有新增参数均较初始化更新，合成反传中没有缺失或全零新增参数梯度，且空间和时间分支各执行一次。已训练 A-dynamic/B-dynamic 对不同构造输入的路由最大差分别约 `4.34e-4 / 7.43e-5`，静态对照为 0。动态不是硬编码常量，但实际 VAL 接近常量是需要如实接受的结果。
- train 模式的按周期 head Dropout 与 D0B 的共享 mask 不同，仍作为已披露限制；static/dynamic 匹配对照使用相同 head 实现。B 初始化的 router 首步预测梯度为零，是零残差设计的预期现象。
