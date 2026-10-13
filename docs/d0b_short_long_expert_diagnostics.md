# 短／长任务 MoE 专家互补性和路由诊断

入口：`cmgm/scripts/d0b_short_long_expert_diagnostics.py`。只评估现有 seed42 冻结 checkpoint，不能训练，没有 TEST 选项。A-dynamic 为辅助对象，B-dynamic 为主对象；其他四组通过已审计的原始结果作参照。

仓库根目录运行合成预检：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python -m cmgm.scripts.d0b_short_long_expert_diagnostics
```

冻结 TRAIN/VAL 诊断命令（每条都是一行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_expert_diagnostics --run --file-index experiments/d0b_short_long_dynamic_moe/summary_seed42_20261013_131817_245598/file_index.json --device cpu --threads 1
```

原实验 GPU 机器可将 `--device cpu` 改成 `--device cuda:0`。`--run` 在这个脚本中仅表示正式冻结诊断，不表示训练。默认建立新的 `experiments/d0b_short_long_dynamic_moe/expert_diagnostics_<timestamp>/`；可通过 `--output` 指定新的目录，已有目录拒绝覆盖。`--file-index` 可省略，脚本按 checkpoint 身份检查六组索引；存在相互冲突的索引时停止，不按修改时间猜模型。

## 恢复与计算位置

从索引读取每组真实 result/config/history/checkpoint 和后处理 provenance，检查完成状态、seed42、variant、checkpoint SHA256、最佳 epoch 与原 Huber 最小值；`strict=True` 加载，恢复 best epoch 的 native Switch KL 调度状态。原配置和训练源码哈希、当前评估源码哈希分别记录。

数据复用限制读取的 TRAIN/VAL 构建函数；只用 TRAIN 拟合统计，必须复现整个原 data_audit，包括输入与标签指纹、节点/商品顺序，不能只比样本数。日历定位扫描 CSV 的日期元数据，未来价格行在 parsing 阶段排除，不构建 TEST dataset。

先完整复算两个动态模型的 TRAIN/VAL 正常预测、路由及指标，核对原保存数组：预测采用本机 PyTorch float32 默认 `assert_close` 容差（atol=`1e-5`、rtol=`1.3e-6`）；路由仍为绝对容差 `1e-6`；总体 MAE/MSE/RMSE 采用更严格的绝对 `1e-8`、Hit 必须完全一致；目标、日期、商品顺序完全一致。复现失败时停止，不继续解释专家能力。

初次使用额外严格的纯绝对 `1e-6` 检查，在 A 的 VAL 上停止。后续 CUDA 核查确认最大差 `2.205e-6`，只有 21/25728 个元素超出该初始检查，平均差 `3.04e-8`；四周期 MAE 差异均小于 `1e-9`、Hit 完全一致，输入数据和参数指纹完全匹配。保留失败目录并采用经过本机 PyTorch 2.7 定义核验的标准 float32 容差，同时收紧总体指标检查。这是数值复现容差，不是预测收益门槛，不改变模型、标签、checkpoint 或性能筛选规则。

原特征复算很慢，成功完成指纹审计后额外保存 `validated_train_val_cache.npz`。可用 `--data-cache /path/to/validated_train_val_cache.npz` 复用已计算数组；缓存不含 TEST，只含两个区间的原 prices/feature_matrix/raw_prices 和日期，必须重新验证原数据指纹、源文件哈希和参考 config 哈希。标签仍由原 Dataset 构造，不重新拟合 VAL 统计，也不近似替代特征公式。

正常 forward 的融合 pre-hook 捕获本次实际产生的 `s/t/H/Z/z_base`。冻结诊断重用这些张量，调用**原融合模块的 override 干预及原预测头**，不从 detach 日志缓存重建专家，不改模型公式。三个 one-hot `[1,0,0] / [0,1,0] / [0,0,1]` 分别同时作用于两组，获得三个候选的所有周期预测。每组两个周期采用同一候选。每个区间第一批另做完整模型 one-hot forward，与复用融合方式比较，并检查单组干预不会改变另一组。

- A：原 `fusion_norm(sum(pi * E))` 保持不变。
- B：共享候选为共同训练后的 `z_base`，状态/交互候选是 `z_base + delta`。共享 one-hot 的融合表示必须逐元素等于 z_base；没有再次加基座，也没有额外 LN。它不等于独立训练的 D0B。
- 模型 `eval()`、`requires_grad_(False)`；所有预测采集使用 `inference_mode()`，没有 optimizer、backward 或参数更新。
- 校验模型 state_dict 哈希及所有原实验输入产物、checkpoint 的前后 SHA256。

## 损失、并列与策略

周期顺序 `[1,5,10,20]`，组索引 `short=[0,1]`、`long=[2,3]`。Huber delta 从各运行的原 protocol 中读取。

`L[i,g,k] = mean_{组内两个周期 × 全部商品} Huber(pred[i,k] - target[i])`。

组内汇总先对每个样本计算，再按 origin 等权平均。它既不是两个周期的 Huber 求和，也不是历史选模的四周期求和＋batch-mean。不能直接拿这些损失数值与历史选择损失比较。MAE/MSE/RMSE/Hit 仍按每个周期的全部 origin×commodity 汇总，Hit 包括零目标，JSON 存比例，展示用百分数。

离线统计全部使用 float64。并列定义为距最小损失不超过 `1e-12 + 1e-9 * abs(minimum)`，容差内选最小候选编号。保存并列候选、数量和第一/第二损失的实际差值；胜出率分别提供“含并列”“并列均分”“确定性选择”三种口径。不能只靠胜出计数把微小损失差称为明显分工。

策略：

1. 正常动态融合。
2. TRAIN 最佳固定专家：两组各按 TRAIN 平均 L 选一次，原样用于 VAL。
3. VAL 最佳固定专家：两组各按 VAL 平均 L 选一次，**只作使用 VAL 标签的事后参照**。
4. 逐样本理想选择：在对应区间按该样本两个周期共同的 L 择优，**使用真实标签，只能作诊断**。
5. TRAIN 全部 origins 的平均路由替换；float64 累积。
6. 当前区间样本的整对短/长路由固定 seed42 置换，专家和目标仍留在原样本。
7. 强制共享专家。

Oracle 仅是在三个单专家候选中按组逐样本择优，不是任意软融合模型的理论性能上界；正常软融合可能更好。按组 Huber 择优不保证 5d MAE 最低。TRAIN 选择后的 TRAIN 结果有 in-sample 选择优势，不能当成泛化证据。

强制单专家使用共同训练后的原预测头，并非独立训练的专家模型。one-hot 表示可能偏离预测头通常接收的混合表示分布；因此较差的单专家分数不能单独证明该专家没有为正常混合提供信息。硬选择一致率也只描述单候选反事实，不能代替软混合干预证据。

## 输出和字段

- `diagnostic_summary.md`：先列结论证据，再给策略表、参考指标和解释边界。
- `expert_competence.csv`：模型、split、组、候选、周期的组内 Huber、MAE/MSE/RMSE/Hit、胜出与并列比例。
- `strategy_comparison.csv`：全部策略的组内 Huber、四周期指标和相对正常路由差值；明确使用 VAL 标签的策略。
- `expert_pairwise.csv`：每对候选、每周期的预测平均绝对差、signed-error Pearson、日期 MAE Pearson、成对胜负；JSON 字段列出分歧最大的五个 origin/日期及损失差。相关性以 sample×commodity 的 signed error 为主；近零方差（std≤`1e-12`）时为空，不填零。共同目标会影响误差相关，不能只凭高相关就断言专家同质化。
- `routing_alignment.csv`：非并列样本数、argmax 路由与理想候选一致率、硬选择相对理想/严格最小损失的平均差。它们描述的是 argmax 硬选择，不是正常软融合损失。
- `per_origin_diagnostics.csv`：每个 origin、组的日期、三权重、三损失、理想编号、并列候选、router argmax、第一/第二损失差。
- `analysis_summary.json`：原始统计分布、策略指标、理想选择相对两种固定选择的绝对/相对 Huber 降幅。
- `reference_results.json`：四组参照及两个动态模型的原始结果，不用交接或 TEST 数字补位。
- `audit.json`、`provenance.json`：加载/来源、数据和正常预测复现、one-hot 隔离检查、参数不变性、训练/评估源码、环境和计时。
- `a-dynamic/b-dynamic_train/val_predictions.npz`：预测、目标、路由、商品顺序、origin 日期；专家预测轴顺序为 `(N,3,4,24)`；均值/打乱路由预测；逐样本 L `(N,2,3)`、理想编号和并列标记等，足以复算统计。

## 解释范围

观察输出分歧、交替胜出、损失差和理想选择收益后，才讨论互补潜力。若一个候选占优或输出几乎相同，优先检视专家设计；若有互补潜力但均值/打乱路由无损，当前 router 未显示有效利用，仍不能保证可观测输入能预测优势；若分歧大而理想收益小，不建议扩容。不会事后设一个有利收益阈值，不作统计显著性或经济因果贡献声明。

该工具到冻结诊断和报告结束为止，不新增或训练模型。合成检查只验证实现，不证明真实专家有效。

## 本次执行记录

真实诊断已完成，目录：

```text
experiments/d0b_short_long_dynamic_moe/expert_diagnostics_20261013_144741_463264/
```

完整报告为 `diagnostic_summary.md`，索引为 `diagnostic_file_index.json`。TRAIN1396、VAL268全部origins覆盖，A/B参数和六组原始checkpoint/实验输入文件SHA256前后一致。7项合成测试通过；四组预测NPZ的专家损失与Oracle组内Huber复算精确一致。

B的VAL理想选择相对最佳固定候选有约2.079%/3.234%的short/long组内Huber降低，当前动态路由的均值替换与打乱几乎不改变5d误差。A的相应收益为0%/0.075%，共享候选几乎一直最优；正常软融合仍优于其单候选Oracle，这不矛盾，也不证明专属函数在软融合中没有作用。当前结论见报告，未开展优势分类器拟合、结构修改、训练或TEST。

源数据复算后已保存精确缓存。如需要独立复现本次冻结诊断，下面命令创建新的目录，不能覆盖本次产物：

```bash
../.venv/bin/python -u -m cmgm.scripts.d0b_short_long_expert_diagnostics --run --file-index experiments/d0b_short_long_dynamic_moe/summary_seed42_20261013_131817_245598/file_index.json --data-cache experiments/d0b_short_long_dynamic_moe/expert_diagnostics_20261013_144741_463264/validated_train_val_cache.npz --device cuda:0 --threads 1 --compare-cpu
```

GPU曾因沙箱设备访问限制而不可用，真实完成运行在获准的沙箱外GPU0执行。A的CPU/CUDA数值差异与最初严格检查的失败记录已保留；没有用重训或更换checkpoint解决数值复现问题，也未改动性能判定规则。
