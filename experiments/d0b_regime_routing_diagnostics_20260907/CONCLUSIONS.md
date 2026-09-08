# D0B routing 机制结论

本轮唯一后续研究优先级：**A. Separate Generator Routing Temperature**。

证据支持小幅、horizon-dependent 的 routing dilution，尚不足以称它为 D0B 最大的性能瓶颈。该优先级只表示三个给定方向中最直接受到本轮隔离干预支持的研究假设；本轮没有实现温度参数、修改模型或重新训练。

完整数值、全部指定干预及逐 horizon 结果见 [REPORT.md](REPORT.md)；原始结果见 [results.json](results.json)。

## 1. Q1：soft mixture 是否稀释已有 candidate 分化？

**主归类为 Routing Case A，证据较弱且局限于 5/10/20d；同时存在明显的低 routing 利用率。**

- T=.75 的 5d MAE 在 VAL / TEST 分别相对降低 **0.02173% / 0.01493%**，RMSE 也同向降低。T=.50 分别降低 **0.06027% / 0.04372%**。这些是确定性数值变化，没有经过显著性验证。
- Flattening 的 T=1.25、T=2.00 在 VAL 和 TEST 的 5d MAE/RMSE 均变差。Uniform 同样变差：5d MAE 从 VAL 0.023200622 → 0.023217632，TEST 0.021994783 → 0.022004856。
- TEST native posterior entropy **1.091357**，接近 ln(3)=1.098612；mean max p **0.374794**，margin **0.041928**。T=.75 保持 p/prior/evidence 完全不变，仅将 q entropy 降到 **1.085802**、margin 提高到 **0.056867**，因而变化确实来自 generator mixture 的使用方式。
- G0/G1/G2 并未输出同一个向量。TEST candidate L2 norm 为 **0.5662 / 0.8843 / 0.6752**；pairwise cosine 为 **0.2246 / 0.1320 / 0.3100**，pairwise mean absolute difference 为 **0.09408 / 0.08295 / 0.08910**。这证明输出分化，不能单独证明经济意义上的状态专门化。
- 低 confidence 组的 candidate disagreement **5.7531**，高 confidence 组 **5.5693**。低置信度并非因为 candidates 都接近。margin 与 dominant-candidate-vs-soft-mixture L1 gap 的相关系数为 **−0.6992**，方向与更强 confidence 更接近 dominant candidate 一致，仅为描述统计。
- T=.75 固定批次的 mean |ΔZ_t| 从 t=0 的 **0.000838** 增至 t=5 的 **0.001507**，t=19 为 **0.001545**。T=.50 为 **0.002485 → 0.004402 → 0.004562**。干预在最初几步递推中累积后大致稳定，不呈持续爆炸。

但作用量级很有限：完整 TEST 的 all-horizon native-vs-uniform impact 为 **0.00017919**，zero-micro impact 为 **0.00856385**，RoutingFraction 仅 **2.09%**；VAL 为 **2.32%**，固定 TEST batch 为 **2.61%**。5d TEST 的比例为 **2.66%**。此比例是非线性干预响应比，不是可加性贡献或解释方差。

因此，微状态的重要性主要不能由当前 p 相对 uniform 的加权变化来解释。保留三个 generator 平均表示时，绝大多数 zero-micro 干预响应仍然保留。Native routing 有可检测的小作用，但没有证据表明它支配有效微状态表示。

其他 routing controls：

- Hard top1 在 5d 的 VAL/TEST MAE、RMSE、Hit 都改善；其中 TEST MAE 降到 **0.021940956**。但 **1d MAE 在 VAL/TEST 恶化 1.69% / 2.96%**。这不支持直接改成 hard training；不确定性混合对短 horizon 可能有价值。
- 固定 TEST batch 的 hard prediction impact 为 **0.00250291**，uniform 为 **0.00022269**，zero-micro 为 **0.00854792**，forced state0/1/2 分别为 **0.00272560 / 0.00152417 / 0.00194936**。三个分化的 generator 有改变预测的能力，但正常低-margin posterior 很少充分使用这种差异。
- 原式 batch-shuffled p 的 fixed prediction mean/max diff 为原始表所列；完整 VAL 的 MAE 微降、RMSE 微升，完整 TEST 的 MAE/RMSE 均微升。样本 routing identity 的作用较弱、方向不完全一致，不能据此认定必须重做 evidence。此项经用户授权，明确是非日历时间因果压力测试。
- Lagged p 的 VAL/TEST 5d MAE 几乎不变，TEST Hit 也不变；当前 posterior 相对上一时刻 posterior 的时点精度不是本轮看到的强敏感因素。

Case B 不符合 sharpening 的一致方向；Case D 不符合 mild sharpening 也改善的结果。Case C 所描述的“低利用率”现象有支持，但不能称为完全 null：uniform 对预测和 5/10/20d 指标有一致的小影响。采用 Case A 的弱证据归类，同时明确报告 Case C 式弱利用，避免把小幅可调作用夸大为主要瓶颈。

## 2. Q2：固定 α=.5 是否限制 dynamics？

**Persistence Case A 有有限支持；不符合“p 大变而 prediction 基本不变”的纯 Case B。**

Native A 平均对角线为 **0.666618**。α=.5 是 identity mixture 系数，并非对角线本身为 .5；learned softmax(L) 接近均匀矩阵。

| α | TEST posterior entropy | TEST temporal p L1 | TEST mean abs Δp | TEST prediction impact | VAL 5d MAE | TEST 5d MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.096217 | 0.030776 | 0.015176 | 0.0000909 | 0.023208952 | 0.022000058 |
| .25 | 1.094833 | 0.029957 | 0.009772 | 0.0000603 | 0.023206061 | 0.021998249 |
| .5 | 1.091357 | 0.030923 | 0 | 0 | 0.023200622 | 0.021994783 |
| .75 | 1.078036 | 0.033534 | 0.023031 | 0.0001710 | 0.023188258 | 0.021984348 |
| 1 | 0.945191 | 0.044163 | 0.110448 | 0.0012248 | 0.023169838 | 0.021917269 |

α=.75 和 α=1 在 VAL/TEST 的 5d MAE/RMSE 均改善，但 α=1 同样损害 1d：VAL/TEST MAE 增加 **0.25% / 1.09%**。由此只能认为当前 window 内 persistence/evidence accumulation 的强度有影响，不能认为更强 persistence 对所有 horizon 更好。

α=0 仍有 learned Markov prior。α=1 是 A=I，posterior 仍由 evidence 更新，实测 p temporal L1 反而增加。原模型每个 20 日窗口将 p_prev 重新设为 uniform，因此这一干预含有更强的窗口内 evidence 累积作用。

尤其不能将此结果直接推导成 **Regime-Specific Latent Persistence**：本轮改变的是 posterior 的公共转移矩阵 mixture 系数，没有改变 Z 的状态专属递推系数，也没有比较不同 G_k 的递推保持偏好。C 方向要求的“不同 state 明显不同 recurrence preference”证据尚未建立。

## 3. Q3：horizon 的依赖是否不同？

**Horizon Case A 有机制差异证据；尚不能证明 shared readout 已发生欠拟合。Case B 的“四个 horizon 都相似”不成立。**

完整 TEST：

| Horizon | Zero-micro impact | Zero-long impact | Micro/Long | Uniform-routing impact | RoutingFraction |
| --- | --- | --- | --- | --- | --- |
| 1d | 0.00655276 | 0.00687588 | 0.9530 | 0.00011254 | 1.72% |
| 5d | 0.00583711 | 0.00600596 | 0.9719 | 0.00015521 | 2.66% |
| 10d | 0.01116714 | 0.00583301 | 1.9145 | 0.00021384 | 1.91% |
| 20d | 0.01069840 | 0.00822675 | 1.3004 | 0.00023518 | 2.20% |

VAL 比例为 **1.056 / 1.028 / 1.902 / 1.343**，和 TEST 的相对形态一致：1/5d 两条路径近似平衡，10d 更依赖 micro，20d 也偏 micro。结果并非“短期 micro、长期 long”的简单划分。

更关键的是单一固定 TEST batch 的 prediction-only 梯度：

| Module | cos(5d,1d) | cos(5d,10d) | cos(5d,20d) |
| --- | --- | --- | --- |
| Regime evidence | −0.9801 | 0.9833 | 0.9747 |
| Transition logits | −0.9895 | 0.9889 | 0.9833 |
| Generators | 0.2435 | 0.4605 | 0.4474 |
| Balanced readouts | 0.3032 | 0.0660 | −0.0063 |

Regime evidence 梯度范数随 horizon 为 **1.71e−5 / 3.00e−5 / 1.67e−4 / 2.52e−4**。这表明在该批次上 1d 对共享 regime 路径的优化方向与其他 horizons 明显竞争；5/10/20d 的方向高度一致。该局部梯度结果和实际 sharpening/hard/α 对 1d 与较长 horizons 的相反性能方向相符。

梯度测量来自一个批次，不能推广成整个训练集恒定冲突。损失为单 horizon 原 Huber，没有辅助损失、没有重加权、没有 optimizer step。本轮没有实现 horizon-specific head/readout 或修改训练 loss。

## 4. 唯一优先级及边界

**选择 A. Separate Generator Routing Temperature。**

选择依据是严格保持 H、p、prior、evidence 不变后，仅改变 generator weighting，就得到 5/10/20d 在 VAL/TEST 方向一致的 sharpening/flattening 响应；native-vs-uniform 有非零预测作用，且 experts 的输出确实分化。这比直接修改 evidence 更接近本轮已经隔离验证的机制。

低 margin 和 hard routing 的结果使 evidence calibration 成为可能解释，但尚不能区分“识别不足”和“共享 posterior 在 horizon 间折中”；本轮没有 evidence 本身的干预证据，因此不把 B 列为最高优先级。全局 α 敏感性也不足以满足 C 对 state-specific latent recurrence 的证据要求。

该选择不意味着 routing 是最大的剩余误差来源：T=.75 的主指标收益只有百分之几百分之一，routing utilization 约为 micro 干预响应的 2%–3%，且 1d 会受损。更准确的结论是：**当前 micro 表示有用，p 对其有弱但真实的控制；过软加权对较长 horizons 有轻微稀释，而共享 posterior 的 horizon 冲突比单看 5d MAE 更值得纳入机制解释。**

没有择优温度重训，没有新增正式 variant，没有更改 D0B 正常 forward、spatial branch、fusion/head、optimizer 或 checkpoint。所有结果保持为诊断产物。**本轮到此停止。**
