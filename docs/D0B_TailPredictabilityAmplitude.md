# D0B-TailPredictabilityAmplitudeDiagnostic

只加载正式 `switching_latent_balanced_readout_best.pt`，使用原始三份 split、21-feature
预处理和四 horizon 输出，动态提取 `MULTI_HORIZONS.index(5)`。没有训练、backward、
optimizer、model variant 注册，也不保存 calibrated checkpoint。

## 运行

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m cmgm.scripts.d0b_tail_predictability_amplitude \
  --no-cuda --prepared-data /tmp/d0b_error_data_verified.npz
```

缓存是上一轮原预处理产生的 numeric NPZ；每次验证原 CSV 和预处理源码指纹。
省略 `--prepared-data` 即重新执行原始预处理。缓存指纹不匹配会拒绝使用。
`--checkpoint /actual/path/to/official_best.pt` 可指定正式权重；默认读取已有路径登记/metadata。

结果写入 `experiments/d0b_tail_predictability_amplitude/<timestamp>/`：

- `REPORT.md`、`results.json`：完整指标、阈值、系数、方向、metadata、校验值与解释。
- `observation_tail_diagnostics.csv`：全部起点×商品的 native/scaled prediction、目标、误差和因果信号。
- `signal_tail_predictability.csv`、`signal_quantile_tail_rates.csv`、`top_signal_capture.csv`。
- `amplitude_quantiles.csv`、`conditional_amplitude.csv`、`tail_response.csv`。
- `calibration_metrics.csv`、`commodity_calibration_diagnostics.csv`、`sample_signal_associations.csv`。
- `inference_records.csv`、`inference_metadata.json`：本轮实际重新读取权重后的 inference 中间记录。

## 固定定义

只有一个用于预测的系数：TRAIN pooled `sum(pred*y)/(sum(pred**2)+1e-12)`。
不加 intercept、不 clipping、不搜索，不将描述性的 TRAIN halves/commodity slopes 应用到预测。
负 global coefficient 触发明确异常并停止，绝不悄悄翻转预测符号。

Tail90/95 阈值只来自 TRAIN |target|。所有 17 个信号的方向只由 TRAIN
Pearson(signal, Tail90) 决定；Tail95 使用同一方向。AUROC/AP 保留并正确处理分数 ties。
AUPRC 指非插值的 average precision，并报告 prevalence 与 AP/prevalence。

Top10/20 采用 oriented TRAIN P90/P80 冻结阈值，报告实际 evaluation coverage；
不以 VAL/TEST 自己的分位数重选固定比例。原始信号分位组也只在 TRAIN 拟合。
sample-level 信号的分位数按起点取一次，避免重复商品权重改变 quantile 定义。

关键 AUROC 与 calibration MAE/MSE 提供 1000 draws、seed42 的 forecast-origin cluster
bootstrap：同一日期全部商品一起抽样。它不是时间块 bootstrap，仍受重叠5d窗口的序列依赖限制。
市场/regime 信号另按每起点的 TailShare、MaxAbsTarget、sample MSE 计算关联。

比率分母加入 1e-12；普通样本里的零目标会夸大平均比率，因此同时记录零目标数和非零目标均值。
尾部/正确符号分组是事后解释，不把这些标签作为 predictors。

## 本次分类解释

按请求第77条优先规则，TRAIN scalar 若在 VAL/TEST 的 MAE/MSE 四项均稳定改善，
优先报告 Case A，并逐项给出效果大小与 CI。本次实际 c<1，所以要严格区分：
“scalar calibration 有改善”与“放大不足是主要瓶颈”。前者成立不意味着后者成立。
不得把校准系数写回正式模型，也不以报告为授权自动训练下一模块。

## 验证

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m pytest -q \
  tests/test_d0b_tail_amplitude.py tests/test_d0b_5d_error_regime.py
```

覆盖 TRAIN-only fitting、冻结方向、AUROC/AP ties、加权 bootstrap 与显式重复等价、
同起点商品聚类、正 scalar 符号一致、负 coefficient 停止和完整记录导出。
