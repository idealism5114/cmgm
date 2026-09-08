# D0B-HuberHorizonScaleDiagnostic

仅加载正式 `switching_latent_balanced_readout_best.pt`，使用原预处理和划分。TRAIN/VAL/TEST 全部窗口依次推理；TRAIN 包含尾 batch。模型始终 eval，forward 使用 no_grad；固定 TEST batch 的 raw per-horizon Huber 梯度仅使用 autograd.grad。

从 Commedities 目录运行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m cmgm.scripts.d0b_huber_horizon_scale_diagnostic --device cpu --threads 4
```

默认输出 `experiments/d0b_huber_horizon_scale/<timestamp>/`，含完整 prediction/target/residual NPZ、results.json、REPORT.md、residual_summary.csv 与 commodity_5d_error.csv。输出目录必须不存在，避免覆盖历史结果。

统计包括固定 delta=.02 的工作区间、loss/MSE share、解析梯度压缩、TRAIN-only std/MAD 标准化、商品异质性、所有模块梯度范数/比值/cosine。`REPORT.md` 的 Interpretation/Decision 由对实测结果的审阅填写；脚本不设无依据的分类阈值。所有原始数字始终自动生成。

解析梯度为 unreduced 单 residual 导数；network probe 是 raw mean Huber。精确 abs(e)=delta 时属于 quadratic 区且 abs(g)=delta，因此 saturated=linear+boundary。JSON 显式记录边界质量。所有 std 为 population std（ddof=0）。

脚本验证 strict checkpoint load、前后 state_dict 完全一致、checkpoint SHA256 不变、parameter.grad 无写入、全模块 eval。没有训练或下一版 loss 实现。指标遵循 [PERFORMANCE_REPORTING_STANDARD.md](PERFORMANCE_REPORTING_STANDARD.md)。
