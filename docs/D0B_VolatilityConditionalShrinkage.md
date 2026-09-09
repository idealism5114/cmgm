# D0B-VolatilityConditionalShrinkageDiagnostic

正式唯一比较：`Conditional=(a+b*z)*p` 相对于 `Global=c*p` 的增量价值。
所有标准化、阈值、方向与正式系数只来自 TRAIN；保持 D0B checkpoint、结构、参数不变。

## 运行

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m cmgm.scripts.d0b_volatility_conditional_shrinkage \
  --no-cuda --prepared-data /tmp/d0b_error_data_verified.npz
```

原预处理的 numeric NPZ 缓存核验原 CSV 和源码 SHA256；省略该选项即可重新预处理。
每次仍重新加载正式 checkpoint 并执行全量 inference，不复用旧预测作为本次 baseline。
若实际 native TEST 指标与参考明显不一致，脚本在拟合 calibration 前停止。

输出在 `experiments/d0b_volatility_conditional_shrinkage/<timestamp>/`，包含 REPORT、results、
全量 observation CSV、overall/tail/commodity/bin/scale/risk-decomposition/bootstrap 表。
没有任何新模型或 calibrated checkpoint。

## 固定统计逻辑

- Commodity vol20 复用 feature_builder 的 raw-unit 语义：past pct_change，rolling std
  ddof=1/min_periods=1，同一 clipping；不使用 market mean 作为正式 conditioning signal。
- `log(vol20+1e-8)` 的 pooled TRAIN mean/std（ddof=0）冻结，
  `z=(log_vol-mu)/(sigma+1e-8)`。
- Global scalar 复用上一诊断：`sum(p*y)/(sum(p*p)+1e-12)`，差异化 epsilon 明确记录。
- Conditional 使用 `np.linalg.lstsq([p,z*p],y,rcond=None)`，lambda=0，无 additive intercept，
  无 clipping、ridge tuning 或任何 gradient-based fitting。
- 半段 `(a,b)` 使用同一 full-TRAIN z 单位，仅作稳定性比较，不应用于正式预测。
- Within/between commodity risk scores 的参考统计与方向只来自 TRAIN，仅评估 tail 排序，
  不构造第二个 calibration。
- Delta = Conditional − comparator，负数为改善。统一 float64 pooled MAE/MSE/RMSE/Hit，
  零目标也计入 Hit。
- Bootstrap 固定 1000 draws、seed42，同一 origin 全部商品一起抽样。
  重叠窗口仍有序列依赖，CI 是近似 origin-level uncertainty diagnostic。

## 解释边界

必须同时检查 MAE 与 MSE，不能只凭优于 Native 或 tail AUC 高就宣称成功。
负 scale 保留原式，并列出符号翻转；若翻转发生在零目标上，两个非零预测都 miss，
所以 Hit 不变不意味着没有 sign flip。

报告包括零波动覆盖与商品误差贡献的加总分解；不剔除这些样本，不修改数据或重拟合系数。
零波动的 log epsilon 值可能影响 TRAIN mean/std 与静态 risk proxy，需限制机制结论。
只对预定义的两参数形式作结论，不推论所有形式的 volatility conditioning 都无效。

## 验证

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m pytest -q \
  tests/test_d0b_volatility_conditional.py tests/test_d0b_tail_amplitude.py
```

验证两列无截距拟合、rank deficiency 报告、TRAIN-only 标准化、VAL/TEST target 不影响预测、
负 scale 不被裁剪、半段统一 z、origin bootstrap、baseline mismatch 停止和各商品贡献加总。
