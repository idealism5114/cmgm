# PRIMARY PERFORMANCE REPORTING STANDARD

从 D0B-HuberHorizonScaleDiagnostic（2026-09-08）开始，所有未来实验对 VAL / TEST 的 1d、5d、10d、20d 都必须报告 **MAE / MSE / RMSE / Hit%**。TRAIN 若报告 performance，采用同一标准。商品级分解至少包含 MAE、MSE、RMSE。

使用 `cmgm.training.metric_standard.population_metrics`，拼接完整 loader 的预测和 target 后计算。统计总体是 sample × commodity，不能先等权平均各 batch 的指标。MSE 直接取误差平方均值；RMSE = sqrt(MSE)，并核对 RMSE²−MSE；Hit 是所有 target 上 sign 相等的比例。JSON 中 Hit 是 0–1，展示乘 100。MSE 至少 6 位小数，优先 9 位或科学计数法。

| Split | Horizon | MAE | MSE | RMSE | Hit% |
| --- | --- | --- | --- | --- | --- |
| VAL / TEST | 1d / 5d / 10d / 20d | 必报 | 必报 | 必报 | 必报 |

Ablation summary 指 **TEST primary horizon = 5d**，列为 Variant / Params / TrainTime / MAE / MSE / RMSE / Hit%。

旧报告 RMSE 是商品级 RMSE 的平均，Hit 排除 abs(target)≤1e-8。历史文件保留原值，不可将口径变化解释为模型改善。`compute_metrics` 额外提供明确标注的 `RMSE_mean_asset_legacy` 和 `Hit_Ratio_masked_legacy`，仅供历史核对。旧 results.json 若无 MSE，不可由旧 RMSE 平方补算，须重新读取预测计算。

这是 evaluation/reporting 变更；训练 loss、validation loss、scheduler 与 checkpoint selection 均保持原逻辑。
