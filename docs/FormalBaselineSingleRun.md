# FORMAL BASELINE BENCHMARK V2 — FIXED CONFIG SINGLE RUN

替代旧 V2 grid、Stage A/B、三个 seeds 和 D0B 重训。旧入口已禁用；历史目录、结果、checkpoint 不覆盖。单次固定配置比较，不宣称多 seed 稳健性或统计显著性。

## 当前续跑审计

旧进程 PID 3937534 的完整命令核验后已发送 SIGTERM，并确认退出。只停止了旧 V2 benchmark。

- 复用 Random Forest：旧 `A_rf_seed42_candidate2`，300 trees/depth20/sqrt，seed42，完整119280维输入、96输出及 sanity 均符合。记录原路径与 SHA256。
- 复用 D0B：正式历史 checkpoint，epoch85，520549 parameters。只读统一 evaluator 复现。
- Ridge 重跑固定 alpha1、lsqr；旧 solver=auto 不符合。
- XGBoost 重跑固定200 trees、depth4、lr.05；旧配置不符合。
- LSTM、TCN、VanillaTransformer、GraphWaveNet、MTGNN 各补跑一次固定 lr1e-4。旧 V2 使用 VAL5 MAE checkpoint/early stopping，且 train drop_last=False；不符合本次历史 D0B 的 multi-horizon Huber selection 和 loader。

总计：7个待拟合模型，RF和D0B直接复用。旧35个 grid runs和24个multi-seed runs不再执行。旧 XGBoost计划7×96=672个output regressors，新协议只拟合96个（每个200棵树）。

## 运行

在 Commedities 项目根目录、激活原 Python 环境后执行。正式训练仍由用户启动。以下命令不含斜杠。

已经准备好当前 single-run 目录时，直接续跑：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.formal_baseline_single_run --resume latest --run >> formal_single_run.log 2>&1 &
```

查看日志：

```bash
tail -f formal_single_run.log
```

首次在没有准备目录的另一环境运行，先准备输入审计、模型 sanity，并只读评估可复用模型：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_single_run
```

然后使用上述 `--resume latest --run`。默认入口不做新拟合；`--run` 只拟合缺失模型，完成的 artifact 不重复训练。完整 checkpoint 已保存但后处理被中断时，可自动恢复后处理。真正中断的未完成 fit 需要检查原因后显式 `--retry-invalid`，不能因结果差使用此选项。

五个神经模型在 GPU 上逐个运行，不自动回退 CPU。Ridge/RF 是标准 sklearn CPU实现；XGBoost采用 hist、外层1 worker、内层8 threads（默认）避免嵌套并行。`--cpu-jobs` 仅是资源配置，恢复时应与已保存值一致。每分钟有 classical fit 心跳；耗时不等于卡死。

## 固定协议

输入原始 model-ready `(B,20,N,21)`，实际N由dataset读取；ML无损flatten，sequence无损flatten每时刻node/feature，graph仅transpose。禁止284→28、删除features或重复标准化。

Ridge: sklearn Ridge(alpha=1, fit_intercept=True, solver="lsqr")。
RF: sklearn RandomForestRegressor(300, max_depth=20, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)。
XGB: MultiOutputRegressor(XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, objective="reg:squarederror", reg_lambda=1, reg_alpha=0, tree_method="hist", random_state=42))。

Neural: Adam1e-4, WD1e-5, batch64, max200epochs, patience10; sum4 Huber(.02)。checkpoint、early stopping、ReduceLROnPlateau都使用原 batch-mean multi-horizon VAL Huber；VAL5仅日志。保持原chronological loader、train drop_last=True，evaluation完整样本。

神经结构及官方graph源码不改。Graph WaveNet原实现最终输出有效receptive field=13，输入仍完整20步；MTGNN原LayerNorm覆盖合法observed window，不冒称其归一化后的内部timeline严格streaming prefix-invariant。这些官方架构特性在报告中披露，不改变block或隐藏差异。

## 输出与恢复

新目录 `experiments/formal_baseline_benchmark_v2_single_run/<timestamp>`，checkpoint `checkpoints/formal_baselines_single_run/<timestamp>`。

每个完成模型立即写 checkpoint、results.json、partial_results.json、RUN_STATUS.md、REPORT.md、CSV；若某个后续模型失败，不丢已完成结果。进程级锁禁止本入口多个任务同时训练。历史D0B hash前后核验。

主表固定9行；未完成模型显示PENDING，不填假数据、不据此排名。正式指标MAE/MSE/RMSE/Hit，Hit表格用百分比、JSON用fraction。全部origin×commodity pooled，含zero targets，float64计算。

结果无论好坏都如实保存，禁止恢复搜索、换seed或修改D0B。9行完成后STOP。
