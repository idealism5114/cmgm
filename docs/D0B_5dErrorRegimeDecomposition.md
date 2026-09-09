# D0B-5dErrorRegimeDecomposition

纯 checkpoint inference 与误差分解，无训练、backward、optimizer、结构修改或 variant 注册。

## 当前可完成的范围

用户确认 `+TempWeighted / temporal_weighted_graph` 当时没有保存 checkpoint。
因此目前只能实际运行 D0B 分解；双模型互补性、平均预测、oracle 与配对 bootstrap
没有真实权重就不能计算。缺失 checkpoint 不应被解释为 Case G，也不补训。

从仓库根目录运行 D0B 部分：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m cmgm.scripts.d0b_5d_error_regime_diagnostic \
  --no-cuda --allow-missing-comparator
```

输出在 `experiments/d0b_5d_error_regime/<timestamp>/`，报告明确标记 `INCOMPLETE`。
默认不加 `--allow-missing-comparator` 时，缺失比较 checkpoint 会在准备数据前报错。

如以后找回实际历史权重，可以运行同一诊断：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m cmgm.scripts.d0b_5d_error_regime_diagnostic \
  --no-cuda --tempweighted-checkpoint /actual/path/to/official_best_checkpoint.pt
```

不要将其他模型权重改名冒充。脚本 strict load，并验证已有 variant metadata。
支持 `checkpoints/registry.json` 中以 internal variant 为 key、路径字符串或
`{"path": "..."}` 为 value 的登记；否则查找真实文件名与内嵌 metadata。
多个匹配 checkpoint 要明确指定路径，不自动选性能最好的文件。

`--prepared-data /tmp/d0b_error_data_verified.npz` 可复用本次原始预处理缓存。
缓存仅包含数值数组和 JSON，禁止 pickle；每次核验原始 CSV 和预处理源码 SHA256。
文件不存在则用原 `build_data` 创建；指纹改变则拒绝加载。无需缓存也能运行。

## 统计定义

- Primary index 动态使用 `MULTI_HORIZONS.index(5)`；原四 horizon 输出不变。
- 统一 float64 pooled MAE/MSE/RMSE/Hit；Hit 不屏蔽零值，CSV/JSON 中为比例。
- 每个 observation 的日期、商品和目标都从原始价格重建核验；双模型按完整身份 join，
  而不是按 loader 顺序配对。保留尾 batch，TRAIN/VAL/TEST 全量 inference。
- quantile 一律来自 TRAIN。sample-level latent/market 变量按每个 origin 取一次拟合阈值，
  target/past-return 按 observation 拟合。相等值归入较低区间，空组保持显式计数。
- Vol5/20 重算原 feature_builder 中因果 volatility 的原始单位值，并验证标准化后与 X 一致。
  不把各资产各自标准化的 volatility 平均值误当收益率波动率。新诊断列不输入模型。
- `TransitionScore` 固定为最后 5 次 posterior L1 movement 均值；不是未来 regime transition。
- target magnitude、未来方向和 reversal 是 ex-post 变量，不能用于部署时选择模型。
- 高/极高组有少于 20 个预测起点时标记 LOW SAMPLE SIZE；不会按 TEST 重切阈值填满组别。
- Delta = D0B error − TempWeighted error，正值支持 TempWeighted；Advantage 使用相反方向。
- Bootstrap 固定 1000 draws、seed 42；每次重采样 forecast origins，并保留该日期全部商品。
  这是 IID-origin 近似 CI；重叠 5d 窗口仍有时间依赖，不声称已做时间块 bootstrap。
- Oracle 使用目标选择误差较小的模型，是不可部署上界；平均预测严格固定 0.5/0.5。

## 文件与验证

`REPORT.md`、`results.json`、observation/sample/commodity CSV，核心分组 CSV，
`all_groups.csv`、`aligned_observations.csv`、`model_complementarity.csv`。
缺失 TempWeighted 时比较文件仅保存缺失原因，不填伪数值。
完整权重和模型 state_dict 的 SHA256 在 inference 前后必须一致。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  ../.venv/bin/python -m pytest -q tests/test_d0b_5d_error_regime.py
```

测试覆盖身份对齐与目标不一致拒绝、TRAIN-only 阈值、零/近零趋势、因果时间线、
movement 计算、pooled 指标、尾部贡献、日期聚类 bootstrap、win consistency、oracle、
真实 checkpoint 发现及缺失比较模型处理。双模型代码目前仅以合成预测验证，
未宣称已完成真实 TempWeighted checkpoint 验证。报告保留人工机制解释入口；不自动训练下一实验。
