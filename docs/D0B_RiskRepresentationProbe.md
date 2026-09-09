# D0B-RiskRepresentationProbeDiagnostic

冻结正式 D0B checkpoint，导出原生 forward 的 `h_temporal`、`h_spatial`、真实 head 输入 `h_fused`、balanced `h_long/h_micro`、pooling 前 `(B,24,64)` 商品节点。只用 TRAIN 的 `|y_5d|` 拟合附件指定的 10 个带截距 OLS，加 MeanRisk 共11项；CommodityTrainMeanRisk 单独作为静态对照。D0B 无训练、无 backward、无 optimizer、无参数更新。

从 Commedities 目录运行：

```bash
cd /home/yangxiaotong/projects/myresearch/Commedities
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m cmgm.scripts.d0b_risk_representation_probe --no-cuda
```

可以用 `--prepared-data /tmp/d0b_error_data_verified.npz` 复用本机已存在、经源码和数据指纹验证的原始 preprocessing 缓存；缓存不存在时省略该参数。checkpoint 默认 `checkpoints/switching_latent_balanced_readout_best.pt`，可以使用 `--checkpoint` 指定实际路径。原 TEST5d pooled 指标无法复现时，在拟合 probe 前停止。

所有 feature mean/std、常量维度筛选、OLS 系数和 Tail90/95 阈值均由 TRAIN 确定，VAL/TEST 仅评估。固定 `np.linalg.lstsq(rcond=None)`，不调正则、不裁剪负风险、不更换目标。TRAIN halves 仅描述系数稳定性，不用于 evaluation prediction。R² 分母按请求使用 TRAIN pooled mean；MSE 直接汇总全部观测的平方误差，RMSE 为其平方根。

结果保存在 `experiments/d0b_risk_representation_probe/<timestamp>/`。包含完整表示、origin/商品映射、逐观测原始信号、probe 输出、所有统计 CSV、结果 JSON 和 REPORT。1000次 seed42 bootstrap 以 origin 为簇保留整组商品；相邻滑动窗口仍存在时间相关，CI 并非消除了序列依赖。

报告中的结论是对本次正式 checkpoint 结果的证据审阅。若后续运行的关键比较方向改变，脚本保留统计结果并要求重新审阅，不机械沿用 Case G、不自动增加 probe。代码不会生成生产 risk checkpoint 或集成 risk head。

验证：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_d0b_risk_representation_probe.py tests/test_d0b_tail_amplitude.py -q
```
