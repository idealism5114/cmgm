# D0B FORMAL ABLATION STUDY

13个唯一配置：FullD0B-Control；CommodityOnly、w/o Stock、w/o Bond；w/o Spatial、w/o Temporal、w/o Graph Propagation、w/o TempWeighted、w/o Markov Switching、w/o Microstate、w/o Switch KL、w/o Balanced Readout、w/o Adaptive Fusion Gate。

FullD0B-Control必须本轮训练一次，两张表共用。历史D0B只做统一evaluator sanity，不作正式delta参考。每配置seed42一次，不搜索、不因为结果差或消融更好而重跑。

## 运行命令

在项目根目录Commedities、激活原Python环境后运行。以下命令不含斜杠。

已完成只读预检时启动GPU训练：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.formal_ablation_study --resume latest --run >> formal_ablation.log 2>&1 &
```

```bash
tail -f formal_ablation.log
```

在另一环境首次使用时，先创建目录、校验历史checkpoint及13个配置的初始sanity（不训练）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_ablation_study
```

随后使用上面的续跑命令。默认不拟合，`--run`只启动本轮未完成配置。每次只训练一个模型；本实验入口有进程锁。`--cpu-check`只适用于只读验证，不能与`--run`一起使用。不会自动回退CPU训练。

中断后完整checkpoint会恢复后处理而非重训；未完成fit必须核实无效原因后才可显式使用`--retry-invalid`。不能因为性能差使用该选项。原始checkpoint永久保留。

## 实现与控制变量

`FormalD0BAblation`先完整调用原生D0B构造函数，再设置无参数的消融flags；所有配置实例化参数量相同（当前520549），并核对shared initialization。保留被bypass模块，另报实际prediction autograd connectivity与nonzero梯度元素数，不能用实例化总数冒充有效容量。

- Multimodal：正式build_data对每node/feature使用TRAIN均值/std中心化，零值就是TRAIN中心。只遮蔽指定市场动态输入，不删nodes，不变commodity targets。
- NoSpatial：直接`lstm_proj(h_temporal)`进原head；NoTemporal：直接`gcn_proj(h_spatial)`进原head，关闭被删除temporal branch的KL。
- NoGraph：TempWeighted之后直接gcn_norm+type_pool；NoTempWeighted：只把time维加权改为mean(dim=1)。
- NoMarkov：绕过transition/evidence，p和prior都固定uniform，仍执行三个generators和原Z递归；KL=0且无routing梯度。
- NoMicro：post-LN micro channel置零，保留原KL。NoKL：prediction不变，仅auxiliary contribution独立置零。
- NoBalanced：仅跳过两个readout LayerNorm。NoGate：保持两projection，固定0.5/0.5。

Full与旧nativeD0B的初始state_dict和eval prediction精确对照。默认flags不改变旧variants行为。

## 训练及评估

直接复用原train_epoch/validate_epoch。Adam1e-4、WD1e-5、batch64、200epochs、patience10、ReduceLROnPlateau factor.5/patience5；四horizon Huber(.02)求和。原KL epoch1 beta0，epoch20达到5e-4。checkpoint、early stopping和scheduler全用原multi-horizon VAL Huber batch mean；VAL5仅logging。

数据、TRAIN/VAL/TEST、20步window、21features、target clipping不改。所有评估包含完整origin×24commodity，用MAE/MSE/RMSE/Hit；zero targets included、unmasked sign hit。Hit表格为百分比，JSON为fraction。所有正式delta只相对于新FullD0B-Control。

Prefix causality检查temporal E/H/p/Z以及effective long/micro/readout。Spatial/final prediction是当前整个合法observed window的forecast，不要求它对window内已观察timesteps的扰动不变；不会把这种条件错误地作为时间泄漏判据。

## 结果

`experiments/formal_ablation_study/<timestamp>`含RUN_STATUS、两张固定主表、branch/fusion subtable、所有horizon、参数/init/masking/sanity、每商品CSV、runtime与最小mechanism数据。`checkpoints/formal_ablation/<timestamp>`独立保存本轮权重，不覆盖历史D0B。

每个配置完成立即保存并释放模型。只报告Full/NoKL的regime概率/entropy/occupancy，以及Full/NoBalanced的long/micro norm ratio。没有额外risk/tail/gradient-conflict诊断。

没有训练结果时表格明确PENDING；不预填贡献YES。单seed结果不能声明统计显著性。完成固定13项后STOP。
