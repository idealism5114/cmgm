# FORMAL BASELINE BENCHMARK V2

独立入口：`cmgm.scripts.formal_baseline_benchmark_v2`。不覆盖 V1、历史 D0B 权重或其他实验。默认仅 preflight，不会意外训练。

## 命令

在 Commedities 目录、已激活的原 GPU Python 环境中使用以下命令（均不含斜杠）。本次代码验证使用 XGBoost3.4.1；若尚未安装：

```bash
python -m pip install xgboost==3.4.1
```

先在 GPU 上检查来源、输入、神经网络 sanity，并复现历史 D0B（不正式训练）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --stage preflight
```

在该目录继续完整流程：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --resume latest --stage all
```

也可以一次建立新 V2 suite 并执行完整流程：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --stage all
```

不要对已有有效实验再次建立新 suite 以重新比较性能。正常恢复使用 `--resume latest`，已完成任务通过 checksum 后直接复用。分阶段命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --resume latest --stage tune
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --resume latest --stage final
```

`final` 只接受全部 Stage A selection 已冻结的 suite。若代码/数值/资源问题中断，先检查并修复原因，才使用：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_baseline_benchmark_v2 --resume latest --stage all --retry-invalid
```

它不重跑 DONE 的差结果。完整训练 artifact 已保存时优先恢复其评估，不创建第二次训练；不完整 artifact 留存为 invalid 后，才重新执行因中断而无效的任务。CPU/GPU资源错误记录到 results.json，然后停止，不减小输入或 batch。正式神经训练要求 GPU；传统模型的标准 sklearn/XGBoost hist 拟合使用 CPU。RF/XGB 默认最多8个可用 CPU worker，外层96输出 XGBoost 串行，避免嵌套线程爆炸；`--cpu-jobs` 可在创建 suite 时指定并冻结。

## 九个正式模型与输入

Ridge、Random Forest、XGBoost、LSTM、TCN、Vanilla Transformer、Graph WaveNet、MTGNN、D0B。没有 Zero、GRU 或 V1 简化图模型。

全部接收原 builder 的 `(B,20,N,21)`，N 来自 dataset。当前 N284，每 origin119,280个历史 scalar。只有无损 flatten/transpose 或模型内可训练 projection，没有 market pooling、PCA、feature selection、额外标准化/裁剪或 target normalization。

commodity mapping、逐模型输入可逆审计、split/global-origin边界写入 input_equality_audit.json。当前 builder 不暴露日历日期，记录精确全局行号，而不猜测日期。

全部方法使用每个 TRAIN origin；神经模型保留最后一个不足64的 batch（V2与历史 loader 的 drop_last=True 有此明确区别，目的是匹配传统模型使用的样本集合）。历史 data/feature/target construction 不变。除历史 D0B sanity 外，所有正式 D0B 权重重新训练。

## 官方代码

Graph WaveNet：作者 `nnzhan/Graph-WaveNet`，commit `6b162e80c59a1d494809252eca055cff93dc66b1`。

MTGNN：作者 `nnzhan/MTGNN`，commit `f811746fa7022ebf336f9ecd2434af5f365ecbf6`。

均保留 MIT LICENSE、README、SHA256及精确 patch。见 `third_party/baselines/ADAPTATION_NOTES.md`。Graph WaveNet仅修复旧 Conv1d接受4D输入的兼容问题；MTGNN仅改为包内相对 import。Graph learned adjacency 使用其各自原始机制，不读取 D0B 权重。

官方结构的有效 receptive field 与输入窗口不应混为一谈：Graph WaveNet保持官方默认13步 receptive field，输入仍是20步，取最后 origin输出。MTGNN保留 official time-spanning LayerNorm；其卷积核是 causal，但不能声称整条中间 hidden timeline 是 streaming prefix-invariant。MTGNN eval top-k扰动仍保留，batch/single sanity使用相同 RNG realization。没有为通过检查而替换官方模块。

## 训练和选择

Stage A仅 seed42：Ridge5组、RF8组、XGBoost4组、六个神经模型各3个 LR，共35个候选任务。候选 grid与实际 estimator参数固定为用户要求，所有选择只基于 pooled VAL5 MAE。

神经模型统一四 horizon Huber(delta=.02)，D0B额外保留正式 switch KL与warmup。Adam、WD1e-5、batch64、200epoch上限、patience10。Checkpoint与 early stopping监控 VAL5 MAE；scheduler监控每 batch多 horizon VAL Huber的平均值，factor.5/patience5。

Stage A全部完成才写入并锁定 selected_hyperparameters.json。Stage B随机模型以所选配置重新训练 seed42、2025、3407，共24个任务。Ridge复用 Stage A选中的确定性拟合，不伪造重复seed。因此完整方案最多59次实际 fit/train，不含preflight；Stage B seed42是明确的独立重新拟合。

全部25个正式 artifact（24随机run +1 Ridge）固定之后才运行正式 TEST。不存在逐候选查看TEST。历史D0B standardized TEST sanity是用户指定的独立例外，发生在 Stage A之前且不进入正文主表。

## 保存结果

结果在 `experiments/formal_baseline_benchmark_v2/<timestamp>`，权重在 `checkpoints/formal_baselines_v2/<same timestamp>`。后者包含 A/B、model、seed、candidate标识，防止覆盖。

REPORT与所有要求的JSON/CSV会逐阶段更新。未完成的九行表明确写 pending，不是论文结果。完整后显示随机模型全部三个seed的 mean±sample std（ddof=1），Ridge单次且std为—。RMSE先按每run计算sqrt(MSE)，再跨seed汇总。Hit包括零target、无mask。每seed、每commodity与prediction range全部留存。

只比较已注册模型，不因结果差扩大搜索或替换 baseline，不声明三seed结果自动具有统计显著性，完成后停止。
