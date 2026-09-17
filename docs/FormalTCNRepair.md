# FORMAL TCN BASELINE REPAIR + SINGLE RERUN

本轮仅修复 TCN architecture fidelity。旧 TCN 标记为 `INVALID_FOR_FINAL_BASELINE_TABLE`，原权重和历史数值保留；原报告顶部附失效声明，`TCN_INVALIDATION.json` 指向修复后的独立报告。其它八个正式模型的结果、checkpoint和训练均不改变。

## 来源与结构

官方 locuslab/TCN，commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`。核心代码、MIT license、原文件副本、patch和SHA256在 `third_party/baselines/tcn`。详见其中 `ADAPTATION_NOTES.md`。

完整输入 B,20,N,21 无损转为 B,N*21,20；直接进入 TemporalConvNet(N*21,[128,128,128],kernel_size=3,dropout=.1)。三块 dilation1/2/4，各两次 WeightNorm Conv1d → Chomp → ReLU → Dropout，再 residual + ReLU。第一个block的1×1仅用于residual downsample，无独立input projection。取最后时间步，唯一Linear128→96，再reshape B,4,24。

使用现代等价WeightNorm API；对底层v做N(0,.01)，令g=norm(v)，确保有效卷积权重在第一次forward后仍为指定的小正态初始化。未改变bias或head初始化。实际N284时参数量3,313,376。

训练完全调用原 `baseline_protocol.train_one`：seed42、Adam1e-4、WD1e-5、batch64、200epochs上限、patience10、ReduceLROnPlateau(.5,5)、sum4 Huber(.02)。multi-horizon VAL Huber选择checkpoint/early-stop/scheduler；VAL5仅logging。TRAIN shuffleFalse/drop_lastTrue。无gradient clipping，无额外head，无其它调参。

## 运行命令

在Commedities项目根目录、原Python环境运行。以下命令不含斜杠。

已准备修复报告时，只启动TCN一次GPU重跑：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m cmgm.scripts.formal_tcn_repair --resume latest --run >> formal_tcn_repair.log 2>&1 &
```

```bash
tail -f formal_tcn_repair.log
```

没有准备目录的环境，先只读预检（不训练）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cmgm.scripts.formal_tcn_repair
```

然后使用上述续跑命令。默认不训练。`--cpu-check`只用于只读预检，不能配合`--run`。入口只调用TCN，不遍历baseline suite。完成的corrected checkpoint会跳过，不重复fit；真正中断的fit停止，不能自动重试或因为性能不好重跑。

## 输出

独立目录 `experiments/formal_tcn_repair/<timestamp>`，权重 `checkpoints/formal_tcn_repair/<timestamp>/tcn_seed42_best.pt`。

完整九行amended benchmark表保留其它八行原值；修复TCN完成后只填入其新结果。旧TCN数值只保存在 `results.json → invalidated_TCN`，不得进入最终论文主表。`repair`字段记录source manifest及实现hash。新结果无论好坏都保留，不更改config、不再次重跑。

STOP after one corrected fit and the unchanged standardized evaluator.
