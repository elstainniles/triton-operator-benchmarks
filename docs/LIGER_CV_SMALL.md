# Liger 六个小尺寸 CV 用例：BH 合并与 batch matmul

这组用例单独调用 LK07–LK12，固定 **B=2、H=4、S=D=32、FP32/IEEE**。
六个原始 kernel 已在仓库中；新增 `cases-cv-small.json` 将本轮独立 kernel 测试的输入、输出、
步长、标量参数、grid 和参考检查明确登记，并复用各目录的 `kernels.py`。
原 `cases.json` 中同尺寸的 transformer 正确性测试、大尺寸配置及历史数据均保留。

来源为 [Liger-Kernel 781a8aef64bf](https://github.com/linkedin/Liger-Kernel/commit/781a8aef64bf6105bdefb5f4c991e0d14c08e793)，
BSD-2-Clause。它们都是 generic 实现的 CV kernel，独立计时，不包含 PyTorch 参考时间。

## 六项输入与配置

| ID | kernel | 方向 | 输入 |
| --- | --- | --- | --- |
| LK07 | `_fused_neighborhood_attention_qk_kernel` | fwd | Q、K: [2,4,32,32]；mask: [32,32] |
| LK08 | `_fused_neighborhood_attention_av_kernel` | fwd | Attn、V: [2,4,32,32] |
| LK09 | `_fused_neighborhood_attention_grad_qk_kernel` | bwd | grad_attn、K: [2,4,32,32] |
| LK10 | `_fused_neighborhood_attention_grad_k_kernel` | bwd | grad_attn、Q: [2,4,32,32] |
| LK11 | `_fused_neighborhood_attention_grad_v_kernel` | bwd | Attn、grad_output: [2,4,32,32] |
| LK12 | `_fused_neighborhood_attention_grad_attn_kernel` | bwd | grad_output、V: [2,4,32,32] |

全部输入/输出为连续 float32；mask 也是 float32，邻域窗口 7、dilation=1。
BM=BN=BK=32，num_warps=4、num_stages=2，原 grid=(8,1,1)。axis=0 对应独立的 batch/head。
合并 factor=2/4/8 后实际 grid 分别为 (4,1,1)/(2,1,1)/(1,1,1)。
输入无需外部文件，seed=20261031；CPU float64 方程生成参考，输出按 float32 比较，
rtol=atol=0.005。另检查输入不变和输出前后各 64 个哨兵元素。bwd 直接调用独立 kernel。

## 一键运行

在仓库根目录查看配置、校验源码/配置指纹和历史样本；只需 Python 3.11+，不用卡：

```bash
python tools/run_liger_cv_small.py plan
python tools/run_liger_cv_small.py verify
```

先按 [Liger 环境说明](../operators/liger_kernel/README.md)准备隔离 Ascend runtime、固定 Liger wheel
和 `environment.local.json`。入口复用仓库的包路径/二进制核验、独立缓存、卡锁及健康/占用检查，
每个 kernel 独立进程串行运行；不会安装包或下载依赖。

六个原始小尺寸基线（factor=1，关闭下层 batch matmul）：

```bash
"$RUNTIME_PYTHON" -I tools/run_liger_cv_small.py run --operator all \
  --physical-card 6 --environment environment.local.json --output runs/liger-cv-small-base-001
```

支持 `--operator LK07,LK10` 或精确 kernel 名称、`--correctness-only`。
可额外用 `--task-lock /path/to/task.lock --card-lock /path/to/card-6.lock` 接入已有任务锁。
异常保存日志后停止；恢复时选择未完成的 ID，指定新输出目录。

使用**支持 `rank3_full_tile_batch_v2` 的实验 CV 编译器**时，一键运行 factor=2：

```bash
"$RUNTIME_PYTHON" -I tools/run_liger_cv_small.py run --operator all --factor 2 \
  --physical-card 6 --environment environment.local.json --output runs/liger-cv-small-f2-001
```

factor 可选 2/4/8，入口自动启用 `block_batching_cv=True`、`block_batching_axis=0`、
`enable_hivm_batch_matmul=True`；始终传入原 grid，由编译器合并。
`--factor 1 --batch-control` 仅打开下层 batch matmul，不合并 program。
普通 TA 不保证支持这些实验选项；该仓库不分发实验编译器 wheel。
入口检查运行时 APPLIED、实际 factor/grid、完整 mask 守卫、rank-3 dot、下层开关及 IEEE 精度，
并要求该合并候选的 auto blockify 关闭。回退、未知路径或精度错误不会记作有效合并性能。

新运行输出 `summary.json`、`performance.csv`、各 ID 的日志、数值检查、编译指纹和 profiler 证据。
每项采用 15 个目标 Triton device kernel 样本的中位数；通过源码/编译产物及 CPU→NPU flow 归因，
排除首次编译、参考、Torch/厂商 kernel 和主机调度时间。
这是单次指定模式的复现入口，**不会自动执行下述四轮收益确认**。
新导出的入口仅做离线校验，未重新实跑 NPU；已有设备结果来自下面的原实验框架。
导出校验：Linux 隔离 runtime 的 65 项 CPU 单测全部通过，`tools/verify_repository.py` 全仓库审计通过；
大小尺寸配置、原函数 AST、1920 个小尺寸样本及 fallback 拒绝逻辑均已检查。

## 已完成实验的历史结果

索引：[cv-small-manifest.json](../operators/liger_kernel/cv-small-manifest.json)。
汇总：[cv-small-performance.csv](../operators/liger_kernel/cv-small-performance.csv)。
逐 kernel 的 `results/20261031-cv-small/measurement.json` 保留 control、F2/F4/F8 的全部结果与逐轮样本，
包括变慢候选，不以基线覆盖它们。

| ID | 配对基线 (µs) | BH 合并 (µs) | factor | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| LK07 | 4.1850 | 3.2885 | 2 | 1.2726× |
| LK08 | 4.1095 | 3.5500 | 2 | 1.1576× |
| LK09 | 4.2500 | 3.4560 | 2 | 1.2297× |
| LK10 | 4.1225 | 3.6695 | 2 | 1.1235× |
| LK11 | 4.2095 | 3.7510 | 2 | 1.1222× |
| LK12 | 4.0525 | 3.2410 | 2 | 1.2504× |

初筛每候选两轮，每轮相邻交错运行基线/候选，各 15 个样本，轮间交换顺序。
两轮都超过 1.05× 才换独立 seed=20261101 再测两轮；四轮均超过 1.05× 才确认收益。
表内是四轮中位数再取中位数，每个候选对应自己的配对基线。
18/18 合并候选数值正确，8 项候选通过独立 seed 复核，6/6 kernel 的最优均为 F2；
F4 只有 LK07/LK12 确认收益，F8 全部变慢。
连同 6 项 batch control，公开保留 1920 个设备样本。采纳轮次中未观测到外来 PID。

历史环境：Ascend950PR_9579、CANN 9.1.0，TA CV 编译器
`329a9708f9077e0938f966faf9c86c8772ddfdfe`，原测试仓库
`20cc2f0ad0efafbd494341bff65cbaff0d89d69e`。wheel/libtriton SHA256 保存在各 measurement 中。
18 个合并产物均通过 batch MMA 审计，诊断 binary 与实测 binary SHA256 一致。
12 项负例验证 S=29/D=23 和 BH=3/F2 分别触发完整 mask 未证明、grid 不整除的完整基线回退。
这些负例不参与性能比较。

原始 trace/IR 含私有运行路径，未公开上传；保留样本、实际 launch、关键 lower 证据和原文件/归档 SHA256。
历史结果不表示任意 shape/dtype 都有收益，也不表示自动 factor 选择已验证。
