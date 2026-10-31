# Liger 六个大尺寸 CV 用例与第一版合并结果

2026-10-31 增加 LK07–LK12 的 `matrix128` 配置：**B=8、H=16、S=128、D=64，float32**。
这些是六个精确 kernel 的独立调用；源函数保持与固定 Liger 提交
`781a8aef64bf6105bdefb5f4c991e0d14c08e793` 一致。
旧 `cases.json`、小尺寸正确性用例和历史结果继续保留。

每个算子目录新增 `cases-cv-large.json` 和 `results/20261031-cv-large/measurement.json`。
统一索引是 [cv-large-manifest.json](../operators/liger_kernel/cv-large-manifest.json)，
四列原始数据为 [cv-large-performance.csv](../operators/liger_kernel/cv-large-performance.csv)。

## 输入与 launch

`shape=[B,H,S,D]` 描述 attention 的逻辑规模，不意味着全部输入具有相同形状。
Q/K/V/grad_output 为 `[8,16,128,64]`；Attn/grad_attn 为 `[8,16,128,128]`；
QK 的邻域 mask 为 `[128,128]`、float32，窗口 7、dilation 1。
所有张量连续，完整的指针形状、步长、输出形状和标量参数在各配置文件内。

| ID | kernel | BM / BN / BK | 原 grid | factor=8 后 grid |
| --- | --- | --- | --- | --- |
| LK07 | `_fused_neighborhood_attention_qk_kernel` (fwd) | 32 / 32 / 64 | (128,4,4) | (16,4,4) |
| LK08 | `_fused_neighborhood_attention_av_kernel` (fwd) | 32 / 32 / 128 | (128,4,2) | (16,4,2) |
| LK09 | `_fused_neighborhood_attention_grad_qk_kernel` (bwd) | 32 / 32 / 128 | (128,4,2) | (16,4,2) |
| LK10 | `_fused_neighborhood_attention_grad_k_kernel` (bwd) | 32 / 32 / 128 | (128,4,2) | (16,4,2) |
| LK11 | `_fused_neighborhood_attention_grad_v_kernel` (bwd) | 32 / 32 / 128 | (128,4,2) | (16,4,2) |
| LK12 | `_fused_neighborhood_attention_grad_attn_kernel` (bwd) | 32 / 32 / 64 | (128,4,4) | (16,4,4) |

BK 覆盖完整归约长度，使 K 循环只有一次迭代，符合第一版合并的能力范围。
这是显式的大尺寸测试配置，**不是原 wrapper 的 BK=32 配置**。
同一项的基线和合并候选采用完全相同的输入、dtype、tile、num_warps=4、num_stages=2。
seed 固定为 20261031；未修改 dot 精度或数值容差。

## 已有实测结果

“原始性能”指本次大尺寸、完整 K tile、未启用 CV 合并的配对基线；不是旧小尺寸历史耗时。
“第一版合并最优”是在实际 APPLIED 的 factor=2/4/8 中选耗时最低者，六项均为 **factor=8**；
即使慢于基线也保留真实合并耗时，不用 fallback 冒充合并结果。

| 算子名称 | 输入 shape / dtype | 原始性能 (µs) | 第一版合并最优性能 (µs) |
| --- | --- | ---: | ---: |
| `_fused_neighborhood_attention_qk_kernel` (fwd) | Q,K: [8,16,128,64]; mask: [128,128]; f32 | 140.5385 | 164.8830 |
| `_fused_neighborhood_attention_av_kernel` (fwd) | Attn: [8,16,128,128]; V: [8,16,128,64]; f32 | 72.4695 | 81.8490 |
| `_fused_neighborhood_attention_grad_qk_kernel` (bwd) | grad_attn: [8,16,128,128]; K: [8,16,128,64]; f32 | 73.4860 | 86.2620 |
| `_fused_neighborhood_attention_grad_k_kernel` (bwd) | grad_attn: [8,16,128,128]; Q: [8,16,128,64]; f32 | 69.0575 | 83.3215 |
| `_fused_neighborhood_attention_grad_v_kernel` (bwd) | Attn: [8,16,128,128]; grad_output: [8,16,128,64]; f32 | 72.4025 | 81.3305 |
| `_fused_neighborhood_attention_grad_attn_kernel` (bwd) | grad_output,V: [8,16,128,64]; f32 | 140.4790 | 155.1695 |

18/18 候选实际合并、数值正确且与基线逐位相同；0/6 配置确认性能提升。
每个候选两轮相邻交错采样，每轮基线与候选各 15 个设备样本；表中为两轮中位数再取中位数，
每个候选使用它自己的配对基线。合计 1080 个有效设备样本，全部随 measurement.json 保存。
计时通过 profiler CPU→NPU flow 只归属目标 Triton kernel，不包含主机调度、参考或 Torch/厂商 kernel。
正确性采用 CPU float64 方程参考、FP32 输出和 `rtol=atol=0.005`，并检查输入不变、输出两侧各 64 元素哨兵。

第一版 `static_masked_lanes_v1` 沿 grid axis 0 把 F 个独立 batch/head 的原二维计算静态展开到一个 program，
并非拼成一个更大的 dot。编译器在该模式下关闭普通 auto blockify，故这些数据衡量的是完整机制；
没有将静态展开与普通 grid 映射变化的贡献单独分离。

历史环境：Ascend950PR_9579、物理卡 6、CANN 9.1.0，TA CV 分支
`7edb1aafc3d99f9fb11103a33cdb71db8ba868c1`（implementation 21，`ta_cv_structured_v2`），
BishengIR `8995ba08834c`。wheel SHA256：
`facdf6ddf3b1171df82c1484bce1dabda30851cbe8bda5106198f09f44775990`；
libtriton SHA256：`c3bd9fceb6f9f88f3978a5513860ddcbf00cb41203a568e16e3df6df938c4bcc`。
历史采样只忽略过经单独核验的固定残留；公开入口默认拒绝任何外来占用，不携带该例外。
原始私有 trace 未上传；导入记录保留 trace/result/scope 的 SHA256 和全部样本。

## 一键运行这六项

先按 [Liger 环境说明](../operators/liger_kernel/README.md)准备并核验隔离 runtime 和固定 Liger wheel。
此入口自生成输入、直接调用本仓库 kernel，绕过旧小尺寸 transformer wrapper；
每个 kernel 独立进程/缓存、串行运行，沿用环境/源码哈希审计、卡锁和占用监控。
不会安装包、构建编译器、清理进程或重置设备。

离线核验和查看计划，不导入 Torch/Triton、不访问设备：

```bash
python tools/run_liger_cv_large.py verify
python tools/run_liger_cv_large.py plan
python tools/verify_repository.py
python -m unittest discover -s tests -v
```

六项大尺寸原始基线：

```bash
"$RUNTIME_PYTHON" -I tools/run_liger_cv_large.py run \
  --physical-card 6 --environment environment.local.json \
  --output runs/liger-cv-large-baseline
```

可用 `--operator LK07,LK12` 只选择未完成项，或 `--correctness-only` 只校验。
若测试站点另有全任务锁/卡锁，必须追加已有锁路径，不能绕开现有调度器：
`--task-lock /absolute/path/to/validation-task.lock --card-lock /absolute/path/to/npu-6.lock`。
占用或锁冲突会退出并保留证据；数值/编译/执行失败也会停止，审阅后用新的输出目录续跑未完成项。

复现第一版 factor=8 需要安装上述支持手动 CV 合并的编译器 wheel：

```bash
"$CV_RUNTIME_PYTHON" -I tools/run_liger_cv_large.py run --factor 8 \
  --physical-card 6 --environment environment.cv.local.json \
  --output runs/liger-cv-large-factor8
```

该命令要求 runtime 报告 APPLIED、正确 factor/axis/grid 和 `static_masked_lanes_v1`，
否则拒绝候选计时。缺少该功能的编译器不能重放合并候选；普通基线不传合并选项。
新跑输出是单版的 15 样本中位数，**不会从两个独立 run 自动宣称加速比**；
上表来自另行完成的双轮交错配对实验。

本次仓库更新完成离线源码、参数、原始样本与 host 测试核验；没有占用 NPU 重新跑测。
历史实测证明冻结 kernel/配置在所列环境通过，不等于新打包入口已重新执行过 NPU 验证。
