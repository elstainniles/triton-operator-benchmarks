# 数据来源与发布范围

公开代码来自 [FlagGems](https://github.com/flagos-ai/FlagGems)，固定提交 `61f3372e817e8bf4e9279df9ef21c0c453f718b8`，Apache-2.0。每个目录的 `operator.json` 记录上游文件路径、所选函数、完整源码快照 SHA256。快照按 LF 规范化；`kernels.py` 由 AST 提取所选函数，保留函数体和 `triton.jit`，移除外部 dispatch/autotune 装饰器。Where/GELU 增加显式连续探针；数学 shim 与历史 Ascend 实验一致。

## 历史记录

来自 2026-10-26 手改 tile/grid 泛化扫描。只导入 P04–P11 FlagGems 条目：35 FP32 主扫描 + 8 BF16 补充 + 8 smoke，51 个运行内比较组。218 条候选记录包含 189 次通过与 29 次编译失败。相同 case 在 smoke 和主扫描出现时，靠 run/comparison_group 区分；不能跨 run 把 baseline 和候选拼成新的加速比。

各目录 `results/20261026/measurements.json` 保留历史 measurement_id、比较组、输入、配置、grid、正确性、每次 kernel 样本、轮间中位数、错误诊断。`comparison.json` 保留逐候选相对 baseline 的收益与失败原因。

`samples_ms`/`median_ms` 的单位是 ms；比较表、轮中位数的单位是 μs。报告值是“轮中位数的中位数”，不是所有 samples 直接取一次中位数。smoke 为 2 轮×3 次，主测试为 3 轮×5 次。脚本核查样本数、轮中位数与比较配对。

历史完整输入 artifact 的 SHA256 与提取范围见 [import-manifest.json](import-manifest.json)。原 artifact 包含其它项目，未整份发布。这个仓库可离线核查导入数据内部一致性、函数体一致性和配置/grid，不含完整历史 profiler trace，不能仅凭这些 JSON 独立重做历史 trace 归因。

公开版移除了私有绝对路径、账号和内部编译器 URL。错误诊断保留编译阶段、错误文本、行号、文件 basename，并记录原始诊断的 SHA256。原始私有 artifact 保留在原工作区；这里不会声称脱敏后的文件与原 artifact 字节相同。

## 环境与限制

固定硬件 Ascend950PR_9579，TA `c4d742a1e51fe259e9039dde1abc708b6df32a76`，CANN 9.1.0，BishengIR `8995ba08834c`。详细公开版本/hash 见 [historical-environment.json](historical-environment.json)。这个 TA 历史环境含当时的 patch，并不是“当前 main 性能基线”；基线一词在本仓库是指同次实验中固定原 tile 的对照版本。

历史设备监控存在查询缺失，不是全程独占的证明。原扫描已排除一个被干扰的 RMSNorm run，它不属于本次 FlagGems 范围；本次没有把排除记录的值重新当成有效测量。小于 0.1 μs 的差值和接近阈值的结果尤其需要再次测量，历史 >5% 计数不是统计显著性声明。

新入口每 case 播种，历史每进程播种；随机张量不保证逐字节相同。新入口使用同样公式/形状/候选/launch 参数并额外预填输出。当前只有离线验证；新跑结果应进入独立目录，不能覆盖历史文件。发布仓库本身不会启动任何 NPU 测试。
