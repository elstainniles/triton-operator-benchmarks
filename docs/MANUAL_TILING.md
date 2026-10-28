# 手改 tile/grid 的范围

这批候选保留原计算公式与 dtype。`baseline.py` 固定对照 tile；同目录 `manual_tile.py` 是手改版本，显式保存其它 tile；两者通过 `launch.py` 绑定相同 kernel 和输入。不同算子目录不交叉引用实现，可独立新增或修改。

| 条目 | 基线 | 手改候选 | grid |
| --- | --- | --- | --- |
| P04/P05 连续/标量 Mul | B1024 | B256、B4096、B16384 | `(ceil(N/B),)` |
| P06 广播 Mul | B1024 固定对照 | B256、B4096、B16384 | `(ceil(M*N/B),)` |
| P07/P08/P09 Where/GELU 探针 | B1024 固定对照 | B256、B4096、B16384 | `(ceil(N/B),)` |
| P10/P11 SwiGLU | BM64/BH64 | BM1/BH512、BM4/BH512、BM16/BH256、BM4/BH2048 | `(ceil(M/BM), ceil(H/BH))` |

`B` 对应 kernel 的 BLOCK_SIZE；`BM/BH` 对应 BLOCK_SIZE_M/H。grid 根据 tile 同步改变并保留尾部 mask，不是仅减少 program 数而漏算数据。所有候选固定 Ascend `num_warps=32, num_stages=2`；不对这些选项再做调参。

P04/P05 的 B1024、SwiGLU 的 64×64 来自源码 tile；均使用受控 NPU launch，不能视为未改上游 GPU wrapper 的端到端基线。P06 的 B1024 是显式控制值，不代表上游 autotune 结果。Where/GELU 仅保留计算函数，使用连续向量适配探针；没有复制 `pointwise_dynamic` 生成器的调度与 stride 支持。

## 如何阅读收益

每个 case 的 `comparison.json` 记录全部候选和 baseline，以及该 case 的 best。未改善时 best 就是 baseline，减少百分比为 0%。失败候选没有耗时；慢的候选保留负收益。不能只保留每个算子的峰值。

`speedup = baseline_us / candidate_us`；`time_reduction_pct = 100 × (1 - candidate_us / baseline_us)`。1.25× 加速对应耗时减少 20%，两种百分比不能混用。

例如 P10 SwiGLU fwd 的 (1024,8192) FP32：64×64 → 4×2048，64.945 → 25.259 μs，2.571×、耗时减少 61.11%。这说明该配置有较大调度改进空间，不说明 4×2048 对所有形状都最好。

反例：P09 GELU bwd、N=8193 FP32，B4096 是 8.093 μs，而 B1024 基线为 3.262 μs，耗时增加约 148.1%。部分 B16384 候选发生 UB 容量溢出。容量、每个 program 的工作量、program 数量与尾部浪费共同影响结果，不能从“更大 tile”直接推出“更快”。

当前仓库保存的是手工候选集合和历史逐 case 选择，不实现自动 tile 选择规则。未来 compiler 优化或新候选必须作为单独版本、隔离环境、同口径重测，保留 baseline 与失败证据。
