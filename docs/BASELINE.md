# FlagGems 历史基线与手改 tile 对照

这批数据是 2026-10-26 的历史实测，不是本次重新跑分。8 个方向/实现条目中，每项至少一个 FP32 配置测得耗时下降超过 5%；按算法名称去重只有 Mul、Where、GELU、SwiGLU 4 类，不能把同族拆分当成 8 个独立泛化算法。

35 个 FP32 主配置中，23 个（65.7%）测得 >5% 耗时减少。逐配置取已测最优（含基线），几何平均加速 **1.224×**、算术平均加速 1.259×、平均耗时减少 16.42%。该统计是候选中的 best-of-measured，不代表一个固定 tile 或自动策略的可交付收益。

| 条目 | FP32 配置 | 耗时下降 >5% | 几何平均加速 | 峰值配置 | baseline → best (μs) | 峰值下降 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| [mul_contiguous_fwd](../operators/flaggems/mul_contiguous_fwd/) | 4 | 2/4 | 1.167× | 1048576 / B4096 | 6.594 → 4.366 | 33.79% |
| [mul_scalar_fwd](../operators/flaggems/mul_scalar_fwd/) | 4 | 3/4 | 1.324× | 16777216 / B16384 | 82.807 → 45.595 | 44.94% |
| [mul_broadcast_fwd](../operators/flaggems/mul_broadcast_fwd/) | 5 | 4/5 | 1.146× | 1024×8192 / B4096 | 270.055 → 215.380 | 20.25% |
| [where_fwd](../operators/flaggems/where_fwd/) | 4 | 2/4 | 1.111× | 1048576 / B4096 | 6.552 → 4.984 | 23.93% |
| [gelu_tanh_fwd](../operators/flaggems/gelu_tanh_fwd/) | 4 | 3/4 | 1.162× | 127 / B256 | 2.114 → 1.449 | 31.46% |
| [gelu_tanh_bwd](../operators/flaggems/gelu_tanh_bwd/) | 4 | 3/4 | 1.259× | 127 / B256 | 2.947 → 1.734 | 41.16% |
| [swiglu_fwd](../operators/flaggems/swiglu_fwd/) | 5 | 2/5 | 1.361× | 1024×8192 / M4H2048 | 64.945 → 25.259 | 61.11% |
| [swiglu_bwd](../operators/flaggems/swiglu_bwd/) | 5 | 4/5 | 1.266× | 1024×8192 / M4H2048 | 153.994 → 93.534 | 39.26% |

只统计原 kernel 的 P04/P05/P06/P10/P11：23 个 FP32 配置，15 个 >5%，几何平均加速 1.250×；Where/GELU 的 12 个配置属于适配探针，单独识别。P05 某配置的 >5% 差值仅 0.067 μs；若另要求绝对改善 >0.1 μs，计数降为全部 22/35、原 kernel 14/23。没有做显著性检验，小差值必须重测确认。

全部导入记录是 51 组（35 FP32 主扫描 + 8 BF16 + 8 smoke）、218 个候选检查：189 通过、29 编译失败。失败不计为零耗时，也不从候选列表删除。BF16 和 smoke 单独保留，不并入 FP32 收益统计。

全部 8 项为 vv：没有 matmul。fwd/bwd 分别测对应 Triton kernel，没有将 Python/PyTorch wrapper 或 autograd 总耗时混入结果。

[环境与溯源](DATA_PROVENANCE.md) · [手改说明](MANUAL_TILING.md) · [机器可读汇总](historical-summary.json)
