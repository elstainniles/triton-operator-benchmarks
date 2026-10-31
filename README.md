# Triton Operator Benchmarks

用于保存 Triton 算子的原始实现、手工 tile/grid 对照、正确性测试和可追溯性能数据。每个算子方向/实现独立存放，后续按来源加入更多算子。

首批来自 [FlagGems](https://github.com/flagos-ai/FlagGems)，固定提交 `61f3372e817e8bf4e9279df9ef21c0c453f718b8`。包含 Mul 三种布局、Where、GELU fwd/bwd、SwiGLU fwd/bwd，共 **8 个方向/实现条目**，按算法名称去重为 4 类。

| ID | 目录 | 内容 | 历史测试方式 |
| --- | --- | --- | --- |
| P04 | [mul_contiguous_fwd](operators/flaggems/mul_contiguous_fwd/) | 连续张量乘法 | 归档 Triton kernel，手改 BLOCK_SIZE/grid |
| P05 | [mul_scalar_fwd](operators/flaggems/mul_scalar_fwd/) | 张量乘标量 | 归档 Triton kernel，手改 BLOCK_SIZE/grid |
| P06 | [mul_broadcast_fwd](operators/flaggems/mul_broadcast_fwd/) | 二维广播乘法 | 归档 Triton kernel，手改 BLOCK_SIZE/grid |
| P07 | [where_fwd](operators/flaggems/where_fwd/) | Where | 原计算函数的连续适配探针 |
| P08 | [gelu_tanh_fwd](operators/flaggems/gelu_tanh_fwd/) | GELU tanh fwd | 原计算函数的连续适配探针 |
| P09 | [gelu_tanh_bwd](operators/flaggems/gelu_tanh_bwd/) | GELU tanh bwd | 原计算函数的连续适配探针 |
| P10 | [swiglu_fwd](operators/flaggems/swiglu_fwd/) | SwiGLU fwd | 归档二维 Triton kernel，手改 BM/BH/grid |
| P11 | [swiglu_bwd](operators/flaggems/swiglu_bwd/) | SwiGLU bwd | 归档二维 Triton kernel，手改 BM/BH/grid |

Where/GELU 没有接入 FlagGems 的 `pointwise_dynamic` 生成器，不能用探针结果宣称原生成器获得同等加速。上游完整文件放在各算子的 `upstream/`，执行版保留所选 kernel/计算函数的函数体，移除外部调优装饰器并使用受控 NPU launch。

第二批来自 [TritonGraphOptimizationTests](https://gitcode.com/weizhan4/TritonGraphOptimizationTests)，独立放在 [operators/triton_graph_optimization](operators/triton_graph_optimization/)，与 `flaggems` 并列。包含 7 组实际测试代码：两种 gather/scale、FP8 matmul、dacs_segsum、segsum_dadt bwd、dadt_cumsum_fused bwd 和 fused_moe。另保留 3 项仅有外部运行说明的引用，未计入已导入实现。

**第二批测试前必须关闭图优化：`enable_graph_optimize=False`。** 默认入口已自带输入和 CPU 参考，全部 7 组均无需外部 `.pt`。按 [第二批测试说明](docs/GRAPH_OPTIMIZATION_TESTS.md) 可直接测 kernel 性能。

2026-10-28 已在 Ascend950PR / TA main 完成上述两批 **15/15 个条目、18/18 个 smoke 配置的正确性及纯 kernel 计时**。TG04–TG06 显式使用 SIMT；这是固定基线验证，较大尺寸和新入口 manual 候选未在本轮全面验证。实测表、失败记录和验证范围见 [VALIDATION.md](docs/VALIDATION.md)。

新增 [Liger-Kernel 通用优先批次](operators/liger_kernel/)：46 个精确 kernel，38 generic、7 Ascend-only、1 experimental，ID 为 LK01–LK46。每个 kernel 独立目录，提供原函数、正确性测试、自生成输入、原始 grid/tile/autotune 和历史性能；历史有效 42/46，其余编译失败/依赖阻断保留。见[完整清单](operators/liger_kernel/catalog-table.md)与[新入口验证](docs/LIGER_VALIDATION.md)。上述三批共 61 个条目，Liger 额外需要固定源码 wheel 和测试依赖。

新增 [FLA Triton 抽取批次](operators/flash_linear_attention/)：从固定版本的
`TritonAscendTest/flash-linear-attention` 导入 296 个具有静态 launch 引用的定义，
按来源文件和符号分别登记。122 个来自 `fla/`，174 个来自原仓库 Ascend 测试；
同名或相同函数体仍保留来源区别，不将条目数当作独立算法数。
未适配项明确为 `source_archive`，运行入口会返回 `not_executable` 和原因。
已接入的生成式用例提供固定种子输入和独立 CPU 参考；新入口的设备验证仍待完成。
覆盖范围、依赖及执行方法见 [FLA 说明](docs/FLA_KERNELS.md)。

## 资料入口

- [测试说明](docs/TESTING.md)：环境、输入、数值校验、计时、命令和限制。
- [图优化测试批次](docs/GRAPH_OPTIMIZATION_TESTS.md)：必须关闭图优化、7 组入口及输入依赖。
- [Liger 批跑说明](operators/liger_kernel/README.md)：通用优先的 46 个 kernel、依赖与目标 kernel 计时。
- [Liger 六个大尺寸 CV 用例](docs/LIGER_CV_LARGE.md)：B=8/H=16/S=128/D=64，独立批跑入口与第一版 factor 合并实测对照。
- [手改 tile/grid 说明](docs/MANUAL_TILING.md)：基线、候选、grid 联动及退化例子。
- [历史基线汇总](docs/BASELINE.md)：FP32/BF16、每个算子的历史收益。
- [新增算子约定](CONTRIBUTING.md)：目录、接口、数据和验证要求。
- [整理校验记录](docs/VALIDATION.md)：源码、结果和离线测试的校验范围。
- [FLA 抽取与生成式用例](docs/FLA_KERNELS.md)：完整静态清单、来源区别和当前可执行范围。
- [来源及许可](NOTICE)：FlagGems Apache-2.0 归属与改动范围。

FlagGems 各目录都有 `kernels.py`、`baseline.py`、`manual_tile.py`、`case.py`、`cases.json`、`operator.json` 和 `results/20261026/`。图优化测试批次通过各自 `benchmark.py` 和 `cases.json` 提供固定基线、自生成输入和 CPU 参考，原 JIT 函数保持不变；旧脚本另作来源归档。Liger 使用 `kernels.py`、`test_case.py` 与 `cases.json`。各批共用 `tools/run.py` 入口，各算子独立目录。

## 快速查看与离线验证

以下命令只需 Python 3.11+，不导入 Torch/Triton，也不使用加速卡：

```bash
python tools/run.py list
python tools/run.py plan --operator swiglu_fwd --suite fp32
python tools/verify_repository.py
python -m unittest discover -s tests -v
```

已配置隔离 Ascend runtime、Liger 依赖和环境清单后，一条命令串行测试 Liger 46 个条目：

```bash
python -I tools/run.py run --operator liger --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/liger-second-001
```

`--operator graph` 选图优化批次 7 组，`flaggems` 选 8 个，`liger` 选 46 个，`fla` 选 FLA 抽取批次，`all` 选全部登记条目，也支持单个 ID 或逗号分隔列表。FLA 源码归档项会明确报告未适配，不计入成功执行或性能统计。`--suite all` 测所选算子的全部登记配置。Liger 的 `smoke` 仍是第二批原尺寸，可能占较多显存。整批输出 `performance.csv` 和逐算子证据；统计只包含数值通过、设备占用检查通过的 Triton kernel 设备耗时。

NPU 运行方式见 [TESTING.md](docs/TESTING.md)。不要把上游源码直接安装进测试 runtime，也不要覆盖已有环境或已有测量结果。

## 历史结果与当前代码验证的区别

已导入 **51 个运行内比较组、218 个候选检查**：35 个 FP32 主扫描组、8 个 BF16 补充组、8 个 smoke 组；189 个候选通过、29 个编译失败。全部候选的配置、grid、正确性、样本、失败和退化都保留。

固定环境为 Ascend950PR、TA `c4d742a1e51fe259e9039dde1abc708b6df32a76`、CANN 9.1.0、BishengIR `8995ba08834c`。历史硬件监控存在已记录的缺口，小差值不作为确定收益；这不是最新编译器或其它卡型的结论。

历史 FP32 35 个配置逐配置选已测最优后，几何平均加速为 **1.224×**；23/35 个配置测得耗时减少超过 5%。这是已测候选中的最佳值，包含保留基线，不是统一策略或自动调优的收益保证。

历史结果来自原测试框架，新入口的设备验证单独记录于 [VALIDATION.md](docs/VALIDATION.md)。历史结果不能自动证明新入口或未测配置通过；新运行始终生成独立结果。
