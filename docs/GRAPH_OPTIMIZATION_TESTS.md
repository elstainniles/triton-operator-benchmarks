# TritonGraphOptimizationTests 测试说明

本批位于 [operators/triton_graph_optimization](../operators/triton_graph_optimization/)，与 FlagGems 分开。固定来源提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`，7 组实际测试代码；另外 3 项只有外部链接，单独登记。当前尚未运行这些 NPU 测试，不填入通过率或性能数字。

## 必须先关闭图优化

**这批测试必须在图优化关闭的条件下编译和运行，才能用于此次验证。** 对照与将来的手改 tile/grid 候选都要使用同一设置，避免图优化重写计算图后改变比较对象。

在每个 Triton kernel launch 显式传入 TA 编译选项：

```python
kernel[grid](
    # 原有输入、shape、tile 与 launch 参数
    ...,
    enable_graph_optimize=False,
)
```

本仓库执行版的全部 9 处 launch 已加该选项，覆盖预调用、正确性测试和 profiler 中的调用。原始 `upstream/` 快照没有该修改，不要直接运行它。源码校验会拒绝开启、缺失或改变此选项的执行版。

此开关对应 Ascend backend 的 `enable_graph_optimize` 和 TTIR `add_graph_optimize` pass。不是 PyTorch `torch.compile`、`use_cuda_graph=False` 或设备 Graph replay 的开关；也不等于 `force_simt_only=True`、`TRITON_ALL_BLOCKS_PARALLEL` 或关闭 autotune。后几个选项按原测试保留，不能代替关闭图优化。不要猜测不存在的环境变量。

专用入口在导入测试前要求 backend 实际解析 `enable_graph_optimize=False`。若所用 TA 版本不支持/忽略该选项，停止并记录环境版本，不删掉参数勉强运行。测试使用全新专属缓存；归档 compiler metadata/编译日志时再次核对该字段为 false。不复用曾开启图优化的编译结果。

## 环境与运行方式

沿用 [通用环境要求](TESTING.md)：Linux、Python 3.11+、隔离 runtime、固定外部 TA wheel、显式 CANN、`-I`、禁用 user site、审计包路径/版本/libtriton/compiler hash，先检查设备健康与占用。必要依赖是 torch、torch_npu、triton-ascend、pytest、numpy；TG04 原文件还导入 einops。不要为准备这批测试修改另一任务的 runtime。

这批保留原测试框架与 autotune，使用独立入口，不套用 FlagGems 的 FP32/BF16 tile 候选矩阵。`list`/`plan` 仍可查看全部条目：

```bash
python tools/run.py plan --operator TG04

/path/to/isolated-runtime/bin/python -I tools/run_graph_tests.py \
  --operator TG04 --physical-card 6 \
  --environment environment.local.json --output runs/tg04-graph-off-001
```

命令从仓库根目录运行。仅当物理卡 6 已预约且健康空闲时使用这个示例；本次整理不会启动它。入口与 FlagGems 共用每卡锁及只读占用监控，不清理外来进程。输出目录必须全新，不能覆盖已有证据。

入口将该算子全部执行文件复制到输出目录的 `work/` 后再运行，原脚本的相对 profiler 路径和启动时清理旧 profiler 的代码只作用在新工作目录内。测试源目录与其它算子的结果不作为工作目录。`work/` 只加入测试模块搜索路径，不用 TA/TD 源码覆盖已安装编译器。

输出包含 `result.json`、`test.log`、`device.jsonl`、环境指纹、专属 cache，以及上游生成的 profiler/结果。stdout 已写入 `test.log`，过程查看该文件。编译、精度或运行异常保留原 traceback；不放宽容差、不替换算子公式。

## 输入与完整性限制

| ID | 数据与入口 | 当前已知限制 |
| --- | --- | --- |
| TG01 | gather+scale；`.pt` replay | 缺 `test__fbgemm_gather_scale_dense_tokens_v2.pt` |
| TG02 | gather+scale+FP8 row quant；`.pt` replay | 缺 `test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_v2.pt` |
| TG03 | FP8 row matmul；`.pt` replay | 缺 `test__kernel_matmul_fp8_row_v2_ieee.pt`；上游 autotune key=`BLOCK_D`，但 kernel 没有同名参数，待实测定位，未擅自修复 |
| TG04 | 自生成 FP32 da[16,32,2048]，chunk=16 | 原完整输出 max_abs<1e-4 检查后 profiling |
| TG05 | 自生成 FP32 dSSdA/SSdA[16,32,128,16,16] | 原比较及 profiler 混有 PyTorch 参考；检查失败后退出 |
| TG06 | 自生成 4 个 FP32 [16,32,2048] 输入 | 正确性阶段额外缩放 -log2(e)，原 profiler 测未缩放 wrapper，需标明范围 |
| TG07 | 100 个生成配置；支持 case ID 0–99 | 原 PASS 表示执行并保存输出，缺 `compare_results.py`；不是数值通过 |

TG01–TG03 的 shape/dtype/grid 由 `.pt` 数据决定，不能从文件名编造。需要可信来源的原始 `.pt`（上游用 `torch.load(weights_only=False)`，不要加载不可信文件）：

```bash
/path/to/isolated-runtime/bin/python -I tools/run_graph_tests.py \
  --operator TG01 --physical-card 6 --environment environment.local.json \
  --input-pt /path/to/trusted/test__fbgemm_gather_scale_dense_tokens_v2.pt \
  --output runs/tg01-graph-off-001
```

专用入口先调用该文件的 `test_get_last_loc_kernel`，成功后才调用 `test_perf`，并保存输入 SHA256。缺输入会在 NPU 初始化前报错。上游未提供 pytest 的 `ptfile_path` fixture，因此不建议直接全目录 pytest 采集。

即使只测试性能，这 3 个 `.pt` 仍是运行依赖：`test_perf` 会读取其中的 `input_data` 和 `grid`，不仅仅使用它们做精度对比。`gpu_output` 才是原数值比较需要的参考结果。因此当前是“7/7 代码已收录、4/7 输入自给、0/7 本次设备实测”，不能声称 7 组都能开箱完整复现。

若目标是新增性能摸测，可以以后为 TG01–TG03 明确定义 shape、dtype、strides、索引合法范围、量化 scale、grid/constexpr，并生成合成输入，不一定需要取得原 `.pt`。这些必须标记为新的合成用例，独立保存 seed、规格与结果；不能伪造原输入或宣称恢复了原测试覆盖。这次导入尚未补充合成输入。

MoE 可以逐 case 保存输出与原 profiler：

```bash
/path/to/isolated-runtime/bin/python -I tools/run_graph_tests.py \
  --operator TG07 --case-id 0 --physical-card 6 \
  --environment environment.local.json --output runs/tg07-case0-graph-off-001
```

省略 `--case-id` 会使用上游全部 100 配置；多个 `--case-id` 可选择多个。MoE 的 `numerical_status` 固定标记为 `OUTPUT_SAVED_REFERENCE_MISSING`，即使原日志写 PASS，也不能算已验证精度。

## 如何处理性能数据

本批沿用原 profiler，仅归档原始资料，不自动生成已采纳的 kernel 性能表。专用入口不会把 wrapper 计时或 PyTorch 参考计时写成 `median_us`。

采纳新性能前，必须先通过数值验证、确认关闭图优化、检查 autotune 实际选定配置，再从 profiler 按对应 Triton kernel 名精确筛选设备 `Duration(us)`。排除 PyTorch 参考、输出清零/分配、缩放、拷贝、编译及 host 调度。TG05 尤其不能对 profiler 所有行求和；TG06 的缩放范围也必须注明。MoE 缺参考时只能作为待复核原始 profiler，不能宣称有效性能收益。

`performance_status=RAW_PROFILE_REQUIRES_REVIEW` 表示脚本完成后的原始资料还需要人工核验；出现占用或监控异常时拒绝采用计时。`UPSTREAM_CHECKS_PASSED` 仅表示原脚本断言通过，不代表已通过 FlagGems 的独立参考/哨兵门禁。新入口目前只做离线验证，尚未 NPU 重验。
