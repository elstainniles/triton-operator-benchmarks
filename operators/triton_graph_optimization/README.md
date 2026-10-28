# TritonGraphOptimizationTests 性能算子

来源：[weizhan4/TritonGraphOptimizationTests](https://gitcode.com/weizhan4/TritonGraphOptimizationTests/tree/main)，固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。与 `operators/flaggems/` 分目录存放。

7 组均自带输入生成、CPU 参考和直接 Triton kernel 计时入口，默认无需 `.pt`。**测试必须关闭图优化：`enable_graph_optimize=False`**，运行器显式设置并验证编译后的选项。

| ID | 算子 | 类型 | 默认输入来源 |
| --- | --- | --- | --- |
| TG01 | [gather + scale fwd](fbgemm_gather_scale_dense_tokens_fwd/) | vv | 新增确定性合成输入 |
| TG02 | [gather + scale + FP8 quant fwd](fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_fwd/) | vv | 新增确定性合成输入 |
| TG03 | [FP8 row matmul fwd](matmul_fp8_row_fwd/) | cv | 新增确定性合成输入 |
| TG04 | [dacs_segsum fwd](dacs_segsum_fwd/) | vv | 小配置及上游尺寸，固定 seed |
| TG05 | [segsum_dadt bwd](segsum_dadt_bwd/) | vv | 小配置及上游尺寸，固定 seed |
| TG06 | [dadt_cumsum_fused bwd](dadt_cumsum_fused_bwd/) | vv | 小配置及上游尺寸，固定 seed |
| TG07 | [fused_moe fwd](fused_moe_fwd/) | cv | 修正路由准备，四种量化模式，CPU 参考 |

cv/vv 按计算中是否包含 matmul（`tl.dot`）分类。共 119 个生成配置，smoke 包含 10 个配置；实际设备验证范围见 [验证记录](../../docs/VALIDATION.md)。

```bash
python tools/run.py plan --operator graph --suite smoke --variants baseline
python -I tools/run.py run --operator graph --suite smoke --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/graph-001
```

从仓库根目录、已配置的隔离 Ascend runtime 运行。各目录的 `benchmark.py` 和 `cases.json` 完整定义输入、参考及 tile/grid，公共运行器只加载原始 JIT 函数体，采用固定基线配置。数值失败不记录性能，PyTorch 准备/参考计算不计入 kernel 耗时。

原 11 个 Python 文件与图优化关闭版继续保留，来源哈希见 [source-manifest.json](source-manifest.json)。原 `.pt` 未公开的事实仅影响可选旧用例回放，不阻塞默认测试。另有 [3 个外部测试链接](external_references/README.md)，未包含实现，不计入 7 组。

[运行说明与限制](../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [来源许可](SOURCE_NOTICE.md)
