# TritonGraphOptimizationTests 导入批次

来源：[weizhan4/TritonGraphOptimizationTests](https://gitcode.com/weizhan4/TritonGraphOptimizationTests/tree/main)，固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。本目录与 `operators/flaggems/` 并列，各算子独立存放。

**本批测试前必须关闭图优化：`enable_graph_optimize=False`。** 执行版的每个 kernel launch 已显式设置。详见 [测试说明](../../docs/GRAPH_OPTIMIZATION_TESTS.md)。`upstream/` 是未修改原始快照，不作为默认运行入口。

| ID | 目录 | 类型 | 输入与当前条件 |
| --- | --- | --- | --- |
| TG01 | [fbgemm_gather_scale_dense_tokens_fwd](fbgemm_gather_scale_dense_tokens_fwd/) | vv | 需要上游未提交的 `.pt` 输入 |
| TG02 | [fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_fwd](fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_fwd/) | vv | 需要上游未提交的 FP8 `.pt` 输入 |
| TG03 | [matmul_fp8_row_fwd](matmul_fp8_row_fwd/) | cv | 需要 `.pt` 输入；上游 autotune key 问题保留并记录 |
| TG04 | [dacs_segsum_fwd](dacs_segsum_fwd/) | vv | FP32 (16,32,2048)，chunk_size=16 |
| TG05 | [segsum_dadt_bwd](segsum_dadt_bwd/) | vv | FP32 (16,32,128,16,16)，chunk_size=16 |
| TG06 | [dadt_cumsum_fused_bwd](dadt_cumsum_fused_bwd/) | vv | 4 个 FP32 (16,32,2048)，chunk_size=16 |
| TG07 | [fused_moe_fwd](fused_moe_fwd/) | cv | 100 个生成配置；保存输出，不含完整数值对比程序 |

cv/vv 按源计算中是否有 matmul（`tl.dot`）分类；TG03 即使保留 `force_simt_only=True`，仍是 cv 计算类型。

共导入 7 组测试、11 个 Python 源文件。每个目录有原始 `upstream/`、同名图优化关闭版和 `operator.json`。执行版仅增加 `enable_graph_optimize=False`，原数学函数、tile/grid、autotune、输入和容差均保留。文件 SHA256、Git blob 与改动位置数量见 [source-manifest.json](source-manifest.json)。

另外保留 [3 个外部测试引用](external_references/README.md)，它们在上游只有运行说明，不包含算子实现，不计入已导入的 7 组代码。

本批状态为 **NOT_RUN / NO_MEASUREMENTS**：仅完成代码归档与离线检查，没有新的 NPU 通过率或性能数据，也未新增手改 tile 版本。上游 wrapper profiler 输出需要筛选与数值核验，不能并入 FlagGems 已核验历史表。

完整性分开计算：代码收录 7/7 组；输入可自行生成 4/7 组（TG04–TG07）；3/7 组缺少 `.pt`（TG01–TG03）；本次 NPU 实测 0/7 组。输入自给不等于编译/执行通过。

```bash
python tools/run.py list
python tools/run.py plan --operator TG04
python tools/verify_repository.py
```

[测试命令、环境和限制](../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [来源许可记录](SOURCE_NOTICE.md)
