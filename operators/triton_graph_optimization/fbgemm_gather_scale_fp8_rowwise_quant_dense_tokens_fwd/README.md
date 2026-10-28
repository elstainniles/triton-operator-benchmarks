# fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_fwd

TG02 · fwd · vv（按是否包含 matmul 分类）

来源：[_fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens](https://gitcode.com/weizhan4/TritonGraphOptimizationTests/tree/2e67d5b6723b1d472dcef91adb8a83c5b6761765/_fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens)，固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。

**测试前必须关闭图优化：所有执行版 Triton kernel launch 显式传入 `enable_graph_optimize=False`。** 原始源码位于 `upstream/`，不可直接当作关闭图优化的执行版；本目录同名 `.py` 是只增加该选项的版本。

输入 shape/dtype/grid 与 FP8 类型来自外部 .pt；本地 test_common.py 一起保留。

入口：[test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_npu.py](test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_npu.py)；文件 hash、kernel 名和依赖见 [operator.json](operator.json)。未运行 NPU 测试、未记录性能基线、尚未新增手改 tile 候选。

[测试说明与关闭图优化方法](../../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [本批总览](../README.md) · [来源许可记录](../SOURCE_NOTICE.md)
