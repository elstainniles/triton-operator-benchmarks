# segsum_dadt_bwd

TG05 · bwd · vv（按是否包含 matmul 分类）

来源：[bwd_segsum_dadt_kernel](https://gitcode.com/weizhan4/TritonGraphOptimizationTests/tree/2e67d5b6723b1d472dcef91adb8a83c5b6761765/bwd_segsum_dadt_kernel)，固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。

**测试前必须关闭图优化：所有执行版 Triton kernel launch 显式传入 `enable_graph_optimize=False`。** 原始源码位于 `upstream/`，不可直接当作关闭图优化的执行版；本目录同名 `.py` 是只增加该选项的版本。

FP32 dSSdA/SSdA[16,32,128,16,16]；S=2048，chunk_size=16，输出[16,32,2048]。原 profiler 循环含 PyTorch 参考与缩放，不能整体当成 Triton 时间。

入口：[test_bwd_segsum_dadt_kernel_autotune.py](test_bwd_segsum_dadt_kernel_autotune.py)；文件 hash、kernel 名和依赖见 [operator.json](operator.json)。未运行 NPU 测试、未记录性能基线、尚未新增手改 tile 候选。

[测试说明与关闭图优化方法](../../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [本批总览](../README.md) · [来源许可记录](../SOURCE_NOTICE.md)
