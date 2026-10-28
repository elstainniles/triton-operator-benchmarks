# fused_moe_fwd

TG07 · fwd · cv（按是否包含 matmul 分类）

来源：[fused_moe](https://gitcode.com/weizhan4/TritonGraphOptimizationTests/tree/2e67d5b6723b1d472dcef91adb8a83c5b6761765/fused_moe)，固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。

**测试前必须关闭图优化：所有执行版 Triton kernel launch 显式传入 `enable_graph_optimize=False`。** 原始源码位于 `upstream/`，不可直接当作关闭图优化的执行版；本目录同名 `.py` 是只增加该选项的版本。

25 组 (M,N,K,E,top_k) ×4 量化模式=100 个上游配置。FP16/BF16/INT8/FP8 混合；PASS 只表示运行与保存输出成功，缺少 compare_results.py，不能记作数值通过。

入口：[test_fused_moe_kernel.py](test_fused_moe_kernel.py)；文件 hash、kernel 名和依赖见 [operator.json](operator.json)。未运行 NPU 测试、未记录性能基线、尚未新增手改 tile 候选。

[测试说明与关闭图优化方法](../../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [本批总览](../README.md) · [来源许可记录](../SOURCE_NOTICE.md)
