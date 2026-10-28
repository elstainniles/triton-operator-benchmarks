# TG03 · matmul_fp8_row_fwd

默认性能入口自带输入和 CPU 参考，无需外部 `.pt`。输入规格和固定 seed 见 [cases.json](cases.json)，准备与绑定见 [benchmark.py](benchmark.py)。共 3 个配置，其中 1 个 smoke 配置。

```bash
python -I tools/run.py run --operator TG03 --suite smoke --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/tg03-001
```

从仓库根目录、已配置的隔离 Ascend runtime 执行；`--suite all` 运行全部登记配置。本批 **必须关闭图优化**，新入口显式传入 `enable_graph_optimize=False` 并核对编译后 metadata。

新基线只加载 `upstream/` 中的 JIT 函数，函数体及签名不变；省略 autotune 装饰器，使用 `cases.json` 中固定 tile/launch 参数。默认不执行上游脚本的分配、profiler 或文件清理逻辑。旧脚本和图优化关闭版保留供来源核查；原数据回放另用 `tools/replay_graph_tests.py`。

输入是新增合成用例，不是缺失 `.pt` 的复现。数值通过且设备独占检查通过后才记录直接 Triton kernel 耗时；失败保留状态、耗时为空。尚未添加手改 tile 候选。

[统一测试说明](../../../docs/GRAPH_OPTIMIZATION_TESTS.md) · [来源与许可](../SOURCE_NOTICE.md) · [验证范围](../../../docs/VALIDATION.md)
