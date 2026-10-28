# where_fwd

P07 · where · fwd · vv（无 matmul）

原计算函数的连续点算适配探针，**不是** FlagGems pointwise_dynamic 生成 kernel。

输入：X[N], Y[N], C[N] bool；O[N]；C[i]=(i%3==0)。FP32 和 BF16 配置见 [cases.json](cases.json)。bwd 是独立反向 kernel，不包含 fwd 或 autograd 框架时间。

来源：[FlagGems `61f3372e817e`](https://github.com/flagos-ai/FlagGems/blob/61f3372e817e8bf4e9279df9ef21c0c453f718b8/src/flag_gems/ops/where.py)，符号 `where_inner`；完整文件见 [upstream/where.py](upstream/where.py)。执行提取版 [kernels.py](kernels.py) 保留函数体。

基线 [baseline.py](baseline.py)：`B1024`。手改版 [manual_tile.py](manual_tile.py)：B256, B4096, B16384。仅改变 tile 和随之联动的 grid，不改公式、精度或算法。所有候选共享本目录 [launch.py](launch.py) 与 [case.py](case.py)。

基线口径：explicit B=1024 adapter control, not original wrapper/autotune baseline。不是上游 autotune 最优性能承诺。

历史数据：4 个 FP32 主配置、1 个 BF16 补充配置和 1 个 smoke；共 24 条候选记录。查看 [完整结果表](results/20261026/README.md)，包含慢于基线和编译失败的候选。

```bash
python tools/run.py plan --operator where_fwd --suite fp32
python -I tools/run.py run --operator where_fwd --suite fp32 --physical-card 6 \
  --environment environment.local.json --output runs/unique-run-name
```

命令从仓库根目录运行。先按 [测试说明](../../../docs/TESTING.md) 准备隔离 runtime 和显式工具链。新入口仅完成离线检查，尚未 NPU 重验；不可将历史结果当作新入口的设备验证结果。
