# swiglu_fwd

P10 · swiglu · fwd · vv（无 matmul）

归档原 Triton kernel 函数体，去掉外部调优/dispatch 装饰器，固定 Ascend launch options。

输入：AB[M,2H]；O[M,H]；shape 参数表示 (M,H)。FP32 和 BF16 配置见 [cases.json](cases.json)。bwd 是独立反向 kernel，不包含 fwd 或 autograd 框架时间。

来源：[FlagGems `61f3372e817e`](https://github.com/flagos-ai/FlagGems/blob/61f3372e817e8bf4e9279df9ef21c0c453f718b8/src/flag_gems/fused/swiglu.py)，符号 `swiglu_kernel`；完整文件见 [upstream/swiglu.py](upstream/swiglu.py)。执行提取版 [kernels.py](kernels.py) 保留函数体。

基线 [baseline.py](baseline.py)：`M64H64`。手改版 [manual_tile.py](manual_tile.py)：M1H512, M4H512, M16H256, M4H2048。仅改变 tile 和随之联动的 grid，不改公式、精度或算法。所有候选共享本目录 [launch.py](launch.py) 与 [case.py](case.py)。

基线口径：source tile/row mode with controlled NPU launch options; broadcast B1024 is fixed control。不是上游 autotune 最优性能承诺。

历史数据：5 个 FP32 主配置、1 个 BF16 补充配置和 1 个 smoke；共 35 条候选记录。查看 [完整结果表](results/20261026/README.md)，包含慢于基线和编译失败的候选。

```bash
python tools/run.py plan --operator swiglu_fwd --suite fp32
python -I tools/run.py run --operator swiglu_fwd --suite fp32 --physical-card 6 \
  --environment environment.local.json --output runs/unique-run-name
```

命令从仓库根目录运行。先按 [测试说明](../../../docs/TESTING.md) 准备隔离 runtime 和显式工具链。新入口仅完成离线检查，尚未 NPU 重验；不可将历史结果当作新入口的设备验证结果。
