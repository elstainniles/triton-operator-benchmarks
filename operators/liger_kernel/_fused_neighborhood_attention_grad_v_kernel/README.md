# _fused_neighborhood_attention_grad_v_kernel

LK11 · generic · bwd · CV

来源：[Liger-Kernel 781a8aef64bf](https://github.com/linkedin/Liger-Kernel/blob/781a8aef64bf6105bdefb5f4c991e0d14c08e793/src/liger_kernel/ops/fused_neighborhood_attention.py)，BSD-2-Clause。

`kernels.py` 保留目标函数、签名、装饰器和函数体；完整模块在 `upstream/`。
执行使用固定 Liger wheel 的原始 wrapper/辅助函数与本目录的目标 JIT，保留原始 grid/tile/autotune；没有新增优化候选。
测试前将公开入口绑定到所选源码，精确匹配目标源码/函数后才采纳计时。

正确性用例：`test/transformers/test_fused_neighborhood_attention.py::test_fused_neighborhood_attention_correctness[dtype0-0.005-0.005-False-2-32-128-4-7]`。输入自行生成，不需要外部 `.pt`。
本条目登记第二批实际采用的一个配置；`smoke` 也运行该配置，不意味着小尺寸。
实际张量和固定参数见 `cases.json`；前向/反向参考计算不计入目标 kernel 时间。

历史结果：4.169 µs。历史与新入口验证分别登记。

```bash
python -I tools/run.py run --operator LK11 --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/LK11-001
```

从仓库根执行；先按 [Liger 运行说明](../README.md) 配置隔离依赖。

## 大尺寸 CV 配置（2026-10-31）

`cases-cv-large.json` 登记 B=8、H=16、S=128、D=64 的独立 kernel 配置。使用完整 K tile，并保留原 kernel 函数；不走旧小尺寸 transformer wrapper。一键入口、第一版合并结果和口径见 [大尺寸说明](../../../docs/LIGER_CV_LARGE.md)。历史小尺寸配置与结果继续保留。
