# FLA Triton 抽取批次

该批次保存原始 FLA 仓库中已经抽取、有静态 launch 引用的 Triton 定义，便于逐项审阅和补充可复现测试。来源为 [TritonAscendTest/flash-linear-attention](https://gitcode.com/TritonAscendTest/flash-linear-attention)，固定提交 `5752e83ce7e469cfc0a3145bf20d3c902b716d48`。MIT 许可证及上游版权说明随源码保留。

## 范围与目录

296 个条目按“来源文件 + kernel 符号”区分：122 个来自 `fla/`，174 个来自原仓库 `tests/`。原 Ascend 测试中存在与库实现不同的函数体，元数据保留这个区别。辅助 JIT 函数仅作为依赖保存，不另外算作 benchmark。重复函数体通过清单关联；目录数不代表独立算法数。

每项包含 `operator.json`、原始 `upstream/` 快照、提取的 `kernels.py`、`cases.json` 和 README。公共部分只提供清单、检查、运行和计时；每项的输入、参考数学及算子实现留在自己的目录。未公开的 `.pt`、本地环境、账号配置和本轮机器日志未导入。

## 可执行范围

`source_archive` 表示源码已归档，但输入、参考结果、launcher 或依赖仍待适配。它不是可执行成功，也不是编译器拒绝。`plan` 显示原因；`run` 保存 `not_executable`，不调用 Torch 或设备，也不提供耗时。

`generated_benchmark` 提供独立入口：固定种子生成输入、CPU 数学参考、完整输出及边界检查，然后仅计时目标 Triton kernel。当前生成式入口覆盖 sigmoid、logsigmoid、softmax 的前向与反向。采用原 autotune 候选中的固定配置；移除 autotune 装饰器的范围单独记录，函数签名、函数体和 JIT 装饰器保持不变。新增形状用于可复现基线验证，不冒充历史缺失 `.pt` 的同一份数据。

这里没有导入本地持续队列的计时或 block 加速比，也不以旧队列结果证明这些新入口通过。当前新入口为 **未设备验证**；正确性、编译拒绝、运行失败和设备干扰由新运行分别记录。

## 使用

离线检查不需要 Torch、Triton 或 NPU：

```bash
python tools/run.py plan --operator fla --suite smoke --variants baseline
python tools/verify_repository.py
python -m unittest discover -s tests -v
```

安装了 PyTorch 的环境还会执行 `tests/test_fla_generated.py` 中的 CPU 输入、独立参考、哨兵及 launch 参数检查；无需 Torch-NPU。未安装 PyTorch 时仅跳过这五项张量检查，源码 AST、元数据及手工候选为空的检查仍执行。CPU 扩展检查不代表 NPU 验证。

在已经配置的独立 Ascend runtime 中，按 [测试说明](TESTING.md) 生成环境清单，选择物理卡并运行；设备编号由调用者指定：

```bash
python -I tools/run.py run --operator fla --suite smoke --variants baseline \
  --physical-card <card> --environment environment.local.json --output runs/fla-001
```

也可使用 `catalog.json` 中的单个 FLA ID，避免遍历尚未适配条目。运行输出目录必须是新目录。编译失败保留原因，未生成有效样本时耗时为 null；不得用源码审核或 CPU 参考验证替代设备执行结果。
