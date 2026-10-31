# Liger-Kernel：第二批通用优先基线

46 个精确 kernel 函数，固定 [Liger-Kernel 781a8aef64bf](https://github.com/linkedin/Liger-Kernel/commit/781a8aef64bf6105bdefb5f4c991e0d14c08e793)：**38 generic、7 Ascend-only、1 experimental；CV 7、VV 39**。同名选择顺序是 generic > Ascend > experimental，不会因通用版失败自动换成 Ascend 版。[完整名称和历史性能表](catalog-table.md)按用户第二批清单排序，ID 为 LK01–LK46。

每个目录包含 `kernels.py`、`test_case.py`、`cases.json`、`operator.json`、`upstream/` 和 `results/20261028/`。目标 JIT 函数、装饰器、算法、原始 grid/tile/autotune 均未修改。完整上游模块保留在 `upstream/`，执行仅抽取目标函数；wrapper、其它辅助 kernel 和参考模型来自固定 wheel。本目录不提供手改优化候选，也不测 PyTorch/ACLNN 的性能。

## 环境准备（一次）

运行平台是 Linux + Ascend NPU；“generic”指源码目录，并不意味着本测试工具支持 CUDA。需要 Python 3.11、配套 Torch/Torch-NPU/Triton-Ascend/CANN，以及 `npu-smi`、profiler 和驱动 host-PID 查询。使用已配置的隔离 runtime，按[公共环境说明](../../docs/TESTING.md)核验编译器和 CANN。不同编译器/候选使用不同 runtime，不能覆盖全局 Triton namespace。

额外需要固定提交的 Liger wheel 和 [测试依赖](requirements-test.txt)。若没有 wheel，在**单独 build 环境**构建纯 Python Liger 包，再将 wheel 安装到 runtime；测试脚本不会自动安装或下载：

```bash
# BUILD_PYTHON 与 RUNTIME_PYTHON 分别指向隔离 build/runtime 的 Python。
# build 环境预先安装 setuptools、wheel、setuptools-scm。
"$BUILD_PYTHON" -I -m pip wheel --no-deps --no-build-isolation \
  'https://github.com/linkedin/Liger-Kernel/archive/781a8aef64bf6105bdefb5f4c991e0d14c08e793.tar.gz' \
  --wheel-dir wheelhouse
"$RUNTIME_PYTHON" -I -m pip install --no-deps wheelhouse/liger_kernel-0.8.5-py3-none-any.whl
"$RUNTIME_PYTHON" -I -m pip install -r operators/liger_kernel/requirements-test.txt
```

测试入口逐文件核对固定提交的 285 个 Liger Python 源文件，不依赖版本号相同。**不要用普通 `pip install liger-kernel` 替代固定源码 wheel。** 源码归档 SHA256 为 `29ab242a600e724389e9360121bb475a67a1d2766ad408aa4d4e5b0605599cc8`。

历史已测试组合：Torch 2.10.0+cpu、Torch-NPU 2.10.0.post7、TA 3.6.0 main `b6dda4fa17bb00780199aa52f0549262cf2e1d1b`、CANN 9.1.0、Ascend950PR；二进制指纹见 [historical-environment.json](historical-environment.json)。Liger 的 NPU wheel metadata 仍声明 Torch/Torch-NPU 2.9、TA 3.2.2，因此这个历史组合的 `pip check` 有已知冲突；42 个目标的实测通过不代表整个依赖组合通过。上面的 `--no-deps` 用于保留自行核验的 TA/Torch 配套，不能据此忽略未核实的依赖问题。

在干净 shell 中显式选择 CANN、runtime 和工具链，再生成本机清单：

```bash
source /absolute/path/to/cann/set_env.sh
unset PYTHONPATH
# PATH / LD_LIBRARY_PATH 应包含本 runtime 及其配套 BishengIR/LLVM/CANN。
"$RUNTIME_PYTHON" -I tools/capture_environment.py --output environment.local.json
```

## 一键批跑

在仓库根目录执行；输入全部自行生成，无需上游 checkout 或 `.pt`：

```bash
python tools/run.py plan --operator liger --variants baseline
"$RUNTIME_PYTHON" -I tools/run.py run --operator liger --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/liger-second-001
```

单条使用 `--operator LK03` 或精确 kernel 名称；多条使用 `LK03,LK14,LK46`。支持 `--correctness-only` 和可选 `--timing mspti`。每个 kernel 独立进程和缓存，串行运行；同一用户不同 checkout 共用卡锁，另有设备健康/占用监控，不杀进程、不重置设备。

每个条目默认重放第二批实际采用的一个 shape/dtype。原计划优先较大候选，部分项目最终采用较小配置；**这些配置不是经搜索确认的最大安全尺寸**。`smoke` 是同一配置，不是小尺寸模式；有的配置包含数亿元素，需预留显存。`fp32`/`bf16` 按登记的主输入 dtype 筛选；`all` 包括 INT8/UINT8 的 experimental matmul。`historical-plan.json` 保存原大尺寸优先候选计划，仅供溯源；新入口不自动降级 shape，也不将不同形状比较成加速比。

编译失败、精度失败、跳过和依赖编译阻断会记录后继续下一条；运行/设备/计时归因异常会保存证据并停止。续跑时用逗号列表选择 `summary.json` 尚未完成的 ID，并指定新的输出目录；已完成数据不会覆盖或自动重测。

## 正确性、计时与结果

先跑固定的上游正确性用例或本批直接参考用例，setup/call/teardown 全部通过才计时。公开入口在测试进程内绑定到明确选择的源码，目标 JIT 替换为本目录 `kernels.py`；其它 kernel 只是准备/校验依赖。原有数值阈值保留，测试中参考实现使用 PyTorch，不计时。部分上游用例容差较宽，应结合 `test_case.py` 评估适用范围；本次没有加严或放宽。

记录实际 launch 输入、grid、constexpr，重放最终目标 JIT。通过编译缓存源码位置和 profiler 的 CPU→NPU flow，要求每个样本只命中一个同名目标 Triton kernel。**15 个设备 Duration(us) 的中位数**作为耗时；排除参考、Torch/厂商 kernel、host 开销、首次编译。未知归因、缺样本或外来占用不产生有效耗时。重放保持原始参数/缓冲区，包括原位 kernel，不将其当作第二次完整数值测试。bwd 直接计时，不通过 fwd+bwd 相减。

MSPTI 模式对每次重放独立采集，要求恰有一条同名 kernel 记录；取每轮五次的中位数，再取三轮中位数。保留与默认 profiler 相同的编译产物和占用检查，结果中记录采样方法和原始样本。

新结果在 `runs/<批次>/`：`performance.csv`、`summary.json`、每个 ID 的日志及 `result.json`、trace/CSV、源码/编译缓存哈希、占用证据。`completed` 表示调度完成，`all_passed` 才表示全部通过；失败的性能为空。`runs/` 被 Git 忽略，提交新结果前需去除本机路径和主机身份。

## 历史数据与已知失败

`results/20261028/` 是原框架第二批历史导入，46/46 已尝试、42 个有效性能、3 个编译失败、1 个依赖阻断。保存全部 42×15 个设备样本及来源哈希；原始 trace 含私有运行路径，未公开上传。新入口验证单独记录于[验证说明](../../docs/LIGER_VALIDATION.md)，不能把历史结果当作新脚本实测。

- Multi-block Softmax fwd/bwd：`tl.float32(...)` 引起 `'dtype' object is not callable`，两条分别尝试。
- Sparsemax fwd：BiSheng `MaxThreads out of range`。
- Sparsemax bwd：正确性用例先调用 fwd，被上述编译失败阻断；未独立编译或执行 bwd。

许可为 [BSD-2-Clause](LICENSE)，保留 [上游 NOTICE](UPSTREAM_NOTICE) 和 `licenses/`。通用源码中可能有上游 NPU 分支；本次“通用”不表示删除那些分支。
