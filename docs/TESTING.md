# 测试说明

各批共用 `tools/run.py`，都能自行生成输入。图优化批次详见 [TritonGraphOptimizationTests 专项说明](GRAPH_OPTIMIZATION_TESTS.md)，**测之前必须关闭图优化，入口显式传入 `enable_graph_optimize=False`**。Liger 的固定依赖、原始 grid/tile 和独立计时约定见 [Liger 批跑说明](../operators/liger_kernel/README.md)。下文的具体形状与候选矩阵属于第一批 FlagGems。

本仓库第一批仅测试 FlagGems 的 8 个 Triton 方向/实现条目，全部为 vv（无 matmul）。数据准备、CPU 参考计算和输出检查使用 PyTorch，但**不会计入性能数据**。这里不测 PyTorch 算子、上游完整 Python wrapper 或 autograd 的端到端耗时。

`results/20261026/` 是已完成的历史实测。新入口已验证的配置范围另见 [VALIDATION.md](VALIDATION.md)，不能从历史结果推定未测配置已通过。

## 先查看计划

从仓库根目录执行，Python 3.11+ 即可，无需 Torch/Triton：

```bash
python tools/run.py list
python tools/run.py plan --operator P10 --suite fp32
python tools/verify_repository.py
python -m unittest discover -s tests -v
```

ID 或目录名都可选择算子，也支持 `all`、`flaggems`、`graph` 或逗号分隔 ID 列表。`--suite smoke|fp32|bf16|all`；`--variants baseline|manual|all`，第二批尚无 manual 候选。第一批 smoke 选择每算子的第一个登记配置，全部计划是 43 个 shape/dtype、184 个候选；第二批是 119 个生成配置，其中 10 个 smoke。TG 的 `plan` 会列出输入规格、固定 tile 和图优化关闭设置。

## Ascend 环境

使用已有且已审计的独立 runtime。不同编译器版本分别建环境；build/runtime 分开。runtime 仅安装从外部 build 环境得到的固定 wheel 和依赖，不进行 TA/TD 源码构建、editable install、`pip --user` 或源码 PYTHONPATH 覆盖。不要用 Conda base、全局 Python 或 `venv --system-site-packages`。这里不提供自动安装脚本，避免覆盖共享 Triton namespace。

历史环境指纹见 [historical-environment.json](historical-environment.json)；它是结果身份，不是自动创建环境的锁文件。硬件和驱动/CANN 不会因 Python 环境独立而自动隔离。

可以复制 [environment.example.json](environment.example.json) 并手动填写，或在已选择 CANN 的干净 shell 中自动采集当前隔离 runtime：

```bash
# 先按平台选择具体 CANN；清除 set_env.sh 加入的 Python 源码搜索路径。
source /absolute/path/to/cann/set_env.sh
unset PYTHONPATH
# PATH/LD_LIBRARY_PATH 需含该 runtime 及其配套 BishengIR/LLVM/CANN。
/absolute/path/to/isolated-runtime/bin/python -I tools/capture_environment.py \
  --output environment.local.json
```

采集工具只读取已安装的包，不安装、不修改环境，不初始化 NPU；会检查隔离路径、CANN linker 和实际 compiler。采集记录当前环境身份，不能代替对 wheel 来源/版本配套的独立核实。清单（被 Git 忽略）包含：

- runtime 绝对路径；`torch`、`torch_npu`、`triton` 的模块版本和 `triton-ascend` 包版本。
- 已验证 wheel 安装后的 `libtriton.so` SHA256、实际解析的 `bishengir-compile` 路径和 SHA256。
- 显式 CANN 根目录，以及审计后的完整 `PATH`、`LD_LIBRARY_PATH`。PATH 必须能解析 `npu-smi` 和该 CANN 的 `tools/bisheng_compiler/bin/ld.lld`。

在清洁 shell 中按平台安装方式选择该 CANN 与 runtime，使 `ASCEND_HOME_PATH` 等于 manifest 的 CANN 根目录。不要继承其它任务的 `PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD`、`TRITON_*`、`TORCHINDUCTOR_*` 或分布式 rank/HCCL 变量。入口会检查这些条件、Python/module 来源、版本、二进制 hash 和编译器实际解析路径；不匹配即停止。manifest 的占位符必须替换，不能把“能 import”当作设备测试通过。

每个运行创建全新的 cache/tmp/dump/override 目录。NPU 可见设备由 `--physical-card` 控制，在 Torch 导入前设置。运行前、用例边界、运行期间约每 5 秒和结束时读取设备健康/进程；查询失败、未知格式、其它占用都会阻止计时采纳。约 5 秒采样不是连续独占的数学证明，复现实验仍需要协调独占卡。

监控通过驱动 `drvDeviceGetBareTgid()` 获取当前进程的宿主机 PID，不把容器 PID 数字相同当作归属证明。缺少该接口会停止，不绕过核实。同一 Linux 用户的运行通过每卡文件锁串行；跨用户/容器依靠独占协调和占用检测，不能仅依赖这个用户级锁。工具不杀进程、不重置设备。

## 运行命令

选择已预约且健康空闲的物理卡，例如 6。运行必须使用 Linux 隔离 runtime 的 Python，带 `-I`：

```bash
/path/to/isolated-runtime/bin/python -I tools/run.py run \
  --operator swiglu_fwd --suite fp32 --variants all \
  --physical-card 6 --environment environment.local.json \
  --output runs/swiglu-fwd-fp32-run-001
```

首次确认可加 `--correctness-only`。输出目录不能覆盖，不自动重试失败。全部 smoke 用 `--operator all --suite smoke --variants baseline`，程序自动串行调度；批次保留逐算子结果和汇总 `performance.csv`。失败后可用逗号分隔 ID 继续尚未完成算子，不必重跑已完成结果。

输出包括 `result.json`、原始 profiler trace/CSV、`device.jsonl` 和当前运行的专属 cache。`result.json` 保存当前源码 hash、实际环境、每候选配置/grid、数值状态、原始计时及错误。`completed` 表示调度完所有所选候选，不等价于所有候选通过；`all_passed` 和逐项状态才表示实际通过范围。批次 `counts` 区分各状态。失败记录没有性能值。

状态含义：`correct`（仅数值通过）、`measured`、`compile_failed`、`accuracy_failed`、`runtime_failed`、`timing_rejected_interference`。编译/精度失败保留后继续其它候选；设备运行错误或哨兵损坏停止本次运行。计时归因失败也停止并保留 trace，不能改用 wrapper 总时间填充。

## 输入与正确性

固定形状来自原手工泛化扫描，不声称覆盖 FlagGems 的全部 shape/dtype。每个算子的 `cases.json` 是本次重测计划。

| 类别 | FP32 shape | BF16 shape | shape 的含义 |
| --- | --- | --- | --- |
| 连续/标量 Mul、Where、GELU | 127；8193；1048576；16777216 | 8193 | 一维 N |
| 广播 Mul | (1,2048)；(17,513)；(128,1024)；(2048,128)；(1024,8192) | (17,513) | 输出 (M,N)，输入分别 (M,1)/(1,N) |
| SwiGLU | 同上二维列表 | (17,513) | (M,H)，实际 AB 输入 (M,2H)，bwd 另有 dY(M,H) |

输入是 CPU normal(0,1)，先舍入到被测 dtype 再传 NPU。标量 Mul 的 scalar=0.75；Where 的条件是 `i % 3 == 0`。参考值在 CPU FP64 中按原公式计算，再按原输出 dtype 舍入。FP32 `rtol=2e-4, atol=2e-5`；BF16 两者均 `2e-2`。全量输出比较，输出前后各 16 个 sentinel 检查越界；每个候选测试前用 NaN 填充输出，防止前一候选的值掩盖漏写。

seed=20261026。历史框架每进程只播种一次，新入口每 case 重置 seed；输入分布和参数相同，随机张量**不是历史运行的逐字节重放**。同次 case 所有候选共享相同输入。历史只记录前后 guard 检查，新入口额外加入输出预填充以增强漏写检查。

## 性能口径

正确后 warmup 3 次，每个候选 3 轮×5 次；每轮随机化候选顺序。每个直接 Triton launch 单独打 CPU 标签，通过 `async_npu` flow 与 `kernel_details.csv` 的精确十进制时间戳关联，要求每个标签对应一个且仅一个预期 Triton kernel。缺标签、重复 flow、未知 kernel、多 kernel 或无法匹配都会拒绝计时。

计时值是 CSV 的设备 `Duration(us)`。先取每轮 5 次的中位数，再取 3 轮中位数的中位数。输入分配/拷贝、PyTorch 参考值、JIT 首次编译、host launch、显式 synchronize、正确性检查都在统计外；profiler 和同步仍可能影响执行条件，所以只能与同口径同环境数据比较。这不是设备 event 包住 wrapper 的总时长。

bwd 条目直接测指定反向 Triton kernel，没有先测 fwd 再减时间；此批不存在 fwd+bwd 合并计时。性能比较只在同 case、同设备、同环境、同次运行内进行。最佳值包含基线，不能把逐 case 最优当作固定策略的实际收益。

新结果如需提交，先审核并移除运行路径、主机身份、账号、内部地址等；保留可追溯 hash、公开环境版本和原始数值。`runs/` 默认不进入 Git。

## 可选 MSPTI 计时

FlagGems 和图优化批次可用 `--timing mspti` 替代默认 profiler，例如在上述运行命令中加入 `--timing mspti`。支持单算子及 `--operator flaggems`、`--operator graph`；Liger 和包含 Liger 的批次暂不支持。运行环境需提供 `mspti` Python 模块和兼容的 CANN/MSPTI；本入口禁止继承 `LD_PRELOAD`，因此需要依赖无需预加载的环境（例如 CANN 9.1 或更新版本），较旧环境不在支持范围。

正确性检查、设备监控、warmup 和三轮各五次的统计口径保持一致。每次调用前同步并单独启动 KernelMonitor，调用后同步并停止；仅接受恰好一条名称与已编译 kernel 一致、耗时为正的设备记录。MSPTI 时间戳以纳秒计，转换为微秒，原始样本和各轮中位数保存在 `result.json` 的 `profile` 字段。缺失、额外或名称不匹配的记录会停止计时；不会回退到其他计时方式。此方法与 profiler 方法的运行条件不同，结果应分别标记，不应直接混合比较。MSPTI 路径尚需在目标 NPU 环境验证。
