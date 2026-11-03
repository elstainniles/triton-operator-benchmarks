# FLA VV 按表格顺序批跑

入口是 [`tools/run_fla_vv.py`](../tools/run_fla_vv.py)。清单按
`FLA已确认测试结果_最新.xlsx` 的 **`11.1全量结果`** 工作表冻结：
317 条 VV 规格，原 Excel 第 **203–519 行**，对应 VV 序号 **1–317**。
这是规格数量，不是独立算法数或已通过测试数。VV 标签沿用原表，未重新按是否有 `dot` 分类。

源工作簿 SHA256 为 `ce1ced530438ff63251e2ca7fb5b1d0048502f88bda7c0a400df11f602611280`。
查看[按表格顺序的清单](../operators/flash_linear_attention/vv-catalog.md)或
[`vv-manifest.json`](../operators/flash_linear_attention/vv-manifest.json)。
执行时不需要 Excel、外部 `.pt` 或原作者的测试目录。

## 目录与映射

与 CV 的 `cv-manifest.json` / `cases-cv.json` 组织方式一致：

- `operators/flash_linear_attention/vv-manifest.json`：稳定行序和算子目录索引。
- 各现有算子目录的 `cases-vv.json`：shape/dtype、表格配置、精确测试 ID、源文件哈希及验收方式。
- 各算子已有的 `tests/<完整ID>/test.json`、`source/`、`generated/fixture.py`：原测试、源码闭包、自生成输入与参考。
- `opbench/fla_vv.py` / `fla_vv_runner.py`：选择、核验、串行调度与汇总。

130 个目录、127 个 kernel 名称参与此批。`additional_tests/` 中的已有入口也纳入统一索引。
同名 kernel 不按名字合并；不同来源、shape、tile、`save_kl` 和尾块规格各自保留。
sigmoid/logsigmoid 原测试的 `CONFIGS` 循环按该表行的 `original_config_N` 限定为一个原配置，
并在执行前核对 shape、dtype、temperature；不会把其它表行的 shape 顺带执行。
原 CV 用例、kernel 函数体和历史测量不变；VV 复用导入的测试源码，避免再复制一套待维护的实现。

原表第 488 行（VV 序号 286）的 `num_warps` 未记录，现有测试有 4/8 两个匹配配置，
因此这一行展开成两个独立记录：整批共有 **318 个执行配置**。
不选最快值代替缺失的历史配置。其它行若表中参数为 null，实际固定值在 `configurations` 中显式列出。

## 先查看要跑的区间

在仓库根目录运行，以下命令无需 Torch/Triton 或加速卡：

```bash
python tools/run_fla_vv.py verify
python tools/run_fla_vv.py plan --cases 1-20
python tools/run_fla_vv.py plan --rows 203-222
python tools/run_fla_vv.py plan --cases 1-20,31-40
python tools/run_fla_vv.py plan --operator FLA077
```

`--cases` 使用 **VV 序号**，`--rows` 使用 **Excel 实际行号**，只能选一个。
范围均包含首尾；例如 `1-20` 是 20 条规格，不是前 20 个去重 kernel。
多段选择自动去掉重复序号，始终按表格原顺序运行。省略选择参数表示全部 317 条。
可用 `--operator` 进一步筛选目录 ID，序号不会因筛选而重新编号。

## 批跑基线或 Block 对照

使用 Linux、现有隔离 Triton-Ascend runtime 和显式 CANN 环境。
按 [环境说明](TESTING.md#ascend-环境) 准备 `environment.local.json`；
本入口不会构建或安装任何包。原测试还需要同一 runtime 中的 `pytest`、`numpy`、`einops`，
版本说明见 [FLA 原用例复测](FLA_IMPORTED_TESTS.md)。

测试前 20 条的基线：

```bash
"$RUNTIME_PYTHON" -I tools/run_fla_vv.py run --cases 1-20 \
  --physical-card 6 --environment environment.local.json \
  --task-lock /path/to/validation-task.lock --card-lock /path/to/npu-6.lock \
  --output runs/fla-vv-001-020
```

下一批换 `--cases 21-40` 和新的输出目录即可。测试全部用 `--cases all`。
示例中的锁路径应替换为现场已使用的锁；不要为不同批次分别新建同名锁目录。
没有其它调度器时可省略这两个参数，入口仍会持有仓库共用的每用户卡锁。

支持 block batching 的编译器可做基线与候选对照：

```bash
"$RUNTIME_PYTHON" -I tools/run_fla_vv.py run --cases 1-20 --factor 2 --axis 0 \
  --physical-card 6 --environment environment.local.json \
  --task-lock /path/to/validation-task.lock --card-lock /path/to/npu-6.lock \
  --output runs/fla-vv-factor2-001-020
```

默认 `--factor 1` 只跑基线；`--factor >=2` 按每个配置先基线、后候选。
默认合并轴为 0，可显式指定 `--axis 0|1|2`；原表 axis 只是历史记录，不自动混用。
基线和候选使用同一精确测试配置、固定种子、输入生成器和数值阈值，
目标 launch 的 grid 保持原脚本定义，tile/编译参数由清单固定，关闭旧包装器遗留的 Block 选项。
基线编译、精度、运行或计时资格失败时，跳过该候选并继续下一条。
候选的编译拒绝、控制流等拒绝原因、运行时回退和无合并机会分别保留，不能算作有效合并性能。

每个基线/候选在新进程、独立编译缓存中执行。源码按哈希复制到当次输出目录再运行，
原脚本按 `__file__` 写出的 profiler 文件不会污染仓库。控制器与 worker 共同持有继承的锁，
避免控制器意外退出后第二个批次与遗留 worker 并行。设备健康、实际占用和环境包/二进制身份
继续由仓库现有检查把关；此入口不清理进程、不重置设备。

## 计时、结果与续跑

默认每个版本 **2 轮，每轮预热 5 次、采样 15 次**，汇总为两轮中位数的平均值，单位 µs。
可用 `--rounds`、`--samples`、`--warmup` 调整；只做正确性加 `--correctness-only`。
目标 kernel 通过编译对象名称匹配 profiler 的 `kernel_details.csv`，每个样本必须恰好一个目标。
只记录该 Triton kernel 的设备时间，输入生成、参考计算、编译、张量恢复及其它 kernel 不计入。
原测试内打印的耗时和历史 Excel 耗时不参与新比较；不会用总耗时相减构造 kernel 时间。
这是串行基线/候选比较，不是交错多次复核后的确定收益结论。

输出内容：

- `performance.csv`：按原表顺序的 kernel、shape/dtype、配置、基线/候选耗时、加速比、状态与证据路径，可由 Excel 打开。
- `summary.json`：选择范围、源码/执行器/环境指纹、完成状态、每次执行结果和分类计数。
- `vv-row-NNN/<测试ID>/baseline|block/attempt-NNN/`：`result.json`、执行副本、独立缓存、占用日志和 profiler 原始样本。
- 同级 `attempt-NNN.log`：每个独立进程的 stdout/stderr。

`completed=true` 表示所选配置处理完，不等于所有配置编译或精度通过；请同时看 `all_passed` 和各行状态。
41 个现有配置的验收标注为 diagnostic/profiling/requires-review：它们只执行原检查，
不会输出合格性能。其余 277 个带数值断言的配置也必须实际通过当次检查才能计时。
这里沿用各原测试验收与阈值，不声称新增了所有输出的独立数学参考或哨兵检查。
一次测试捕获多个不同目标配置、缺少数值断言、受干扰或回退，都不能获得有效加速比。

继续同一批次，在原命令后添加 `--resume`：

```bash
"$RUNTIME_PYTHON" -I tools/run_fla_vv.py run --cases 1-20 \
  --physical-card 6 --environment environment.local.json \
  --task-lock /path/to/validation-task.lock --card-lock /path/to/npu-6.lock \
  --output runs/fla-vv-001-020 --resume
```

必须保持区间、卡、环境文件、源码、factor、axis、采样与锁参数一致。
成功和已有终态的编译/精度/运行失败均不重复执行；刻意复测失败配置应另选范围和新输出目录。
设备忙时保存证据并暂停，待空闲后 `--resume` 会创建新 attempt；不会把设备忙记为精度失败。
Ctrl+C 会停止本次创建的 worker 进程组，保存记录后退出，可续跑。
每个 worker 默认超时 1800 秒，可用 `--worker-timeout` 调整；超时保留失败记录后处理下一条。
如果机器/控制器异常退出，已有完整 worker 结果会自动收回；没有终态记录的孤立 attempt
要求先核实进程和证据，不能因一次 resume 重复启动未知状态的测试。

批次处理完返回 0（可能包含用例失败），等待设备/环境检查时返回 2，Ctrl+C 返回 130。
执行器没有产生完整结果等异常会中止调度并保留 `review_required` 状态。

## 本次整理的验证范围

本次只做清单、源码完整性、区间选择、失败/续跑状态机和原接口回归检查。
Linux 隔离 Python 下的 `tools/verify_repository.py` 和 131 项 host 单元测试通过，
包含真实子进程锁继承、超时清理和 CPU 张量参考检查；没有跳过项。
**没有在 NPU 上重跑 318 个配置**，新入口的设备可执行性和性能以以后实际生成的结果为准。
原表中的错误和原 wrapper 的 grid/tile 不匹配不会在整理时被静默修正；运行时应如实失败，
不能以修改算法、输入或放宽验收换取“通过”。
