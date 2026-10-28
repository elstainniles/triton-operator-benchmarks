# TritonGraphOptimizationTests：自带输入的性能测试

本批 7 组算子全部提供输入生成、CPU 数值参考和直接 Triton kernel 计时入口，**默认不需要 `.pt`、GPU 输出文件或上游 pytest fixture**。代码在 [operators/triton_graph_optimization](../operators/triton_graph_optimization/)，与 FlagGems 分开。

固定来源提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`。新用例有固定 shape/dtype、seed、tile/grid，不宣称恢复上游缺失 `.pt` 的数据分布。NPU 实测范围见 [验证记录](VALIDATION.md)，未测配置不能视为已验证。

## 运行

前提为 Linux、隔离 Ascend runtime 中已经安装配套 Torch/Torch-NPU/TA wheel，并显式选择 CANN。环境配置见 [通用测试说明](TESTING.md)。默认入口不需要 pytest、einops、FLA、FlagGems 包或 compare_results.py。

```bash
python tools/run.py plan --operator graph --suite smoke --variants baseline

# 7 组共 10 个小配置；MoE 四种模式各一个。
python -I tools/run.py run --operator graph --suite smoke --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/graph-smoke-001

# 单个算子的全部登记配置。
python -I tools/run.py run --operator TG01 --suite all --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/tg01-001

# 支持逗号分隔选择；--operator all 为全部 15 个条目，flaggems 为首批 8 个。
python -I tools/run.py run --operator TG03,TG07 --suite smoke --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/matmul-moe-001
```

使用已预约且健康空闲的物理卡；工具只监控占用、不清理进程。输出目录必须全新。`tools/run_graph_tests.py` 是同一入口的别名，省略 `run` 即可。单算子可用 `--case-id <cases.json中的ID>` 精确选择，可重复指定；`--correctness-only` 只检查数值。

整批串行运行，各算子独立子进程、输出目录和编译缓存。输出 `summary.json`、`performance.csv`、每算子日志及 `result.json`、原始 profiler 和设备监控。编译失败/精度失败单独记录、不填耗时；设备错误、越界、归因错误或占用冲突会停止批次，先检查证据，再选择未完成的算子续跑。

## 输入和覆盖

| ID | 内容 | 配置数 | 输入要点 |
| --- | --- | --- | --- |
| TG01 | gather + scale fwd | 3 | FP32/FP16/BF16；合法 token/expert 索引；有效行数边界 |
| TG02 | gather + scale + FP8 row quant fwd | 3 | FP32/FP16/BF16 输入，E4M3 输出与 FP32 scale；含 clamp |
| TG03 | FP8 row matmul fwd | 3 | E4M3 A/B、行 scale、FP32 输出；含尾块及 bias |
| TG04 | dacs_segsum fwd | 2 | FP32，小配置及上游 (16,32,2048)、chunk=16 |
| TG05 | segsum_dadt bwd | 2 | FP32，dSSdA/SSdA 为 (B,H,S/16,16,16) |
| TG06 | dadt_cumsum_fused bwd | 2 | 4 个 FP32 (B,H,S) 输入 |
| TG07 | fused_moe fwd | 104 | 4 个 smoke + 上游 25 组尺寸×4 模式的重新生成用例 |

合计 119 个生成配置，smoke 为 10 个。shape 简写：TG01/02 为 `[输入token数,D,输出行数,专家数]`；TG03 为 `[M,N,K]`；TG04–06 为 `[B,H,S]`；TG07 为 `[M,N,K,E,top_k]`。实际所有输入的 shape、dtype、stride、SHA256 写入结果；简写 dtype 不代表所有输入张量的 dtype。

MoE 包含 INT8 w8a16、block INT8 w8a8、tensor FP8 w8a8 和不量化。新输入准备修正上游脚本的契约问题：A 仅含原始 M 个 token 行，kernel 把路由位置除以 top_k；`num_tokens_post_padded` 为实际 padding 后长度，expert IDs 与 BLOCK_SIZE_M=32 对齐，每个有效路由恰好出现一次。CPU 参考按 expert 独立计算 matmul、量化 scale 和路由权重，并处理 sorted 输出。

上游 3 项仅有链接的 [外部测试](../operators/triton_graph_optimization/external_references/) 不属于已收录的 15 个可执行条目。

## 编译选项和计算范围

**必须关闭图优化：`enable_graph_optimize=False`。** 每次直接调用均传入，先核对 backend 能解析，再核对编译后 metadata 为 false。缓存按运行隔离。该开关不是 torch.compile、设备 graph replay、`force_simt_only` 或 `TRITON_ALL_BLOCKS_PARALLEL`。

原始快照及原图优化关闭脚本不变。新入口只加载哈希审计过的 JIT 函数和 helper，保留函数体/签名，不执行上游顶层文件清理、环境设置或 wrapper。默认固定 tile，省略 autotune 装饰器，TG01/TG03 的无效 `BLOCK_D` key 因此不再阻碍启动。num_warps=4、num_stages=2；TG04–TG06 显式使用 `force_simt_only=True`（当前 TA 将其规范化为 `compile_mode=simt_only`）。原默认 SIMD 路径中，TG04/TG06 报 `MaxThreads out of range`，TG05 的小用例首次编译超过 10 分钟。原失败/超时和显式 SIMT 的结果分别保留。这是新的固定基线，不是上游 autotune 最优结果，不能混称为默认 SIMD 性能。全部参数在配置和结果中记录。

TG05 会累加旧输出，每次正确性/预热/计时调用前在计时标签外清零。TG05/TG06 的参考对应 kernel 本身，不包含 wrapper 的 `-log2(e)` 缩放。TG04 的 segsum load 没有尾部 mask，只接受 S 能被 chunk 整除的输入。

## 正确性与性能口径

全量输出和首尾 guard 通过后才计时。CPU 参考使用实际传入的已舍入/量化输入。FP8 quant 输出允许一个 FP8 量化步长的舍入差异，scale 独立校验；容差在 `benchmark.py` 及结果中固定记录。

预热 3 次，测 3 轮×5 次。CPU 标签只包住一次直接 Triton launch，按 async_npu flow 与精确十进制时间戳匹配 `kernel_details.csv` 的设备 `Duration(us)`；要求一对一关联和精确名称匹配。运行时名取自 backend 的 `pack_metadata`，兼容长名称截断，不做模糊后缀匹配。先取每轮中位数，再取三轮中位数。

分配/拷贝、CPU/PyTorch 参考、输出清零、额外缩放、host launch、JIT 编译及同步均不计入 `median_us`。标签外准备 kernel 单独记录并排除；标签内额外 kernel 或无法归因的行均拒绝。受干扰计时保留但不采纳。

## 旧脚本回放

仅在复现原 `.pt` 用例时使用 `tools/replay_graph_tests.py --input-pt <可信原始文件>`；TG01–03 的原文件仍未由上游提供。旧入口保存原 profiler，不自动生成有效性能表。默认生成入口不读取 pickle；旧文件缺失不再阻塞性能测试，也不宣称复现原始 `.pt` 用例。
