# 验证记录

## 自带输入入口：设备 smoke 验证（2026-10-28）

当前 **15/15 个算子条目、18/18 个代表性 smoke 配置数值通过并取得有效计时**：FlagGems 8 个，TritonGraphOptimizationTests 7 个（其中 MoE 四种模式）。这是新入口的固定基线验证，不是新一轮手改 tile/grid 调优；未比较优化收益。

每个条目可用 `tools/run.py run --operator <ID>` 独立运行，整仓用 `--operator all --suite smoke --variants baseline`。所有默认输入均自行生成，无需外部 `.pt`。第二批 119 个配置已完成 CPU 输入、路由完整性和参数签名绑定检查，设备仅验证其中 10 个 smoke；较大 shape/dtype、FlagGems 新入口的 manual 候选尚未在本轮全面重测。

环境：Ascend950PR_9579、物理卡 6，TA main `b6dda4fa17bb00780199aa52f0549262cf2e1d1b` / `3.6.0+main.b6dda4fa`，Triton 3.6.0、Torch 2.10.0+cpu、Torch-NPU 2.10.0.post7、CANN 9.1.0。复用隔离 runtime，未重新构建/安装。每个运行核验导入路径、libtriton 和编译器哈希，使用独立缓存；采纳的运行设备监控均通过。

数值通过后 warmup 3 次，采集 3 轮×5 次。表中数值为每轮中位数再取中位数，单位 µs，**仅来自直接 Triton kernel 的设备 Duration**；不含 PyTorch 准备/参考、拷贝、重置输出、JIT、host launch 和同步耗时。完整配置、grid、数值误差、15 次原始样本、环境和证据哈希见 [结构化实测数据](standalone-smoke-20261028.json)。

| ID | 算子 | smoke shape / dtype（简写） | kernel µs |
| --- | --- | --- | ---: |
| P04 | mul_contiguous_fwd | N=127 / FP32 | 1.197 |
| P05 | mul_scalar_fwd | N=127 / FP32 | 1.074 |
| P06 | mul_broadcast_fwd | M,H=1,2048 / FP32 | 2.890 |
| P07 | where_fwd | N=127 / FP32 | 1.255 |
| P08 | gelu_tanh_fwd | N=127 / FP32 | 2.129 |
| P09 | gelu_tanh_bwd | N=127 / FP32 | 2.991 |
| P10 | swiglu_fwd | M,H=1,2048 / FP32 | 2.509 |
| P11 | swiglu_bwd | M,H=1,2048 / FP32 | 2.828 |
| TG01 | gather_scale_fwd | tokens,D,rows,experts=16,128,24,4 / FP32 | 2.058 |
| TG02 | gather_scale_fp8_quant_fwd | 16,128,24,4 / FP32→E4M3 | 2.975 |
| TG03 | matmul_fp8_row_fwd | M,N,K=32,64,64 / E4M3→FP32 | 6.831 |
| TG04 | dacs_segsum_fwd | B,H,S=1,2,32 / FP32 | 2.377 |
| TG05 | segsum_dadt_bwd | B,H,S=1,2,32 / FP32 | 2.590 |
| TG06 | dadt_cumsum_fused_bwd | B,H,S=1,2,32 / FP32 | 2.314 |
| TG07 | fused_moe_fwd, INT8 w8a16 | M,N,K,E,top_k=4,64,64,4,2 / FP16 activations, INT8 weights | 9.800 |
| TG07 | fused_moe_fwd, block INT8 w8a8 | 4,64,64,4,2 / INT8 A/B→FP16 | 9.927 |
| TG07 | fused_moe_fwd, tensor FP8 w8a8 | 4,64,64,4,2 / E4M3 A/B→FP16 | 10.808 |
| TG07 | fused_moe_fwd, no quant | 4,64,64,4,2 / FP16 | 11.213 |

输入简写不代表每个张量的 shape/dtype；例如索引为 INT32、scale 单独记录。各算子 `benchmark.py`/`case.py` 定义完整输入契约。

必须保留的限制和失败证据：

- TG01–TG07 全部显式 `enable_graph_optimize=False`，并核对编译后的 metadata；未修改原始 JIT 函数体。
- TG04、TG06 的默认 SIMD 配置编译失败，报 `MaxThreads out of range`。TG05 默认 SIMD 首次编译超过 665 秒后中止。三者当前登记配置显式使用 `force_simt_only=True`，最终 smoke 通过；此项是编译模式选择，不表示默认 SIMD 已修复或获得调优收益。
- TG02 首次计时解析因编译器截断运行时 kernel 名称失败。根据 `backend.pack_metadata()` 的实际名称重新解析同一份原始 trace，恢复 15 个精确对应样本，数值及设备监控均通过；没有重复执行来挑选结果。
- 设备冲突阻止启动的尝试、默认 SIMD 编译失败/超时和原始 profiler/监控均保留。导出的成功汇总不抹除这些失败；`completed` 仅表示调度结束，须同时查看 `all_passed` 和逐项状态。
- 自生成输入不等同于缺失 `.pt` 的逐字节重放。当前支持在已配套的 Ascend Linux runtime 直接运行，未验证其它卡型或编译器版本。

本地完整性检查及 20 项 host 单元测试通过，包括批次编译失败不能误报为通过、设备错误必须停止后续调度、输出重置 kernel 不混入直接 launch 计时。上游快照及图优化关闭副本与导入时一致；FlagGems 51 组/218 条历史数据保持不变。

## 首次整理校验（2026-10-28，历史）

本地 Windows、Python 3.12.14 完成以下离线验证，不导入 Torch/Triton、不使用 NPU：

- `python tools/verify_repository.py`：8 个条目，51 组比较、218 条记录（189 passed / 29 failed）；核对源码 SHA256、提取函数 AST、baseline/候选配置、全部 grid、case 覆盖、样本数、轮中位数、组内比较和加速比、环境指纹、Markdown 链接及 Python 语法。
- `python -m unittest discover -s tests -v`：11 项通过；覆盖精确 trace 时间戳匹配、漏测/重复/非预期 kernel 拒绝、多个 kernel 拒绝、PID namespace 数值碰撞、未知设备占用格式，以及 8 个实际 launch 的尾部 shape 与源码参数绑定。
- `list` 和 `plan` 可离线读取目录和候选计划。

归档的 4 个源文件及许可证已通过 GitHub API 与上游固定提交的 Git blob SHA 对照一致：

| 上游路径 | Git blob SHA |
| --- | --- |
| src/flag_gems/ops/mul.py | 9133dd1bf8a69ef9693d0872a83dc306fea8266b |
| src/flag_gems/ops/where.py | f997ae0eb7133d34c7ddd900ae3edda6b70f42e0 |
| src/flag_gems/ops/gelu.py | 5dff0a4c428594c0cf700aaa13090444e14e3321 |
| src/flag_gems/fused/swiglu.py | 9e1193915c34bc311283f6efcc36b591eb30d3a8 |
| LICENSE | f433b1a53f5b830a205fd2df78e2b34974656c7b |

GitHub Actions 配置在 Ubuntu/Python 3.11 上运行同样的 host 校验；不运行设备测试。新 NPU 入口仍需在隔离环境和空闲设备上验证。离线成功不等于 NPU 编译、执行或性能验证通过。

## 第二批导入校验（2026-10-28，历史）

TritonGraphOptimizationTests 固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`：7 组实际测试、11 个源文件，9 处 kernel launch 显式设置 `enable_graph_optimize=False`。原始 Git blob/SHA256 与快照一致；执行版去掉新增的关闭开关后，整文件 AST 必须与原始源码一致。3 项只有外部运行说明的引用单独记录，不作为已导入实现。

统一目录登记为 15 个条目（8 FlagGems + 7 新批次）。FlagGems 的 51 组/218 条历史结果仍单独核查。16 项 host 测试覆盖此前的检查以及图优化误开启拒绝、输入缺失、MoE case ID 边界、导入批次不能误用 FlagGems 手改计划。NPU 执行数为 0，没有新增性能数据。
