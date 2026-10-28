# 上游仅提供说明的 3 个外部测试

原始说明归档为 [upstream_run.md](upstream_run.md)。以下代码不在本次固定的 TritonGraphOptimizationTests 提交中，因此不登记为已导入、可执行或已测量的算子：

| 名称 | 外部仓库 | 上游说明中的入口 |
| --- | --- | --- |
| chunk_scaled_dot_kkt_fwd_kernel | [Q2TritonKernel](https://gitcode.com/TritonAscendTest/Q2TritonKernel) | tests/fla/test_chunk_scaled_dot_kkt_fwd_perf.py |
| chunk_local_cumsum | [Q2TritonKernel](https://gitcode.com/TritonAscendTest/Q2TritonKernel) | tests/fla/test_chunk_local_cumsum_perf.py |
| matmul_kernel（int8/int2） | [LigerKernel](https://gitcode.com/TritonAscendTest/LigerKernel) | benchmark/scripts/benchmark_mm_int8int2.py |

目前没有固定这些外部仓库的 revision、实现或依赖，不把原说明中的目录名推测为已验证分支。后续导入应另行固定提交、检查依赖和来源，并同样在每个被测 kernel launch 关闭图优化。

原说明包含清理 cache 和 PYTHONPATH 的历史命令；这里只保留溯源，不应直接复制到共享环境执行。使用独立 runtime、全新 cache 和本项目的环境检查要求，不能覆盖其它任务的数据或环境。
