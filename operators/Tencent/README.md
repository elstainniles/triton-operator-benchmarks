# Tencent：CE / RMSNorm / SiTU-GLU v2

用户提供的三个算子族，按 **fwd / bwd 拆成六个独立入口**。全部为 VV，无 matmul。
`Tencent` 是用户指定的目录名，不表示已核实腾讯官方仓库来源；来源及许可状态见 [SOURCE_NOTICE.md](SOURCE_NOTICE.md)。

| ID | 目录 | 精确 kernel | 输入 shape / dtype | 原始 grid / tile |
| --- | --- | --- | --- | --- |
| TX01 | [ce_fwd](ce_fwd/) | `_cross_entropy_fwd_kernel` | logits `[4096,163840]` FP32；labels `[4096]` INT64 | `(4096,80)` / BLOCK_SIZE=2048 |
| TX02 | [ce_bwd](ce_bwd/) | `_cross_entropy_bwd_kernel` | logits/dlogits `[4096,163840]` FP32；labels INT64 | `(4096,80)` / BLOCK_SIZE=2048 |
| TX03 | [rmsnorm_fwd](rmsnorm_fwd/) | `_rmsnorm_fwd_fused` | x/y `[2048,512]` BF16；weight/rrms FP32 | `(2048,)` / D_ROUNDED=512 |
| TX04 | [rmsnorm_bwd](rmsnorm_bwd/) | `_rmsnorm_bwd_dxdw_fused` | x/dy/dx `[2048,512]` BF16；weight/rrms/partial_dw FP32 | `(64,)` / BLOCK_S=32, D_ROUNDED=512 |
| TX05 | [situ_fwd](situ_fwd/) | `_fused_situgluv2_fwd_triton_kernel` | a/b/out 各 `[128,1024]` BF16 | `(128,8)` / BLOCK=128 |
| TX06 | [situ_bwd](situ_bwd/) | `_fused_situgluv2_bwd_triton_kernel` | a/b/dout/da/db 各 `[128,1024]` BF16 | `(128,8)` / BLOCK=128 |

RMSNorm **eps=10**，保留附件中 `16-6` 的实际值。SiTU 的 beta、linear_beta 均为 0.5，倒数均为 2；公式为
`(beta*tanh(a/beta)*sigmoid(a)) * (linear_beta*tanh(b/linear_beta))`，不是 SwiGLU。
`_tanh` 是 SiTU fwd 的内部辅助函数，不是第七个测试入口。

## 一键批跑

从仓库根目录执行。Python 3.11+、Torch、Torch-NPU、Triton-Ascend 使用已有隔离 runtime；不需要安装 FLA/Liger 或下载 `.pt`。
按 [环境说明](../../docs/TESTING.md#ascend-环境) 配置并采集 `environment.local.json`。
卡号由使用者选择健康空闲卡，以下 `6` 只是示例；脚本不会杀进程或重置设备。

```bash
# 只查看六项计划，不需要 Torch、Triton 或 NPU
python tools/run.py list --operator Tencent
python tools/run.py plan --operator Tencent --variants baseline

# 在已配置的隔离 runtime 中串行跑全部六项
python -I tools/run.py run --operator Tencent --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/tencent-001

# 也支持单项或指定列表；输出目录必须是新的
python -I tools/run.py run --operator TX03,TX04 --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/tencent-rms-001
```

可加 `--correctness-only` 跳过计时。`tencent` 小写也可用，单项可用 `TX01` 或 `tencent_ce_fwd` 等唯一名称。
每项只登记表中一个尺寸；`--suite smoke` **仍然是该尺寸**，不会自动缩小 CE 或 RMSNorm。
`--suite fp32` 选择 CE，`--suite bf16` 选择 RMSNorm 和 SiTU。
此批只有原始 baseline；默认 `--variants all` 等价于 baseline，`--variants manual` 明确报未提供。
当前只接入默认 `--timing profiler`；MSPTI 不支持本批的输入重置协议，会在执行前报错。

CE 单个 logits 张量为 2.5 GiB；bwd 还保留一个同尺寸原地输出缓冲区。
CPU 参考按 16 行分块计算，避免创建完整 FP64 logits；bwd 仍需完整 FP32 参考结果。
执行时须预留数据、校验和运行库所需的主机内存与显存。若出错，结果和日志保留，设备错误会停止整批。
已有结果不会覆盖；可用未完成 ID 列表继续下一次运行。

## 测量和精度范围

- 仅计单个目标 Triton kernel 的设备时间。正确后 warmup 3 次，测 3 轮 × 5 次；每轮取中位数，再取三轮中位数的中位数。
- seed=42，自生成输入；CE 包含正常标签、ignore_index=-1、首尾类别与 tile 边界标签。
- **CE bwd 每次调用前恢复 logits**，包括首次验证、warmup 和全部 15 次计时；恢复在独立计时标签外并同步完成。目标 logits 和 dlogits 传入同一缓冲区。
- CE fwd 的 row_max 预计算、最终 sum/log/reduction，以及 RMSNorm bwd 的 partial_dw 最终归约，均不计入目标 kernel 耗时。反向输入直接生成，不计前向调用，不用时间相减得到 bwd。
- profiler 要求每个标签恰好归因到一个指定 kernel；标签外的准备操作单独排除，标签内出现其它 kernel、归因不明或设备干扰均拒绝计时。
- CPU FP64 数学/Autograd 参考检查完整输出。每次调用前独立输出填 NaN，原地输出恢复源输入；检查输出两端各 64 个哨兵和只读输入。计时后再次校验最终输出。

| 检查 | rtol | atol |
| --- | --- | --- |
| CE partial/final sum_exp | 1e-4 | 1e-2 |
| CE partial_loss、loss、lse、dlogits | 1e-5 | 1e-5 |
| RMSNorm y / dx | 0.02 | 0.02 |
| RMSNorm rrms | 1e-5 | 1e-6 |
| RMSNorm partial_dw / dw_sum | 1e-4 | 1e-3 |
| SiTU out / da / db | 0.02 | 0.002 |

六个目录保留原始 JIT 函数体/签名/装饰器和原 tile/grid；使用编译器默认 launch options，不显式关闭图优化或 AutoBlockify。
每个入口独立保留源码快照、生成式 case、launch 和 metadata，不导入另一个算子的实现。
公共框架提供环境指纹/卡锁/占用门禁/计时。整批生成 `performance.csv`、`summary.json`、逐项日志和 `result.json`/原始 trace。

## 验证状态

本次上传为脚本整理：已做源码一致性、launch 绑定、输入重置/计时边界和仓库离线检查。
Linux 全仓库完整性检查及 141 项离线/CPU 单元测试通过，详见 [验证记录](../../docs/TENCENT_VALIDATION.md)。
**新仓库入口未重新执行 NPU 性能测试**；不将之前 standalone harness 的结果当作此入口的设备验证。
RMSNorm 本次固定 `[2048,512]`，没有从 `[1024,2048]` 或 `[4096,512]` 移用性能数值。

```bash
python tools/verify_repository.py
python -m unittest discover -s tests -v
```
