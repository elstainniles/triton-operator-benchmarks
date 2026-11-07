# Tencent 入口验证记录（2026-11-07）

本批在仓库 main `7d3aa28bdbc0cd2b9508ce845f65ccffe6f338f5` 上新增六个独立入口：
TX01–TX06，CE / RMSNorm / SiTU-GLU v2 各 fwd/bwd。具体尺寸与命令见 [Tencent 说明](../operators/Tencent/README.md)。

## 本次已验证

- 六个目标 kernel 与所存原始快照的函数签名、函数体、装饰器 AST 一致；SiTU fwd 的 `_tanh` 依赖也一致。
- shape、dtype、输入输出绑定、原 grid/tile、RMSNorm eps=10 和 CE bwd 原地别名正确登记。
- CE bwd 在每一次调用前恢复 logits；单元测试检查连续 15 次的恢复、同步、标签和 kernel 调用顺序。
- CE 的 CPU FP64 参考与 PyTorch native cross_entropy/autograd 对照通过，包含 ignore_index 与跨 tile 标签。
- 六个生成式 fixture 的 CPU 检查通过，包括 `[2048,512]` RMSNorm、`[128,1024]` SiTU、CE 小规格 `[7,2051]`。
  验证了输出未写、输入误改、哨兵破坏会被检出；这不等价于执行对应 Triton kernel。
- Linux `tools/verify_repository.py` 通过，全仓库 363 个登记入口的既有检查及新增 Tencent 检查完成。
- Linux 共 **141 项离线/CPU 单元测试通过**。环境为 Python 3.11、PyTorch 2.10.0+cpu；
  禁止 Torch 后端自动加载，既有 FLA mock 测试所需的 `torch.npu` 仅提供空命名空间供 patch 使用。
  没有初始化 NPU，没有编译或执行设备 kernel。

Windows 首次运行遇到既有 FLA 测试的 260 字符路径限制和临时文件 CRLF/hash 不一致；
确认原因后在 Docker 的 Linux 环境复核通过。没有为这些平台差异改写既有 FLA 测试或结果。
新增来源快照使用仓库规定的 LF，同时保留原附件/提取字节哈希。

## 尚未验证的范围

此次未重新执行新仓库入口的 NPU 正确性或性能，也没有在本批导入历史计时。
尤其不能把其它 RMSNorm 尺寸的历史计时赋给本次 `[2048,512]`。
当前状态为 `HOST_CHECKED_NOT_NPU_REVALIDATED`；设备结果必须来自使用者随后运行时生成的独立证据。

`--operator Tencent` 可独立或整批运行，采用默认 profiler；MSPTI 在执行前明确拒绝。
旧批次的计时标签边界保持原有行为，新增输入准备协议只应用于实现了 `prepare()` 的 fixture。
