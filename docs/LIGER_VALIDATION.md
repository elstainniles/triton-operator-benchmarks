# Liger 第二批导入验证

范围为 46 个精确 kernel，固定 Liger 提交 `781a8aef64bf6105bdefb5f4c991e0d14c08e793`。历史记录为 42 条正确且有性能、3 条编译失败、1 条被前向编译阻断；数据来源是旧框架，不是新入口的验证结果。

离线审计检查全部 46 条源码哈希、抽取函数的 AST（含装饰器）、测试源码、目录/名称唯一性、实现选择、15 个原始样本中位数、285 文件的依赖清单和无外部输入的计划。25 项 host 测试通过。

新入口已在原隔离 Ascend950PR / TA main 环境、独占卡 6 完成 **25/46 个 kernel 的正确性与目标计时验证，25 个均通过**。按用户要求，确认入口能运行后停止剩余调度，当前用例正常结束后退出；未执行剩余 21 个。它们继续标为 `HISTORICAL_ONLY`，不声称新入口已全量通过。

已完成条目为 LK01–LK25，覆盖 generic 与 Ascend-only；experimental 新入口尚未实跑。逐项状态与耗时见 [验证索引](liger-validation-20261028.json)和 [CSV](../operators/liger_kernel/export-validation.csv)，每条原始样本及输入/grid/constexpr 保存在该目录的 `results/20261028-export/measurement.json`，与原 `results/20261028/` 分开。

证据审计重新解析全部 25×15=375 个 profiler 设备样本，核对编译产物/trace 哈希、抽取源码位置、285 文件依赖身份、libtriton/编译器身份及采样期间的占用记录。未改变已安装包或编译器。环境版本和 metadata pin 冲突与 [历史环境](../operators/liger_kernel/historical-environment.json)相同；本次不宣称整个依赖环境通过。

每个条目采用第二批最终记录的 shape/dtype。原计划优先较大候选，但部分项目使用较小配置；未搜索显存、索引或编译器允许的最大尺寸，因此不能把登记用例称为“最大且不越界”。
