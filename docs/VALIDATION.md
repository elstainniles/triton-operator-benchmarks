# 首次整理校验（2026-10-28）

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

## 第二批导入校验（2026-10-28）

TritonGraphOptimizationTests 固定提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`：7 组实际测试、11 个源文件，9 处 kernel launch 显式设置 `enable_graph_optimize=False`。原始 Git blob/SHA256 与快照一致；执行版去掉新增的关闭开关后，整文件 AST 必须与原始源码一致。3 项只有外部运行说明的引用单独记录，不作为已导入实现。

统一目录登记为 15 个条目（8 FlagGems + 7 新批次）。FlagGems 的 51 组/218 条历史结果仍单独核查。16 项 host 测试覆盖此前的检查以及图优化误开启拒绝、输入缺失、MoE case ID 边界、导入批次不能误用 FlagGems 手改计划。NPU 执行数为 0，没有新增性能数据。
