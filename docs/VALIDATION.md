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
