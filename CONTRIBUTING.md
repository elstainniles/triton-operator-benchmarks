# 加入算子与结果

每个算子方向/实现放到 `operators/<来源>/<名称_方向>/`，不同算子不互相导入实现。fwd、bwd 分目录，相关条目用 `family` 关联；同族不能因拆目录而被当作独立泛化样本。公共 `opbench/` 只负责配置、测试与计时。

目录内提供：

- `operator.json`：唯一 ID、名称、方向、family、来源提交/路径/许可证、适配范围、baseline ID、候选与计算符号。
- `upstream/`：固定来源的文件，保留版权与许可。
- `kernels.py`：执行代码；改动范围写在文件头和算子 README。
- `baseline.py` / `manual_tile.py`：固定基线和显式手工候选。
- `launch.py`：输入绑定和 grid 公式。
- `case.py`：`make_case(shape, dtype, seed)` 返回输入、独立参考输出、带哨兵的输出和实际输出列表。
- `cases.json`：有理由选择的输入配置，区分 shape 的含义和各张量真实形状，不盲目扩展笛卡尔积。
- `results/<批次>/`：不可覆盖的比较记录、原始样本、环境指纹、失败和退化；未计时用 null。
- `README.md`：原始版本、手改版本、形状、数据来源及已知限制。

新增目录后在 `catalog.json` 登记。保持老 ID 稳定。运行 `python tools/verify_repository.py` 与 host 单元测试；新的 NPU 执行放入独立 runtime/cache，核对包来源、libtriton/工具链哈希、设备健康与占用。build/runtime 分离，使用构建好的 wheel，不使用全局安装、`pip --user`、editable install 或 TA/TD 源码 PYTHONPATH。

每个可执行条目必须能自生成输入并运行性能测试，不能默认依赖未公开的 `.pt` 或外部参考输出。TritonGraphOptimizationTests 采用 `standalone_generated_benchmark`：各算子提供 `benchmark.py`/`cases.json`，原始快照保留；公共 loader 只加载审计过的 JIT 函数，省略 autotune，函数体与签名不变。使用 `enable_graph_optimize=False`，旧图优化关闭脚本仅作归档/可选回放。输入适配、固定 tile、校验容差及实测范围必须记录；没有手改候选或数据时如实说明。源码缺少许可证时如实登记，不将根许可证自动套到第三方来源。

先检查完整输出与哨兵，正确才计时。不得放宽容差、改原算子来迁就优化候选。只比较同一环境、输入、卡和计时口径下的结果。受干扰计时保留但拒绝采用；不重置设备、不终止其它任务。

历史导入与新跑结果必须标明来源。不要提交账号凭据、内部网络地址、机器登录配置、运行环境目录或未授权的第三方私有代码。第三方代码必须记录可公开来源与许可。
