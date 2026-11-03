# FLA 原用例复测

已抽取 kernel 的原测试和后续适配保存在各算子的 `tests/` 子目录。每个配置包含原测试、运行依赖、固定种子的输入生成函数及独立 CPU 参考，不需要下载或提供 `.pt` 文件。

全量索引见 [test-sources.json](../operators/flash_linear_attention/test-sources.json)。同一原用例可能有不同 shape、dtype、tile 或修订后的测试实现，因此使用完整 `id` 选择配置；`case_id` 用于关联原用例。`test.json` 记录来源、原验收方式、生成器及各文件哈希。此前静态清单未登记的动态调用目标保存在 `additional_tests/<kernel>/`，来源另行列明。

本批覆盖已有可复测清单中的 351 个原用例、631 个配置变体，关联 286 个源码目标。原始 352 项中，`cv_eb0c60313edf85dd` 未加入执行入口：`fla/ops/path_attn/intra_chunk_preprocess_bwd_prepare.py` 的原 wrapper 在 `return_h=True` 时分配的 `h` 与 kernel 写地址不匹配，原规格存在越界及重叠写入；这不是缺少输入文件，需要先修复原实现。

## 查看和检查输入

以下命令均从仓库根目录执行：

```bash
python tools/run_fla_test.py --list
python tools/run_fla_test.py --list --case vv_2899008551315181
python tools/verify_repository.py
python -m unittest discover -s tests -v
```

从列表复制一个完整测试 `id`，只检查数据生成时需要 CPU PyTorch，无需 NPU：

```bash
python tools/run_fla_test.py --case <完整测试id> --check-fixtures
```

`generated/fixture.py` 的 `make_fixtures()` 直接返回内存中的输入和参考输出。旧文件名仅作为字典键，`load_fixture()` 不读取文件；每次返回独立张量副本，保留原 stride、view 和别名关系。输入生成和 CPU 参考计算均在设备计时之前完成。原测试已经自行生成输入的配置使用空 fixture，由原测试完成生成。

## 设备运行

按照 [环境说明](TESTING.md#ascend-环境) 安装匹配 Python 的 Torch-NPU 和 Triton-Ascend wheel、选择 CANN 并生成 `environment.local.json`。在所选物理卡运行单个配置：

原测试还使用 `pytest`、`numpy` 和 `einops`，请安装到同一 runtime；本批检查使用 `pytest==8.3.2`、`numpy==1.26.4`、`einops==0.8.1`。

```bash
/path/to/runtime/bin/python -I tools/run_fla_test.py \
  --case <完整测试id> --physical-card <物理卡号> \
  --environment environment.local.json --output runs/fla-test-001
```

默认预热 5 次，采样 5 次并取平均值，可用 `--warmup` 和 `--samples` 调整。只做数值检查可加 `--correctness-only`。换编译器版本时，使用对应 runtime Python、重新采集环境清单并指定新的输出目录。

Block 版本使用支持 block batching 的独立 runtime，额外传入：

```bash
--block-factor 2 --axis 0
```

原 Python grid 和 tile 保持在对应测试配置中。Block 计时要求实际报告 `APPLIED`，并记录 requested/applied factor、selected axis、original grid 和 batch grid；回退、没有足够块及编译拒绝分别记录。比较两个版本时必须选择同一完整测试配置，使用相同的输入、tile、采样设置和设备。

## 输出与限制

`result.json` 保存正确性结果、错误全文、环境身份、实际 launch，以及成功计时的 `samples_us` 和 `mean_us`。`profiles/` 保存原始 profiler 数据，`device.jsonl` 保存设备状态。耗时只取目标 kernel 的 `Duration(us)`，不包括输入生成、CPU 参考、编译和数据恢复。

原测试的精度标准随配置保留。经批准由 CUDA 比较改为独立 CPU FP64 比较的配置，在 `validation_standard` 中标明；不能把这些结果解释为 CUDA 逐位一致。一次测试如果运行多个不同的目标配置，入口报告 `multiple_target_configurations`，不会把最后一次 launch 的耗时当作整个用例的耗时。编译失败和精度失败仍有可检查的结果，性能字段为 null。

输入生成通过 CPU 核对不代表所有配置均已通过设备验证。新运行的 `result.json` 是该配置在所选版本上的执行证据；旧队列的耗时不作为这批公开入口的测量结果。
