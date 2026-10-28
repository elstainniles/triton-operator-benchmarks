# 上游来源与许可记录

来源：<https://gitcode.com/weizhan4/TritonGraphOptimizationTests>，提交 `2e67d5b6723b1d472dcef91adb8a83c5b6761765`，于 2026-10-28 按仓库维护者的导入要求归档。

该固定提交没有仓库级 `LICENSE` 文件。部分源文件带有 `Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.`；其余文件未声明统一许可证。原版权/作者注释全部保留，metadata 记为 `NOASSERTION`。不能将“来源可公开访问”解释为上游授予 Apache-2.0 许可，也不能由本仓库替上游重授许可。

本目录导入的第三方 Python 文件、原始快照及 `external_references/upstream_run.md` **不适用根目录对自有测试工具的 Apache-2.0 声明**，其权利声明随原始来源保留。本仓库新增的测试说明、登记信息和公共工具依根目录许可提供。

`upstream/` 与对应 Git blob 字节一致。同目录执行版仅在 9 处 Triton kernel 调用中添加 `enable_graph_optimize=False`；没有更改 kernel 计算体或声称这些是新增手工优化。
