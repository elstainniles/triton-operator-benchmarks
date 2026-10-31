# 小尺寸 BH 合并历史结果

B=2、H=4、S=D=32，FP32/IEEE，BM=BN=BK=32，原 grid=(8,1,1)。

`measurement.json` 保留 batch control、factor=2/4/8 全部候选、逐轮样本、独立 seed 复核、实际 grid、binary/IR 指纹与守卫负例；含慢于基线的候选。

这些是已完成实验的历史导入，不表示本次导出的入口已重新在 NPU 运行。原始 trace/IR 包含私有路径，未上传；公开数据保留其来源 SHA256。输入、运行方式、环境与性能解释见[小尺寸说明](../../../../../docs/LIGER_CV_SMALL.md)。
