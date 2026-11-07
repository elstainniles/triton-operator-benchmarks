# rmsnorm_fwd

`TX03` · `_rmsnorm_fwd_fused` · **VV**（无 matmul）。独立 fwd kernel，采用附件原始函数；函数体、签名、装饰器未改。

x/y[2048,512] BF16；weight[512]、rrms[2048] FP32；eps=10（保留附件 16-6 的实际值）。grid=(2048,)，D_ROUNDED=512。

自生成 seed=42 输入，无需 `.pt`。CPU FP64 参考、逐输出检查、输入未改写检查和输出两端哨兵均在计时外。
运行方式、精度阈值、原地重置及最终归约范围见 [Tencent 说明](../README.md)。

```bash
python -I tools/run.py run --operator TX03 --variants baseline \
  --physical-card 6 --environment environment.local.json --output runs/rmsnorm_fwd-001
```

从仓库根目录执行，卡号由使用者按空闲情况选择。`smoke` 仍使用上面的固定尺寸。

此目录只有原始 baseline；`manual_tile.py` 留空，不将其它尺寸的手改结果充当本尺寸性能。
本次完成离线校验，未重跑新仓库入口的 NPU 性能。新计时仅写入新运行的结果目录。

来源为用户提供代码，`Tencent` 是用户指定目录名；并非已核实的官方仓库来源。详见 [来源说明](../SOURCE_NOTICE.md)。
