# softmax_fwd_kernel

Extracted from FLA commit `5752e83ce7e469cfc0a3145bf20d3c902b716d48`,
`fla/ops/utils/softmax.py` (MIT). The source snapshot is in `upstream/`; the MIT license is retained at `../LICENSE`.

`kernels.py` preserves the exact function signature, body, masks, strides and JIT
decorator. The original autotune decorator is removed: the baseline fixes an
original candidate (`num_warps=4`, `num_stages=2`, `B=next_power_of_2(D)`).
The local `exp`/`log` helpers, where used, are copied unchanged from the default
`FLA_USE_FAST_OPS=0` branch of `fla/ops/utils/op.py`; fast-op mode is not selected.
`manual_tile.py` records that no hand-tuned candidate is registered. This entry
provides the extracted baseline only.

`benchmark.py`/`case.py` implement `make_case(config, device='npu')` using fixed
CPU random seeds and independent FP64 mathematics. No external `.pt` files or
historical GPU outputs are required. Shapes mean `[rows, columns]`; the activation
upstream shape flattens `[2, 1024, 512]` to `[2048, 512]`. Only FP32 is currently
supported. Inputs and outputs have prefix/suffix sentinels. All output elements
start as NaN and must be written. Inputs, sentinels and any output row padding
must remain unchanged.

Softmax rows remain contiguous as required by the original kernel. Non-power-of-two D exercises the original masked B-D lanes; this does not introduce an unsupported row-stride parameter.

Validation uses the original activation RMS-error gate (`max_abs <= 1e-6` or
`RMS(error)/(RMS(reference)+1e-8) < 1e-3`) **and** the original softmax FP32
elementwise `atol=rtol=1e-5` gate. The original Ascend FP32 test also uses `atol=rtol=1e-5`. Both gates and finite-output checks apply to
these generated cases, so neither original threshold is relaxed. The relevant
original acceptance test is archived at `upstream/tests/ops/ascend_ops/vv_operator/softmax_fwd_kernel/test_softmax_fwd_kernel_npu.py`.

`baseline.py` calls the direct Triton launch in `launch.py`;
CPU generation/reference, transfers, reset and validation are outside target
kernel timing. `enable_graph_optimize=False` is fixed. The repository FLA runner
records compile/runtime errors and target-device duration; these generated
entrypoints have CPU contract tests only and **have not been NPU validated**.
No historical measurements are imported as results for these new entrypoints.
