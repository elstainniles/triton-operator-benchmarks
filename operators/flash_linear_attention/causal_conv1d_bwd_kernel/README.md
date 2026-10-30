# FLA010: `causal_conv1d_bwd_kernel`

Source archive from `fla/modules/conv/triton/kernels.py` at `5752e83ce7e469cfc0a3145bf20d3c902b716d48`. Origin: `fla_source`. The original kernel, including its decorators, is unchanged.

**Status: source archive; no executable benchmark or device validation is claimed.** The original kernel and dependency sources are archived; an independently checked, self-generated benchmark launch is not yet registered.

`kernels.py` contains the selected function. `upstream/` contains the complete source, dependency snapshots and MIT license. Absolute data paths in upstream files are historical source text; personal checkout prefixes in provenance comments are removed when listed in `source.snapshot_changes`; `source.original_sha256` records original bytes and `source.sha256` records the published snapshot. the runner never imports or executes these archived files. `cases.json` stays empty until a self-generated input and independent reference are adapted.

Original test associations:

- `tests/ops/ascend_ops/vv_operator/causal_conv1d_states_fwd_kernel/test_causal_conv1d_states_fwd_npu.py::test_causal_conv1d_states_fwd_kernel_functiontest`
- `tests/ops/ascend_ops/vv_operator/causal_conv1d_states_fwd_kernel/test_causal_conv1d_states_fwd_npu.py::test_causal_conv1d_states_fwd_kernel_perftest`
- `tests/ops/ascend_ops/vv_operator/compute_dh0_kernel/test_compute_dh0_npu.py::test_compute_dh0_kernel_functiontest`
- `tests/ops/ascend_ops/vv_operator/compute_dh0_kernel/test_compute_dh0_npu.py::test_compute_dh0_kernel_perftest`
- `tests/ops/ascend_ops/vv_operator/test_causal_conv1d_bwd_kernel/test_causal_conv1d_bwd_kernel_npu.py::test_causal_conv1d_bwd_kernel`
- `tests/ops/ascend_ops/vv_operator/test_causal_conv1d_fwd_kernel/test_causal_conv1d_fwd_kernel_npu.py::test_causal_conv1d_fwd_kernel`
- `tests/ops/ascend_ops/vv_operator/test_causal_conv1d_update_kernel/test_causal_conv1d_update_kernel_npu.py::test_causal_conv1d_update_kernel`
