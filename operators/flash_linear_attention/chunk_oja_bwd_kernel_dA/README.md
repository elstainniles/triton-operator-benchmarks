# FLA056: `chunk_oja_bwd_kernel_dA`

Source archive from `fla/ops/gated_oja_rule/chunk_o.py` at `5752e83ce7e469cfc0a3145bf20d3c902b716d48`. Origin: `fla_source`. The original kernel, including its decorators, is unchanged.

**Status: source archive; no executable benchmark or device validation is claimed.** The original kernel and dependency sources are archived; an independently checked, self-generated benchmark launch is not yet registered.

`kernels.py` contains the selected function. `upstream/` contains the complete source, dependency snapshots and MIT license. Absolute data paths in upstream files are historical source text; personal checkout prefixes in provenance comments are removed when listed in `source.snapshot_changes`; `source.original_sha256` records original bytes and `source.sha256` records the published snapshot. the runner never imports or executes these archived files. `cases.json` stays empty until a self-generated input and independent reference are adapted.

Original test associations:

- `tests/ops/ascend_ops/cv_operator/chunk_oja_bwd_kernel_dA/test_chunk_oja_bwd_kernel_dA_npu.py::test_chunk_oja_bwd_kernel_dA_function`
- `tests/ops/ascend_ops/cv_operator/chunk_oja_bwd_kernel_dA/test_chunk_oja_bwd_kernel_dA_npu.py::test_chunk_oja_bwd_kernel_dA_performance`
- `tests/ops/ascend_ops/cv_operator/chunk_oja_bwd_kernel_dA/test_chunk_oja_bwd_kernel_dA_simple_npu.py::test_chunk_oja_bwd_kernel_dA_simple_function`
- `tests/ops/ascend_ops/cv_operator/chunk_oja_bwd_kernel_dA/test_chunk_oja_bwd_kernel_dA_simple_npu.py::test_chunk_oja_bwd_kernel_dA_simple_performance`

## Frozen CV workbook cases

`cases-cv.json` adds workbook rows 118, 119, 164, 165, 181, 182, 195, 196; `cv_kernels.py` preserves the original JIT function bodies and binds fixed launch settings. These cases use generated inputs and a CPU execution of the kernel semantics, not an independent end-to-end mathematical oracle.

Run `python -I tools/run_fla_cv.py run --operator FLA056 --physical-card <card> --environment environment.local.json --output runs/fla-cv-001` from the repository root. See [CV batch instructions](../../../docs/FLA_CV.md) for control-flow exclusions, compiler factors, accuracy and resume behavior. The existing generic entry remains unchanged.

`results/20261102-cv/` is sanitized historical evidence from the original task harness; it does not claim device validation of this newly packaged runner.
