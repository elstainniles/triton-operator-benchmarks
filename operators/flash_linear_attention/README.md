# Flash Linear Attention Triton kernels

This batch contains **296 statically launch-referenced definitions** extracted from
[TritonAscendTest/flash-linear-attention](https://gitcode.com/TritonAscendTest/flash-linear-attention)
at `5752e83ce7e469cfc0a3145bf20d3c902b716d48`, under the retained [MIT license](LICENSE).
The original repository includes both library code and Ascend tests: 122 entries
come from `fla/`, and 174 from `tests/`. A test kernel is not implicitly treated
as an identical library kernel. Every entry has a source path, symbol, source
hash, archived upstream code, extracted function, dependency records and original
test associations. See the [catalog](catalog-table.md) and [scope](scope.json).

The 296 entries are source-path/symbol identities, not distinct algorithms or
completed device tests. The scope records 288 unique full function ASTs and
seven duplicate groups. The other 168 inventoried JIT definitions are helpers or
lack a statically resolved launch; they are not counted as additional operators.
Families connect related forward/backward implementations.

## Execution status

- `source_archive`: extraction is available, but no independently checked input,
  reference and launch adapter is registered. Running it writes `not_executable`
  with a reason and null timing; it does not import Torch or access a device.
- `generated_benchmark`: a self-contained adapter generates seeded inputs and
  independent CPU references. The sigmoid, logsigmoid and softmax forward/backward
  entries provide this interface. Their README files specify fixed original
  autotune candidates, shapes, strides, precision thresholds and guards.

New generated entry points have not been validated on an NPU. No local continuous
queue timing, block speedup or external `.pt` is imported as evidence for them.
Full-output correctness and guard checks must pass before a new run is timed.

```bash
python tools/run.py plan --operator fla --suite smoke --variants baseline
python tools/verify_repository.py
```

Commands run from the repository root. See [FLA testing](../../docs/FLA_KERNELS.md)
for the runtime environment and selected-device command.

## Reproducing the source extraction

`tools/import_fla_kernels.py` consumes the committed `import-manifest.json` and a
checkout of the exact original revision via `--source-root`. It reproduces the
source archive, not subsequent handwritten benchmark adapters. It does not
build a compiler, generate historical data or execute kernels. The manifest
retains the original test and existing extraction provenance without machine
workspace paths.

Original personal checkout paths in nine upstream test comments are redacted;
the original and published snapshot hashes and changes are recorded separately.
Function ASTs remain unchanged. Public upstream data-path strings inside archived
tests are provenance only; generated entry points do not load those files.
