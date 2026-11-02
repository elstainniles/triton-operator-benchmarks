# FLA CV workbook batch

The 201 specifications in `11.1全量结果`, rows 2-202, are frozen in
[`cv-manifest.json`](../operators/flash_linear_attention/cv-manifest.json).
Each source-backed row lives in the existing `FLA###` operator directory as
`cases-cv.json` and `cv_kernels.py`. The original `kernels.py`, catalog IDs and
other benchmark cases remain intact. The CV adapter keeps original JIT function
signatures and bodies, removes autotune/heuristic selection, and binds the fixed
shape, scalar, grid and launch configuration in its case file. The source
archive and function bodies are checked by SHA256/AST before execution.

Seven named variants are not in the pinned source repository; no approximate
implementation is substituted. The historical compiler rejected 152 rows at a
normalized-IR control-flow precheck. They were **not** NPU baseline executions;
the batch keeps them labelled `CONTROL_FLOW_BLOCKED_PRECHECK_ONLY` by default.
Use `--include-control-flow` only to explicitly recheck those adapters with a
compiler that supports the control flow. This switch makes no claim that they
will compile or merge.

The tested rows use deterministic inputs (seeds `20261102`, `20261103`) made
from the recorded shape/dtype. No external `inputs.pt` or `outputs.pt` is
needed. The reference subprocess executes the original kernel semantics in
Triton's CPU interpreter with pointer bounds and complete-output checks;
`dot` accumulates in FP64 and then rounds to its declared accumulator dtype.
For rows 24, 26 and 85, `exp` is computed in FP64 before FP32 rounding, as
independently checked in the original task. This is **not** an independent
end-to-end mathematical oracle for every FLA kernel. The device runner checks
both seeds, all inputs and outputs, and sentinel guards before timing. FP32
uses `rtol=atol=1e-5`, FP16 `1e-3`, BF16 `5e-3`. A baseline failure skips the
candidate. Accuracy thresholds are not widened to make a candidate pass.

## Run

Use Linux, a pinned isolated Triton-Ascend runtime, and an explicit CANN
selection. Generate `environment.local.json` with the existing
[`capture_environment.py`](../tools/capture_environment.py) command described
in [TESTING.md](TESTING.md). The runner never builds or installs packages.

```bash
python -I tools/run_fla_cv.py verify
python -I tools/run_fla_cv.py plan --cases tested
"$RUNTIME_PYTHON" -I tools/run_fla_cv.py run --cases tested \
  --physical-card 6 --environment environment.local.json \
  --task-lock /path/to/validation-task.lock \
  --card-lock /path/to/npu-6.lock --output runs/fla-cv-001
```

`--cases all` includes all 201 rows but records the control-flow and missing
source exclusions without a device launch. `--cases 24,26` selects workbook
row numbers, and `--operator FLA188` selects an existing catalog entry. A
fresh output directory is required. `--resume` on the same directory skips
recorded terminal rows and refuses to silently repeat an incomplete or failed
attempt. Review that attempt first, then use a new output directory for any
intentional retry. `--factor 1` measures baseline only; the default factor 2
tries grid axes with enough programs. `--axis 0|1|2` narrows the candidate.

Each row runs in a new process with fresh compiler caches. The site task/card
locks are optional CLI arguments; use them wherever other suites share a card.
The runner also acquires the repository's per-user card lock. Device health
and occupancy must be known and clear. It never kills processes or resets the
device. A known driver residual in the original environment is not a general
allowlist for a new machine or run.

Only an `APPLIED` candidate with a bound plan and passing accuracy can be
timed as a merge. Fallbacks have no candidate time. Profiler labels and async
flows attribute each of 15 samples per round to exactly one compiled Triton
device kernel; the result is the mean of two round medians, in microseconds.
Foreign or unknown occupancy rejects timing. See `summary.json`,
`performance.csv`, each row's `result.json`, CPU references and profiler traces.

## Historical result boundary

`results/20261102-cv/` contains sanitized results from the earlier task
harness on physical card 6 with compiler code
`c06b1a4df3d710c6dba23f696da577e499a7feb9` and libtriton SHA256
`5840f711f87bef983c074ab44242b7ae74a5b5ca751a3876ae3429be085f547d`.
Of 201 rows, 41 had valid baseline timings, 1 failed baseline accuracy,
152 were control-flow prechecks and 7 had no exact source. There were **zero
actually applied and timed CV merges**. Five source-labelled CV rows did not
contain a `dot` in normalized IR. The historical timings do not validate the
new portable runner or other compiler revisions. New measurements must use
the runner and its own evidence.
