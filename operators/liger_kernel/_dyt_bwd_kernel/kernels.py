# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_stages=ns, num_warps=nw)
        for bn in [1024, 2048, 4096]
        for ns in [1, 2]
        for nw in [4, 8, 16]
    ],
    key=["N"],
    # DA is indexed by program_id(0), so different BLOCK_N configs write to
    # different slot counts per SM. Autotune trials don't zero outputs between
    # runs, so stale slots from a prior trial would leak into da.sum(). Reset
    # DA between trials to isolate each config's writes.
    reset_to_zero=["DA"],
)
@triton.jit
def _dyt_bwd_kernel(
    DY, DX, DA, DG, DB, X, Alpha, Gamma, HAVE_BETA: tl.constexpr, M, N: tl.constexpr, BLOCK_N: tl.constexpr
):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    start_row_id = tl.cast(tl.program_id(1), tl.int64)

    alpha = tl.load(Alpha).to(tl.float32)
    da = 0.0
    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)
    dg = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAVE_BETA:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row_id in range(start_row_id, M, tl.num_programs(1)):
        x = tl.load(X + row_id * N + col, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row_id * N + col, mask=mask, other=0.0).to(tl.float32)
        tanh_x = tanh(alpha * x)
        if HAVE_BETA:
            db += dy
        dg += dy * tanh_x
        tmp = (1 - tanh_x * tanh_x) * dy * gamma
        da += tl.sum(x * tmp, 0)
        dx = alpha * tmp
        tl.store(DX + row_id * N + col, dx, mask=mask)

    tl.store(DG + start_row_id * N + col, dg, mask=mask)
    if HAVE_BETA:
        tl.store(DB + start_row_id * N + col, db, mask=mask)
    tl.store(DA + start_row_id * tl.cdiv(N, 512) + tl.program_id(0), da)
