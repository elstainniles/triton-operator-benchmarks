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
)
@triton.jit
def _dyt_fwd_kernel(X, Y, Alpha, Gamma, Beta, HAVE_BETA: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    row_id = tl.cast(tl.program_id(1), tl.int64)

    X += row_id * N
    Y += row_id * N
    alpha = tl.load(Alpha).to(tl.float32)

    gamma = tl.load(Gamma + col, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(X + col, mask=mask, other=0.0).to(tl.float32)

    tanh_x = tanh(alpha * x)
    y = tanh_x * gamma
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.0).to(tl.float32)
        y += beta
    tl.store(Y + col, y, mask=mask)
