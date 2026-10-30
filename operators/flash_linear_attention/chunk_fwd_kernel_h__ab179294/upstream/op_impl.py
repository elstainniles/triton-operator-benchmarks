import torch
import triton
import triton.language as tl
from einops import reduce
import os
import functools
import inspect
from packaging import version

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    import triton.language.extra.libdevice as tldevice
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))

from verify_torch_kernels import chunk_gsa_bwd_k_kernel_dA_torch, chunk_gsa_bwd_k_kernel_intra_dvg_torch, chunk_gsa_bwd_k_kernel_dqkvg_torch
os.environ['TRITON_F32_DEFAULT'] = 'ieee'
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({})
        #for num_warps in [2, 4, 8]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_dA(
    v,
    g,
    do,
    dA,
    chunk_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    p_dA = tl.make_block_ptr(dA+((i_v*all+bos)*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j*BC), (BC, BC), (1, 0))

    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + (bos*H+i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t*BT + i_j*BC), (BV, BC), (0, 1))
        p_gv = tl.make_block_ptr(g + (bos*H+i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t*BT + i_j*BC), (BV, BC), (0, 1))
        p_gn = g + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
        p_g = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * exp(b_g - b_gn[None, :])).to(b_do.dtype)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        b_dA = tl.dot(b_do, b_vg) * scale
    elif i_i == i_j:
        p_g = tl.make_block_ptr(g + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_v = v + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_dAj = tl.sum(b_do * b_v[None, :] * exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += H*V
            p_gv += H*V
        b_dA = tl.where(m_dA, b_dA, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}) # num_warps=num_warps, num_stages=num_stages
        # for num_warps in [2, 4]
        # for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    dgv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + (bos*HQ+i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos*H+i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + ((i_k*all+bos)*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_g = tl.make_block_ptr(g + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gn = g + (bos + o_t - 1) * H*V + i_h * V + o_v
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + ((i_k*all+bos)*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dg = tl.make_block_ptr(dg + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dgv = tl.make_block_ptr(dgv+((i_k*all+bos)*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + (i_tg * HQ + i_hq) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        m_v = o_v < V

        b_gn = tl.load(p_gn, mask=m_v, other=0)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_gv = exp(b_gn[None, :] - b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * exp(b_g)).to(b_do.dtype)
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dg = tl.sum(tl.trans(b_h) * b_dh, 0) * exp(b_gn)

        b_dh = b_dh.to(b_k.dtype)
        b_dq += tl.dot(b_do, b_h.to(b_k.dtype)) * scale
        b_dk += tl.dot((b_v * b_gv).to(b_v.dtype), tl.trans(b_dh))
        b_dv = tl.dot(b_k, b_dh) * b_gv
        b_dg += tl.sum(b_dv * b_v, 0)

        if i_k == 0:
            b_dgv = tl.load(p_dg, boundary_check=(0, 1)) + b_dg[None, :]
        else:
            b_dgv = tl.zeros([BT, BV], dtype=tl.float32) + b_dg[None, :]

        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + (bos*HQ + i_hq) * BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos*HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos*HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_intra_dvg(
    v,
    g,
    o,
    A,
    do,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    p_gv = tl.make_block_ptr(g + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T)-1)*H*V + i_h*V + o_v
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + (bos*HQ+i_hq) * BT, (BT, T), (1, HQ*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT + i_j*BC, i_v*BV), (BC, BV), (1, 0))

        m_j = (i_t * BT + i_j * BC + o_i) < T
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * tl.where(m_j[:, None], exp(b_g - b_gn[None, :]), 0)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do.to(b_A.dtype))
    b_dv *= exp(b_gn[None, :] - b_gv)

    p_g = g + (bos + i_t * BT + i_i * BC) * H*V + i_h * V + o_v
    p_A = A + (bos + i_t*BT + i_i*BC) * HQ*BT + i_hq * BT + i_i * BC + o_i
    p_do = do + (bos + i_t*BT + i_i*BC) * HQ*V + i_hq * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.load(p_A)
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += H * V
        p_A += HQ * BT
        p_do += HQ * V
    p_o = tl.make_block_ptr(o + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))

    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(torch.diff(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(torch.diff(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    import torch.nn.functional as F
    return F.pad(triton.cdiv(torch.diff(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}) # num_warps=num_warps, num_stages=num_stages
        for BK in [32, 64]
        for BV in [32, 64]
        # for num_warps in [1, 2, 4, 8]
        # for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % NTS == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        if USE_G_GAMMA:
            b_g = b_gamma * (tl.arange(0, BT) + 1 + i_t * BT)
            b_h = b_h * exp(b_g[-1] - b_g[-BT])[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv)).to(b_v.dtype)

        b_h = b_h + tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return h, ht


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}) # num_warps=num_warps, num_stages=num_stages
        for BK in [32, 64]
        for BV in [32, 64]
        # for num_warps in [1, 2, 4, 8]
        # for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // H, i_nh % H
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_hq)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_s = i_t // NTS
        p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_dh = tl.make_block_ptr(dh + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % NTS == NTS - 1:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G_GAMMA:
            b_g = b_gamma * (tl.arange(0, BT) + 1 + i_t * BT)
            b_dh = b_dh * exp(b_g[-1] - b_g[-BT])[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = (b_do * exp(b_gv)).to(b_do.dtype)

        b_dh = b_dh + tl.dot(b_q, b_do)

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    NG = HQ // H

    dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        NG=NG,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return dh, dh0


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}) # num_warps=num_warps
        for BS in [32, 64]
        # for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum(
    s: torch.Tensor,
    chunk_size: int = 64,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    scale: float | None = None,
    head_first: bool = False,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    o = torch.empty_like(s)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_local_cumsum_vector_kernel[grid](
        s,
        o,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        B,
        H,
        S,
        BT,
        REVERSE=reverse,
        HEAD_FIRST=head_first,
    )
    return o


def chunk_gsa_bwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[2]

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NG = HQ // H

    if h is None:
        print("=============> h is None ===============")
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=None,
            gv=g,
            h0=h0,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            states_in_fp32=False,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        do=do,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        states_in_fp32=True,
    )
    dA = q.new_empty(NV, B, T, HQ, BT)
    grid = (NV, NT * NC * NC, B * HQ)
    chunk_gsa_bwd_k_kernel_dA[grid](
        v,
        g,
        do,
        dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
    )
    dA = dA.sum(0, dtype=dA.dtype)
    # dA = chunk_gsa_bwd_k_kernel_dA_torch(v, g, do, BT=BT, BC=BC, BV=BV, scale=scale)

    A = do.new_empty(NK, B, T, HQ, BT)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, T, HQ, K)
    dv = v.new_empty(NK, B, T, HQ, V)
    dgv = g.new_empty(NK, B, T, HQ, V, dtype=torch.float)
    grid = (NK, NT, B * HQ)
    chunk_gsa_bwd_k_kernel_dqkvg[grid](
        q,
        k,
        v,
        h,
        g,
        A,
        do,
        dh,
        dq,
        dk,
        dv,
        dg,
        dgv,
        dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NG=NG,
    )
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    dgv = dgv.sum(0, dtype=dgv.dtype)
    # A, dq, dk, dv, dgv = chunk_gsa_bwd_k_kernel_dqkvg_torch(
    #     q, k, v, h, g, do, dh, dA, dg, BT=BT, BK=BK, BV=BV, scale=scale
    # )
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT * NC, B * HQ)
    chunk_gsa_bwd_k_kernel_intra_dvg[grid](
        v,
        g,
        o,
        A,
        do,
        dv,
        dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        num_warps=4,
        num_stages=2,
    )
    # dv, dg = chunk_gsa_bwd_k_kernel_intra_dvg_torch(
    #     v, g, o, A, do, dv, dg, BT=BT, BC=BC, BV=BV
    # )
    dg = dgv.add_(chunk_local_cumsum(dg, chunk_size=BT, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices))

    return dq, dk, dv, dg, dh0, dA, A
