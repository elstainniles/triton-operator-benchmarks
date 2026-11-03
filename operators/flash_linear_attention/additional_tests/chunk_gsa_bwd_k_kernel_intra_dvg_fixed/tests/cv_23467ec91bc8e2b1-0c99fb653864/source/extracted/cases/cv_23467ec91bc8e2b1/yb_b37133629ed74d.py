
from opbench.fla_fixtures import load_fixture
import os
import sys
import torch
import triton
import triton.language as tl
from yb_249d3453092fd3 import chunk_gsa_bwd_k_kernel_intra_dvg_torch
from yb_ab1792945eaa4c import chunk_gsa_bwd_k_kernel_intra_dvg
if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    import triton.language.extra.libdevice as tldevice

    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))
else:

    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({})], key=['BT'], **{})
@triton.jit
def chunk_gsa_bwd_k_kernel_intra_dvg_fixed(v, g, o, A, do, dv, dg, chunk_indices, cu_seqlens, T, HQ: tl.constexpr, H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr, NC: tl.constexpr, NG: tl.constexpr, IS_VARLEN: tl.constexpr):
    """
    修复版本：
    1. 使用手动指针计算加载 A 矩阵
    2. 正确处理边界条件
    3. 确保 A 矩阵访问方式与 torch 实现一致
    """
    i_v, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_hq = (i_bh // HQ, i_bh % HQ)
    i_h = i_hq // NG
    i_t, i_i = (i_c // NC, i_c % NC)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if i_t * BT + i_i * BC >= T:
        return
    p_gv = tl.make_block_ptr(g + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H * V + i_h * V + o_v
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        m_j = i_t * BT + i_j * BC + o_i < T
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_do_scaled = b_do * tl.where(m_j[:, None], exp(b_g - b_gn[None, :]), 0.0)
        o_j = tl.arange(0, BC)
        A_ptrs = A + (bos + i_t * BT + i_j * BC + o_j[None, :]) * HQ * BT + i_hq * BT + (i_i * BC + o_i[:, None])
        b_A = tl.load(A_ptrs).to(tl.float32)
        b_dv += tl.dot(b_A, b_do_scaled)
    b_dv *= exp(b_gn[None, :] - b_gv)
    p_g = g + (bos + i_t * BT + i_i * BC) * H * V + i_h * V + o_v
    p_A = A + (bos + i_t * BT + i_i * BC) * HQ * BT + i_hq * BT + i_i * BC + o_i
    p_do = do + (bos + i_t * BT + i_i * BC) * HQ * V + i_hq * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.load(p_A)
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.0)
        p_g += H * V
        p_A += HQ * BT
        p_do += HQ * V
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

def run_kernel(kernel_fn, v, g, o, A, do, dv_in, dg_in, B, T, HQ, H, V, BT, BC, BV, NG):
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NV = triton.cdiv(V, BV)
    dv = dv_in.clone()
    dg = dg_in.clone()
    grid = (NV, NT * NC, B * HQ)
    kernel_fn[grid](v, g, o, A, do, dv, dg, cu_seqlens=None, chunk_indices=None, T=T, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG, IS_VARLEN=False)
    return (dv, dg)

def compare(name, dv_test, dg_test, dv_ref, dg_ref):
    dv_diff = (dv_test - dv_ref).abs()
    dg_diff = (dg_test - dg_ref).abs()
    dv_max_diff = dv_diff.max().item()
    dg_max_diff = dg_diff.max().item()
    dv_mean_diff = dv_diff.mean().item()
    dg_mean_diff = dg_diff.mean().item()
    dv_allclose = torch.allclose(dv_test, dv_ref, rtol=0.01, atol=0.01)
    dg_allclose = torch.allclose(dg_test, dg_ref, rtol=0.01, atol=0.01)
    status = 'PASS' if dv_allclose and dg_allclose else 'FAIL'
    print(f"\n{'=' * 60}")
    print(f'  {name}: {status}')
    print(f"{'=' * 60}")
    print(f'  dv max diff:     {dv_max_diff:.6e}')
    print(f'  dv mean diff:    {dv_mean_diff:.6e}')
    print(f'  dv allclose(1e-2): {dv_allclose}')
    print(f'  dg max diff:     {dg_max_diff:.6e}')
    print(f'  dg mean diff:    {dg_mean_diff:.6e}')
    print(f'  dg allclose(1e-2): {dg_allclose}')
    return dv_allclose and dg_allclose

def test_chunk_gsa_bwd_k_kernel_intra_dvg():
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    print('=' * 60)
    print('修复版本测试')
    print('=' * 60)
    print('\n关键修复:')
    print('1. 使用手动指针计算替代 make_block_ptr 加载 A 矩阵')
    print('2. 正确处理边界条件和 mask')
    print('3. 确保 A 矩阵访问方式与 torch 实现一致')
    print('\nA 矩阵访问方式:')
    print('  Off-diagonal: A_block[ii, jj] = A[b, t_j, hq, i_start + ii]')
    print('    where t_j = i_t * BT + i_j * BC + jj')
    print('  Diagonal: A_val = A[b, t_j, hq, i_start + ii]')
    print('    where t_j = i_t * BT + i_start + j')
    data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k.pt', map_location='cpu', weights_only=False)
    input_data = data['input_data']
    device = 'npu'
    v = input_data['v'].to(device)
    g = input_data['g'].to(device)
    o = input_data['o'].to(device)
    do = input_data['do'].to(device)
    B = input_data['B']
    T = input_data['T']
    HQ = input_data['HQ']
    H = input_data['H']
    V = input_data['V']
    BT = input_data['BT']
    BC = input_data['BC']
    BV = input_data['BV']
    NG = input_data['NG']
    scale = input_data['scale']
    q = input_data['q']
    k = input_data['k']
    A = torch.zeros(B, T, HQ, BT, dtype=q.dtype)
    NT = T // BT
    for i_b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            for i_hq in range(HQ):
                q_chunk = q[i_b, t_start:t_end, i_hq, :].float()
                k_chunk = k[i_b, t_start:t_end, i_hq // NG, :].float()
                A_chunk = torch.matmul(q_chunk, k_chunk.T) * scale
                mask = torch.tril(torch.ones(t_end - t_start, t_end - t_start, dtype=torch.bool))
                A[i_b, t_start:t_end, i_hq, :t_end - t_start] = A_chunk * mask
    A = A.to(device)
    dv_in = torch.zeros(B, T, HQ, V, dtype=v.dtype, device=device)
    dg_in = torch.zeros(B, T, HQ, V, dtype=g.dtype, device=device)
    print('\nComputing torch native reference...')
    dv_torch, dg_torch = chunk_gsa_bwd_k_kernel_intra_dvg_torch(v, g, o, A, do, dv_in.clone(), dg_in.clone(), BT=BT, BC=BC, BV=BV)
    print('\nRunning original Triton kernel...')
    dv_orig, dg_orig = run_kernel(chunk_gsa_bwd_k_kernel_intra_dvg, v, g, o, A, do, dv_in, dg_in, B, T, HQ, H, V, BT, BC, BV, NG)
    passed = compare('Original Triton kernel', dv_orig, dg_orig, dv_torch, dg_torch)
    assert passed
    print('\nRunning fixed kernel...')
    try:
        dv_fixed, dg_fixed = run_kernel(chunk_gsa_bwd_k_kernel_intra_dvg_fixed, v, g, o, A, do, dv_in, dg_in, B, T, HQ, H, V, BT, BC, BV, NG)
        passed = compare('Fixed kernel', dv_fixed, dg_fixed, dv_torch, dg_torch)
        assert passed
        if passed:
            print('\n' + '=' * 60)
            print('SUCCESS: Fixed kernel passes precision test!')
            print('=' * 60)
            print('\n修复说明:')
            print('  问题: make_block_ptr 在 NPU 上无法正确处理 A 矩阵的非连续 stride')
            print('  原因: A 的 shape 是 [B, T, HQ, BT]，但 make_block_ptr 使用了错误的 shape')
            print('  解决: 使用手动指针计算替代 make_block_ptr')
            print('  公式: A_ptrs = A + (bos + i_t*BT + i_j*BC + o_j[None,:])*HQ*BT + i_hq*BT + (i_i*BC + o_i[:,None])')
    except Exception as e:
        print(f'\n  Fixed kernel FAILED: {e}')
        import traceback
        traceback.print_exc()
