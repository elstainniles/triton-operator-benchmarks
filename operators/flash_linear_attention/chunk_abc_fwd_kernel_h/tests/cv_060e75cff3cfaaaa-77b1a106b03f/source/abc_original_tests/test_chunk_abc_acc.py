import os
import sys
import torch
import pytest
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_abc import chunk_abc_fwd
from chunk_abc_torch_impl import chunk_abc_pytorch_fwd

from chunk_abc import (
    _chunk_abc_fwd_impl, softmax_bwd,
    chunk_abc_bwd_kernel_dh, chunk_abc_bwd_kernel_V,
    chunk_abc_bwd_kernel_intra_V,
    chunk_abc_bwd_kernel_intra_K, chunk_abc_bwd_kernel_K,
    chunk_abc_bwd_kernel_intra_KV, chunk_abc_bwd_kernel_rcum_inter,
    chunk_abc_bwd_kernel_rcum_intra,
    )
from chunk_abc_torch_impl import (
    chunk_abc_bwd_kernel_dh_pytorch,
    chunk_abc_bwd_kernel_V_pytorch,
    chunk_abc_bwd_kernel_intra_V_pytorch,
    softmax_bwd_kernel_pytorch,
    chunk_abc_bwd_kernel_intra_K_pytorch,
    chunk_abc_bwd_kernel_K_pytorch,
    chunk_abc_bwd_kernel_intra_KV_pytorch,
    chunk_abc_bwd_kernel_rcum_inter_pytorch,
    chunk_abc_bwd_kernel_rcum_intra_pytorch,
)



def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

device = get_device()

def test_chunk_abc_fwd_vs_torch(rtol=5e-3, atol=5e-3):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    B, H, T, K, V, M = 2, 4, 512, 64, 64, 32
    dtype = torch.bfloat16

    q = torch.randn((B, H, T, K), dtype=dtype, device=device)
    k = torch.randn((B, H, T, K), dtype=dtype, device=device)
    v = torch.randn((B, H, T, V), dtype=dtype, device=device)
    s = torch.randn((B, H, T, M), dtype=dtype, device=device)

    ov_triton, _ = chunk_abc_fwd(q, k, v, s, None, False)
    ov_torch, _ = chunk_abc_pytorch_fwd(q, k, v, s, None, False)

    torch.testing.assert_close(ov_triton, ov_torch, rtol=rtol, atol=atol, equal_nan=True)

def prepare_bwd_data():
    torch.manual_seed(42)
    B, H, T, K, V, M = 2, 4, 512, 64, 64, 32
    dtype = torch.bfloat16 if device != 'cpu' else torch.float16

    q = torch.randn((B, H, T, K), dtype=dtype, device=device)
    k = torch.randn((B, H, T, K), dtype=dtype, device=device)
    v = torch.randn((B, H, T, V), dtype=dtype, device=device)
    s = torch.randn((B, H, T, M), dtype=dtype, device=device)

    # forward
    ov, _, intermediates = _chunk_abc_fwd_impl(q, k, v, s, None, False)
    z, ok, p, hk, hv, Av = intermediates
    dov = torch.randn_like(ov)

    # tiling
    BT, BC = 64, 16
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    BM = min(64, triton.next_power_of_2(M))
    NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
    NK, NM = triton.cdiv(K, BK), triton.cdiv(M, BM)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    # ---- Step 1: chunk_abc_bwd_kernel_dh (dhv, NORMK=True) ----
    scale_v = 1.0
    qv = p.to(q.dtype)
    dhv = q.new_empty(B, H, NT * M, V)
    grid_dhv = (NM, triton.cdiv(V, BV), B * H)
    chunk_abc_bwd_kernel_dh[grid_dhv](
        qv, z, dov, dhv,
        scale=scale_v, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
        NORMK=True, num_warps=num_warps, num_stages=num_stages,
    )

    # ---- Step 2: chunk_abc_bwd_kernel_V ----
    dp1 = torch.empty_like(p)
    dsv1 = torch.empty_like(s, dtype=torch.float)
    dv_V = v.new_empty(NM, *v.shape)
    dAv = q.new_zeros(B, H, T, BT)
    grid_V = (NM, NT, B * H)
    chunk_abc_bwd_kernel_V[grid_V](
        s, v, z, hv, Av, dov, dhv, dp1, dsv1, dv_V, dAv,
        scale=scale_v, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT,
        num_warps=num_warps, num_stages=num_stages,
    )
    dv_V = dv_V.sum(0)

    # ---- Step 3: chunk_abc_bwd_kernel_intra_V ----
    dp0 = torch.empty_like(p)
    dsv0 = s.new_zeros(s.shape, dtype=torch.float)
    grid_intra_V = (NM, NT * NC, B * H)
    chunk_abc_bwd_kernel_intra_V[grid_intra_V](
        qv, s, z, dAv, dp0, dsv0,
        T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC,
        num_warps=2, num_stages=num_stages,
    )
    dp = dp1.add_(dp0)
    dsv = dsv1.add_(dsv0)

    # ---- Step 4: softmax_bwd ----
    dok = softmax_bwd(p, dp, dtype=ok.dtype)

    # ---- Step 5: chunk_abc_bwd_kernel_dh (dhk, NORMK=False) ----
    scale_k = K ** -0.5
    dhk = q.new_empty(B, H, NT * K, M)
    grid_dhk = (NK, NM, B * H)
    chunk_abc_bwd_kernel_dh[grid_dhk](
        q, z, dok, dhk,
        scale=scale_k, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
        NORMK=False, num_warps=num_warps, num_stages=num_stages,
    )

    # ---- Step 6: chunk_abc_bwd_kernel_intra_K ----
    dAk = q.new_zeros(NM, B, H, T, BT)
    grid_intra_K = (NM, NT * NC * NC, B * H)
    chunk_abc_bwd_kernel_intra_K[grid_intra_K](
        s, z, dok, dAk,
        scale=scale_k, T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
        num_warps=2, num_stages=num_stages,
    )
    dAk = dAk.sum(0)

    # ---- Step 7: chunk_abc_bwd_kernel_K ----
    Ak = q.new_zeros(NK, B, H, T, BT)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dsk1 = s.new_empty(NK, *s.shape, dtype=torch.float)
    grid_K = (NK, NT, B * H)
    chunk_abc_bwd_kernel_K[grid_K](
        q, k, s, z, hk, Ak, dok, dhk, dq, dk, dsk1, dAk,
        scale=scale_k, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
        num_warps=num_warps, num_stages=num_stages,
    )
    Ak = Ak.sum(0)
    dsk1 = dsk1.sum(0)

    # ---- Step 8: chunk_abc_bwd_kernel_intra_KV ----
    dsk0 = torch.empty_like(s, dtype=torch.float)
    grid_intra_KV = (NM, NT * NC, B * H)  # 原始分核
    chunk_abc_bwd_kernel_intra_KV[grid_intra_KV](
        s, z, Ak, dok, dsk0,
        T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC,
        num_warps=2, num_stages=num_stages,
    )

    # ---- Step 9: chunk_abc_bwd_kernel_rcum_inter ----
    ss = ok * dok + p * dp
    doo = torch.empty_like(s)
    grid_rcum_inter = (NM, B * H)
    chunk_abc_bwd_kernel_rcum_inter[grid_rcum_inter](
        s, z, ss, doo,
        T=T, S=M, BT=BT, BS=BM, NT=NT,
        num_warps=num_warps, num_stages=num_stages,
    )
    doo_inter = doo.clone()

    # ---- Step 10: chunk_abc_bwd_kernel_rcum_intra ----
    grid_rcum_intra = (NM, NT * NC, B * H)
    chunk_abc_bwd_kernel_rcum_intra[grid_rcum_intra](
        s, z, ss, doo,
        T=T, S=M, BT=BT, BC=BC, BS=BM, NC=NC,
        num_warps=num_warps, num_stages=num_stages,
    )

    result = {
        'q': q, 'k': k, 'v': v, 's': s,
        'z': z, 'ok': ok, 'p': p, 'hk': hk, 'hv': hv, 'Av': Av,
        'dov': dov, 'qv': qv,
        'dhv': dhv, 'dp1': dp1, 'dsv1': dsv1, 'dv_V': dv_V, 'dAv': dAv,
        'dp0': dp0, 'dsv0': dsv0, 'dp': dp, 'dsv': dsv,
        'dok': dok, 'dhk': dhk, 'dAk': dAk,
        'Ak': Ak, 'dq': dq, 'dk': dk, 'dsk1': dsk1,
        'dsk0': dsk0, 'ss': ss, 'doo': doo, 'doo_inter': doo_inter,
        'B': B, 'H': H, 'T': T, 'K': K, 'V': V, 'M': M,
        'BT': BT, 'BC': BC, 'BK': BK, 'BV': BV, 'BM': BM,
        'NT': NT, 'NC': NC, 'NK': NK, 'NM': NM,
        'num_warps': num_warps, 'num_stages': num_stages,
        'scale_v': scale_v, 'scale_k': scale_k,
    }
    return result

# https://codehub-y.huawei.com/c00564736/AscendNPU_IR/issues/19
def test_chunk_abc_bwd_kernel_dh(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NM = data['NT'], data['NM']

    # --- Test NORMK=True (dhv) ---
    qv = data['qv']
    z = data['z']
    dov = data['dov']
    dhv_triton = data['dhv']
    dhv_pt = chunk_abc_bwd_kernel_dh_pytorch(
        qv, z, dov, dhv_triton.dtype,
        scale=1.0, T=T, K=M, V=V_dim, BT=BT, BK=BM, BV=BV, NT=NT, NORMK=True,
    )
    torch.testing.assert_close(dhv_pt, dhv_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  dhv: PASS")

    # --- Test NORMK=False (dhk) ---
    q = data['q']
    dok = data['dok']
    dhk_triton = data['dhk']
    dhk_pt = chunk_abc_bwd_kernel_dh_pytorch(
        q, z, dok, dhk_triton.dtype,
        scale=K_dim ** -0.5, T=T, K=K_dim, V=M, BT=BT, BK=BK, BV=BM, NT=NT, NORMK=False,
    )
    torch.testing.assert_close(dhk_pt, dhk_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  dhk: PASS")


def test_chunk_abc_bwd_kernel_V(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NM = data['NT'], data['NM']

    s = data['s']
    v = data['v']
    z = data['z']
    hv = data['hv']
    Av = data['Av']
    dov = data['dov']
    dhv = data['dhv']
    dp1_triton = data['dp1']
    dsv1_triton = data['dsv1']
    dv_triton = data['dv_V']
    dAv_triton = data['dAv']

    dp0 = data['dp0']
    dsv0 = data['dsv0']
    dp1_pt, dsv1_pt, dv_pt, dAv_pt = chunk_abc_bwd_kernel_V_pytorch(
        s, v, z, hv, Av, dov, dhv,
        scale=1.0, T=T, K=M, V=V_dim, BT=BT, BK=BM, BV=BV, NT=NT,
        dp0=dp0, dsv0=dsv0,
    )

    for name, pt, tr in [('dp1', dp1_pt, dp1_triton), ('dsv1', dsv1_pt, dsv1_triton),
                          ('dv', dv_pt, dv_triton), ('dAv', dAv_pt, dAv_triton)]:
        torch.testing.assert_close(pt.to(torch.float32), tr.to(torch.float32), rtol=rtol, atol=atol, equal_nan=True)
        print(f"  {name}: PASS")

# https://codehub-y.huawei.com/c00564736/AscendNPU_IR/issues/12
def test_chunk_abc_bwd_kernel_intra_V(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NC = data['NT'], data['NC']

    qv = data['qv']
    s = data['s']
    z = data['z']
    dAv = data['dAv']
    dp0_triton = data['dp0']
    dsv0_triton = data['dsv0']

    dp0_pt, dsv0_pt = chunk_abc_bwd_kernel_intra_V_pytorch(
        qv, s, z, dAv,
        T=T, K=M, BT=BT, BC=16, BK=BM, NC=NC,
    )

    for name, pt, tr in [('dp0', dp0_pt, dp0_triton), ('dsv0', dsv0_pt, dsv0_triton)]:
        torch.testing.assert_close(pt, tr, rtol=rtol, atol=atol, equal_nan=True)
        print(f"  {name}: PASS")


def test_softmax_bwd_kernel(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    p = data['p']
    dp = data['dp']
    dok_triton = data['dok']

    dok_pt = softmax_bwd_kernel_pytorch(p, dp, dok_triton.dtype)

    torch.testing.assert_close(dok_pt, dok_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  dok: PASS")


def test_chunk_abc_bwd_kernel_intra_K(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NC = data['NT'], data['NC']

    s = data['s']
    z = data['z']
    dok = data['dok']
    dAk_triton = data['dAk']

    dAk_pt = chunk_abc_bwd_kernel_intra_K_pytorch(
        s, z, dok, dAk_triton,
        scale=K_dim ** -0.5, T=T, V=M, BT=BT, BC=16, BV=BM, NC=NC,
    )
    dAk_pt = dAk_pt.sum(0)

    torch.testing.assert_close(dAk_pt, dAk_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  dAk: PASS")


def test_chunk_abc_bwd_kernel_K(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NM = data['NT'], data['NM']

    q = data['q']
    k = data['k']
    s = data['s']
    z = data['z']
    hk = data['hk']
    dok = data['dok']
    dhk = data['dhk']
    dAk = data['dAk']
    dq_triton = data['dq']
    dk_triton = data['dk']
    dsk1_triton = data['dsk1']

    dq_pt, dk_pt, dsk1_pt = chunk_abc_bwd_kernel_K_pytorch(
        q, k, s, z, hk, dok, dhk, dAk,
        scale=K_dim ** -0.5, T=T, K=K_dim, V=M, BT=BT, BK=BK, BV=BM, NT=NT,
    )

    for name, pt, tr in [('dq', dq_pt, dq_triton), ('dk', dk_pt, dk_triton),
                          ('dsk1', dsk1_pt, dsk1_triton)]:
        torch.testing.assert_close(pt.to(torch.float32), tr.to(torch.float32), rtol=rtol, atol=atol, equal_nan=True)
        print(f"  {name}: PASS")


def test_chunk_abc_bwd_kernel_intra_KV(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BK, BV, BM = data['BT'], data['BK'], data['BV'], data['BM']
    NT, NC = data['NT'], data['NC']

    s = data['s']
    z = data['z']
    Ak = data['Ak']
    dok = data['dok']
    dsk0_triton = data['dsk0']

    dsk0_pt = chunk_abc_bwd_kernel_intra_KV_pytorch(
        s, z, Ak, dok,
        T=T, V=M, BT=BT, BC=16, BV=BM, NC=NC,
    )

    torch.testing.assert_close(dsk0_pt, dsk0_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  dsk0: PASS")


def test_chunk_abc_bwd_kernel_rcum_inter(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BM = data['BT'], data['BM']
    NT, NM = data['NT'], data['NM']

    s = data['s']
    z = data['z']
    ss = data['ss']

    doo_inter_triton = torch.empty_like(s)
    from chunk_abc import chunk_abc_bwd_kernel_rcum_inter
    grid = (NM, B * H)
    chunk_abc_bwd_kernel_rcum_inter[grid](
        s, z, ss, doo_inter_triton,
        T=T, S=M, BT=BT, BS=BM, NT=NT,
        num_warps=data['num_warps'], num_stages=data['num_stages'],
    )

    doo_inter_pt = chunk_abc_bwd_kernel_rcum_inter_pytorch(
        s, z, ss,
        T=T, S=M, BT=BT, BS=BM, NT=NT,
    )

    torch.testing.assert_close(doo_inter_pt, doo_inter_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  doo_inter: PASS")


def test_chunk_abc_bwd_kernel_rcum_intra(rtol=5e-3, atol=5e-3):
    data = prepare_bwd_data()
    B, H, T = data['B'], data['H'], data['T']
    K_dim, V_dim, M = data['K'], data['V'], data['M']
    BT, BM = data['BT'], data['BM']
    NT, NC = data['NT'], data['NC']

    s = data['s']
    z = data['z']
    ss = data['ss']
    doo_triton = data['doo']
    doo_inter = data['doo_inter']
    doo_pt = chunk_abc_bwd_kernel_rcum_intra_pytorch(
        s.clone(), z.clone(), ss.clone(), doo_inter.clone(),
        T=T, S=M, BT=BT, BC=16, BS=BM, NC=NC,
    )

    torch.testing.assert_close(doo_pt, doo_triton, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  doo: PASS")

