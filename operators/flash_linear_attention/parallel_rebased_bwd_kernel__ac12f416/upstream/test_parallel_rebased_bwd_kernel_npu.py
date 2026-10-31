import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional


@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c*BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c*BTL + tl.arange(0, BTL)) < T)

    for _ in range(0, i_c*BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, i_c*BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dkv(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)

    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        # [BK, BTS]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BV, BTS]
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        # [BTL, BTS]
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))  # [BD, BQ]
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        # [BK, BQ]
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        # [BK, BD]
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit(do_not_specialize=['T'])
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dq,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dk,
        dv,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )



def parallel_rebased_bwd_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dz: torch.Tensor,
    scale: float,
    B: int,
    H: int,
    T: int,
    K: int,
    V_dim: int,
    NK: int,
    NV: int,
    BK: int,
    BV: int,
):
    q_fp64 = q.to(torch.float64)
    k_fp64 = k.to(torch.float64)
    v_fp64 = v.to(torch.float64)
    do_fp64 = do.to(torch.float64)
    dz_fp64 = dz.to(torch.float64)

    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))

    dq_parts = []
    dk_parts = []
    dv_parts = []

    for i_v in range(NV):
        v_start = i_v * BV
        v_end = min(v_start + BV, V_dim)
        do_v = do_fp64[:, :, :, v_start:v_end]
        v_v = v_fp64[:, :, :, v_start:v_end]
        dz_v = dz_fp64 if i_v == 0 else None

        dq_i = torch.zeros(B, H, T, K, dtype=torch.float64, device=q.device)
        for i_k in range(NK):
            k_start = i_k * BK
            k_end = min(k_start + BK, K)
            q_k = q_fp64[:, :, :, k_start:k_end]
            k_k = k_fp64[:, :, :, k_start:k_end]

            s = torch.einsum('bhti,bhji->bhtj', q_k * scale, k_k)
            s2 = s * s
            s2_causal = s2 * causal_mask

            ds = torch.einsum('bhtv,bhjv->bhtj', do_v, v_v)
            if i_v == 0:
                ds = ds + dz_v.unsqueeze(-1)
            ds_causal = ds * causal_mask * scale

            dq_k = 2 * torch.einsum('bhtj,bhji->bhti', ds_causal * s, k_k) * scale
            dq_i[:, :, :, k_start:k_end] += dq_k

        dq_parts.append(dq_i)

    for i_v in range(NV):
        v_start = i_v * BV
        v_end = min(v_start + BV, V_dim)
        do_v = do_fp64[:, :, :, v_start:v_end]
        v_v = v_fp64[:, :, :, v_start:v_end]

        dk_i = torch.zeros(B, H, T, K, dtype=torch.float64, device=q.device)
        dv_i = torch.zeros(B, H, T, V_dim, dtype=torch.float64, device=q.device)

        for i_k in range(NK):
            k_start = i_k * BK
            k_end = min(k_start + BK, K)
            k_k = k_fp64[:, :, :, k_start:k_end]

            s = torch.einsum('bhti,bhji->bhtj', q_fp64 * scale, k_k)
            s2 = s * s
            s2_causal = s2 * causal_mask

            dv_k = torch.einsum('bhtj,bhtv->bhjv', s2_causal, do_v)
            dv_i[:, :, :, v_start:v_end] += dv_k

            ds = torch.einsum('bhtv,bhjv->bhtj', do_v, v_v)
            if i_v == 0:
                ds = ds + dz_fp64.unsqueeze(-1)
            ds_causal = ds * causal_mask * scale

            dk_k = 2 * torch.einsum('bhtj,bhti->bhji', ds_causal * s, q_fp64)
            dk_i[:, :, :, k_start:k_end] += dk_k

        dk_parts.append(dk_i)
        dv_parts.append(dv_i)

    dq_fp64 = torch.stack(dq_parts, dim=0)
    dk_fp64 = torch.stack(dk_parts, dim=0)
    dv_fp64 = torch.stack(dv_parts, dim=0)

    return dq_fp64, dk_fp64, dv_fp64


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_parallel_rebased_bwd_kernel():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_rebased_bwd_kernel/test_parallel_rebased_bwd_kernel_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = data["input_data"]
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    do_tensor = input_data['do']
    dz = input_data['dz']
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V_dim = input_data['V']
    BTL = input_data['BTL']
    BTS = input_data['BTS']
    BK = input_data['BK']
    BV = input_data['BV']
    NK = input_data['NK']
    NV = input_data['NV']
    scale = input_data['scale']

    dq_fp64, dk_fp64, dv_fp64 = parallel_rebased_bwd_ref(
        q=q, k=k, v=v, do=do_tensor, dz=dz, scale=scale,
        B=B, H=H, T=T, K=K, V_dim=V_dim, NK=NK, NV=NV, BK=BK, BV=BV,
    )

    device = torch.device('npu')
    q_npu = q.to(device)
    k_npu = k.to(device)
    v_npu = v.to(device)
    do_npu = do_tensor.to(device)
    dz_npu = dz.to(device)

    dq = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=device)
    dk = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=device)
    dv = torch.empty(NK, B, H, T, V_dim, dtype=q.dtype, device=device)

    grid = (NK * NV, triton.cdiv(T, BTL), B * H)
    num_warps = 4
    num_stages = 2

    parallel_rebased_bwd_kernel[grid](
        q_npu,
        k_npu,
        v_npu,
        do_npu,
        dz_npu,
        dq,
        dk,
        dv,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V_dim,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dq_npu = dq.cpu()
    dk_npu = dk.cpu()
    dv_npu = dv.cpu()

    gpu_output = data["gpu_output"]
    dq_gpu = gpu_output['dq']
    dk_gpu = gpu_output['dk']
    dv_gpu = gpu_output['dv']

    dq_fp64_f16 = dq_fp64.to(q.dtype)
    dk_fp64_f16 = dk_fp64.to(q.dtype)
    dv_fp64_f16 = dv_fp64.to(q.dtype)

    print("\n=== Comparing dq ===")
    print("--- NPU vs fp64 ---")
    dq_rel_max_npu, dq_rel_mean_npu, dq_rmse_npu = cal_precision(dq_npu.to(torch.float32), dq_fp64_f16.to(torch.float32))
    print("--- GPU vs fp64 ---")
    dq_rel_max_gpu, dq_rel_mean_gpu, dq_rmse_gpu = cal_precision(dq_gpu.to(torch.float32), dq_fp64_f16.to(torch.float32))
    print(f"dq 最大相对误差比: {dq_rel_max_npu / dq_rel_max_gpu}, 平均相对误差比: {dq_rel_mean_npu / dq_rel_mean_gpu}, 均方根误差比: {dq_rmse_npu / dq_rmse_gpu}")
    assert dq_rel_max_npu / dq_rel_max_gpu <= 2, "dq 最大相对误差比 check failed"
    assert dq_rel_mean_npu / dq_rel_mean_gpu <= 1.2, "dq 平均相对误差比 check failed"
    assert dq_rmse_npu / dq_rmse_gpu <= 1.2, "dq 均方根误差比 check failed"

    print("\n=== Comparing dk ===")
    print("--- NPU vs fp64 ---")
    dk_rel_max_npu, dk_rel_mean_npu, dk_rmse_npu = cal_precision(dk_npu.to(torch.float32), dk_fp64_f16.to(torch.float32))
    print("--- GPU vs fp64 ---")
    dk_rel_max_gpu, dk_rel_mean_gpu, dk_rmse_gpu = cal_precision(dk_gpu.to(torch.float32), dk_fp64_f16.to(torch.float32))
    print(f"dk 最大相对误差比: {dk_rel_max_npu / dk_rel_max_gpu}, 平均相对误差比: {dk_rel_mean_npu / dk_rel_mean_gpu}, 均方根误差比: {dk_rmse_npu / dk_rmse_gpu}")
    assert dk_rel_max_npu / dk_rel_max_gpu <= 2, "dk 最大相对误差比 check failed"
    assert dk_rel_mean_npu / dk_rel_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert dk_rmse_npu / dk_rmse_gpu <= 1.2, "dk 均方根误差比 check failed"

    print("\n=== Comparing dv ===")
    print("--- NPU vs fp64 ---")
    dv_rel_max_npu, dv_rel_mean_npu, dv_rmse_npu = cal_precision(dv_npu.to(torch.float32), dv_fp64_f16.to(torch.float32))
    print("--- GPU vs fp64 ---")
    dv_rel_max_gpu, dv_rel_mean_gpu, dv_rmse_gpu = cal_precision(dv_gpu.to(torch.float32), dv_fp64_f16.to(torch.float32))
    print(f"dv 最大相对误差比: {dv_rel_max_npu / dv_rel_max_gpu}, 平均相对误差比: {dv_rel_mean_npu / dv_rel_mean_gpu}, 均方根误差比: {dv_rmse_npu / dv_rmse_gpu}")
    assert dv_rel_max_npu / dv_rel_max_gpu <= 2, "dv 最大相对误差比 check failed"
    assert dv_rel_mean_npu / dv_rel_mean_gpu <= 1.2, "dv 平均相对误差比 check failed"
    assert dv_rmse_npu / dv_rmse_gpu <= 1.2, "dv 均方根误差比 check failed"

    print("\nTest parallel_rebased_bwd_kernel passed!")


if __name__ == "__main__":
    test_parallel_rebased_bwd_kernel()
