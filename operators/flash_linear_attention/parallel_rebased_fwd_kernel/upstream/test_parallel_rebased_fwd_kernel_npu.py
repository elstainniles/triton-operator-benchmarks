import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional


@triton.jit(do_not_specialize=['T'])
def parallel_rebased_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
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
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v*BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c*BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, i_c*BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c*BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c*BTL + tl.arange(0, BTL)) < T))



def parallel_rebased_fwd_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    B: int,
    H: int,
    T: int,
    K: int,
    V_dim: int,
    NK: int,
    BK: int,
):
    q_fp64 = q.to(torch.float64)
    k_fp64 = k.to(torch.float64)
    v_fp64 = v.to(torch.float64)

    o_parts = []
    z_parts = []

    for i_k in range(NK):
        k_start = i_k * BK
        k_end = min(k_start + BK, K)
        q_part = q_fp64[:, :, :, k_start:k_end]
        k_part = k_fp64[:, :, :, k_start:k_end]

        s = torch.einsum('bhti,bhji->bhtj', q_part * scale, k_part)
        s = s * s

        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        s = s * causal_mask

        z_part = s.sum(dim=-1)
        o_part = torch.einsum('bhtj,bhjv->bhtv', s, v_fp64)

        o_parts.append(o_part)
        z_parts.append(z_part)

    o_fp64 = torch.stack(o_parts, dim=0)
    z_fp64 = torch.stack(z_parts, dim=0)

    return o_fp64, z_fp64


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


def test_parallel_rebased_fwd_kernel():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_rebased_fwd_kernel/test_parallel_rebased_fwd_kernel_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = data["input_data"]
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
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

    o_fp64, z_fp64 = parallel_rebased_fwd_ref(
        q=q, k=k, v=v, scale=scale,
        B=B, H=H, T=T, K=K, V_dim=V_dim, NK=NK, BK=BK,
    )

    device = torch.device('npu')
    q_npu = q.to(device)
    k_npu = k.to(device)
    v_npu = v.to(device)

    o = torch.empty(NK, B, H, T, V_dim, dtype=torch.float32, device=device)
    z = torch.empty(NK, B, H, T, dtype=torch.float32, device=device)

    num_stages = 2
    num_warps = 4
    grid = (NK * NV, triton.cdiv(T, BTL), B * H)

    parallel_rebased_fwd_kernel[grid](
        q_npu,
        k_npu,
        v_npu,
        o,
        z,
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

    o_npu = o.cpu()
    z_npu = z.cpu()

    gpu_output = data["gpu_output"]
    o_gpu = gpu_output['o']
    z_gpu = gpu_output['z']

    o_fp64_f32 = o_fp64.to(torch.float32)
    z_fp64_f32 = z_fp64.to(torch.float32)

    print("\n=== Comparing o ===")
    print("--- NPU vs fp64 ---")
    o_rel_max_npu, o_rel_mean_npu, o_rmse_npu = cal_precision(o_npu, o_fp64_f32)
    print("--- GPU vs fp64 ---")
    o_rel_max_gpu, o_rel_mean_gpu, o_rmse_gpu = cal_precision(o_gpu, o_fp64_f32)
    print(f"o 最大相对误差比: {o_rel_max_npu / o_rel_max_gpu}, 平均相对误差比: {o_rel_mean_npu / o_rel_mean_gpu}, 均方根误差比: {o_rmse_npu / o_rmse_gpu}")
    assert o_rel_max_npu / o_rel_max_gpu <= 2, "o 最大相对误差比 check failed"
    assert o_rel_mean_npu / o_rel_mean_gpu <= 1.2, "o 平均相对误差比 check failed"
    assert o_rmse_npu / o_rmse_gpu <= 1.2, "o 均方根误差比 check failed"

    print("\n=== Comparing z ===")
    print("--- NPU vs fp64 ---")
    z_rel_max_npu, z_rel_mean_npu, z_rmse_npu = cal_precision(z_npu, z_fp64_f32)
    print("--- GPU vs fp64 ---")
    z_rel_max_gpu, z_rel_mean_gpu, z_rmse_gpu = cal_precision(z_gpu, z_fp64_f32)
    print(f"z 最大相对误差比: {z_rel_max_npu / z_rel_max_gpu}, 平均相对误差比: {z_rel_mean_npu / z_rel_mean_gpu}, 均方根误差比: {z_rmse_npu / z_rmse_gpu}")
    assert z_rel_max_npu / z_rel_max_gpu <= 2, "z 最大相对误差比 check failed"
    assert z_rel_mean_npu / z_rel_mean_gpu <= 1.2, "z 平均相对误差比 check failed"
    assert z_rmse_npu / z_rmse_gpu <= 1.2, "z 均方根误差比 check failed"

    print("\nTest parallel_rebased_fwd_kernel passed!")


if __name__ == "__main__":
    test_parallel_rebased_fwd_kernel()
