import os

import torch
import triton
import triton.language as tl

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))

@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & (o_q[None, :] - o_k[:, None] < W) & m_q[None, :],
                exp2(b_s - b_lse[None, :]), 0
            )
        else:
            b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WINDOW:
            b_p = tl.where((o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    b_dk = b_dk * scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


LOAD_DIR = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_attn_bwd_kernel_dkv/"
TORCH_CPU_OUTPUTS_NAME = "torch_cpu_outputs.pt"


def _load_torch_cpu_outputs(load_dir: str, device: torch.device):
    torch_cpu_path = os.path.join(load_dir, TORCH_CPU_OUTPUTS_NAME)
    if not os.path.exists(torch_cpu_path):
        raise FileNotFoundError(
            f"{torch_cpu_path} does not exist. Run "
            "test_parallel_attn_bwd_kernel_dkv_npu_mod2.py first."
        )
    outputs = torch.load(torch_cpu_path, map_location="cpu")
    return outputs["dk"].to(device), outputs["dv"].to(device)


def _check_same_shape(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"{name} shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")


def test_parallel_attn_bwd_kernel_dkv():
    inputs_dict = torch.load(os.path.join(LOAD_DIR, "inputs.pt"))

    device = torch.device("npu:0")
    q = inputs_dict["q"].to(device)
    k = inputs_dict["k"].to(device)
    v = inputs_dict["v"].to(device)
    lse = inputs_dict["lse"].to(device)
    delta = inputs_dict["delta"].to(device)
    do = inputs_dict["do"].to(device)
    dk = inputs_dict["dk"].to(device)
    dv = inputs_dict["dv"].to(device)
    dg_cumsum = inputs_dict["dg_cumsum"]
    g_cumsum = inputs_dict["g_cumsum"]
    scale = inputs_dict["scale"]
    B = inputs_dict["B"]
    T = inputs_dict["T"]
    H = inputs_dict["H"]
    HQ = inputs_dict["HQ"]
    G = inputs_dict["G"]
    K = inputs_dict["K"]
    V = inputs_dict["V"]
    BT = inputs_dict["BT"]
    BS = inputs_dict["BS"]
    BK = inputs_dict["BK"]
    BV = inputs_dict["BV"]

    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * HQ)

    print("\nExecuting kernel...")
    parallel_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        T=T,
        W=None,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=4,
        enable_dynamic_cv_pipeline=False,
    )

    outputs_dict = torch.load(os.path.join(LOAD_DIR, "outputs.pt")) # os.path.join(LOAD_DIR, "outputs.pt")
    dk_gpu_fp64 = outputs_dict["dk_fp64"].to(device)
    dv_gpu_fp64 = outputs_dict["dv_fp64"].to(device)
    dk_torch_cpu, dv_torch_cpu = _load_torch_cpu_outputs(LOAD_DIR, device)

    _check_same_shape("dk vs dk_fp64", dk, dk_gpu_fp64)
    _check_same_shape("dv vs dv_fp64", dv, dv_gpu_fp64)
    _check_same_shape("dk_torch_cpu vs dk_fp64", dk_torch_cpu, dk_gpu_fp64)
    _check_same_shape("dv_torch_cpu vs dv_fp64", dv_torch_cpu, dv_gpu_fp64)

    print("\nComparing outputs with expected values...")
    print("Comparing dk:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk, dk_gpu_fp64)
    relative_max_torch_cpu, relative_mean_torch_cpu, rmse_torch_cpu = cal_precision(
        dk_torch_cpu, dk_gpu_fp64
    )
    print(
        f"最大相对误差比 :{relative_max_npu / relative_max_torch_cpu}, "
        f"平均相对误差比 :{relative_mean_npu / relative_mean_torch_cpu}, "
        f"均方根误差比 :{rmse_npu / rmse_torch_cpu}"
    )
    assert relative_max_npu / relative_max_torch_cpu <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_torch_cpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_torch_cpu <= 1.2, "dk 均方根误差比 check failed"

    print("Comparing dv:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv, dv_gpu_fp64)
    relative_max_torch_cpu, relative_mean_torch_cpu, rmse_torch_cpu = cal_precision(
        dv_torch_cpu, dv_gpu_fp64
    )
    print(
        f"最大相对误差比 :{relative_max_npu / relative_max_torch_cpu}, "
        f"平均相对误差比 :{relative_mean_npu / relative_mean_torch_cpu}, "
        f"均方根误差比 :{rmse_npu / rmse_torch_cpu}"
    )
    assert relative_max_npu / relative_max_torch_cpu <= 2, "dv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_torch_cpu <= 1.2, "dv 平均相对误差比 check failed"
    assert rmse_npu / rmse_torch_cpu <= 1.2, "dv 均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == "__main__":
    test_parallel_attn_bwd_kernel_dkv()