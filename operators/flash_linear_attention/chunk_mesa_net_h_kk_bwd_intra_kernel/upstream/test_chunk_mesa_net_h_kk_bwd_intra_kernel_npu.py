# coding=utf-8
import sys
import os
import pytest
import triton
import torch
import torch.nn.functional as F
import triton.language as tl
import torch_npu
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kk_bwd_intra_kernel(
    k,
    beta,
    h,
    dh,
    g,
    q_star,
    dq,
    dk,
    dg,
    dbeta,
    dk_beta,
    dlamb,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    q_star += (bos * H + i_h) * V
    dq += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    k += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dlamb += (i_tg * H + i_h).to(tl.int64) * K
    beta += (bos * H + i_h)
    dbeta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_gk = tl.where(m_t, exp(b_g_last - b_g), 0)

    p_q_star = tl.make_block_ptr(q_star, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q_star = tl.load(p_q_star, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))
    b_dlamb = -tl.sum(b_q_star * b_dq, axis=0)
    p_dlamb = tl.make_block_ptr(dlamb, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_dlamb, b_dlamb.to(p_dlamb.dtype.element_ty), boundary_check=(0,))

    p_h = tl.make_block_ptr(h, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_v = tl.load(p_k, boundary_check=(0, 1))
    b_k = (b_v * b_beta[:, None]).to(b_v.dtype)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_s = tl.dot(b_q_star, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_dq, tl.trans(b_v))
    b_dv += tl.dot(tl.trans(b_s.to(b_dq.dtype)), b_dq)
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_ds = b_ds * b_m
    b_dk += tl.dot(tl.trans(b_ds.to(b_q_star.dtype)), b_q_star)

    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dg += tl.sum(tl.dot(b_dq, tl.trans(b_h)) * tl.exp(b_g)[:, None] * b_q_star, axis=1)
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_dk2 = tl.dot(b_v, b_dh.to(b_v.dtype)) * b_gk[:, None]
    b_dg -= tl.sum(b_dk2 * b_k, axis=1)
    b_dg_last += tl.sum(b_dk2 * b_k)
    b_dk += b_dk2
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_gk[:, None]
    b_dh = b_dh * b_h
    b_dg_last += tl.sum(b_dh) * exp(b_g_last)

    p_dk_beta = tl.make_block_ptr(dk_beta, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dk -= tl.load(p_dk_beta, boundary_check=(0, 1))
    b_dbeta = tl.sum(b_dk * b_v, axis=1)
    b_dk = b_dk * b_beta[:, None] + b_dv
    b_dk = -b_dk

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, -b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    p_dbeta = tl.make_block_ptr(dbeta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, -b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def naive_chunk_mesa_net_h_kk_bwd_intra(k, beta, g, h, dh, q_star, dq, dk_beta, chunk_size=64):
    B, T, H, D = k.shape
    k = k.float().clone()
    beta = beta.float().clone()
    g = g.float().clone()
    h = h.float().clone()
    dh = dh.float().clone()
    q_star = q_star.float().clone()
    dq = dq.float().clone()
    dk_beta = dk_beta.float().clone()

    NT = triton.cdiv(T, chunk_size)
    dk = torch.zeros(B, T, H, D, dtype=torch.float32)
    dg = torch.zeros(B, T, H, dtype=torch.float32)
    dbeta = torch.zeros(B, T, H, dtype=torch.float32)
    dlamb = torch.zeros(B, NT, H, D, dtype=torch.float32)

    for b in range(B):
        for h_idx in range(H):
            for t in range(NT):
                t_start = t * chunk_size
                t_end = min(t_start + chunk_size, T)
                BT = t_end - t_start

                g_chunk = g[b, t_start:t_end, h_idx]
                g_last = g_chunk[-1]
                gk = torch.where(torch.arange(BT) < BT, torch.exp(g_last - g_chunk), 0.0)

                q_star_chunk = q_star[b, t_start:t_end, h_idx, :]
                dq_chunk = dq[b, t_start:t_end, h_idx, :]
                k_chunk = k[b, t_start:t_end, h_idx, :]
                beta_chunk = beta[b, t_start:t_end, h_idx]

                dlamb[b, t, h_idx, :] = -(q_star_chunk * dq_chunk).sum(dim=0)

                v_chunk = k_chunk
                k_beta_chunk = k_chunk * beta_chunk.unsqueeze(-1)

                h_mat = h[t + b * NT, h_idx, :, :]
                dh_mat = dh[t + b * NT, h_idx, :, :]

                m = torch.zeros(BT, BT)
                for i in range(BT):
                    for j in range(i + 1):
                        m[i, j] = math.exp(g_chunk[i].item() - g_chunk[j].item())

                s = q_star_chunk @ k_beta_chunk.T * m
                ds = dq_chunk @ v_chunk.T
                dv = s.T @ dq_chunk
                dm = s * ds
                dm = torch.tril(dm)
                dg_chunk = dm.sum(dim=1) - dm.sum(dim=0)
                ds = ds * m
                dk_chunk = ds.T @ q_star_chunk

                dg_chunk += (dq_chunk @ h_mat * torch.exp(g_chunk).unsqueeze(-1) * q_star_chunk).sum(dim=1)

                dk2 = v_chunk @ dh_mat.T * gk.unsqueeze(-1)
                dg_chunk -= (dk2 * k_beta_chunk).sum(dim=1)
                dg_last = (dk2 * k_beta_chunk).sum()
                dk_chunk += dk2
                dv += k_beta_chunk @ dh_mat * gk.unsqueeze(-1)

                dh_prod = dh_mat * h_mat
                dg_last += dh_prod.sum() * math.exp(g_last.item())

                dk_chunk -= dk_beta[b, t_start:t_end, h_idx, :]
                dbeta_chunk = (dk_chunk * v_chunk).sum(dim=1)
                dk_chunk = dk_chunk * beta_chunk.unsqueeze(-1) + dv
                dk_chunk = -dk_chunk

                if BT > 0:
                    dg_chunk[-1] += dg_last

                dk[b, t_start:t_end, h_idx, :] = dk_chunk
                dg[b, t_start:t_end, h_idx] = -dg_chunk
                dbeta[b, t_start:t_end, h_idx] = -dbeta_chunk

    return dk, dg, dbeta, dlamb


def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata


def cal_precision(input: torch.Tensor, golden: torch.Tensor, name: str = ""):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"{name} - max relative error: {relative_max:.6f}")
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"{name} - mean relative error: {relative_mean:.6f}")
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f"{name} - rmse: {rmse:.6f}")
    return relative_max, relative_mean, rmse


def third_part_cmp(y_npu, y_gpu, ref, name):
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(y_npu.to("cpu").float(), ref, f"{name} NPU")
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(y_gpu.to("cpu").float(), ref, f"{name} GPU")
    print(
        f"{name} - 最大相对误差比(NPU/GPU): {relative_max_npu / relative_max_gpu:.4f}, "
        f"平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}, "
        f"均方根误差比: {rmse_npu / rmse_gpu:.4f}")


def generate_test_data(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)

    B, T, H, D = 2, 512, 8, 64
    chunk_size = 64
    IS_VARLEN = False

    k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    g = torch.rand(B, T, H, dtype=dtype, device=device).float().uniform_(0.8, 0.99).log()

    NT = triton.cdiv(T, chunk_size)
    h = torch.randn(NT * B, H, D, D, dtype=dtype, device=device)
    dh = torch.randn(NT * B, H, D, D, dtype=dtype, device=device)

    q_star = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dq = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dk_beta = torch.randn(B, T, H, D, dtype=dtype, device=device)

    dk = torch.empty(B, T, H, D, dtype=k.dtype, device=device)
    dg = torch.empty(B, T, H, dtype=g.dtype, device=device)
    dbeta = torch.empty(B, T, H, dtype=beta.dtype, device=device)
    dlamb = torch.empty(B, NT, H, D, dtype=torch.float32, device=device)

    cu_seqlens = None
    chunk_indices = None

    input_data = {
        'k': k,
        'beta': beta,
        'g': g,
        'h': h,
        'dh': dh,
        'q_star': q_star,
        'dq': dq,
        'dk_beta': dk_beta,
        'dk': dk,
        'dg': dg,
        'dbeta': dbeta,
        'dlamb': dlamb,
        'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.empty(0, dtype=torch.int32),
        'chunk_indices': chunk_indices if chunk_indices is not None else torch.empty(0, dtype=torch.int32),
        'B': B,
        'T': T,
        'H': H,
        'K': D,
        'V': D,
        'BT': chunk_size,
        'BK': max(triton.next_power_of_2(D), 16),
        'BV': max(triton.next_power_of_2(D), 16),
        'IS_VARLEN': IS_VARLEN,
    }

    grid = (NT, B * H)

    return input_data, grid


def test_chunk_mesa_net_h_kk_bwd_intra_kernel_npu_thirdparty():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_h_kk_bwd_intra_kernel/test_chunk_mesa_net_h_kk_bwd_intra_kernel_fp16_v2.pt'
    if not os.path.exists(ptfile_path):
        pytest.skip(f"{ptfile_path} not found")

    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data_cpu = data["input_data"]
    gpu_output = data["gpu_output"]

    # 1> CPU fp32 高精度参考计算
    print("=" * 70)
    print("Step 1: CPU fp32 reference computation (naive PyTorch)")
    print("=" * 70)
    dk_ref, dg_ref, dbeta_ref, dlamb_ref = naive_chunk_mesa_net_h_kk_bwd_intra(
        k=input_data_cpu['k'],
        beta=input_data_cpu['beta'],
        g=input_data_cpu['g'],
        h=input_data_cpu['h'],
        dh=input_data_cpu['dh'],
        q_star=input_data_cpu['q_star'],
        dq=input_data_cpu['dq'],
        dk_beta=input_data_cpu['dk_beta'],
        chunk_size=input_data_cpu['BT'],
    )
    print("CPU reference computation done.")
    for name, ref in [('dk', dk_ref), ('dg', dg_ref), ('dbeta', dbeta_ref), ('dlamb', dlamb_ref)]:
        print(f"  {name}: dtype={ref.dtype}, range=[{ref.min():.4f}, {ref.max():.4f}]")

    # 2> NPU fp16 计算
    print("\n" + "=" * 70)
    print("Step 2: NPU fp16 computation (Triton kernel)")
    print("=" * 70)
    input_data_npu = convert_tensor_with_device_type(input_data_cpu, device_type='npu')
    chunk_mesa_net_h_kk_bwd_intra_kernel[data["grid"]](
        k=input_data_npu['k'],
        h=input_data_npu['h'],
        dh=input_data_npu['dh'],
        g=input_data_npu['g'],
        q_star=input_data_npu['q_star'],
        beta=input_data_npu['beta'],
        dbeta=input_data_npu['dbeta'],
        dq=input_data_npu['dq'],
        dk=input_data_npu['dk'],
        dk_beta=input_data_npu['dk_beta'],
        dg=input_data_npu['dg'],
        dlamb=input_data_npu['dlamb'],
        cu_seqlens=None,
        chunk_indices=None,
        B=input_data_npu['B'],
        T=input_data_npu['T'],
        H=input_data_npu['H'],
        K=input_data_npu['K'],
        V=input_data_npu['V'],
        BT=input_data_npu['BT'],
        BK=input_data_npu['BK'],
        BV=input_data_npu['BV'],
    )
    print("NPU computation done.")

    # 3> 三方精度对比: NPU vs GPU vs CPU(fp32)
    print("\n" + "=" * 70)
    print("Step 3: Three-way precision comparison (NPU fp16 vs GPU fp16 vs CPU fp32)")
    print("=" * 70)

    for name, ref in [('dk', dk_ref), ('dg', dg_ref), ('dbeta', dbeta_ref), ('dlamb', dlamb_ref)]:
        npu_out = input_data_npu[name].cpu()
        gpu_out = gpu_output[name]
        print(f"\n--- {name} (dtype: GPU={gpu_out.dtype}, CPU={ref.dtype}) ---")
        third_part_cmp(npu_out, gpu_out, ref, name)


if __name__ == '__main__':
    test_chunk_mesa_net_h_kk_bwd_intra_kernel_npu_thirdparty()
