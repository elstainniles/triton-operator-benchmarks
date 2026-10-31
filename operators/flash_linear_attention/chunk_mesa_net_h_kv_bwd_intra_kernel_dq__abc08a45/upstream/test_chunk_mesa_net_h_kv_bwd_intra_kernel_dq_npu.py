# coding=utf-8
import sys
import os
import triton
import torch
import triton.language as tl
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import is_arch35


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dkv(
    q_star, k, v, beta, h_kv, g, do, dh_kv, dk_beta, dg, dv,
    cu_seqlens, chunk_indices,
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr, IS_VARLEN: tl.constexpr,
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

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_v in tl.static_range(0, NV):
        v_off = i_v * BV
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (v_off, 0), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (v_off, 0), (BV, BK), (0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dg_last += tl.sum(b_h * b_dh) * exp(b_g_last)

        b_ds = tl.dot(b_do, tl.trans(b_v))
        b_dm = b_s * b_ds
        b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)
        b_dg += tl.sum(b_dm, axis=1)
        b_dg -= tl.sum(b_dm, axis=0)

        b_ds = b_ds * b_m
        b_dk_inter = tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
        b_dk += b_dk_inter
        b_dg_last += tl.sum(b_dk_inter * b_k)
        b_dg -= tl.sum(b_dk_inter * b_k, axis=1)

        b_dv = tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
        b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dq(
    q_star, k, v, beta, h_kv, g, do, dq, dg_prev, dg,
    cu_seqlens, chunk_indices,
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr,
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

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg_prev += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg_prev = tl.make_block_ptr(dg_prev, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_ds = tl.dot(b_do, tl.trans(b_v)) * b_m
    b_g_exp_q = exp(b_g)
    b_dq = tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dg = tl.sum(b_dq * b_q, axis=1) + tl.load(p_dg_prev, boundary_check=(0,))
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def naive_chunk_mesa_net_h_kv_bwd_intra_dkv(q_star, k, v, beta, h_kv, g, do, dh_kv, B, T, H, K, V, BT):
    NT = (T + BT - 1) // BT
    dk = torch.zeros(B, T, H, K, dtype=torch.float32)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32)
    dg = torch.zeros(B, T, H, dtype=torch.float32)

    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                i_tg = i_b * NT + i_t
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                bt = t_end - t_start
                if bt <= 0:
                    continue

                q = q_star[i_b, t_start:t_end, i_h].float()
                kk = k[i_b, t_start:t_end, i_h].float()
                vv = v[i_b, t_start:t_end, i_h].float()
                bbeta = beta[i_b, t_start:t_end, i_h].float()
                gg = g[i_b, t_start:t_end, i_h].float()
                ddo = do[i_b, t_start:t_end, i_h].float()
                h = h_kv[i_tg, i_h].float()
                dh = dh_kv[i_tg, i_h].float()

                g_last = gg[-1]

                dg_last = (h * dh).sum() * torch.exp(g_last)

                g_diff = gg[:, None] - gg[None, :]
                m = torch.tril(torch.exp(g_diff))

                k_beta = kk * bbeta[:, None]

                s = q @ k_beta.T * m
                ds_full = ddo @ vv.T
                dm = torch.tril(s * ds_full)
                b_dg = dm.sum(dim=1) - dm.sum(dim=0)

                g_exp_k = torch.exp(-gg + g_last)

                ds = ds_full * m

                dk_inter = vv @ dh.T * g_exp_k[:, None]
                b_dk = dk_inter

                dg_last = dg_last + (dk_inter * k_beta).sum()
                b_dg = b_dg - (dk_inter * k_beta).sum(dim=1)

                b_dv = k_beta @ dh * g_exp_k[:, None] + s.T @ ddo

                b_dk = b_dk + ds.T @ q

                b_dg[-1] = b_dg[-1] + dg_last

                dk[i_b, t_start:t_end, i_h] = b_dk
                dv[i_b, t_start:t_end, i_h] = b_dv
                dg[i_b, t_start:t_end, i_h] = b_dg

    return dk, dv, dg


def naive_chunk_mesa_net_h_kv_bwd_intra_dq(q_star, k, v, beta, h_kv, g, do, dg_prev, B, T, H, K, V, BT):
    NT = (T + BT - 1) // BT
    dq = torch.zeros(B, T, H, K, dtype=torch.float32)
    dg_final = torch.zeros(B, T, H, dtype=torch.float32)

    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                i_tg = i_b * NT + i_t
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                bt = t_end - t_start
                if bt <= 0:
                    continue

                q = q_star[i_b, t_start:t_end, i_h].float()
                kk = k[i_b, t_start:t_end, i_h].float()
                vv = v[i_b, t_start:t_end, i_h].float()
                bbeta = beta[i_b, t_start:t_end, i_h].float()
                gg = g[i_b, t_start:t_end, i_h].float()
                ddo = do[i_b, t_start:t_end, i_h].float()
                h = h_kv[i_tg, i_h].float()
                dg_p = dg_prev[i_b, t_start:t_end, i_h].float()

                g_diff = gg[:, None] - gg[None, :]
                m = torch.tril(torch.exp(g_diff))

                k_beta = kk * bbeta[:, None]

                ds = (ddo @ vv.T) * m
                g_exp_q = torch.exp(gg)

                b_dq = ddo @ h.T * g_exp_q[:, None]
                b_dg = (b_dq * q).sum(dim=1) + dg_p
                b_dq = b_dq + ds @ k_beta

                dq[i_b, t_start:t_end, i_h] = b_dq
                dg_final[i_b, t_start:t_end, i_h] = b_dg

    return dq, dg_final


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


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    return relative_max, relative_mean, rmse


def third_part_cmp(npu_out, gpu_out, cpu_ref, name):
    npu_cpu = npu_out.cpu().float()
    gpu_cpu = gpu_out.cpu().float()
    ref = cpu_ref.float()

    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_cpu, ref)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_cpu, ref)

    max_diff_npu = (npu_cpu - ref).abs().max().item()
    max_diff_gpu = (gpu_cpu - ref).abs().max().item()

    print(f"\n  {name}:")
    print(f"    NPU vs CPU fp32: max_diff={max_diff_npu:.2e}, max_rel_err={relative_max_npu:.4f}, mean_rel_err={relative_mean_npu:.6f}, rmse={rmse_npu:.6f}")
    print(f"    GPU vs CPU fp32: max_diff={max_diff_gpu:.2e}, max_rel_err={relative_max_gpu:.4f}, mean_rel_err={relative_mean_gpu:.6f}, rmse={rmse_gpu:.6f}")
    print(f"    误差比(NPU/GPU) - 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}, 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}, 均方根误差比: {rmse_npu / rmse_gpu:.4f}")

    return relative_max_npu, relative_max_gpu, rmse_npu, rmse_gpu


def run_npu_kernels(data):
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    dkv_bv = 64 if is_arch35() else 32
    dkv_nv = triton.cdiv(input_data['V'], dkv_bv)
    bk = input_data['BK'] if is_arch35() else 64
    bt = input_data['BT'] if is_arch35() else 16

    chunk_mesa_net_h_kv_bwd_intra_kernel_dkv[data["grid"]](
        q_star=input_data['q_star'], k=input_data['k'], v=input_data['v'],
        beta=input_data['beta'], h_kv=input_data['h_kv'], g=input_data['g'],
        do=input_data['do'], dh_kv=input_data['dh_kv'],
        dk_beta=input_data['dk'], dg=input_data['dg'], dv=input_data['dv'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
        B=input_data['B'], T=input_data['T'], H=input_data['H'],
        K=input_data['K'], V=input_data['V'], BT=bt,
        BK=bk, BV=dkv_bv, NV=dkv_nv,
    )

    chunk_mesa_net_h_kv_bwd_intra_kernel_dq[data["grid"]](
        q_star=input_data['q_star'], k=input_data['k'], v=input_data['v'],
        beta=input_data['beta'], h_kv=input_data['h_kv'], g=input_data['g'],
        do=input_data['do'], dq=input_data['dq'],
        dg_prev=input_data['dg'], dg=input_data['dg_final'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
        B=input_data['B'], T=input_data['T'], H=input_data['H'],
        K=input_data['K'], V=input_data['V'], BT=input_data['BT'],
        BK=input_data['BK'], BV=input_data['BV'],
    )

    return input_data


def test_3way_precision():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_h_kv_bwd_intra_kernel_dq/test_chunk_mesa_net_h_kv_bwd_intra_separate_kernel_fp16_v2.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = data["input_data"]
    gpu_output = data["gpu_output"]
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    BT = input_data['BT']

    print(f"Config: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}")
    print(f"Grid: {data['grid']}")

    # 1) CPU fp32 reference
    print("\n[1/3] Computing CPU fp32 reference ...")
    q_star_fp32 = input_data['q_star'].float()
    k_fp32 = input_data['k'].float()
    v_fp32 = input_data['v'].float()
    beta_fp32 = input_data['beta'].float()
    h_kv_fp32 = input_data['h_kv'].float()
    g_fp32 = input_data['g'].float()
    do_fp32 = input_data['do'].float()
    dh_kv_fp32 = input_data['dh_kv'].float()

    cpu_dk, cpu_dv, cpu_dg = naive_chunk_mesa_net_h_kv_bwd_intra_dkv(
        q_star_fp32, k_fp32, v_fp32, beta_fp32, h_kv_fp32, g_fp32, do_fp32, dh_kv_fp32,
        B, T, H, K, V, BT)

    cpu_dq, cpu_dg_final = naive_chunk_mesa_net_h_kv_bwd_intra_dq(
        q_star_fp32, k_fp32, v_fp32, beta_fp32, h_kv_fp32, g_fp32, do_fp32, cpu_dg,
        B, T, H, K, V, BT)

    cpu_ref = {'dk': cpu_dk, 'dv': cpu_dv, 'dg': cpu_dg, 'dq': cpu_dq, 'dg_final': cpu_dg_final}

    # 2) NPU fp16
    print("[2/3] Running NPU fp16 kernels ...")
    npu_data = run_npu_kernels(data)

    # 3) GPU fp16 (from pt file)
    print("[3/3] Loading GPU fp16 outputs from pt file ...")

    print("\n" + "=" * 70)
    print("Three-way precision comparison: NPU fp16 vs GPU fp16 vs CPU fp32")
    print("=" * 70)

    all_npu_better_or_equal = True
    for name in ['dk', 'dv', 'dg', 'dq', 'dg_final']:
        npu_out = npu_data[name].cpu()
        gpu_out = gpu_output[name]
        ref = cpu_ref[name]
        rel_max_npu, rel_max_gpu, rmse_npu, rmse_gpu = third_part_cmp(npu_out, gpu_out, ref, name)
        if rel_max_npu > rel_max_gpu:
            all_npu_better_or_equal = False

    print("\n" + "=" * 70)
    print("Summary (NPU/GPU error ratio, <1.0 means NPU is better):")
    print("=" * 70)
    for name in ['dk', 'dv', 'dg', 'dq', 'dg_final']:
        npu_out = npu_data[name].cpu()
        gpu_out = gpu_output[name]
        ref = cpu_ref[name]
        rel_max_npu, _, rmse_npu = cal_precision(npu_out.float(), ref.float())
        rel_max_gpu, _, rmse_gpu = cal_precision(gpu_out.cpu().float(), ref.float())
        ratio_max = rel_max_npu / rel_max_gpu
        ratio_rmse = rmse_npu / rmse_gpu
        verdict = "NPU <= GPU" if ratio_max <= 1.0 else "NPU > GPU"
        print(f"  {name:10s}: max_rel_err_ratio={ratio_max:.4f}, rmse_ratio={ratio_rmse:.4f}  [{verdict}]")

    return all_npu_better_or_equal


if __name__ == '__main__':
    test_3way_precision('')
