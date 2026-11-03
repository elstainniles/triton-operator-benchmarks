import torch

import math

def chunk_update_once_pt(b_p, b_k, b_v, b_m, b_g_exp_q, b_h, b_lamb, use_bf16_matmul=False):
    if use_bf16_matmul:
        def matmul_bf16(a, b):
            return torch.matmul(a.bfloat16().float(), b.bfloat16().float())
        b_o = matmul_bf16(b_p, b_k.transpose(-2, -1)) * b_m
        b_o = matmul_bf16(b_o, b_v)
        b_o = b_o + matmul_bf16(b_p * b_g_exp_q, b_h)
    else:
        b_o = (b_p @ b_k.transpose(-2, -1)) * b_m
        b_o = b_o @ b_v
        b_o = b_o + (b_p * b_g_exp_q) @ b_h
    if b_lamb is not None:
        b_o = b_o + b_lamb.unsqueeze(0) * b_p
    return b_o

def chunk_mesa_cg_fwd_pt(
    q,
    k,
    v,
    h,
    h_kv,
    g,
    beta,
    lamb,
    cu_seqlens=None,
    chunk_size=64,
    max_CG_iteration=30,
    output_dtype=None,
    use_bf16_matmul=False,
):
    B, T, H, K = q.shape
    NT = math.ceil(T / chunk_size)
    compute_dtype = q.dtype

    q_final = torch.zeros(B, T, H, K, dtype=compute_dtype, device=q.device)
    o = torch.zeros(B, T, H, K, dtype=compute_dtype, device=q.device)

    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                bos = i_b * T
                i_tg = i_b * NT + i_t

                t_start = i_t * chunk_size
                t_end = min(t_start + chunk_size, T)
                actual_bt = t_end - t_start

                b_q = q[i_b, t_start:t_end, i_h, :].to(torch.float32)
                b_k_block = k[i_b, t_start:t_end, i_h, :].to(torch.float32)
                b_h_block = h[i_tg, i_h, :, :].to(torch.float32)
                b_g = g[i_b, t_start:t_end, i_h].to(torch.float32)
                b_beta = beta[i_b, t_start:t_end, i_h].to(torch.float32)
                b_lamb = lamb[i_h, :].to(torch.float32)

                o_t = torch.arange(t_start, t_end, device=q.device)
                m_t = o_t < T

                b_m = torch.exp(b_g[:, None] - b_g[None, :]) * b_beta[None, :]
                causal_mask = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
                b_m = torch.where(causal_mask, b_m, torch.zeros_like(b_m))
                b_g_exp_q = torch.exp(b_g)[:, None]

                b_x = torch.zeros(actual_bt, K, dtype=torch.float32, device=q.device)
                b_r = b_q.clone()
                b_p = b_r.clone()
                b_delta_old = (b_r * b_r).sum(dim=1)

                for _ in range(max_CG_iteration):
                    b_o_iter = chunk_update_once_pt(b_p, b_k_block, b_k_block, b_m, b_g_exp_q, b_h_block, b_lamb, use_bf16_matmul)
                    alpha = b_delta_old / ((b_p * b_o_iter).sum(dim=1) + 1e-5)
                    b_x = b_x + alpha[:, None] * b_p
                    b_r = b_r - alpha[:, None] * b_o_iter
                    b_delta_new = (b_r * b_r).sum(dim=1)
                    b_p = b_r + (b_delta_new / (b_delta_old + 1e-5))[:, None] * b_p
                    b_delta_old = b_delta_new

                q_final[i_b, t_start:t_end, i_h, :] = b_x.to(compute_dtype)

                b_h_kv = h_kv[i_tg, i_h, :, :].to(torch.float32)
                b_v_block = v[i_b, t_start:t_end, i_h, :].to(torch.float32)
                b_o_out = chunk_update_once_pt(b_x, b_k_block, b_v_block, b_m, b_g_exp_q, b_h_kv, None, use_bf16_matmul)
                o[i_b, t_start:t_end, i_h, :] = b_o_out.to(compute_dtype)

    return q_final, o
