"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def naive_chunk_gsa_bwd_k_dqkvg(q, k, v, g, h, do, dh, dA, dg, scale, BT, dtype=torch.float64):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    NG = HQ // H
    A_out = torch.zeros(B, T, HQ, BT, dtype=dtype)
    dq_out = torch.zeros(B, T, HQ, K, dtype=dtype)
    dk_out = torch.zeros(B, T, HQ, K, dtype=dtype)
    dv_out = torch.zeros(B, T, HQ, V, dtype=dtype)
    dgv_out = torch.zeros(B, T, HQ, V, dtype=dtype)
    for i_b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_bt = t_end - t_start
            o_t = t_end
            for i_hq in range(HQ):
                i_h = i_hq // NG
                q_block = q[i_b, t_start:t_end, i_hq, :]
                k_block = k[i_b, t_start:t_end, i_h, :]
                v_block = v[i_b, t_start:t_end, i_h, :]
                g_block = g[i_b, t_start:t_end, i_h, :]
                do_block = do[i_b, t_start:t_end, i_hq, :]
                dA_block = dA[i_b, t_start:t_end, i_hq, :actual_bt]
                dg_block = dg[i_b, t_start:t_end, i_hq, :]
                h_block = h[i_b, i_t, i_h, :, :]
                dh_block = dh[i_b, i_t, i_hq, :, :]
                g_last = g[i_b, o_t - 1, i_h, :]
                gv = torch.exp2(g_last.unsqueeze(0) - g_block)
                A_block = torch.mm(q_block * scale, k_block.T)
                causal_mask = torch.tril(torch.ones(actual_bt, actual_bt, dtype=dtype))
                A_block = A_block * causal_mask
                A_out[i_b, t_start:t_end, i_hq, :actual_bt] = A_block
                do_scaled = do_block * torch.exp2(g_block)
                dg_scalar = torch.sum(h_block * dh_block, dim=0) * torch.exp2(g_last)
                dq_chunk = torch.mm(do_scaled, h_block.T) * scale
                dk_chunk = torch.mm(v_block * gv, dh_block.T)
                dv_chunk = torch.mm(k_block, dh_block) * gv
                dg_scalar = dg_scalar + torch.sum(dv_chunk * v_block, dim=0)
                dq_chunk = dq_chunk + torch.mm(dA_block, k_block)
                dk_chunk = dk_chunk + torch.mm(dA_block.T, q_block)
                dgv_chunk = dg_block + dg_scalar.unsqueeze(0)
                dq_out[i_b, t_start:t_end, i_hq, :] = dq_chunk
                dk_out[i_b, t_start:t_end, i_hq, :] = dk_chunk
                dv_out[i_b, t_start:t_end, i_hq, :] = dv_chunk
                dgv_out[i_b, t_start:t_end, i_hq, :] = dgv_chunk
    return (A_out, dq_out, dk_out, dv_out, dgv_out)
