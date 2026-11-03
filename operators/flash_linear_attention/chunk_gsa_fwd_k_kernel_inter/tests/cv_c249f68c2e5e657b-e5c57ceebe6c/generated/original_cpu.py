"""Existing independent CPU equations; file-generation main omitted."""
import torch

def chunk_fwd_h_torch(k, v, g, BT=64):
    """
    Torch native实现chunk_fwd_h
    
    输入:
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, V] (作为gv使用)
        BT: chunk size
    
    输出:
        h: [B, NS, H, K, V]
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    NS = (T + BT - 1) // BT
    h = torch.zeros(B, NS, H, K, V, dtype=torch.float64, device=k.device)
    for b in range(B):
        for h_idx in range(H):
            h_state = torch.zeros(K, V, dtype=torch.float64, device=k.device)
            for i_t in range(NS):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                h[b, i_t, h_idx, :, :] = h_state
                k_chunk = k[b, t_start:t_end, h_idx, :]
                v_chunk = v[b, t_start:t_end, h_idx, :]
                g_chunk = g[b, t_start:t_end, h_idx, :]
                g_last = g_chunk[-1, :]
                h_state *= torch.exp(g_last)[None, :]
                v_scaled = v_chunk * torch.exp(g_last[None, :] - g_chunk)
                h_state += k_chunk.T @ v_scaled
    return h

def chunk_gsa_fwd_k_kernel_inter_torch(q, k, h, g, scale, BT=64):
    """
    Torch native实现chunk_gsa_fwd_k_kernel_inter
    
    输入:
        q: [B, T, HQ, K]
        k: [B, T, H, K]
        h: [B, NS, H, K, V]
        g: [B, T, H, V]
        scale: float
        BT: chunk size
    
    输出:
        o_inter: [B, T, HQ, V]
        A: [B, T, HQ, BT]
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = g.shape[-1]
    NS = h.shape[1]
    o_inter = torch.zeros(B, T, HQ, V, dtype=torch.float64, device=q.device)
    A = torch.zeros(B, T, HQ, BT, dtype=torch.float64, device=q.device)
    for b in range(B):
        for hq_idx in range(HQ):
            h_idx = hq_idx
            for i_t in range(NS):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                q_chunk = q[b, t_start:t_end, hq_idx, :]
                k_chunk = k[b, t_start:t_end, h_idx, :]
                h_chunk = h[b, i_t, h_idx, :, :]
                g_chunk = g[b, t_start:t_end, h_idx, :]
                o_inter_chunk = q_chunk * scale @ h_chunk
                o_inter_chunk = o_inter_chunk * torch.exp(g_chunk)
                o_inter[b, t_start:t_end, hq_idx, :] = o_inter_chunk
                A_chunk = q_chunk * scale @ k_chunk.T
                o_i = torch.arange(BT, device=q.device)
                m_s = o_i[:, None] >= o_i[None, :]
                A_chunk = torch.where(m_s, A_chunk, torch.zeros_like(A_chunk))
                A[b, t_start:t_end, hq_idx, :] = A_chunk
    return (o_inter, A)

def chunk_gsa_fwd_k_kernel_intra_torch(v, g, o_inter, A, BT=64, BC=16):
    """
    Torch native实现chunk_gsa_fwd_k_kernel_intra
    
    输入:
        v: [B, T, H, V]
        g: [B, T, H, V]
        o_inter: [B, T, HQ, V]
        A: [B, T, HQ, BT]
        BT: chunk size
        BC: sub-chunk size
    
    输出:
        o_final: [B, T, HQ, V]
    """
    B, T, H, V = v.shape
    HQ = o_inter.shape[2]
    NS = (T + BT - 1) // BT
    NC = BT // BC
    o_final = o_inter.clone()
    for b in range(B):
        for hq_idx in range(HQ):
            h_idx = hq_idx
            for i_t in range(NS):
                for i_i in range(NC):
                    t_start = i_t * BT + i_i * BC
                    t_end = min(t_start + BC, T)
                    if t_start >= T:
                        continue
                    g_i = g[b, t_start:t_end, h_idx, :]
                    g_n = g[b, min(t_start, T - 1), h_idx, :]
                    o_intra_i = torch.zeros(BC, V, dtype=torch.float64, device=v.device)
                    for i_j in range(i_i):
                        A_ij = A[b, t_start:t_end, hq_idx, i_j * BC:(i_j + 1) * BC]
                        v_j = v[b, i_t * BT + i_j * BC:i_t * BT + (i_j + 1) * BC, h_idx, :]
                        g_j = g[b, i_t * BT + i_j * BC:i_t * BT + (i_j + 1) * BC, h_idx, :]
                        v_scaled = v_j * torch.exp(g_n[None, :] - g_j)
                        o_intra_i += A_ij @ v_scaled
                    o_intra_i *= torch.exp(g_i - g_n[None, :])
                    for j in range(min(BC, T - t_start)):
                        v_j = v[b, t_start + j, h_idx, :]
                        g_j = g[b, t_start + j, h_idx, :]
                        A_ij = A[b, t_start:t_end, hq_idx, i_i * BC + j]
                        v_scaled = v_j[None, :] * torch.exp(g_i - g_j[None, :])
                        o_i_bc = torch.arange(BC, device=v.device)
                        mask = o_i_bc[:, None] >= j
                        o_intra_i += torch.where(mask, A_ij[:, None] * v_scaled, torch.zeros_like(v_scaled))
                    o_final[b, t_start:t_end, hq_idx, :] += o_intra_i
    return o_final

def chunk_gsa_fwd_k_torch(q, k, v, g, scale, BT=64):
    """
    Torch native实现完整的chunk_gsa_fwd_k
    
    输入:
        q: [B, T, HQ, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, V]
        scale: float
        BT: chunk size
    
    输出:
        A: [B, T, HQ, BT]
        h: [B, NS, H, K, V]
        o: [B, T, HQ, V]
    """
    h = chunk_fwd_h_torch(k, v, g, BT)
    o_inter, A = chunk_gsa_fwd_k_kernel_inter_torch(q, k, h, g, scale, BT)
    o = chunk_gsa_fwd_k_kernel_intra_torch(v, g, o_inter, A, BT)
    return (A, h, None, o)
