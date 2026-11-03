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
    
    h = torch.zeros(B, NS, H, K, V, dtype=torch.float32, device=k.device)
    
    for b in range(B):
        for h_idx in range(H):
            h_state = torch.zeros(K, V, dtype=torch.float32, device=k.device)
            
            for i_t in range(NS):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                
                # 存储 h（在处理chunk之前）
                h[b, i_t, h_idx, :, :] = h_state
                
                # 加载当前chunk的数据
                k_chunk = k[b, t_start:t_end, h_idx, :]  # [BT, K]
                v_chunk = v[b, t_start:t_end, h_idx, :]  # [BT, V]
                g_chunk = g[b, t_start:t_end, h_idx, :]  # [BT, V]
                
                # GV模式：使用g作为gv
                g_last = g_chunk[-1, :]  # [V]
                
                # b_h *= exp(b_gv_last)[None, :]
                h_state *= torch.exp(g_last)[None, :]
                
                # b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)
                v_scaled = v_chunk * torch.exp(g_last[None, :] - g_chunk)
                
                # b_h += tl.dot(b_k, b_v)
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
    
    o_inter = torch.zeros(B, T, HQ, V, dtype=torch.float32, device=q.device)
    A = torch.zeros(B, T, HQ, BT, dtype=torch.float32, device=q.device)
    
    for b in range(B):
        for hq_idx in range(HQ):
            h_idx = hq_idx  # 假设HQ = H
            
            for i_t in range(NS):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                
                # 加载数据
                q_chunk = q[b, t_start:t_end, hq_idx, :]  # [BT, K]
                k_chunk = k[b, t_start:t_end, h_idx, :]  # [BT, K]
                h_chunk = h[b, i_t, h_idx, :, :]  # [K, V]
                g_chunk = g[b, t_start:t_end, h_idx, :]  # [BT, V]
                
                # 计算 o_inter
                # b_o = tl.dot(b_q * scale, b_h)
                o_inter_chunk = (q_chunk * scale) @ h_chunk  # [BT, V]
                
                # b_o = b_o * exp(b_g)
                o_inter_chunk = o_inter_chunk * torch.exp(g_chunk)
                
                o_inter[b, t_start:t_end, hq_idx, :] = o_inter_chunk
                
                # 计算 A
                # b_A = tl.dot(b_q * scale, b_k)
                A_chunk = (q_chunk * scale) @ k_chunk.T  # [BT, BT]
                
                # 应用mask: b_A = tl.where(m_s, b_A, 0.)
                o_i = torch.arange(BT, device=q.device)
                m_s = o_i[:, None] >= o_i[None, :]
                A_chunk = torch.where(m_s, A_chunk, torch.zeros_like(A_chunk))
                
                A[b, t_start:t_end, hq_idx, :] = A_chunk
    
    return o_inter, A

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
            h_idx = hq_idx  # 假设HQ = H
            
            for i_t in range(NS):
                for i_i in range(NC):
                    t_start = i_t * BT + i_i * BC
                    t_end = min(t_start + BC, T)
                    
                    if t_start >= T:
                        continue
                    
                    # 加载g
                    g_i = g[b, t_start:t_end, h_idx, :]  # [BC, V]
                    g_n = g[b, min(t_start, T-1), h_idx, :]  # [V]
                    
                    # 计算o_intra_i
                    o_intra_i = torch.zeros(BC, V, dtype=torch.float32, device=v.device)
                    
                    # 处理前面的sub-chunks
                    for i_j in range(i_i):
                        # 加载A, v, g
                        A_ij = A[b, t_start:t_end, hq_idx, i_j*BC:(i_j+1)*BC]  # [BC, BC]
                        v_j = v[b, i_t*BT+i_j*BC:i_t*BT+(i_j+1)*BC, h_idx, :]  # [BC, V]
                        g_j = g[b, i_t*BT+i_j*BC:i_t*BT+(i_j+1)*BC, h_idx, :]  # [BC, V]
                        
                        # b_vg = (b_v * exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
                        v_scaled = v_j * torch.exp(g_n[None, :] - g_j)
                        
                        # b_o += tl.dot(b_A, b_vg)
                        o_intra_i += A_ij @ v_scaled
                    
                    # 应用exp(g_i - g_n)
                    o_intra_i *= torch.exp(g_i - g_n[None, :])
                    
                    # 处理对角线部分
                    for j in range(min(BC, T - t_start)):
                        # 加载v, g
                        v_j = v[b, t_start+j, h_idx, :]  # [V]
                        g_j = g[b, t_start+j, h_idx, :]  # [V]
                        
                        # 加载A
                        A_ij = A[b, t_start:t_end, hq_idx, i_i*BC+j]  # [BC]
                        
                        # b_vg = b_v[None, :] * exp(b_g - b_gv[None, :])
                        v_scaled = v_j[None, :] * torch.exp(g_i - g_j[None, :])  # [BC, V]
                        
                        # 应用mask并累加
                        o_i_bc = torch.arange(BC, device=v.device)
                        mask = o_i_bc[:, None] >= j
                        o_intra_i += torch.where(mask, A_ij[:, None] * v_scaled, torch.zeros_like(v_scaled))
                    
                    # 合并inter和intra
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
    # 步骤1：计算h
    h = chunk_fwd_h_torch(k, v, g, BT)
    
    # 步骤2：计算inter-chunk的o和A
    o_inter, A = chunk_gsa_fwd_k_kernel_inter_torch(q, k, h, g, scale, BT)
    
    # 步骤3：计算intra-chunk的o
    o = chunk_gsa_fwd_k_kernel_intra_torch(v, g, o_inter, A, BT)
    
    return A, h, None, o
