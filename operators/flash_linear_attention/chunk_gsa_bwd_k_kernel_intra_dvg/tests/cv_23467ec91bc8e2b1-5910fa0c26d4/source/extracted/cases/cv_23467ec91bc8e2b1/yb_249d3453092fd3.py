import torch

def chunk_gsa_bwd_k_kernel_intra_dvg_torch(v, g, o, A, do, dv_in, dg_in, BT=64, BC=16, BV=64):
    """Torch implementation of chunk_gsa_bwd_k_kernel_intra_dvg"""
    B, T, H, V_val = v.shape
    HQ = do.shape[2]
    NG = HQ // H
    NC = (BT + BC - 1) // BC
    NV = (V_val + BV - 1) // BV
    NT = (T + BT - 1) // BT
    
    dv = dv_in.clone()
    dg = dg_in.clone()
    
    for i_v in range(NV):
        v_start = i_v * BV
        v_end = min((i_v + 1) * BV, V_val)
        
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min((i_t + 1) * BT, T)
            
            for i_i in range(NC):
                i_start = i_i * BC
                i_end_actual = min((i_i + 1) * BC, t_end - t_start) - i_start
                i_end_actual = max(0, i_end_actual)
                
                if i_end_actual <= 0:
                    continue
                
                for i_bh in range(B * HQ):
                    i_b = i_bh // HQ
                    i_hq = i_bh % HQ
                    i_h = i_hq // NG
                    
                    gv_block = g[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_h, v_start:v_end].float()
                    g_n = g[i_b, min(t_start + i_start + BC, t_end) - 1, i_h, v_start:v_end].float()
                    
                    b_dv = torch.zeros(BC, BV, dtype=torch.float32, device=v.device)
                    
                    for i_j in range(i_i + 1, NC):
                        j_start = i_j * BC
                        j_end = min((i_j + 1) * BC, BT, t_end - t_start)
                        j_end_actual = min(j_end - j_start, t_end - t_start - j_start)
                        j_end_actual = max(0, j_end_actual)
                        
                        if j_end_actual <= 0 or j_start >= BT:
                            continue
                        
                        g_block = g[i_b, t_start + j_start:t_start + j_start + j_end_actual, i_h, v_start:v_end].float()
                        do_block = do[i_b, t_start + j_start:t_start + j_start + j_end_actual, i_hq, v_start:v_end].float()
                        
                        # A shape is [B, T, HQ, BT], accessed as (BT, T) in GPU
                        # From GPU: p_A shape (BT, T), offsets (i_i*BC, i_t*BT + i_j*BC), block (BC, BC)
                        # block[i, j] = A[b, t_j, hq, i_i*BC + i] where t_j = i_t*BT + i_j*BC + j
                        # But we need to match GPU's dot(b_A, b_do), so build as (i_end_actual, j_end_actual)
                        A_block = torch.zeros(i_end_actual, j_end_actual, dtype=torch.float32, device=v.device)
                        for ii in range(i_end_actual):
                            j_rel = i_start + ii  # position in BT dimension
                            for jj in range(j_end_actual):
                                t_j = t_start + j_start + jj
                                if t_j < T and j_rel < BT:
                                    A_block[ii, jj] = A[i_b, t_j, i_hq, j_rel]
                        
                        m_j = (t_start + j_start + torch.arange(j_end_actual, device=v.device)) < t_end
                        m_j_expanded = m_j[:do_block.shape[0]].unsqueeze(1).float()
                        do_scaled = do_block * torch.exp(g_block - g_n.unsqueeze(0)) * m_j_expanded
                        
                        b_dv += torch.matmul(A_block, do_scaled)
                    
                    b_dv *= torch.exp(g_n.unsqueeze(0) - gv_block)
                    
                    for j in range(min(BC, t_end - t_start - i_start)):
                        t_j = t_start + i_start + j
                        j_rel = t_j - t_start
                        
                        # Load A[b, t_j, hq, i_start + arange(0, i_end_actual)]
                        for ii in range(i_end_actual):
                            t_i = t_start + i_start + ii
                            
                            if t_i < T and t_j < T and ii <= j and j_rel < BT:
                                # A[b, t_j, hq, i_start + ii]
                                A_val = A[i_b, t_j, i_hq, i_start + ii]
                                g_val = g[i_b, t_j, i_h, v_start:v_end].float()
                                do_val = do[i_b, t_j, i_hq, v_start:v_end].float()
                                
                                b_dv[ii, :] += torch.exp(g_val - gv_block[ii, :]) * A_val * do_val
                    
                    o_block = o[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    v_block = v[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_h, v_start:v_end].float()
                    do_block = do[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    dv_block = dv[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end].float()
                    
                    b_dv = b_dv + dv_block
                    b_dg = o_block * do_block - v_block * b_dv
                    
                    dv[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end] = b_dv[:i_end_actual, :]
                    dg[i_b, t_start + i_start:t_start + i_start + i_end_actual, i_hq, v_start:v_end] = b_dg[:i_end_actual, :]
    
    return dv, dg
