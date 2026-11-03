import torch

def chunk_gsa_bwd_k_kernel_dA_torch(v, g, do, BT=64, BC=16, BV=64, scale=1.0):
    """Torch implementation of chunk_gsa_bwd_k_kernel_dA"""
    B, T, H, V_val = v.shape
    HQ = do.shape[2]
    NG = HQ // H
    NC = (BT + BC - 1) // BC
    NV = (V_val + BV - 1) // BV
    NT = (T + BT - 1) // BT
    
    dA = torch.zeros(B, T, HQ, BT, dtype=torch.float32, device=v.device)
    
    for i_b in range(B):
        for i_hq in range(HQ):
            i_h = i_hq // NG
            
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                
                for i_i in range(NC):
                    i_start = i_i * BC
                    i_end_actual = min((i_i + 1) * BC, t_end - t_start)
                    
                    for i_j in range(NC):
                        j_start = i_j * BC
                        j_end_actual = min((i_j + 1) * BC, t_end - t_start)
                        
                        if i_i > i_j:
                            for ii in range(i_end_actual):
                                for jj in range(j_end_actual):
                                    t_i = t_start + i_start + ii
                                    t_j = t_start + j_start + jj
                                    
                                    # Check boundary: t_i and t_j must be within current chunk [t_start, t_end)
                                    if t_i >= t_end or t_j >= t_end:
                                        continue
                                    
                                    # Check boundary and lower triangular condition
                                    if t_i < T and t_j < T and j_start + jj < BT:
                                        # dA is lower triangular based on chunk-relative positions
                                        # i = t_i % BT = i_start + ii (relative position in chunk)
                                        # j = j_start + jj (relative position in chunk)
                                        i_rel = i_start + ii
                                        j_rel = j_start + jj
                                        
                                        if i_rel >= j_rel:
                                            do_val = do[i_b, t_i, i_hq, :].float()
                                            v_val = v[i_b, t_j, i_h, :].float()
                                            g_i = g[i_b, t_i, i_h, :].float()
                                            g_j = g[i_b, t_j, i_h, :].float()
                                            
                                            g_n = g_i[-1] if len(g_i) > 0 else 0.0
                                            do_scaled = do_val * torch.exp(g_i - g_n)
                                            v_scaled = v_val * torch.exp(g_n - g_j)
                                            
                                            dA_ij = torch.sum(do_scaled * v_scaled) * scale
                                            dA[i_b, t_i, i_hq, j_rel] = dA_ij
                        
                        elif i_i == i_j:
                            for j in range(min(BC, t_end - t_start - j_start)):
                                t_j = t_start + j_start + j
                                
                                if t_j < T:
                                    for ii in range(i_end_actual):
                                        t_i = t_start + i_start + ii
                                        
                                        # Check boundary: t_i must be within current chunk
                                        if t_i >= t_end:
                                            continue
                                        
                                        # Only compute when i >= j (lower triangular)
                                        if t_i < T and ii >= j and j_start + j < BT:
                                            do_val = do[i_b, t_i, i_hq, :].float()
                                            v_val = v[i_b, t_j, i_h, :].float()
                                            g_i = g[i_b, t_i, i_h, :].float()
                                            g_j = g[i_b, t_j, i_h, :].float()
                                            
                                            do_scaled = do_val * scale
                                            v_scaled = v_val * torch.exp(g_i - g_j)
                                            
                                            dA_ij = torch.sum(do_scaled * v_scaled)
                                            dA[i_b, t_i, i_hq, j_start + j] = dA_ij
    
    return dA
