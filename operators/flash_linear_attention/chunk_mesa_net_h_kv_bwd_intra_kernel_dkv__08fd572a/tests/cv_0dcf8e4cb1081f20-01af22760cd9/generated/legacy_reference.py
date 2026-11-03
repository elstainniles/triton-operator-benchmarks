from original_fp32 import naive_chunk_mesa_net_h_kv_bwd_intra_dkv, naive_chunk_mesa_net_h_kv_bwd_intra_dq

def reference(d):
    common = [d[k].float() for k in ('q_star','k','v','beta','h_kv','g','do')]
    dims = [d[k] for k in ('B','T','H','K','V','BT')]
    dk,dv,dg = naive_chunk_mesa_net_h_kv_bwd_intra_dkv(*common,d['dh_kv'].float(),*dims)
    dq,dgf = naive_chunk_mesa_net_h_kv_bwd_intra_dq(*common,dg,*dims)
    return dict(dk=dk,dv=dv,dg=dg,dq=dq,dg_final=dgf)
