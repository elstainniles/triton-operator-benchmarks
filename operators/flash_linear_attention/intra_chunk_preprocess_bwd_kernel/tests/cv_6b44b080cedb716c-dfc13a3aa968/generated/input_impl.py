"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    'Test function for intra_chunk_preprocess_bwd_kernel'
    B = 4
    T = 256
    HQ = 16
    H = 4
    K = 32
    BT = 32
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    w = torch.randn(B, T, H, K, dtype=dtype, device=device)
    w2 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    dq = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    dk = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    dA_local = torch.randn(B, T, HQ, BT, dtype=dtype, device=device)
    dw1 = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    dw2 = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    A = torch.randn(B, T, H, BT, BT, dtype=dtype, device=device)
    for i in range(BT):
        for j in range(BT):
            if j > i:
                A[:, :, :, i, j] = 0
    L = torch.randn(B, T, HQ, dtype=dtype, device=device)
    D = torch.randn(B, T, HQ, dtype=dtype, device=device)
    do = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    scale = 1.0 / K ** 0.5
    cu_seqlens = None
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'w': w.cpu(), 'w2': w2.cpu(), 'beta': beta.cpu(), 'dq': dq.cpu(), 'dk': dk.cpu(), 'dA_local': dA_local.cpu(), 'dw1': dw1.cpu(), 'dw2': dw2.cpu(), 'A': A.cpu(), 'L': L.cpu(), 'D': D.cpu(), 'do': do.cpu(), 'scale': scale, 'cu_seqlens': cu_seqlens}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'w': w.cpu(), 'w2': w2.cpu(), 'beta': beta.cpu(), 'dq': dq.cpu(), 'dk': dk.cpu(), 'dA_local': dA_local.cpu(), 'dw1': dw1.cpu(), 'dw2': dw2.cpu(), 'A': A.cpu(), 'L': L.cpu(), 'D': D.cpu(), 'do': do.cpu(), 'scale': scale, 'cu_seqlens': cu_seqlens}
        except NameError:
            pass
    return locals()
