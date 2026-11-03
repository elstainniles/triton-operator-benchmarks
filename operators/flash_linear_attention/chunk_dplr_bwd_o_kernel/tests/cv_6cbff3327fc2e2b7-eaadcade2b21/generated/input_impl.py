"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V = (2, 512, 8, 64, 128)
    chunk_size = 64
    scale = 1.0
    device = torch.device('cpu')
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    b = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    v_new = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    gk = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    do = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    h = torch.randn(B, (T + chunk_size - 1) // chunk_size, H, K, V, dtype=torch.float32, device=device)
    dh = torch.randn(B, (T + chunk_size - 1) // chunk_size, H, K, V, dtype=torch.float32, device=device)
    dv = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    inputs_dict = {'k': k.cpu(), 'b': b.cpu(), 'v': v.cpu(), 'v_new': v_new.cpu(), 'gk': gk.cpu(), 'do': do.cpu(), 'h': h.cpu(), 'dh': dh.cpu(), 'dv': dv.cpu(), 'w': w.cpu(), 'chunk_size': chunk_size, 'scale': scale, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
    k_clone = k.clone()
    b_clone = b.clone()
    v_clone = v.clone()
    v_new_clone = v_new.clone()
    gk_clone = gk.clone()
    do_clone = do.clone()
    h_clone = h.clone()
    dh_clone = dh.clone()
    dv_clone = dv.clone()
    w_clone = w.clone()
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'k': k.cpu(), 'b': b.cpu(), 'v': v.cpu(), 'v_new': v_new.cpu(), 'gk': gk.cpu(), 'do': do.cpu(), 'h': h.cpu(), 'dh': dh.cpu(), 'dv': dv.cpu(), 'w': w.cpu(), 'chunk_size': chunk_size, 'scale': scale, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        except NameError:
            pass
    return locals()
