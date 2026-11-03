"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V = (2, 512, 8, 64, 128)
    chunk_size = 64
    device = torch.device('cpu')
    A_qk = torch.randn(B, T, H, chunk_size, dtype=torch.float32, device=device)
    kg = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    do = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    dh = torch.randn(B, (T + chunk_size - 1) // chunk_size, H, K, V, dtype=torch.float32, device=device)
    inputs_dict = {'A_qk': A_qk.cpu(), 'kg': kg.cpu(), 'do': do.cpu(), 'dh': dh.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size, 'chunk_indices': None, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
    A_qk_clone = A_qk.clone()
    kg_clone = kg.clone()
    do_clone = do.clone()
    dh_clone = dh.clone()
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'A_qk': A_qk.cpu(), 'kg': kg.cpu(), 'do': do.cpu(), 'dh': dh.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size, 'chunk_indices': None, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        except NameError:
            pass
    return locals()
