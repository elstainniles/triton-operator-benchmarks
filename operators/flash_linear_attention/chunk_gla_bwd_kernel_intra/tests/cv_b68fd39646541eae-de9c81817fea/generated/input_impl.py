"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    'Test chunk_gla_bwd_kernel_intra'
    B = 2
    T = 1024
    H = 8
    K = 128
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dA = torch.randn(B, T, H, chunk_size, dtype=torch.float32, device=device)
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'dA': dA.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'dA': dA.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size}
        except NameError:
            pass
    return locals()
