"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    torch.manual_seed(0)
    import os
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    B, T, H, D = (2, 256, 4, 64)
    dtype = torch.float16
    device = 'cpu'
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).contiguous()
    g = torch.randn((B, T, H), dtype=torch.float32, device=device).contiguous()
    do = torch.randn((B, T, H, D), dtype=dtype, device=device).contiguous()
    scale = 0.1
    chunk_size = 64
    cu_seqlens = None
    chunk_indices = None
    data = {}
    data['input_data'] = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'do': do.cpu(), 'scale': scale, 'chunk_size': chunk_size, 'cu_seqlens': cu_seqlens, 'chunk_indices': chunk_indices}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'do': do.cpu(), 'scale': scale, 'chunk_size': chunk_size, 'cu_seqlens': cu_seqlens, 'chunk_indices': chunk_indices}
        except NameError:
            pass
    return locals()
