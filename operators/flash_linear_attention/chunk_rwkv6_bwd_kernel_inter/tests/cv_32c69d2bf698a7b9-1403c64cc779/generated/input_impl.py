"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    import os
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    B, T, H, D = (4, 512, 4, 64)
    chunk_size = 32
    IS_VARLEN = False
    device = 'cpu'
    dtype = torch.float32
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device=device))
    u = torch.randn((H, D), dtype=dtype, device=device)
    scale = 0.1
    NT = triton.cdiv(T, chunk_size)
    gi = torch.randn((B, T, H, D), dtype=dtype, device=device)
    ge = torch.randn((B, T, H, D), dtype=dtype, device=device)
    h = torch.randn((B * NT, H, D, D), dtype=torch.float32, device=device)
    do = torch.randn((B, T, H, D), dtype=dtype, device=device)
    dh = torch.randn((B * NT, H, D, D), dtype=torch.float32, device=device)
    dA = torch.randn((B, T, H, chunk_size), dtype=torch.float32, device=device)
    dq = torch.randn((B, T, H, D), dtype=dtype, device=device)
    dk = torch.randn((B, T, H, D), dtype=dtype, device=device)
    dq2 = torch.zeros(B, T, H, D, dtype=dtype, device=device)
    dk2 = torch.zeros(B, T, H, D, dtype=dtype, device=device)
    dg = torch.zeros(B, T, H, D, dtype=dtype, device=device)
    du = torch.zeros(B * NT, H, D, dtype=torch.float32, device=device)
    cu_seqlens = None
    chunk_indices = None
    input_data = {'q': q, 'k': k, 'v': v, 'h': h, 'g': g, 'gi': gi, 'ge': ge, 'u': u, 'do': do, 'dh': dh, 'dA': dA, 'dq': dq, 'dk': dk, 'dq2': dq2, 'dk2': dk2, 'dg': dg, 'du': du, 'cu_seqlens': torch.zeros(0, dtype=torch.int32), 'chunk_indices': torch.zeros(0, dtype=torch.int32), 'scale': scale, 'T': T, 'H': H, 'K': D, 'V': D, 'BT': chunk_size, 'BK': 32, 'BV': 32, 'IS_VARLEN': IS_VARLEN}
    gpu_output = {'dq2': dq2.clone(), 'dk2': dk2.clone(), 'dg': dg.clone(), 'du': du.clone()}
    grid = (triton.cdiv(D, 32), NT, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'q': q, 'k': k, 'v': v, 'h': h, 'g': g, 'gi': gi, 'ge': ge, 'u': u, 'do': do, 'dh': dh, 'dA': dA, 'dq': dq, 'dk': dk, 'dq2': dq2, 'dk2': dk2, 'dg': dg, 'du': du, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'chunk_indices': chunk_indices if chunk_indices is not None else torch.zeros(0, dtype=torch.int32), 'scale': scale, 'T': T, 'H': H, 'K': D, 'V': D, 'BT': chunk_size, 'BK': 32, 'BV': 32, 'IS_VARLEN': IS_VARLEN}
        except NameError:
            pass
    return locals()
