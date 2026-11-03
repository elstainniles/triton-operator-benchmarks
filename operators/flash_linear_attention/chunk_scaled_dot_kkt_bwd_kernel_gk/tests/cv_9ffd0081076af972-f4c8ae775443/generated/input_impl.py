"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    torch.manual_seed(0)
    B, T, H, K = (4, 2048, 8, 64)
    BT = 64
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    g = torch.randn(B, T, H, K, dtype=torch.float32)
    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    dA = torch.randn(B, T, H, BT, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dg = torch.zeros_like(g, dtype=torch.float32)
    db = beta.new_zeros(NK, *beta.shape, dtype=torch.float32)
    NT = triton.cdiv(T, BT)
    input_data = {'k': k, 'g': g, 'beta': beta, 'dA': dA, 'dk': dk, 'dg': dg, 'db': db, 'cu_seqlens': None, 'chunk_indices': None, 'B': B, 'T': T, 'H': H, 'K': K, 'BT': BT, 'BC': BC, 'BK': BK, 'NC': NC}
    gpu_output = {'dk': dk.clone(), 'dg': dg.clone(), 'db': db.clone()}
    grid = (NK, NT * NC, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'k': k, 'g': g, 'beta': beta, 'dA': dA, 'dk': dk, 'dg': dg, 'db': db, 'cu_seqlens': None, 'chunk_indices': None, 'B': B, 'T': T, 'H': H, 'K': K, 'BT': BT, 'BC': BC, 'BK': BK, 'NC': NC}
        except NameError:
            pass
    return locals()
