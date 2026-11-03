"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(20261031)
    torch.manual_seed(0)
    B, T, H, D = (4, 2048, 8, 64)
    BT = 64
    K = D
    V = D
    dtype = torch.float32
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    p = torch.randn(B, T, H, K, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g0 = F.logsigmoid(torch.randn(B, T, H, dtype=dtype))
    g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype))
    A = torch.randn(B, T, H, BT, dtype=dtype)
    dw = torch.randn(B, T, H, K, dtype=dtype)
    du = torch.randn(B, T, H, V, dtype=dtype)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dp = torch.zeros_like(p)
    dbeta = torch.zeros_like(beta)
    dg0 = torch.zeros_like(g0)
    dg = torch.zeros_like(g)
    CONST_TILING = 64
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT)
    input_data = {'k': k, 'v': v, 'p': p, 'beta': beta, 'g0': g0, 'g': g, 'A': A, 'dw': dw, 'du': du, 'dk': dk, 'dv': dv, 'dp': dp, 'dbeta': dbeta, 'dg0': dg0, 'dg': dg, 'cu_seqlens': None, 'chunk_indices': None, 'T': T, 'H': H, 'K': K, 'V': V, 'BT': BT, 'BK': BK, 'BV': BV}
    gpu_output = {'dk': dk.clone(), 'dv': dv.clone(), 'dp': dp.clone(), 'dbeta': dbeta.clone(), 'dg0': dg0.clone(), 'dg': dg.clone()}
    grid = (NT, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'k': k, 'v': v, 'p': p, 'beta': beta, 'g0': g0, 'g': g, 'A': A, 'dw': dw, 'du': du, 'dk': dk, 'dv': dv, 'dp': dp, 'dbeta': dbeta, 'dg0': dg0, 'dg': dg, 'cu_seqlens': None, 'chunk_indices': None, 'T': T, 'H': H, 'K': K, 'V': V, 'BT': BT, 'BK': BK, 'BV': BV}
        except NameError:
            pass
    return locals()
