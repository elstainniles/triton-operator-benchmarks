"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, K, V = (2, 512, 4, 64, 64)
    HQ = H
    BT = 64
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    NG = HQ // H
    NS = triton.cdiv(T, BT)
    IS_VARLEN = False
    device = 'cpu'
    dtype = torch.float32
    q = torch.randn((B, T, HQ, K), dtype=dtype, device=device)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device)
    g = torch.randn((B, T, H, V), dtype=dtype, device=device)
    h = torch.randn((B, NS, H, K, V), dtype=dtype, device=device)
    h0 = None
    o = torch.randn((B, T, HQ, V), dtype=dtype, device=device)
    do = torch.randn((B, T, HQ, V), dtype=dtype, device=device)
    dht = None
    dg = torch.randn((B, T, HQ, V), dtype=dtype, device=device)
    scale = 1.0 / K ** 0.5
    dq = torch.zeros((B, T, HQ, K), dtype=dtype, device=device)
    dk = torch.zeros((B, T, HQ, K), dtype=dtype, device=device)
    dv = torch.zeros((B, T, HQ, V), dtype=dtype, device=device)
    dg_out = torch.zeros((B, T, HQ, V), dtype=dtype, device=device)
    dh0 = None
    input_data = {'q': q, 'k': k, 'v': v, 'g': g, 'h': h, 'h0': h0, 'o': o, 'do': do, 'dht': dht, 'dg': dg, 'dq': dq, 'dk': dk, 'dv': dv, 'dg_out': dg_out, 'dh0': dh0, 'scale': scale, 'T': T, 'B': B, 'H': H, 'HQ': HQ, 'K': K, 'V': V, 'BT': BT, 'BC': BC, 'BK': BK, 'BV': BV, 'NG': NG, 'NS': NS, 'IS_VARLEN': IS_VARLEN}
    gpu_output = {'dq': dq.clone(), 'dk': dk.clone(), 'dv': dv.clone(), 'dg': dg_out.clone(), 'dh0': torch.zeros(0)}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'q': q, 'k': k, 'v': v, 'g': g, 'h': h, 'h0': h0, 'o': o, 'do': do, 'dht': dht, 'dg': dg, 'dq': dq, 'dk': dk, 'dv': dv, 'dg_out': dg_out, 'dh0': dh0, 'scale': scale, 'T': T, 'B': B, 'H': H, 'HQ': HQ, 'K': K, 'V': V, 'BT': BT, 'BC': BC, 'BK': BK, 'BV': BV, 'NG': NG, 'NS': NS, 'IS_VARLEN': IS_VARLEN}
        except NameError:
            pass
    return locals()
