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
    NC = triton.cdiv(BT, BC)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    gk = torch.randn(B, T, H, K, dtype=torch.float32)
    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    A = torch.zeros(B, T, H, BT, dtype=torch.float32)
    NT = triton.cdiv(T, BT)
    input_data = {'k': k, 'g': gk, 'beta': beta, 'A': A, 'cu_seqlens': None, 'chunk_indices': None, 'T': T, 'H': H, 'K': K, 'BT': BT, 'BC': BC, 'NC': NC}
    gpu_output = {'A': A.clone()}
    grid = (NT, NC * NC, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'k': k, 'g': gk, 'beta': beta, 'A': A, 'cu_seqlens': None, 'chunk_indices': None, 'T': T, 'H': H, 'K': K, 'BT': BT, 'BC': BC, 'NC': NC}
        except NameError:
            pass
    return locals()
