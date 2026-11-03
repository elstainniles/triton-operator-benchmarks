"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 4
    T = 1024
    H = 4
    HV = 4
    K = 64
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float16
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, dtype=dtype, device=device)
    beta = torch.randn(B, T, HV, dtype=dtype, device=device).sigmoid()
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=dtype)
    data = {}
    data['input_data'] = {}
    data['input_data']['k'] = k.cpu()
    data['input_data']['g'] = g.cpu()
    data['input_data']['beta'] = beta.cpu()
    data['input_data']['A'] = A.cpu()
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['HV'] = HV
    data['input_data']['K'] = K
    data['input_data']['BT'] = BT
    data['input_data']['NT'] = NT
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {}
        except NameError:
            pass
        try:
            data['input_data']['k'] = k.cpu()
        except NameError:
            pass
        try:
            data['input_data']['g'] = g.cpu()
        except NameError:
            pass
        try:
            data['input_data']['beta'] = beta.cpu()
        except NameError:
            pass
        try:
            data['input_data']['A'] = A.cpu()
        except NameError:
            pass
        try:
            data['input_data']['B'] = B
        except NameError:
            pass
        try:
            data['input_data']['T'] = T
        except NameError:
            pass
        try:
            data['input_data']['H'] = H
        except NameError:
            pass
        try:
            data['input_data']['HV'] = HV
        except NameError:
            pass
        try:
            data['input_data']['K'] = K
        except NameError:
            pass
        try:
            data['input_data']['BT'] = BT
        except NameError:
            pass
        try:
            data['input_data']['NT'] = NT
        except NameError:
            pass
    return locals()
