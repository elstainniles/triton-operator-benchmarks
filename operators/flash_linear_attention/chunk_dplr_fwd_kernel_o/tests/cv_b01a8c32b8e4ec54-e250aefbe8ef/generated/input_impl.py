"""Existing independent CPU equations; file-generation main omitted."""
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V = (4, 512, 8, 128, 128)
    chunk_size = 64
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    device = torch.device('cpu')
    dtype = torch.float16
    qg = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    v_new = torch.randn(B, T, H, V, dtype=dtype, device=device)
    A_qk = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    A_qb = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    h = torch.randn(NT, H, K, V, dtype=dtype, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['qg'] = qg.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['v_new'] = v_new.cpu()
    data['input_data']['A_qk'] = A_qk.cpu()
    data['input_data']['A_qb'] = A_qb.cpu()
    data['input_data']['h'] = h.cpu()
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    if h.ndim == 4 and h.shape[0] != B * (T // chunk_size):
        h = h.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous().reshape(-1, h.shape[-2], h.shape[-1])
        h = h
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
            data['input_data']['qg'] = qg.cpu()
        except NameError:
            pass
        try:
            data['input_data']['v'] = v.cpu()
        except NameError:
            pass
        try:
            data['input_data']['v_new'] = v_new.cpu()
        except NameError:
            pass
        try:
            data['input_data']['A_qk'] = A_qk.cpu()
        except NameError:
            pass
        try:
            data['input_data']['A_qb'] = A_qb.cpu()
        except NameError:
            pass
        try:
            data['input_data']['h'] = h.cpu()
        except NameError:
            pass
        try:
            data['input_data']['chunk_size'] = chunk_size
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
            data['input_data']['K'] = K
        except NameError:
            pass
        try:
            data['input_data']['V'] = V
        except NameError:
            pass
    return locals()
