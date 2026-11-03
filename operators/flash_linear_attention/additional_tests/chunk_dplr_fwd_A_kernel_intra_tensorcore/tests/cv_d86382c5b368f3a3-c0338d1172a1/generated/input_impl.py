"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K = (1, 1024, 8, 64)
    chunk_size = 32
    scale = K ** (-0.5)
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    a = torch.randn(B, T, H, K, dtype=dtype, device=device)
    b = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gi = torch.randn(B, T, H, K, dtype=dtype, device=device)
    ge = torch.randn(B, T, H, K, dtype=dtype, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['a'] = a.cpu()
    data['input_data']['b'] = b.cpu()
    data['input_data']['gi'] = gi.cpu()
    data['input_data']['ge'] = ge.cpu()
    data['input_data']['scale'] = scale
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
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
            data['input_data']['q'] = q.cpu()
        except NameError:
            pass
        try:
            data['input_data']['k'] = k.cpu()
        except NameError:
            pass
        try:
            data['input_data']['a'] = a.cpu()
        except NameError:
            pass
        try:
            data['input_data']['b'] = b.cpu()
        except NameError:
            pass
        try:
            data['input_data']['gi'] = gi.cpu()
        except NameError:
            pass
        try:
            data['input_data']['ge'] = ge.cpu()
        except NameError:
            pass
        try:
            data['input_data']['scale'] = scale
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
    return locals()
