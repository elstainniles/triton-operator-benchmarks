"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, H, T, K, V = (2, 4, 512, 64, 64)
    dtype = torch.float32
    device = torch.device('cpu')
    q = torch.randn(B, H, T, K, dtype=dtype, device=device)
    k = torch.randn(B, H, T, K, dtype=dtype, device=device)
    v = torch.randn(B, H, T, V, dtype=dtype, device=device)
    scale = K ** (-0.5)
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
            data['input_data']['v'] = v.cpu()
        except NameError:
            pass
        try:
            data['input_data']['scale'] = scale
        except NameError:
            pass
        try:
            data['input_data']['B'] = B
        except NameError:
            pass
        try:
            data['input_data']['H'] = H
        except NameError:
            pass
        try:
            data['input_data']['T'] = T
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
