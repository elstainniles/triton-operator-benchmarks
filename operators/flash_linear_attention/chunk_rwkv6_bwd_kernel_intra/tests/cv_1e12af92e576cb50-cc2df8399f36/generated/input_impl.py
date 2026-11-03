"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K = (4, 256, 4, 64)
    dtype = torch.float32
    device = torch.device('cpu')
    chunk_size = 64
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gi = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    ge = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    dA = torch.randn(B, T, H, chunk_size, dtype=torch.float32, device=device)
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
            data['input_data']['gi'] = gi.cpu()
        except NameError:
            pass
        try:
            data['input_data']['ge'] = ge.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dA'] = dA.cpu()
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
