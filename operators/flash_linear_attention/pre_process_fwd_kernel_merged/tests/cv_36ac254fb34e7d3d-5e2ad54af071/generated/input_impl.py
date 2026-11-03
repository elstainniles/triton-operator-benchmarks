"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V, HV = (1, 1024, 4, 64, 128, 8)
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float32
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    w = torch.randn(B, T, HV, K, dtype=dtype, device=device) * 0.1
    u = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    data = {}
    data['input_data'] = {}
    data['input_data']['k'] = k.cpu()
    data['input_data']['w'] = w.cpu()
    data['input_data']['u'] = u.cpu()
    data['input_data']['g'] = None
    data['input_data']['gk'] = None
    data['input_data']['bg'] = None
    data['input_data']['v'] = None
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['cu_seqlens'] = None
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['HV'] = HV
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
            data['input_data']['w'] = w.cpu()
        except NameError:
            pass
        try:
            data['input_data']['u'] = u.cpu()
        except NameError:
            pass
        try:
            data['input_data']['g'] = None
        except NameError:
            pass
        try:
            data['input_data']['gk'] = None
        except NameError:
            pass
        try:
            data['input_data']['bg'] = None
        except NameError:
            pass
        try:
            data['input_data']['v'] = None
        except NameError:
            pass
        try:
            data['input_data']['chunk_size'] = chunk_size
        except NameError:
            pass
        try:
            data['input_data']['cu_seqlens'] = None
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
        try:
            data['input_data']['HV'] = HV
        except NameError:
            pass
    return locals()
