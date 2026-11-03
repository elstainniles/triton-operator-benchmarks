"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K = (4, 2048, 4, 256)
    dtype = torch.float16
    device = torch.device('cpu')
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dq1 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dk1 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    scale = K ** (-0.5)
    reverse = False
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['dq1'] = dq1.cpu()
    data['input_data']['dk1'] = dk1.cpu()
    data['input_data']['scale'] = scale
    data['input_data']['reverse'] = reverse
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
            data['input_data']['dq1'] = dq1.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dk1'] = dk1.cpu()
        except NameError:
            pass
        try:
            data['input_data']['scale'] = scale
        except NameError:
            pass
        try:
            data['input_data']['reverse'] = reverse
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
