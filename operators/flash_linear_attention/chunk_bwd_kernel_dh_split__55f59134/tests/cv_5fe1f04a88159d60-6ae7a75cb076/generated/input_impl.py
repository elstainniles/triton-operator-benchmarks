"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 2048
    H = 8
    HQ = H
    K = 128
    V = 128
    device = torch.device('cpu')
    dtype = torch.float16
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    g = None
    gk = None
    gv = None
    h0 = None
    dht = None
    scale = 1.0 / K ** 0.5
    chunk_size = 64
    split_size = 256
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['do'] = do.cpu()
    data['input_data']['g'] = g
    data['input_data']['gk'] = gk
    data['input_data']['gv'] = gv
    data['input_data']['h0'] = h0
    data['input_data']['dht'] = dht
    data['input_data']['scale'] = scale
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['HQ'] = HQ
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['split_size'] = split_size
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
            data['input_data']['do'] = do.cpu()
        except NameError:
            pass
        try:
            data['input_data']['g'] = g
        except NameError:
            pass
        try:
            data['input_data']['gk'] = gk
        except NameError:
            pass
        try:
            data['input_data']['gv'] = gv
        except NameError:
            pass
        try:
            data['input_data']['h0'] = h0
        except NameError:
            pass
        try:
            data['input_data']['dht'] = dht
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
            data['input_data']['T'] = T
        except NameError:
            pass
        try:
            data['input_data']['H'] = H
        except NameError:
            pass
        try:
            data['input_data']['HQ'] = HQ
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
            data['input_data']['chunk_size'] = chunk_size
        except NameError:
            pass
        try:
            data['input_data']['split_size'] = split_size
        except NameError:
            pass
    return locals()
