"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, H, T, K, V = (2, 8, 1024, 64, 64)
    BTL, BTS = (64, 32)
    BK = min(128, max(triton.next_power_of_2(K), 16))
    BV = min(128, max(triton.next_power_of_2(V), 16))
    device = torch.device('cpu')
    dtype = torch.float16
    q = torch.randn(B, H, T, K, dtype=dtype, device=device)
    k = torch.randn(B, H, T, K, dtype=dtype, device=device)
    v = torch.randn(B, H, T, V, dtype=dtype, device=device)
    do = torch.randn(B, H, T, V, dtype=dtype, device=device)
    dz = torch.randn(B, H, T, dtype=dtype, device=device)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    dq = torch.zeros(NV, B, H, T, K, dtype=dtype, device=device)
    dk = torch.zeros(NV, B, H, T, K, dtype=dtype, device=device)
    dv = torch.zeros(NK, B, H, T, V, dtype=dtype, device=device)
    scale = K ** (-0.5)
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['do'] = do.cpu()
    data['input_data']['dz'] = dz.cpu()
    data['input_data']['B'] = B
    data['input_data']['H'] = H
    data['input_data']['T'] = T
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['BTL'] = BTL
    data['input_data']['BTS'] = BTS
    data['input_data']['BK'] = BK
    data['input_data']['BV'] = BV
    data['input_data']['NK'] = NK
    data['input_data']['NV'] = NV
    data['input_data']['scale'] = scale
    grid = (NK * NV, triton.cdiv(T, BTL), B * H)
    num_warps = 4
    num_stages = 2
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
            data['input_data']['dz'] = dz.cpu()
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
        try:
            data['input_data']['BTL'] = BTL
        except NameError:
            pass
        try:
            data['input_data']['BTS'] = BTS
        except NameError:
            pass
        try:
            data['input_data']['BK'] = BK
        except NameError:
            pass
        try:
            data['input_data']['BV'] = BV
        except NameError:
            pass
        try:
            data['input_data']['NK'] = NK
        except NameError:
            pass
        try:
            data['input_data']['NV'] = NV
        except NameError:
            pass
        try:
            data['input_data']['scale'] = scale
        except NameError:
            pass
    return locals()
