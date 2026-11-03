"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch
import math

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 1024
    G = 1
    K = 128
    H = 8
    V = 64
    L = 7
    BT = 64
    device = torch.device('cpu')
    dtype = torch.float32
    NT = ceil_div(T, BT)
    do = torch.randn(B, T, H, V, dtype=dtype, device=device)
    q = torch.randn(B, T, G, K, dtype=dtype, device=device)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    l = torch.randn(B, T, H, L, dtype=dtype, device=device)
    h_l = torch.randn(B, NT, H, K, V, dtype=dtype, device=device)
    dh = torch.zeros(B, NT, H, K, V, dtype=dtype, device=device)
    dg_last = torch.zeros(B, NT, H, dtype=dtype, device=device)
    cu_seqlens = None
    ell = 0
    data = {}
    data['input_data'] = {}
    data['input_data']['do'] = do.cpu()
    data['input_data']['q'] = q.cpu()
    data['input_data']['g'] = g.cpu()
    data['input_data']['l'] = l.cpu()
    data['input_data']['h_l'] = h_l.cpu()
    data['input_data']['dh'] = dh.cpu()
    data['input_data']['dg_last'] = dg_last.cpu()
    data['input_data']['cu_seqlens'] = cu_seqlens
    data['input_data']['ell'] = ell
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['G'] = G
    data['input_data']['K'] = K
    data['input_data']['H'] = H
    data['input_data']['V'] = V
    data['input_data']['L'] = L
    data['input_data']['BT'] = BT
    data['input_data']['NT'] = NT

    def grid_f(meta):
        return (triton.cdiv(K, meta['BK']), B * H)
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
            data['input_data']['do'] = do.cpu()
        except NameError:
            pass
        try:
            data['input_data']['q'] = q.cpu()
        except NameError:
            pass
        try:
            data['input_data']['g'] = g.cpu()
        except NameError:
            pass
        try:
            data['input_data']['l'] = l.cpu()
        except NameError:
            pass
        try:
            data['input_data']['h_l'] = h_l.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dh'] = dh.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dg_last'] = dg_last.cpu()
        except NameError:
            pass
        try:
            data['input_data']['cu_seqlens'] = cu_seqlens
        except NameError:
            pass
        try:
            data['input_data']['ell'] = ell
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
            data['input_data']['G'] = G
        except NameError:
            pass
        try:
            data['input_data']['K'] = K
        except NameError:
            pass
        try:
            data['input_data']['H'] = H
        except NameError:
            pass
        try:
            data['input_data']['V'] = V
        except NameError:
            pass
        try:
            data['input_data']['L'] = L
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

def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)
