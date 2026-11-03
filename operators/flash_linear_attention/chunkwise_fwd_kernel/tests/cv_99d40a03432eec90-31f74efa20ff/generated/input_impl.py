"""Existing independent CPU equations; file-generation main omitted."""
import math
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import os
import torch

def make_inputs():
    torch.manual_seed(42)
    B = 1
    T = 256
    H = 4
    G = 1
    K = 64
    V = 64
    L = 10
    BT = 64
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    device = torch.device('cpu')
    dtype = torch.float
    q = torch.randn(B, T, G, K, dtype=dtype, device=device)
    k = torch.randn(B, T, G, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    level_scales = torch.randn(B, T, H, L, dtype=dtype, device=device)
    llut = level_lut(BT, device)
    NT = math.ceil(T / BT)
    MAX_LEVEL = math.ceil(math.log(NT, 2)) - 1
    o = torch.zeros((B, T, H, K // BLOCK_K, V), dtype=dtype, device=device)
    h0 = None
    ht = None
    offsets = None
    new_offsets = torch.zeros((B,), dtype=torch.int32, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['g'] = g.cpu()
    data['input_data']['level_scales'] = level_scales.cpu()
    data['input_data']['llut'] = llut.cpu()
    data['input_data']['o_init'] = torch.zeros((B, T, H, K // BLOCK_K, V), dtype=dtype)
    data['input_data']['new_offsets_init'] = torch.zeros((B,), dtype=torch.int32)
    data['input_data']['h0'] = None
    data['input_data']['ht'] = None
    data['input_data']['offsets'] = None
    data['input_data']['cu_seqlens'] = None
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['G'] = G
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['L'] = L
    data['input_data']['BT'] = BT
    data['input_data']['MAX_LEVEL'] = MAX_LEVEL

    def grid(meta):
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
            data['input_data']['g'] = g.cpu()
        except NameError:
            pass
        try:
            data['input_data']['level_scales'] = level_scales.cpu()
        except NameError:
            pass
        try:
            data['input_data']['llut'] = llut.cpu()
        except NameError:
            pass
        try:
            data['input_data']['o_init'] = torch.zeros((B, T, H, K // BLOCK_K, V), dtype=dtype)
        except NameError:
            pass
        try:
            data['input_data']['new_offsets_init'] = torch.zeros((B,), dtype=torch.int32)
        except NameError:
            pass
        try:
            data['input_data']['h0'] = None
        except NameError:
            pass
        try:
            data['input_data']['ht'] = None
        except NameError:
            pass
        try:
            data['input_data']['offsets'] = None
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
            data['input_data']['G'] = G
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
            data['input_data']['L'] = L
        except NameError:
            pass
        try:
            data['input_data']['BT'] = BT
        except NameError:
            pass
        try:
            data['input_data']['MAX_LEVEL'] = MAX_LEVEL
        except NameError:
            pass
    return locals()
BLOCK_K = 64

def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))

def construct_binary_level_mask(level, T):
    if level == 0:
        return torch.diag(torch.ones(T, dtype=torch.bool))
    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T))
    mask = torch.where(torch.logical_and(torch.logical_and(indices[:, 0] % (1 << level) >= 1 << level - 1, indices[:, 1] + (1 << level - 1) >= indices[:, 0] - indices[:, 0] % (1 << level - 1)), indices[:, 1] < indices[:, 0] - indices[:, 0] % (1 << level - 1)).view(T, T), 1, 0)
    return mask

def level_lut(BT, device):
    lut = torch.zeros((BT, BT), dtype=torch.int32, device=device)
    for level in range(1, ceil_log(BT, 2) + 1):
        mask = construct_binary_level_mask(level, BT).to(device)
        lut = torch.where(mask.to(torch.bool), level, lut)
    return lut
