"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 1024
    H = 8
    K = 64
    V = 64
    L = 12
    BT = 32
    G = 1
    import os
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    torch.manual_seed(0)
    NT = ceil_div(T, BT)
    device = torch.device('cpu')
    do = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    q = torch.randn(B, T, G, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, G, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device)
    l = torch.randn(B, T, H, L, dtype=torch.float32, device=device)
    llut = level_lut(BT, device)
    mask = masks(BT, device)
    dq = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32, device=device)
    dg = torch.zeros(B, T, H, dtype=torch.float32, device=device)
    dl = torch.zeros(B, T, H, L, dtype=torch.float32, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['do'] = do.cpu()
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['g'] = g.cpu()
    data['input_data']['l'] = l.cpu()
    data['input_data']['llut'] = llut.cpu()
    data['input_data']['mask'] = mask.cpu()
    data['input_data']['dq'] = dq.cpu()
    data['input_data']['dk'] = dk.cpu()
    data['input_data']['dv'] = dv.cpu()
    data['input_data']['dg'] = dg.cpu()
    data['input_data']['dl'] = dl.cpu()
    data['input_data']['cu_seqlens'] = None
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['L'] = L
    data['input_data']['BT'] = BT
    data['input_data']['G'] = G
    data['input_data']['NT'] = NT
    grid = (NT, B * H)
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
            data['input_data']['l'] = l.cpu()
        except NameError:
            pass
        try:
            data['input_data']['llut'] = llut.cpu()
        except NameError:
            pass
        try:
            data['input_data']['mask'] = mask.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dq'] = dq.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dk'] = dk.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dv'] = dv.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dg'] = dg.cpu()
        except NameError:
            pass
        try:
            data['input_data']['dl'] = dl.cpu()
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
            data['input_data']['L'] = L
        except NameError:
            pass
        try:
            data['input_data']['BT'] = BT
        except NameError:
            pass
        try:
            data['input_data']['G'] = G
        except NameError:
            pass
        try:
            data['input_data']['NT'] = NT
        except NameError:
            pass
    return locals()

def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)

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

def masks(BT, device):
    masks_list = []
    for level in range(0, ceil_log(BT, 2) + 1):
        mask = construct_binary_level_mask(level, BT).to(device).to(torch.int32)
        masks_list.append(mask)
    return torch.stack(masks_list)
