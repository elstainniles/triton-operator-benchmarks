"""Existing independent CPU equations; file-generation main omitted."""
import torch
import torch.nn.functional as F

def make_inputs():
    torch.manual_seed(20261031)
    torch.manual_seed(42)
    device = torch.device('cpu')
    B = 4
    T = 2048
    H = 8
    K = 128
    V = 128
    BT = 64
    dtype = torch.float16
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g_per_token = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_cumsum = chunk_local_cumsum(g_per_token, BT)
    h0 = None
    output_final_state = True
    data = {}
    data['input_data'] = {}
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['g_per_token'] = g_per_token.cpu()
    data['input_data']['h0'] = h0
    data['input_data']['output_final_state'] = output_final_state
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['BT'] = BT
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
            data['input_data']['v'] = v.cpu()
        except NameError:
            pass
        try:
            data['input_data']['g_per_token'] = g_per_token.cpu()
        except NameError:
            pass
        try:
            data['input_data']['h0'] = h0
        except NameError:
            pass
        try:
            data['input_data']['output_final_state'] = output_final_state
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
            data['input_data']['BT'] = BT
        except NameError:
            pass
    return locals()

def chunk_local_cumsum(g, chunk_size):
    B, T, H = g.shape
    NT = (T + chunk_size - 1) // chunk_size
    T_padded = NT * chunk_size
    g_padded = torch.nn.functional.pad(g, (0, 0, 0, T_padded - T))
    g_padded = g_padded.view(B, NT, chunk_size, H)
    g_cumsum = g_padded.cumsum(dim=2)
    g_cumsum = g_cumsum.view(B, T_padded, H)
    return g_cumsum[:, :T]
