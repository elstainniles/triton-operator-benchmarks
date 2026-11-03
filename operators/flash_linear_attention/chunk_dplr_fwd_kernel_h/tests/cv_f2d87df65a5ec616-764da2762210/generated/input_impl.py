"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V = (1, 63, 1, 64, 64)
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float32
    kg = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.1
    bg = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.1
    gk = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['kg'] = kg.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['w'] = w.cpu()
    data['input_data']['u'] = u.cpu()
    data['input_data']['bg'] = bg.cpu()
    data['input_data']['gk'] = gk.cpu()
    data['input_data']['initial_state'] = initial_state.cpu()
    data['input_data']['output_final_state'] = True
    data['input_data']['cu_seqlens'] = None
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['chunk_indices'] = None
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    data['input_data']['K'] = K
    data['input_data']['V'] = V
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
            data['input_data']['kg'] = kg.cpu()
        except NameError:
            pass
        try:
            data['input_data']['v'] = v.cpu()
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
            data['input_data']['bg'] = bg.cpu()
        except NameError:
            pass
        try:
            data['input_data']['gk'] = gk.cpu()
        except NameError:
            pass
        try:
            data['input_data']['initial_state'] = initial_state.cpu()
        except NameError:
            pass
        try:
            data['input_data']['output_final_state'] = True
        except NameError:
            pass
        try:
            data['input_data']['cu_seqlens'] = None
        except NameError:
            pass
        try:
            data['input_data']['chunk_size'] = chunk_size
        except NameError:
            pass
        try:
            data['input_data']['chunk_indices'] = None
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
    return locals()
