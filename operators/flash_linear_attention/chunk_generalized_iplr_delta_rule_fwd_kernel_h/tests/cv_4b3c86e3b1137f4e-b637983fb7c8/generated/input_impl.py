"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, D, scale = (1, 63, 1, 64, 1)
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)
    a = F.normalize(a, p=2, dim=-1)
    b = -a
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, a, b, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, a, b, h0))
    initial_state = h0.clone()
    cu_seqlens = None
    cu_seqlens_cpu = None
    output_final_state = True
    data = {}
    data['input_data'] = {}
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['b'] = b.cpu()
    data['input_data']['initial_state'] = initial_state.cpu()
    data['input_data']['output_final_state'] = output_final_state
    data['input_data']['B'] = B
    data['input_data']['T'] = T
    data['input_data']['H'] = H
    scale = k.shape[-1] ** (-0.5) if scale is None else scale
    chunk_size = min(64, max(triton.next_power_of_2(q.shape[1]), 16))
    chunk_indices = None
    data = {}
    data['input_data'] = {}
    data['input_data']['q'] = q.cpu()
    data['input_data']['k'] = k.cpu()
    data['input_data']['v'] = v.cpu()
    data['input_data']['a'] = a.cpu()
    data['input_data']['b'] = b.cpu()
    data['input_data']['scale'] = scale
    data['input_data']['initial_state'] = initial_state.cpu()
    data['input_data']['output_final_state'] = output_final_state
    data['input_data']['cu_seqlens'] = None
    data['input_data']['chunk_size'] = chunk_size
    data['input_data']['chunk_indices'] = chunk_indices
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
            data['input_data']['b'] = b.cpu()
        except NameError:
            pass
        try:
            data['input_data']['initial_state'] = initial_state.cpu()
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
            data['input_data']['a'] = a.cpu()
        except NameError:
            pass
        try:
            data['input_data']['b'] = b.cpu()
        except NameError:
            pass
        try:
            data['input_data']['scale'] = scale
        except NameError:
            pass
        try:
            data['input_data']['initial_state'] = initial_state.cpu()
        except NameError:
            pass
        try:
            data['input_data']['output_final_state'] = output_final_state
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
            data['input_data']['chunk_indices'] = chunk_indices
        except NameError:
            pass
    return locals()
