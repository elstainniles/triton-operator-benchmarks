"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, D = (2, 2048, 8, 64)
    dtype = torch.float32
    device = 'cpu'
    scale = D ** (-0.5)
    gate_logit_normalizer = 1.0
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = torch.randn((B, T, H), dtype=torch.float32, device=device)
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device)
    dht = torch.randn_like(h0)
    g = F.logsigmoid(g) / gate_logit_normalizer
    do = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g_gamma = None
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'g': g, 'g_gamma': g_gamma, 'do': do, 'scale': scale, 'initial_state': h0, 'dht': dht}, 'gpu_output': {'dq': torch.zeros_like(q), 'dk': torch.zeros_like(k), 'dv': torch.zeros_like(v), 'dg': None, 'dh0': torch.zeros_like(h0)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'g': g, 'g_gamma': g_gamma, 'do': do, 'scale': scale, 'initial_state': h0, 'dht': dht}}
        except NameError:
            pass
    return locals()
