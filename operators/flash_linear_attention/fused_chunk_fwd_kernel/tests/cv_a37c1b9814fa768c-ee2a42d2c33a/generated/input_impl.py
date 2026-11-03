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
    g = F.logsigmoid(g) / gate_logit_normalizer
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'g': g, 'scale': scale, 'initial_state': h0, 'output_final_state': True}, 'gpu_output': {'o': torch.zeros_like(v), 'ht': torch.zeros(B, H, D, D, dtype=torch.float32, device=device)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'g': g, 'scale': scale, 'initial_state': h0, 'output_final_state': True}}
        except NameError:
            pass
    return locals()
