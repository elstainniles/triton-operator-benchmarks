"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    'Test chunk_gla_fwd_kernel_o'
    B = 2
    T = 1024
    H = 4
    D = 60
    K = D
    V = D
    HV = H
    chunk_size = 64
    gate_logit_normalizer = 1
    scale = K ** (-0.5)
    device = torch.device('cpu')
    dtype = torch.float16
    q = torch.rand((B, T, H, K), dtype=dtype, device=device)
    v = torch.rand((B, T, HV, V), dtype=dtype, device=device)
    import torch.nn.functional as F
    g = F.logsigmoid(torch.rand((B, T, HV, K), dtype=dtype, device=device)) / gate_logit_normalizer
    A = torch.rand((B, T, HV, chunk_size), dtype=torch.float32, device=device)
    NT = (T + chunk_size - 1) // chunk_size
    h = torch.rand((NT * B, H, K, V), dtype=torch.float32, device=device)
    inputs_dict = {'q': q.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'A': A.cpu(), 'h': h.cpu(), 'scale': scale, 'chunk_size': chunk_size}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'A': A.cpu(), 'h': h.cpu(), 'scale': scale, 'chunk_size': chunk_size}
        except NameError:
            pass
    return locals()
