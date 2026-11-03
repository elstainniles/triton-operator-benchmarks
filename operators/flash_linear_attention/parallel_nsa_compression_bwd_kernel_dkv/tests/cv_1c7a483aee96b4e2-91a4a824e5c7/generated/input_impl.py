"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, HQ, D = (3, 1024, 2, 32, 60)
    block_size = 32
    scale = 0.1
    TC = T // block_size
    G = HQ // H
    device = torch.device('cpu')
    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.float32)
    k = torch.randn(B, TC, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, TC, H, D, device=device, dtype=torch.float32)
    o = torch.randn(B, T, HQ, D, device=device, dtype=torch.float32)
    lse = torch.randn(B, T, HQ, device=device, dtype=torch.float32) * 2
    do = torch.randn(B, T, HQ, D, device=device, dtype=torch.float32)
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'o': o.cpu(), 'lse': lse.cpu(), 'do': do.cpu(), 'block_size': block_size, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'D': D}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'o': o.cpu(), 'lse': lse.cpu(), 'do': do.cpu(), 'block_size': block_size, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'D': D}
        except NameError:
            pass
    return locals()
