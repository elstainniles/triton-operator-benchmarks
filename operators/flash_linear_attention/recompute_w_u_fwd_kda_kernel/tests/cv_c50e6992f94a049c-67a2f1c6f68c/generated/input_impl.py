"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, HV, D = (2, 256, 2, 4, 64)
    BT = 64
    device = torch.device('cpu')
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, device=device, dtype=torch.float32)
    beta = torch.rand(B, T, HV, device=device, dtype=torch.float32).sigmoid()
    NT = T // BT
    A = torch.randn(B, NT, HV, BT, BT, device=device, dtype=torch.float32)
    A = torch.tril(A, diagonal=-1) + torch.eye(BT, device=device, dtype=torch.float32)
    A = A.permute(0, 1, 3, 2, 4).reshape(B, T, HV, BT).contiguous()
    gk = torch.randn(B, T, HV, D, device=device, dtype=torch.float32) * 0.1
    q = None
    inputs_dict = {'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'A': A.cpu(), 'gk': gk.cpu(), 'q': q, 'BT': BT, 'H': H, 'HV': HV, 'D': D}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'A': A.cpu(), 'gk': gk.cpu(), 'q': q, 'BT': BT, 'H': H, 'HV': HV, 'D': D}
        except NameError:
            pass
    return locals()
