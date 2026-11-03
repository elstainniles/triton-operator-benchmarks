"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, D = (4, 2048, 4, 64)
    BT = 64
    NT = T // BT
    device = torch.device('cpu')
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.float32) * 0.1
    mask = torch.tril(torch.ones(BT, BT, device=device, dtype=torch.bool))
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)
    for i in range(NT):
        A[:, i * BT:(i + 1) * BT, :, :] *= mask.unsqueeze(0).unsqueeze(2)
    do = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    dh = torch.randn(NT, B, H, D, D, device=device, dtype=torch.float32)
    inputs_dict = {'k': k.cpu(), 'g': g.cpu(), 'A': A.cpu(), 'do': do.cpu(), 'dh': dh.cpu(), 'BT': BT, 'B': B, 'T': T, 'H': H, 'D': D}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'k': k.cpu(), 'g': g.cpu(), 'A': A.cpu(), 'do': do.cpu(), 'dh': dh.cpu(), 'BT': BT, 'B': B, 'T': T, 'H': H, 'D': D}
        except NameError:
            pass
    return locals()
