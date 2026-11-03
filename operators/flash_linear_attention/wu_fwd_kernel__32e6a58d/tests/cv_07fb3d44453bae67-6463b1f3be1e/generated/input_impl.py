"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 512
    H = 8
    K = 64
    V = 64
    BT = 64
    device = torch.device('cpu')
    ag = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    A_ak = torch.randn(B, T, H, BT, dtype=torch.float32, device=device)
    A_ab_inv = torch.randn(B, T, H, BT, dtype=torch.float32, device=device)
    cu_seqlens = None
    inputs_dict = {'ag': ag.cpu(), 'v': v.cpu(), 'A_ak': A_ak.cpu(), 'A_ab_inv': A_ab_inv.cpu(), 'cu_seqlens': cu_seqlens, 'chunk_size': BT}
    ag_clone = ag.clone()
    v_clone = v.clone()
    A_ak_clone = A_ak.clone()
    A_ab_inv_clone = A_ab_inv.clone()
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'ag': ag.cpu(), 'v': v.cpu(), 'A_ak': A_ak.cpu(), 'A_ab_inv': A_ab_inv.cpu(), 'cu_seqlens': cu_seqlens, 'chunk_size': BT}
        except NameError:
            pass
    return locals()
