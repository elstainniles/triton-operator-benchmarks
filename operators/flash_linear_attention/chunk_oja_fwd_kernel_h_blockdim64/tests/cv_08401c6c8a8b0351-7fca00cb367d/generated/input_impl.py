"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(20261031)
    DEVICE = 'cpu'
    B, T, H, D = (4, 2048, 8, 128)
    K, V = (D, D)
    BT = 64
    dtype = torch.float32
    scale = D ** (-0.5)
    v = torch.randn(B, T, H, V, dtype=dtype, device=DEVICE) * scale
    w = torch.randn(B, T, H, V, dtype=dtype, device=DEVICE) * scale
    u = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * scale
    gv = F.logsigmoid(torch.randn(B, T, H, V, dtype=dtype, device=DEVICE))
    initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device=DEVICE) * (1.0 / D)
    kernel_name = 'chunk_oja_fwd_kernel_h_blockdim64'
    inputs = {'v': v.cpu(), 'w': w.cpu(), 'u': u.cpu(), 'gv': gv.cpu(), 'initial_state': initial_state.cpu(), 'output_final_state': True, 'chunk_size': BT}
    return locals()
