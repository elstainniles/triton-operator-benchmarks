"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(20261031)
    kernel_name = 'chunk_ttt_linear_bwd_kernel_h'
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE).clamp(-1, 1).to(dtype)
    w = torch.randn(H, V, dtype=torch.float32, device=DEVICE).clamp(-1, 1).to(dtype)
    b = torch.randn(H, V, dtype=torch.float32, device=DEVICE).clamp(-1, 1).to(dtype)
    eta = torch.randn(B, T, H, 1, dtype=dtype, device=DEVICE) * eta_base
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=DEVICE) * (1.0 / D)
    hb0 = torch.randn(B, H, 1, D, dtype=torch.float32, device=DEVICE) * (1.0 / D)
    inputs = {'k': k.cpu(), 'v': v.cpu(), 'w': w.cpu(), 'b': b.cpu(), 'eta': eta.cpu(), 'eps_val': eps, 'h0': h0.cpu(), 'hb0': hb0.cpu(), 'chunk_size': chunk_size}
    return locals()
DEVICE = 'cpu'
B, T, H, D = (4, 2048, 8, 64)
K, V = (D, D)
dtype = torch.float16
eta_base = 0.005
eps = 1e-06
chunk_size = 16
