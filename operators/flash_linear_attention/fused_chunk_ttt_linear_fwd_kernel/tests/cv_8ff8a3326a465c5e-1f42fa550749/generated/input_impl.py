"""Existing independent CPU equations; file-generation main omitted."""
import torch
import torch.nn.functional as F

def make_inputs():
    torch.manual_seed(20261031)
    kernel_name = 'fused_chunk_ttt_linear_fwd_kernel'
    args = make_args()
    q, k, v, w, b, eta, sc, eps_val, h0, hb0, out_final, cu_sl, bt = args
    inputs = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'w': w.cpu(), 'b': b.cpu(), 'eta': eta.cpu(), 'sc': sc, 'eps_val': eps_val, 'h0': h0.cpu(), 'hb0': hb0.cpu(), 'out_final': out_final, 'cu_sl': cu_sl, 'bt': bt}
    return locals()
DEVICE = 'cpu'
B, T, H, D = (4, 2048, 8, 64)
K, V = (D, D)
dtype = torch.float16
eta_base = 0.005
eps = 1e-06
scale = D ** (-0.5)
BT = 16

def make_args():
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype, device=DEVICE)
    w = torch.randn(H, V, dtype=dtype, device=DEVICE)
    b = torch.randn(H, V, dtype=dtype, device=DEVICE)
    eta = torch.randn(B, T, H, 1, dtype=dtype, device=DEVICE) * eta_base
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=DEVICE)
    hb0 = torch.randn(B, H, 1, V, dtype=torch.float32, device=DEVICE)
    return (q, k, v, w, b, eta, scale, eps, h0, hb0, True, None, BT)
