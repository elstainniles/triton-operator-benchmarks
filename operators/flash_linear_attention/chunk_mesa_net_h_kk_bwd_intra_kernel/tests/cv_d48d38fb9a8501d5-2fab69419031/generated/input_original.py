"""Existing independent CPU equations; file-generation main omitted."""
import torch
import torch.nn.functional as F
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())

def generate_test_data(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)
    B, T, H, D = (2, 512, 8, 64)
    chunk_size = 64
    IS_VARLEN = False
    k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    g = torch.rand(B, T, H, dtype=dtype, device=device).float().uniform_(0.8, 0.99).log()
    NT = triton.cdiv(T, chunk_size)
    h = torch.randn(NT * B, H, D, D, dtype=dtype, device=device)
    dh = torch.randn(NT * B, H, D, D, dtype=dtype, device=device)
    q_star = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dq = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dk_beta = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dk = torch.zeros(B, T, H, D, dtype=k.dtype, device=device)
    dg = torch.zeros(B, T, H, dtype=g.dtype, device=device)
    dbeta = torch.zeros(B, T, H, dtype=beta.dtype, device=device)
    dlamb = torch.zeros(B, NT, H, D, dtype=torch.float32, device=device)
    cu_seqlens = None
    chunk_indices = None
    input_data = {'k': k, 'beta': beta, 'g': g, 'h': h, 'dh': dh, 'q_star': q_star, 'dq': dq, 'dk_beta': dk_beta, 'dk': dk, 'dg': dg, 'dbeta': dbeta, 'dlamb': dlamb, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'chunk_indices': chunk_indices if chunk_indices is not None else torch.zeros(0, dtype=torch.int32), 'B': B, 'T': T, 'H': H, 'K': D, 'V': D, 'BT': chunk_size, 'BK': max(triton.next_power_of_2(D), 16), 'BV': max(triton.next_power_of_2(D), 16), 'IS_VARLEN': IS_VARLEN}
    grid = (NT, B * H)
    return (input_data, grid)
