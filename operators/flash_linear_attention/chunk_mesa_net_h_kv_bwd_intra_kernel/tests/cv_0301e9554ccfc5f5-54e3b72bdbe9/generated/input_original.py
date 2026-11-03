"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())

def generate_test_data(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)
    B, T, H, K, V = (2, 64, 8, 64, 64)
    chunk_size = 16
    IS_VARLEN = False
    q_star = torch.rand(B, T, H, K, dtype=dtype, device=device) / 10
    k = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, H, V, dtype=dtype, device=device) / 10
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    g = torch.rand(B, T, H, dtype=dtype, device=device).float().uniform_(0.8, 0.99).log()
    do = torch.rand(B, T, H, V, dtype=dtype, device=device)
    NT = triton.cdiv(T, chunk_size)
    h_kv = torch.rand(NT * B, H, K, V, dtype=torch.float32, device=device)
    dh_kv = torch.rand(NT * B, H, K, V, dtype=torch.float32, device=device)
    dq = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    dk = torch.zeros(B, T, H, K, dtype=k.dtype, device=device)
    dv = torch.zeros(B, T, H, V, dtype=v.dtype, device=device)
    dg = torch.zeros(B, T, H, dtype=g.dtype, device=device)
    cu_seqlens = None
    chunk_indices = None
    input_data = {'q_star': q_star, 'k': k, 'v': v, 'beta': beta, 'h_kv': h_kv, 'dh_kv': dh_kv, 'g': g, 'do': do, 'dq': dq, 'dk': dk, 'dv': dv, 'dg': dg, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'chunk_indices': chunk_indices if chunk_indices is not None else torch.zeros(0, dtype=torch.int32), 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'BT': chunk_size, 'BK': max(triton.next_power_of_2(K), 16), 'BV': max(triton.next_power_of_2(V), 16), 'IS_VARLEN': IS_VARLEN}
    grid = (NT, B * H)
    return (input_data, grid)
