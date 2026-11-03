"""Existing independent CPU equations; file-generation main omitted."""
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F

def make_inputs():
    torch.manual_seed(20261031)
    input_data, gpu_output, grid = generate_test_data(dtype=torch.float32, device='cpu')
    return locals()

def generate_test_data(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)
    B, T, H, D = (2, 1024, 8, 128)
    chunk_size = 64
    split_size = 64
    IS_VARLEN = False
    BK = 64
    BV = 64
    BT = chunk_size
    BS = split_size
    NS = triton.cdiv(T, BS)
    if dtype == torch.float32:
        k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1)
        v = torch.rand(B, T, H, D, dtype=torch.float32, device=device) / 10
        beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid()
        g = torch.rand(B, T, H, dtype=torch.float32, device=device).uniform_(0.8, 0.99).log()
        k_init_rand = torch.nn.functional.normalize(torch.rand(B, H, D, device=device, dtype=torch.float32), dim=-1, p=2)
        h_init = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float()
        h_kv_init = torch.rand(B, H, D, D, dtype=torch.float32, device=device)
        h = torch.zeros(B, NS, H, D, D, dtype=torch.float32, device=device)
        h_kv = torch.zeros(B, NS, H, D, D, dtype=torch.float32, device=device)
        h_final = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
        h_kv_final = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
    elif dtype == torch.float16:
        k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(torch.float16)
        v = torch.rand(B, T, H, D, dtype=torch.float16, device=device) / 10
        beta = torch.rand(B, T, H, dtype=torch.float16, device=device).sigmoid()
        g = torch.rand(B, T, H, dtype=torch.float16, device=device).float().uniform_(0.8, 0.99).log()
        k_init_rand = torch.nn.functional.normalize(torch.rand(B, H, D, device=device, dtype=torch.float16), dim=-1, p=2)
        h_init = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float()
        h_kv_init = torch.rand(B, H, D, D, dtype=torch.float32, device=device)
        h = torch.zeros(B, NS, H, D, D, dtype=torch.float16, device=device)
        h_kv = torch.zeros(B, NS, H, D, D, dtype=torch.float16, device=device)
        h_final = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
        h_kv_final = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    cu_seqlens = None
    split_offsets = None
    if IS_VARLEN:
        seq_lens = [512, 512]
        assert sum(seq_lens) == T
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)])
        split_offsets = torch.cat([torch.tensor([0], dtype=torch.int32), torch.tensor([triton.cdiv(s, BS) for s in seq_lens], dtype=torch.int32).cumsum(dim=0)])
        NS = split_offsets[-1].item()
    input_data = {'k': k, 'v': v, 'beta': beta, 'g': g, 'h_init': h_init, 'h_kv_init': h_kv_init, 'h': h, 'h_kv': h_kv, 'h_final': h_final, 'h_kv_final': h_kv_final, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'split_offsets': split_offsets if split_offsets is not None else torch.zeros(0, dtype=torch.int32), 'T': T, 'H': H, 'K': D, 'V': D, 'BT': BT, 'BS': BS, 'BK': BK, 'BV': BV, 'USE_INITIAL_STATE': True, 'STORE_FINAL_STATE': True, 'IS_VARLEN': IS_VARLEN}
    gpu_output = {'h': h.clone(), 'h_kv': h_kv.clone(), 'h_final': h_final.clone(), 'h_kv_final': h_kv_final.clone()}
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    grid = (triton.cdiv(D, BK), triton.cdiv(D, BV), N * H)
    return (input_data, gpu_output, grid)
