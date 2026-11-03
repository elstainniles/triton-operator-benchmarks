"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, K = (2, 512, 4, 64)
    chunk_size = 64
    max_CG_iteration = 30
    IS_VARLEN = False
    device = 'cpu'
    q = torch.rand(B, T, H, K, dtype=torch.float32, device=device) / 10
    k = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1)
    v = torch.rand(B, T, H, K, dtype=torch.float32, device=device) / 10
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid()
    g = torch.rand(B, T, H, dtype=torch.float32, device=device).uniform_(0.8, 0.99).log()
    lamb = torch.rand(H, K, dtype=torch.float32, device=device).sigmoid() * 0.75 + 0.25
    NT = triton.cdiv(T, chunk_size)
    h = torch.rand(NT * B, H, K, K, dtype=torch.float32, device=device) / 10
    h_kv = torch.rand(NT * B, H, K, K, dtype=torch.float32, device=device) / 10
    q_final = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    o = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    cu_seqlens = None
    chunk_indices = None
    input_data = {'q': q, 'k': k, 'v': v, 'h': h, 'h_kv': h_kv, 'g': g, 'beta': beta, 'lamb': lamb, 'q_final': q_final, 'o': o, 'cu_seqlens': torch.zeros(0, dtype=torch.int32), 'chunk_indices': torch.zeros(0, dtype=torch.int32), 'T': T, 'H': H, 'K': K, 'BT': chunk_size, 'BK': max(triton.next_power_of_2(K), 16), 'max_CG_iteration': max_CG_iteration, 'IS_VARLEN': IS_VARLEN}
    gpu_output = {'q_final': q_final.clone(), 'o': o.clone()}
    grid = (NT, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'q': q, 'k': k, 'v': v, 'h': h, 'h_kv': h_kv, 'g': g, 'beta': beta, 'lamb': lamb, 'q_final': q_final, 'o': o, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'chunk_indices': chunk_indices if chunk_indices is not None else torch.zeros(0, dtype=torch.int32), 'T': T, 'H': H, 'K': K, 'BT': chunk_size, 'BK': max(triton.next_power_of_2(K), 16), 'max_CG_iteration': max_CG_iteration, 'IS_VARLEN': IS_VARLEN}
        except NameError:
            pass
    return locals()
