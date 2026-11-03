"""Existing independent CPU equations; file-generation main omitted."""
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())

def make_inputs():
    torch.manual_seed(20261031)
    B, H, T, K, V = (4, 8, 2048, 128, 128)
    chunk_size = 32
    device = torch.device('cpu')
    q = torch.randn(B, H, T, K, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, K, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, V, device=device, dtype=torch.float32)
    beta = torch.rand(B, H, T, device=device, dtype=torch.float32).sigmoid()
    scale = K ** (-0.5)
    A = fwd_prepare_T(k, beta, chunk_size)
    output_attentions = False
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'A': A.cpu(), 'scale': scale, 'chunk_size': chunk_size, 'output_attentions': output_attentions}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'A': A.cpu(), 'scale': scale, 'chunk_size': chunk_size, 'output_attentions': output_attentions}
        except NameError:
            pass
    return locals()

def fwd_prepare_T(k: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Simplified version of fwd_prepare_T that computes the transition matrix A.
    This is a placeholder implementation since the original function doesn't exist.
    """
    B, H, T, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    A = torch.zeros(B, H, T, BT, device=k.device, dtype=k.dtype)
    for i in range(NT):
        start = i * BT
        end = min((i + 1) * BT, T)
        actual_BT = end - start
        k_chunk = k[:, :, start:end, :]
        beta_chunk = beta[:, :, start:end]
        kkt = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))
        kkt = kkt * beta_chunk.unsqueeze(-1)
        mask = torch.tril(torch.ones(actual_BT, actual_BT, device=k.device, dtype=torch.bool), diagonal=-1)
        kkt = kkt * mask
        A[:, :, start:end, :actual_BT] = kkt
    return A
