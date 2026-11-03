"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    'Test function for parallel_nsa_bwd_kernel_dkv'
    config = get_test_config()
    B, T, H, HQ, K, V = (config['B'], config['T'], config['H'], config['HQ'], config['K'], config['V'])
    block_size = config['block_size']
    block_counts = config['block_counts']
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    o = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    lse = torch.randn(B, T, HQ, dtype=torch.float, device=device)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    NS = triton.cdiv(T, block_size)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                num_blocks = min(block_counts, NS)
                selected_blocks = torch.randperm(NS)[:num_blocks]
                block_mask[b, t, h, selected_blocks] = True
    scale = K ** (-0.5)
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'o': o.cpu(), 'lse': lse.cpu(), 'do': do.cpu(), 'block_mask': block_mask.cpu(), 'block_size': block_size, 'scale': scale}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'o': o.cpu(), 'lse': lse.cpu(), 'do': do.cpu(), 'block_mask': block_mask.cpu(), 'block_size': block_size, 'scale': scale}
        except NameError:
            pass
    return locals()

def get_test_config():
    """Generate test configuration with large shapes"""
    B = 1
    T = 512
    H = 4
    HQ = 64
    K = 64
    V = 64
    block_size = 64
    block_counts = 8
    return {'B': B, 'T': T, 'H': H, 'HQ': HQ, 'K': K, 'V': V, 'block_size': block_size, 'block_counts': block_counts}
