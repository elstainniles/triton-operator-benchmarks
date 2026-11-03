"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch
import torch.nn.functional as F

def make_inputs():
    torch.manual_seed(20261031)
    '\n    Test chunk_kda_fwd_kernel_inter_solve_fused kernel\n    '
    config = get_test_config()
    q, k, v, gk, beta = prepare_inputs(config)
    B, T, H, K, HV = (*k.shape, gk.shape[2])
    BT = config['chunk_size']
    BC = 16
    NC = triton.cdiv(BT, BC)
    scale = config['scale']
    safe_gate = config['safe_gate']
    Aqk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.zeros(B, T, HV, BC, device=k.device, dtype=torch.float32)
    NT = triton.cdiv(T, BT)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'gk': gk.cpu(), 'beta': beta.cpu(), 'Aqk': Aqk.cpu(), 'Akkd': Akkd.cpu()}
        except NameError:
            pass
    return locals()

def get_test_config():
    """
    Generate test configuration with large shapes for performance testing
    """
    config = {'B': 4, 'T': 1024, 'H': 16, 'HV': 32, 'K': 128, 'chunk_size': 64, 'scale': 0.0884, 'safe_gate': False, 'gate_logit_normalizer': 1.0, 'dtype': torch.float32, 'device': 'cpu'}
    return config

def prepare_inputs(config):
    """
    Prepare input tensors for the kernel
    """
    B = config['B']
    T = config['T']
    H = config['H']
    HV = config['HV']
    K = config['K']
    dtype = config['dtype']
    device = config['device']
    gate_logit_normalizer = config['gate_logit_normalizer']
    safe_gate = config['safe_gate']
    q = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gk = F.logsigmoid(torch.randn(B, T, HV, K, dtype=torch.float, device=device)) / gate_logit_normalizer
    if safe_gate:
        gk = gk.clamp(-5.0, 0.0)
    beta = torch.randn(B, T, HV, dtype=dtype, device=device).sigmoid()
    print(f'\nInput statistics:')
    print(f'  q norm: min={q.norm(dim=-1).min().item():.4f}, max={q.norm(dim=-1).max().item():.4f}')
    print(f'  k norm: min={k.norm(dim=-1).min().item():.4f}, max={k.norm(dim=-1).max().item():.4f}')
    print(f'  gk: min={gk.min().item():.4f}, max={gk.max().item():.4f}, mean={gk.mean().item():.4f}')
    print(f'  beta: min={beta.min().item():.4f}, max={beta.max().item():.4f}, mean={beta.mean().item():.4f}')
    return (q, k, v, gk, beta)
