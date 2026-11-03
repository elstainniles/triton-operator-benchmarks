"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K = (1, 256, 4, 64)
    chunk_size = 16
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    a = torch.randn(B, T, H, K, dtype=dtype, device=device)
    b = torch.randn(B, T, H, K, dtype=dtype, device=device)
    gi = torch.randn(B, T, H, K, dtype=dtype, device=device)
    ge = torch.randn(B, T, H, K, dtype=dtype, device=device)
    BT = chunk_size
    dAqk = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    dAqb = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    dAak = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    dAab = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    dqg = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dkg = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dag = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dbg = torch.randn(B, T, H, K, dtype=dtype, device=device)
    NT = triton.cdiv(T, BT)
    dgk_last = torch.randn(B * NT, H, K, dtype=torch.float, device=device)
    scale = 1.0 / K ** 0.5
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'a': a.cpu(), 'b': b.cpu(), 'gi': gi.cpu(), 'ge': ge.cpu(), 'dAqk': dAqk.cpu(), 'dAqb': dAqb.cpu(), 'dAak': dAak.cpu(), 'dAab': dAab.cpu(), 'dqg': dqg.cpu(), 'dkg': dkg.cpu(), 'dag': dag.cpu(), 'dbg': dbg.cpu(), 'dgk_last': dgk_last.cpu(), 'scale': scale, 'cu_seqlens': None, 'chunk_size': chunk_size, 'safe_gate': True, 'chunk_indices': None}
    q_clone = q.clone()
    k_clone = k.clone()
    a_clone = a.clone()
    b_clone = b.clone()
    gi_clone = gi.clone()
    ge_clone = ge.clone()
    dAqk_clone = dAqk.clone()
    dAqb_clone = dAqb.clone()
    dAak_clone = dAak.clone()
    dAab_clone = dAab.clone()
    dqg_clone = dqg.clone()
    dkg_clone = dkg.clone()
    dag_clone = dag.clone()
    dbg_clone = dbg.clone()
    dgk_last_clone = dgk_last.clone()
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'a': a.cpu(), 'b': b.cpu(), 'gi': gi.cpu(), 'ge': ge.cpu(), 'dAqk': dAqk.cpu(), 'dAqb': dAqb.cpu(), 'dAak': dAak.cpu(), 'dAab': dAab.cpu(), 'dqg': dqg.cpu(), 'dkg': dkg.cpu(), 'dag': dag.cpu(), 'dbg': dbg.cpu(), 'dgk_last': dgk_last.cpu(), 'scale': scale, 'cu_seqlens': None, 'chunk_size': chunk_size, 'safe_gate': True, 'chunk_indices': None}
        except NameError:
            pass
    return locals()
