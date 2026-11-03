"""Existing independent CPU equations; file-generation main omitted."""
import torch
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())

def make_inputs():
    torch.manual_seed(20261031)
    input_data, grid = generate_test_data()
    input_data_gpu = convert_tensor_with_device_type(input_data, device_type='cpu')
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': input_data, 'grid': grid}
        except NameError:
            pass
        try:
            data['input_data']['dh'] = torch.zeros(B_val, NT, HQ, K, V, dtype=torch.float16)
        except NameError:
            pass
        try:
            data['input_data']['dh0'] = torch.zeros(B_val, H, K, V, dtype=torch.float32)
        except NameError:
            pass
    return locals()

def generate_test_data():
    torch.manual_seed(42)
    B, T, H, K, V = (2, 512, 4, 64, 64)
    HQ = 4
    chunk_size = 64
    IS_VARLEN = False
    dtype = torch.float16
    gate_scale = 0.5
    q = torch.randn((B, T, HQ, K), dtype=dtype, device='cpu')
    k = torch.randn((B, T, H, K), dtype=dtype, device='cpu')
    v = torch.randn((B, T, H, V), dtype=dtype, device='cpu')
    gi = torch.randn((B, T, H, K), dtype=torch.float32, device='cpu') * gate_scale
    ge = torch.randn((B, T, H, K), dtype=torch.float32, device='cpu') * gate_scale
    do = torch.randn((B, T, HQ, V), dtype=dtype, device='cpu')
    h0 = torch.randn((B, H, K, V), dtype=dtype, device='cpu')
    dht = torch.randn((B, HQ, K, V), dtype=dtype, device='cpu')
    scale = 1.0 / K ** 0.5
    NT = triton.cdiv(T, chunk_size)
    NG = HQ // H
    dh = torch.zeros(B, NT, HQ, K, V, dtype=dtype, device='cpu')
    dh0 = torch.zeros(B, H, K, V, dtype=torch.float32, device='cpu')
    cu_seqlens = None
    chunk_offsets = None
    if IS_VARLEN:
        seq_lens = [256, 256]
        assert sum(seq_lens) == T
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)])
        chunk_offsets = torch.cat([torch.tensor([0], dtype=torch.int32), torch.tensor([triton.cdiv(s, chunk_size) for s in seq_lens], dtype=torch.int32).cumsum(dim=0)])
        NT = chunk_offsets[-1].item()
    input_data = {'q': q, 'gi': gi, 'ge': ge, 'do': do, 'dh': dh, 'dht': dht, 'dh0': dh0, 'h0': h0, 'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.zeros(0, dtype=torch.int32), 'chunk_offsets': chunk_offsets if chunk_offsets is not None else torch.zeros(0, dtype=torch.int32), 'scale': scale, 'T': T, 'HQ': HQ, 'H': H, 'K': K, 'V': V, 'BT': chunk_size, 'BK': 64, 'BV': 64, 'NG': NG, 'STORE_INITIAL_STATE_GRADIENT': True, 'USE_FINAL_STATE_GRADIENT': True, 'IS_VARLEN': IS_VARLEN}
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    grid = (triton.cdiv(K, 64), triton.cdiv(V, 64), N * HQ)
    return (input_data, grid)

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata
