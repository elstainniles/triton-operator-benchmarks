
from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')
USE_CUDA_GRAPH = False
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = 'cache_results' in triton.autotune.__code__.co_varnames
autotune_cache_kwargs = {'cache_results': FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    import triton.language.extra.libdevice as tldevice

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))
else:

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)

def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int, cu_seqlens_cpu: torch.LongTensor | None=None) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device) for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4, 8] for num_stages in [2, 3]], key=['BK', 'BT'], use_cuda_graph=USE_CUDA_GRAPH, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_tensorcore(q, k, a, b, gi, ge, qg, kg, ag, bg, Aqk, Aqb, Aab, Aak, cu_seqlens, chunk_indices, scale: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr, GATHER_SUPPORTED: tl.constexpr):
    i_t, i_b, i_h = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T_len = eos - bos
    else:
        bos = i_b * T
        T_len = T
    if i_t * BT >= T_len:
        return
    offset_base = (bos * H + i_h) * K
    p_q = tl.make_block_ptr(q + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_a = tl.make_block_ptr(a + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_b = tl.make_block_ptr(b + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi_val = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge_val = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)
    valid_len = min(T_len - i_t * BT, BT)
    mid_idx = valid_len // 2
    m_k = tl.arange(0, BK) < K
    p_offset = gi + offset_base + (i_t * BT + mid_idx) * K + tl.arange(0, BK)
    b_offset = tl.load(p_offset, mask=m_k, other=0.0).to(tl.float32)
    b_gi_val = b_gi_val - b_offset[None, :]
    b_ge_val = b_ge_val - b_offset[None, :]
    exp_gi = exp2(b_gi_val)
    inv_exp_gi = exp2(-b_gi_val)
    exp_ge = exp2(b_ge_val)
    b_q = (b_q * scale).to(tl.float32)
    q_ops = (b_q * exp_gi).to(tl.float32)
    k_ops = (b_k * inv_exp_gi).to(tl.float32)
    b_ops = (b_b * inv_exp_gi).to(tl.float32)
    a_ops = (b_a * exp_ge).to(tl.float32)
    last_idx = min((i_t + 1) * BT, T_len) - 1
    p_g_last = gi + offset_base + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0.0)
    exp_offset = exp2(b_offset)
    b_g_centered = b_g_last - b_offset
    exp_g_centered = exp2(b_g_centered)
    p_qg = tl.make_block_ptr(qg + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + offset_base, (T_len, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_qg, (q_ops * exp_offset[None, :]).to(p_qg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_ag, (a_ops * exp_offset[None, :]).to(p_ag.dtype.element_ty), boundary_check=(0, 1))
    b_kg_g = k_ops * exp_g_centered[None, :]
    b_bg_g = b_ops * exp_g_centered[None, :]
    tl.store(p_kg, b_kg_g.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_bg, b_bg_g.to(p_bg.dtype.element_ty), boundary_check=(0, 1))
    k_ops_t = tl.trans(k_ops)
    b_ops_t = tl.trans(b_ops)
    q_ops_h = q_ops.to(b_q.dtype)
    b_A_qk = tl.dot(q_ops_h, k_ops_t.to(b_q.dtype))
    b_A_qb = tl.dot(q_ops_h, b_ops_t.to(b_q.dtype))
    b_A_ak = tl.dot(a_ops, k_ops_t)
    b_A_ab = tl.dot(a_ops, b_ops_t)
    offs_n = tl.arange(0, BT)
    offs_m = tl.arange(0, BT)
    mask_inclusive = offs_m[:, None] >= offs_n[None, :]
    mask_strict = offs_m[:, None] > offs_n[None, :]
    b_A_qk = tl.where(mask_inclusive, b_A_qk, 0.0)
    b_A_qb = tl.where(mask_inclusive, b_A_qb, 0.0)
    b_A_ak = tl.where(mask_strict, b_A_ak, 0.0)
    b_A_ab = tl.where(mask_strict, b_A_ab, 0.0)
    offset_out_base = (bos * H + i_h) * BT
    p_Aqk = tl.make_block_ptr(Aqk + offset_out_base, (T_len, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aqb = tl.make_block_ptr(Aqb + offset_out_base, (T_len, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aak = tl.make_block_ptr(Aak + offset_out_base, (T_len, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aab = tl.make_block_ptr(Aab + offset_out_base, (T_len, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_A_qk.to(p_Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqb, b_A_qb.to(p_Aqb.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aak, b_A_ak.to(p_Aak.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aab, b_A_ab.to(p_Aab.dtype.element_ty), boundary_check=(0, 1))

def chunk_dplr_fwd_intra(q: torch.Tensor, k: torch.Tensor, a: torch.Tensor, b: torch.Tensor, gi: torch.Tensor, ge: torch.Tensor, scale: float, chunk_size: int, safe_gate: bool=False, cu_seqlens: torch.LongTensor | None=None, chunk_indices: torch.LongTensor | None=None):
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    Aqk = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aab = q.new_empty(B, T, H, BT, dtype=torch.float)
    Aak = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, B, H)
    BK = max(triton.next_power_of_2(K), 16)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    if safe_gate:
        chunk_dplr_fwd_A_kernel_intra_func = chunk_dplr_fwd_A_kernel_intra_tensorcore
    else:
        raise NotImplementedError('safe_gate=False mode not implemented in this test')
    chunk_dplr_fwd_A_kernel_intra_func[grid](q=q, k=k, a=a, b=b, gi=gi, ge=ge, Aqk=Aqk, Aqb=Aqb, Aab=Aab, Aak=Aak, qg=qg, kg=kg, ag=ag, bg=bg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T, H=H, K=K, BT=BT, BC=BT, BK=BK, GATHER_SUPPORTED=IS_GATHER_SUPPORTED)
    return (Aab, Aqk, Aak, Aqb, qg, kg, ag, bg)

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str]=None, device_type: Optional[str]=None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == 'saturate':
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=0.001, atol=0.001, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=0.005, atol=0.005, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or (dtype == 'int8'):
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = (set(dict_ref.keys()), set(dict_cal.keys()))
    if not keys_ref.issubset(keys_cal):
        raise ValueError('The keys of dict_ref is not subset of dict_cal')
    for key in dict_ref.keys():
        val_a, val_b = (dict_ref[key], dict_cal[key])
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError('The data type of two dicts are different')
        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError('Non-tensor type is not currently supported')

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

def test_chunk_dplr_fwd_A_kernel_intra_tensorcore():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_fwd_A_kernel_intra_tensorcore/test_chunk_dplr_fwd_A_kernel_intra_tensorcore_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    Aab, Aqk, Aak, Aqb, qg, kg, ag, bg = chunk_dplr_fwd_intra(q=input_data['q'], k=input_data['k'], a=input_data['a'], b=input_data['b'], gi=input_data['gi'], ge=input_data['ge'], scale=scale, chunk_size=chunk_size, safe_gate=True, cu_seqlens=None, chunk_indices=None)
    npu_out = {}
    npu_out['Aab'] = Aab
    npu_out['Aqk'] = Aqk
    npu_out['Aak'] = Aak
    npu_out['Aqb'] = Aqb
    npu_out['qg'] = qg
    npu_out['kg'] = kg
    npu_out['ag'] = ag
    npu_out['bg'] = bg
    try:
        compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    except Exception as e:
        compare_data_precision(data['gpu_output_fp64'], npu_out, device_type='cpu')
    print('Test chunk_dplr_fwd_A_kernel_intra_tensorcore passed!')
