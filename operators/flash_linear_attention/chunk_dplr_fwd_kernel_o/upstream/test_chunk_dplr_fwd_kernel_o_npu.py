import os
import torch
import torch_npu
import triton
import triton.language as tl
from enum import Enum
import functools
from typing import Optional

device_platform = 'cuda'
IS_AMD = (device_platform == 'hip')
IS_NVIDIA = (device_platform == 'cuda')
USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(torch.cuda.device_count())
        ]
    except BaseException:
        return [-1]

class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value

@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False

BK_LIST = [32, 64, 128] if check_shared_mem() else [16, 32]

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BK_LIST
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_o(
    qg,
    v,
    v_new,
    A_qk,
    A_qb,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, b_h)

    p_Aqk = tl.make_block_ptr(A_qk + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aqb = tl.make_block_ptr(A_qb + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
    b_Aqb = tl.load(p_Aqb, boundary_check=(0, 1))
    b_Aqk = tl.where(m_s, b_Aqk, 0)
    b_Aqb = tl.where(m_s, b_Aqb, 0)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
    b_o = b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_v_new.dtype), b_v_new)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        raise NotImplementedError("Varlen mode requires prepare_chunk_indices from fla.ops.utils")

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_fwd_kernel_o[grid](
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


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


def test_chunk_dplr_fwd_kernel_o():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_fwd_kernel_o/test_chunk_dplr_fwd_kernel_o_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    chunk_size = input_data['chunk_size']
    B = input_data['B']
    h = input_data['h']
    if h.ndim == 4 and h.shape[0] != B * (input_data['T'] // chunk_size):
        h = h.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous().reshape(-1, h.shape[-2], h.shape[-1])
        input_data['h'] = h

    o = chunk_dplr_fwd_o(
        qg=input_data['qg'],
        v=input_data['v'],
        v_new=input_data['v_new'],
        A_qk=input_data['A_qk'],
        A_qb=input_data['A_qb'],
        h=input_data['h'],
        cu_seqlens=None,
        chunk_size=chunk_size,
        chunk_indices=None
    )

    npu_out = {}
    npu_out['o'] = o

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test chunk_dplr_fwd_kernel_o passed!")


if __name__ == "__main__":
    test_chunk_dplr_fwd_kernel_o()
