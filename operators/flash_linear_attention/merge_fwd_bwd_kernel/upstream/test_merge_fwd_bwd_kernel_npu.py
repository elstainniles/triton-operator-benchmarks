import os
os.environ['FLA_USE_FAST_OPS'] = '0'

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional

USE_CUDA_GRAPH = False
autotune_cache_kwargs = {}


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr = False,
):
    i_v = tl.program_id(0)
    if INTRACARD_MODE:
        i_seq = tl.program_id(1)
        i_h = tl.program_id(2)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
            if TRANSPOSE_STATE:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * HV + i_h) * V * K,
                    (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            else:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * HV + i_h) * K * V,
                    (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            if TRANSPOSE_STATE:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            p_he = tl.make_block_ptr(
                ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0)
            )
            b_he = tl.load(p_he, boundary_check=(0, 1)).to(tl.float32)
            p_m = tl.make_block_ptr(
                ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0)
            )
            b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)
            else:
                b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)

            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if TRANSPOSE_STATE:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * V * K,
                        (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                    )
                else:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * K * V,
                        (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                    )
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))
    else:
        i_h = tl.program_id(1)
        num_ranks = pre_or_post_num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if TRANSPOSE_STATE:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = tl.make_block_ptr(ag_hm + cur_rank * stride, (K, V), (K + V, 1), (0, i_v * BV), (BK, BV), (1, 0))
            b_ag_h = tl.load(p_ag_h, boundary_check=(0, 1))
            p_ag_m = tl.make_block_ptr(ag_hm + cur_rank * stride + V, (K, K), (K + V, 1), (0, 0), (BK, BK), (1, 0))
            b_ag_m = tl.load(p_ag_m, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)
            else:
                b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_pre_process_merge(
    ag_hm: torch.Tensor,
    HV: int,
    K: int,
    V: int,
    pre_num_ranks: int = 1,
    rank: int = 1,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
):
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    N = 1

    if transpose_state_layout:
        initial_state = ag_hm.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        initial_state = ag_hm.new_zeros(N, HV, K, V, dtype=torch.float32)

    def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
    merge_fwd_bwd_kernel[grid](
        h=initial_state[0],
        ag_hm=ag_hm,
        pre_or_post_num_ranks=pre_num_ranks,
        rank=rank,
        seq_offsets=None,
        init_offsets=None,
        h0_seq_ids=None,
        h0=None,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        FORWARD=True,
        INTRACARD_MODE=False,
        NUM_SEQ_ENTRIES=0,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return initial_state


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


def test_merge_fwd_bwd_kernel():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/merge_fwd_bwd_kernel/test_merge_fwd_bwd_kernel_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    HV = input_data['HV']
    K = input_data['K']
    V = input_data['V']
    pre_num_ranks = input_data['pre_num_ranks']
    rank = input_data['rank']
    chunk_size = input_data['chunk_size']
    transpose_state_layout = input_data['transpose_state_layout']

    initial_state = chunk_gated_delta_rule_fwd_h_pre_process_merge(
        ag_hm=input_data['ag_hm'],
        HV=HV,
        K=K,
        V=V,
        pre_num_ranks=pre_num_ranks,
        rank=rank,
        chunk_size=chunk_size,
        transpose_state_layout=transpose_state_layout,
    )

    npu_out = {}
    npu_out['initial_state'] = initial_state

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test merge_fwd_bwd_kernel passed!")


if __name__ == "__main__":
    test_merge_fwd_bwd_kernel()
