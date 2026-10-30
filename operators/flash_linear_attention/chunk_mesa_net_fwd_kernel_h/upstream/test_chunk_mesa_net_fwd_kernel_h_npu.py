# coding=utf-8
import sys
import os
import pytest
import triton
import torch
import triton.language as tl
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h_init'] is not None,
    'STORE_FINAL_STATE': lambda args: args['h_final'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_fwd_kernel_h(
    k,
    v,
    beta,
    g,
    h,
    h_kv,
    h_init,
    h_kv_init,
    h_final,
    h_kv_final,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_kv = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h_init + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        p_h_kv0 = tl.make_block_ptr(h_kv_init + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h_kv = tl.load(p_h_kv0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // (BS // BT)
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT, ), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_h_kv = tl.make_block_ptr(h_kv + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % (BS // BT) == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_h_kv, b_h_kv.to(p_h_kv.dtype.element_ty), boundary_check=(0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_h *= exp(b_g_last)
        b_h_kv *= exp(b_g_last)
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_k_decay = ((b_k * exp(b_g_last - b_g)[:, None]) * b_beta[:, None]).to(b_k2.dtype)
        b_h += tl.dot(tl.trans(b_k_decay), b_k2)
        b_h_kv += tl.dot(tl.trans(b_k_decay), b_v.to(b_k2.dtype))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(h_final + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        p_h_kv_final = tl.make_block_ptr(h_kv_final + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h_kv_final, b_h_kv.to(p_h_kv_final.dtype.element_ty), boundary_check=(0, 1))

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

def get_tolerances(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1e-3, 1e-3

def compare_output(npu_output: torch.Tensor, gpu_output: torch.Tensor, name: str, atol: float, rtol: float):
    npu_cpu = npu_output.cpu().float()
    gpu_cpu = gpu_output.float()

    max_abs_diff = (npu_cpu - gpu_cpu).abs().max().item()
    max_rel_diff = ((npu_cpu - gpu_cpu).abs() / (gpu_cpu.abs() + 1e-7)).max().item()
    mean_abs_diff = (npu_cpu - gpu_cpu).abs().mean().item()

    print(f"  {name}: max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e}, mean_abs_diff={mean_abs_diff:.2e}")

    match = torch.allclose(npu_cpu, gpu_cpu, atol=atol, rtol=rtol)
    if not match:
        mismatched = (~torch.isclose(npu_cpu, gpu_cpu, atol=atol, rtol=rtol)).sum().item()
        total = npu_cpu.numel()
        print(f"  {name}: FAILED - {mismatched}/{total} elements mismatched (atol={atol}, rtol={rtol})")
    else:
        print(f"  {name}: PASSED (atol={atol}, rtol={rtol})")
    return match

#def test_chunk_mesa_net_fwd_kernel_h_npu_fp32():
#    ptfile_path = 'test_chunk_mesa_net_fwd_kernel_h_v2.pt'
#    _run_npu_test(ptfile_path, 'fp32')

def test_chunk_mesa_net_fwd_kernel_h_npu_fp16():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_fwd_kernel_h/test_chunk_mesa_net_fwd_kernel_h_fp16_v2.pt'
    _run_npu_test(ptfile_path, 'fp16')

def _run_npu_test(ptfile_path: str, dtype_label: str):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    gpu_output = data["gpu_output"]
    h_dtype = gpu_output["h"].dtype
    atol, rtol = get_tolerances(h_dtype)

    print(f"\n=== NPU vs GPU comparison ({dtype_label}, h_dtype={h_dtype}, atol={atol}, rtol={rtol}) ===")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    chunk_mesa_net_fwd_kernel_h[data["grid"]](
        k=input_data['k'],
        v=input_data['v'],
        beta=input_data['beta'],
        g=input_data['g'],
        h=input_data['h'],
        h_kv=input_data['h_kv'],
        h_init=input_data['h_init'],
        h_kv_init=input_data['h_kv_init'],
        h_final=input_data['h_final'],
        h_kv_final=input_data['h_kv_final'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        split_offsets=input_data['split_offsets'] if input_data['split_offsets'].numel() > 0 else None,
        T=input_data['T'],
        H=input_data['H'],
        K=input_data['K'],
        V=input_data['V'],
        BT=input_data['BT'],
        BS=input_data['BS'],
        BK=input_data['BK'],
        BV=input_data['BV'],
    )

    all_passed = True

    # h and h_kv: use h_dtype tolerance
    all_passed &= compare_output(input_data["h"], gpu_output["h"], "h", atol, rtol)
    all_passed &= compare_output(input_data["h_kv"], gpu_output["h_kv"], "h_kv", atol, rtol)

    # h_final and h_kv_final: always fp32, use fp32 tolerance
    all_passed &= compare_output(input_data["h_final"], gpu_output["h_final"], "h_final", 1e-5, 1e-5)
    all_passed &= compare_output(input_data["h_kv_final"], gpu_output["h_kv_final"], "h_kv_final", 1e-5, 1e-5)

    if all_passed:
        print(f"\n=== {dtype_label} NPU test ALL PASSED ===")
    else:
        print(f"\n=== {dtype_label} NPU test FAILED ===")
        pytest.fail(f"NPU vs GPU comparison failed for {dtype_label} test")

if __name__ == '__main__':
    test_chunk_mesa_net_fwd_kernel_h_npu_fp16()
