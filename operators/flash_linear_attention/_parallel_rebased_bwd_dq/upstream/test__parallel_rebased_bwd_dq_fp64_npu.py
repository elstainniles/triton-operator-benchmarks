import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
import os
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c*BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c*BTL + tl.arange(0, BTL)) < T)

    for _ in range(0, i_c*BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, i_c*BTL), (BV, BTS), (0, 1))
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return


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


def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_parallel_rebased_bwd_dq():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/_parallel_rebased_bwd_dq_fp64/test__parallel_rebased_bwd_dq_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/_parallel_rebased_bwd_dq_fp64/test__parallel_rebased_bwd_dq_fp64_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    BTL = input_data['BTL']
    BTS = input_data['BTS']
    BK = input_data['BK']
    BV = input_data['BV']
    scale = input_data['scale']
    
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    
    dq = torch.empty(NV, B, H, T, K, dtype=torch.float32, device=torch.device('npu'))
    
    num_stages = 2
    num_warps = 4
    
    for i_bh in range(B * H):
        for i_c in range(triton.cdiv(T, BTL)):
            for i_kv in range(NK * NV):
                i_k = i_kv // NV
                i_v = i_kv % NV
                i_h = i_bh % H
                
                _parallel_rebased_bwd_dq[(1,)](
                    i_bh,
                    i_c,
                    i_k,
                    i_v,
                    i_h,
                    input_data['q'],
                    input_data['k'],
                    input_data['v'],
                    input_data['do'],
                    input_data['dz'],
                    dq,
                    scale,
                    T=T,
                    B=B,
                    H=H,
                    K=K,
                    V=V,
                    BTL=BTL,
                    BTS=BTS,
                    BK=BK,
                    BV=BV,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
    
    npu_out = {
        'dq': dq.cpu(),
    }
    
    for key in datafp64['gpu_output']:
        if key in npu_out:
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f"Comparing {key}:")
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(
                f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
            assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
            assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
            assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == "__main__":
    test_parallel_rebased_bwd_dq()
