
from opbench.fla_fixtures import load_fixture
import torch
import triton
import triton.language as tl
import torch_npu

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f'NaN detected in {name} tensor')
    else:
        print(f'{name}: {tensor}')

def cal_precision(input: torch.tensor, golden: torch.tensor):
    input = input.to('cpu')
    golden = golden.to('cpu')
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    check_nan(relative_max, 'max_relative_error')
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    check_nan(relative_mean, 'mean_relative_error')
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    check_nan(rmse, 'rmse')
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    check_nan(total_error, 'small_val_error')
    return (relative_max, relative_mean, rmse)

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

@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_fwd_kernel(q, k, v, o, z, scale, T, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_k, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_2o = tl.zeros([BK * BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * T * V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_z = z + (i_bh + i_k * B * H) * T + tl.arange(0, BT)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_0o = 0
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT]).to(b_k.dtype)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(b_k.dtype)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_z = tl.zeros([BT], dtype=tl.float32)
        b_o += b_h_0o
        b_z += k_0o
        b_o += tl.dot(b_q, b_h_1o.to(b_q.dtype), allow_tf32=False)
        b_z += tl.sum(b_q * k_1o, axis=1)
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK]).to(b_k.dtype)
        b_o += tl.dot(b_q_2o, b_h_2o.to(b_q_2o.dtype), allow_tf32=False) * 0.5
        b_z += tl.sum(b_q_2o * k_2o, axis=1) * 0.5
        k_1o += tl.sum(b_k, axis=1)[None, :]
        k_2o += tl.sum(b_k_2o, axis=1)[None, :]
        k_0o += BT
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=i * BT + tl.arange(0, BT) < T)
        b_h_2o = b_h_2o + tl.dot(b_k_2o.to(b_v.dtype), b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_z += BT

def FusedChunkBasedFunction_forward(q, k, v, scale=1):
    B, H, T, K, V = (*k.shape, v.shape[-1])
    scale = scale
    BT = 16
    BK, BV = (min(K, 16), min(V, 32))
    BK, BV = (max(BK, 16), max(BV, 16))
    NK, NV = (triton.cdiv(K, BK), triton.cdiv(V, BV))
    num_warps = 4
    o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
    z = q.new_empty(NK, B, H, T, dtype=torch.float32)
    grid = (NV, NK, B * H)
    fused_chunk_based_fwd_kernel[grid](q, k, v, o, z, scale, T=T, B=B, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, num_warps=num_warps)
    o = o.sum(0)
    z = z.sum(0)
    return (o.to(q.dtype), z.to(z.dtype))

def test_fused_chunk_based_forward():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_based_fwd_kernel/test_fused_chunk_based_fwd_kernel_data.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    scale = input_data['scale']
    print(f'Testing FusedChunkBasedFunction_forward on NPU')
    print(f'Input q shape: {q.shape}, dtype: {q.dtype}')
    print(f'Input k shape: {k.shape}, dtype: {k.dtype}')
    print(f'Input v shape: {v.shape}, dtype: {v.dtype}')
    o_kernel, z_kernel = FusedChunkBasedFunction_forward(q, k, v, scale)
    relative_max_gpu_o, relative_mean_gpu_o, rmse_gpu_o = cal_precision(data['gpu_output']['o'], data['ref_output']['o'])
    relative_max_gpu_z, relative_mean_gpu_z, rmse_gpu_z = cal_precision(data['gpu_output']['z'], data['ref_output']['z'])
    relative_max_npu_o, relative_mean_npu_o, rmse_npu_o = cal_precision(o_kernel, data['ref_output']['o'])
    relative_max_npu_z, relative_mean_npu_z, rmse_npu_z = cal_precision(z_kernel, data['ref_output']['z'])
    mare_thd, mere_thd, rmse_thd = (2, 1.2, 1.2)
    assert relative_max_npu_o / relative_max_gpu_o <= mare_thd
    assert relative_mean_npu_o / relative_mean_gpu_o <= mere_thd
    assert rmse_npu_o / rmse_gpu_o <= rmse_thd
    assert relative_max_npu_z / relative_max_gpu_z <= mare_thd
    assert relative_mean_npu_z / relative_mean_gpu_z <= mere_thd
    assert rmse_npu_z / rmse_gpu_z <= rmse_thd
