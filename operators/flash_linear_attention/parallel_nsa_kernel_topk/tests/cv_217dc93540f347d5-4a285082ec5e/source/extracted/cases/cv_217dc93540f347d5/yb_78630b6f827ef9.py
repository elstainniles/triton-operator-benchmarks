# Test case for parallel_nsa_kernel_topk
# Extracted from generated-data/parallel.py
# 
# 使用 npu/gpu/gpu_fp32 三方比对方式
from opbench.fla_fixtures import load_fixture
from typing import Optional
import torch
import torch_npu
from yb_b716f65a487720 import parallel_nsa_kernel_topk

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

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    """计算精度指标"""
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'max relative error: {relative_max}')
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'mean relative error: {relative_mean}')
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f'rmse: {rmse}')
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f'small val error: {total_error}')
    return (relative_max, relative_mean, rmse)

def test_parallel_nsa_kernel_topk():
    """
    测试 parallel_nsa_kernel_topk
    
    使用 npu/gpu/gpu_fp32 三方比对方式
    """
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_kernel_topk_fp16'
    print(f'Loading inputs from: {load_dir}')
    data_dict = load_fixture(f'{load_dir}/parallel_nsa_kernel_topk.pt')
    input_dict = data_dict['input_data']
    q = input_dict['q'].to(device)
    k_cmp = input_dict['k_cmp'].to(device)
    lse_cmp = input_dict['lse'].to(device)
    scale = input_dict['scale']
    cu_seqlens = input_dict['cu_seqlens'].to(device) if input_dict['cu_seqlens'] is not None else None
    token_indices = input_dict['token_indices'].to(device) if input_dict['token_indices'] is not None else None
    chunk_offsets = input_dict['chunk_offsets'].to(device) if input_dict['chunk_offsets'] is not None else None
    B = input_dict['B']
    T = input_dict['T']
    H = input_dict['H']
    HQ = input_dict['HQ']
    G = input_dict['G']
    K = input_dict['K']
    S = input_dict['S']
    BC = input_dict['BC']
    BS = input_dict['BS']
    BK = input_dict['BK']
    grid = (T, B * H)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    print(f'\n测试配置:')
    print(f'  Batch size (B): {B}')
    print(f'  Sequence length (T): {T}')
    print(f'  Key/Value heads (H): {H}')
    print(f'  Query heads (HQ): {HQ}')
    print(f'  Key dimension (K): {K}')
    print(f'  Scale: {scale}')
    print(f'  Query shape: {q.shape}')
    print(f'  Compressed key shape: {k_cmp.shape}')
    parallel_nsa_kernel_topk[grid](q=q.clone(), k=k_cmp.clone(), lse=lse_cmp.clone(), scale=scale, block_indices=block_indices, cu_seqlens=cu_seqlens, token_indices=token_indices, chunk_offsets=chunk_offsets, T=T, H=H, HQ=HQ, G=G, K=K, S=S, BC=BC, BS=BS, BK=BK)
    torch.npu.synchronize()
    print(f'\n[NPU] Output block_indices shape: {block_indices.shape}')
    print(f'  Output dtype: {block_indices.dtype}')
    print(f'  Sample block indices (first 5): {block_indices[0, 0, 0, :5]}')
    output_dict = data_dict['output_data']
    expected_block_indices = output_dict['block_indices'].to(device)
    output_fp32_dict = data_dict['output_data_fp32']
    expected_block_indices_fp32 = output_fp32_dict['block_indices_fp32'].to(device)
    print('\nComparing block_indices outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(block_indices, expected_block_indices_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_block_indices, expected_block_indices_fp32)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    validate_cmp('float16', block_indices, expected_block_indices)
    print('\n✅ 精度测试通过！')
    print('\n' + '=' * 60)
    print('开始 Profiling 测试')
    print('=' * 60)
    profiling_output_dir = './result_dir_original'
    print('\nWarming up...')
    for _ in range(5):
        _ = parallel_nsa_kernel_topk[grid](q=q.clone(), k=k_cmp.clone(), lse=lse_cmp.clone(), scale=scale, block_indices=block_indices, cu_seqlens=cu_seqlens, token_indices=token_indices, chunk_offsets=chunk_offsets, T=T, H=H, HQ=HQ, G=G, K=K, S=S, BC=BC, BS=BS, BK=BK)
        torch.npu.synchronize()
    print('\nStarting profiling...')
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_output_dir)) as prof:
        prof.start()
        for i in range(35):
            _ = parallel_nsa_kernel_topk[grid](q=q.clone(), k=k_cmp.clone(), lse=lse_cmp.clone(), scale=scale, block_indices=block_indices, cu_seqlens=cu_seqlens, token_indices=token_indices, chunk_offsets=chunk_offsets, T=T, H=H, HQ=HQ, G=G, K=K, S=S, BC=BC, BS=BS, BK=BK)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nProfiling 完成！结果已保存到 {profiling_output_dir}')
    return block_indices
