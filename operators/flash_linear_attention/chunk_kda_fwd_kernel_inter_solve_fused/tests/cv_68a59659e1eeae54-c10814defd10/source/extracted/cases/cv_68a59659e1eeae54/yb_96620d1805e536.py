
from opbench.fla_fixtures import load_fixture
import torch
import torch_npu
import triton
import os
from yb_8b45f3294e9cfe import chunk_kda_fwd_kernel_inter_solve_fused
from yb_8b45f3294e9cfe import chunk_kda_fwd_intra_token_parallel
from yb_8b45f3294e9cfe import chunk_kda_fwd_kernel_intra_sub_chunk

def cal_precision(input: torch.tensor, golden: torch.tensor):
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

def get_test_config():
    """
    Generate test configuration with large shapes for performance testing
    """
    config = {'B': 4, 'T': 1024, 'H': 16, 'HV': 32, 'K': 128, 'chunk_size': 64, 'scale': 0.0884, 'safe_gate': True, 'dtype': torch.float32, 'device': 'npu:0'}
    return config

def load_inputs(config):
    """
    Load input tensors from saved pt file
    """
    device = config['device']
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_kda_fwd_kernel_inter_solve_fused_gate_true'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    gk = inputs_dict['gk'].to(device)
    beta = inputs_dict['beta'].to(device)
    Aqk = inputs_dict['Aqk'].to(device)
    Akkd = inputs_dict['Akkd'].to(device)
    print(f'Inputs loaded from {load_dir}/inputs.pt')
    return (q, k, v, gk, beta, Aqk, Akkd)

def test_kernel():
    """
    Test chunk_kda_fwd_kernel_inter_solve_fused kernel with profiling
    """
    config = get_test_config()
    print(f'Test config: {config}')
    q, k, v, gk, beta, Aqk, Akkd = load_inputs(config)
    B, T, H, K, HV = (*k.shape, gk.shape[2])
    BT = config['chunk_size']
    BC = 16
    NC = triton.cdiv(BT, BC)
    scale = config['scale']
    safe_gate = config['safe_gate']
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](q=q, k=k, g=gk, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, NC=NC, USE_SAFE_GATE=safe_gate)
    torch.npu.synchronize()
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_kda_fwd_kernel_inter_solve_fused_gate_true'
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = load_fixture(os.path.join(load_dir, 'outputs_fp64.pt'))
    expected_Aqk = expected_outputs['Aqk'].to(config['device'])
    expected_Akk = expected_outputs['Akk'].to(config['device'])
    expected_Akkd = expected_outputs['Akkd'].to(config['device'])
    expected_Aqk_fp64 = expected_outputs_fp64['Aqk_fp64'].to(config['device'])
    expected_Akk_fp64 = expected_outputs_fp64['Akk_fp64'].to(config['device'])
    expected_Akkd_fp64 = expected_outputs_fp64['Akkd_fp64'].to(config['device'])
    print('\nComparing Aqk outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(Aqk, expected_Aqk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_Aqk, expected_Aqk_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'Aqk 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'Aqk 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'Aqk 均方根误差比 check failed'
    print('\nComparing Akk outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(Akk, expected_Akk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_Akk, expected_Akk_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'Akk 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'Akk 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'Akk 均方根误差比 check failed'
    print('\nComparing Akkd outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(Akkd, expected_Akkd_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_Akkd, expected_Akkd_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'Akkd 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'Akkd 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'Akkd 均方根误差比 check failed'
    print('All outputs match within tolerance!')

    def kernel_call():
        grid = (NT, B * HV)
        chunk_kda_fwd_kernel_inter_solve_fused[grid](q=q, k=k, g=gk, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, NC=NC, USE_SAFE_GATE=safe_gate)
        torch.npu.synchronize()
    print('Running warmup...')
    for _ in range(5):
        kernel_call()
    print('Running profiling...')
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result_dir_chunk_kda_fwd_kernel_inter_solve_fused')) as prof:
        prof.start()
        for i in range(35):
            kernel_call()
            prof.step()
        prof.stop()
    print('Profiling completed. Results saved to ./result_dir_chunk_kda_fwd_kernel_inter_solve_fused')
    print(f'Output shapes: Aqk={Aqk.shape}, Akk={Akk.shape}')
