
from opbench.fla_fixtures import load_fixture
import torch
import torch_npu
from yb_42903005eb2150 import prepare_wy_repr_bwd

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

def test_prepare_wy_repr_bwd_kda_kernel():
    """
    Test prepare_wy_repr_bwd_kda_kernel with large shape configuration
    """
    B = 2
    T = 1024
    H = 8
    HV = 8
    K = 64
    V = 64
    BT = 64
    device = torch.device('npu:0')
    dtype = torch.float16
    import os
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/prepare_wy_repr_bwd_kda_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    gk = inputs_dict['gk'].to(device)
    A = inputs_dict['A'].to(device)
    dk = inputs_dict['dk'].to(device)
    dw = inputs_dict['dw'].to(device)
    du = inputs_dict['du'].to(device)
    dg = inputs_dict['dg'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_indices = inputs_dict['chunk_indices']
    print('\nExecuting kernel...')
    dk_out, dv, db, dg_out, dA = prepare_wy_repr_bwd(k=k, v=v, beta=beta, gk=gk, A=A, dk=dk, dw=dw, du=du, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    outputs_dict_fp64 = load_fixture(os.path.join(load_dir, 'outputs_fp64.pt'))
    dk_out_expected = outputs_dict['dk_out'].to(device)
    dv_expected = outputs_dict['dv'].to(device)
    db_expected = outputs_dict['db'].to(device)
    dg_out_expected = outputs_dict['dg_out'].to(device)
    dA_expected = outputs_dict['dA'].to(device)
    dk_out_expected_fp64 = outputs_dict_fp64['dk_out_fp64'].to(device)
    dv_expected_fp64 = outputs_dict_fp64['dv_fp64'].to(device)
    db_expected_fp64 = outputs_dict_fp64['db_fp64'].to(device)
    dg_out_expected_fp64 = outputs_dict_fp64['dg_out_fp64'].to(device)
    dA_expected_fp64 = outputs_dict_fp64['dA_fp64'].to(device)
    print('\nComparing dk_out outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk_out, dk_out_expected_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_out_expected, dk_out_expected_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dk_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dk_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dk_out 均方根误差比 check failed'
    print('\nComparing dv outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv, dv_expected_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dv_expected, dv_expected_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dv 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dv 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dv 均方根误差比 check failed'
    print('\nComparing db outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(db, db_expected_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(db_expected, db_expected_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'db 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'db 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'db 均方根误差比 check failed'
    print('\nComparing dg_out outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dg_out, dg_out_expected_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dg_out_expected, dg_out_expected_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dg_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dg_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dg_out 均方根误差比 check failed'
    print('\nComparing dA outputs...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dA, dA_expected_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dA_expected, dA_expected_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dA 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dA 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dA 均方根误差比 check failed'
    print('Output comparison passed!')
    print('\nWarmup runs...')
    for _ in range(5):
        _, _, _, _, _ = prepare_wy_repr_bwd(k=k, v=v, beta=beta, gk=gk, A=A, dk=dk, dw=dw, du=du, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
        torch.npu.synchronize()
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result_dir_original')) as prof:
        prof.start()
        for i in range(35):
            _, _, _, _, _ = prepare_wy_repr_bwd(k=k, v=v, beta=beta, gk=gk, A=A, dk=dk, dw=dw, du=du, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'Test completed successfully!')
    print(f'Input shapes: k={k.shape}, v={v.shape}, beta={beta.shape}, gk={gk.shape}, A={A.shape}')
    print(f'             dk={dk.shape}, dw={dw.shape}, du={du.shape}, dg={dg.shape}')
    print(f'Output shapes: dk={dk_out.shape}, dv={dv.shape}, db={db.shape}, dg={dg_out.shape}, dA={dA.shape}')
    print(f'Profiling results saved to ./result_dir_original')
