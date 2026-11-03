
from opbench.fla_fixtures import load_fixture
import os
os.environ['FLA_USE_FAST_OPS'] = '0'
import torch
import torch_npu
from yb_93f71b12d34293 import chunk_dplr_bwd_dqk_intra

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

def test_chunk_dplr_bwd_kernel_intra_tensorcore():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_bwd_kernel_intra_tensorcore/'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    a = inputs_dict['a'].to(device)
    b = inputs_dict['b'].to(device)
    gi = inputs_dict['gi'].to(device)
    ge = inputs_dict['ge'].to(device)
    dAqk = inputs_dict['dAqk'].to(device)
    dAqb = inputs_dict['dAqb'].to(device)
    dAak = inputs_dict['dAak'].to(device)
    dAab = inputs_dict['dAab'].to(device)
    dqg = inputs_dict['dqg'].to(device)
    dkg = inputs_dict['dkg'].to(device)
    dag = inputs_dict['dag'].to(device)
    dbg = inputs_dict['dbg'].to(device)
    dgk_last = inputs_dict['dgk_last'].to(device)
    scale = inputs_dict['scale']
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    safe_gate = inputs_dict['safe_gate']
    chunk_indices = inputs_dict['chunk_indices']
    dq, dk, da, db, dgk_output = chunk_dplr_bwd_dqk_intra(q=q, k=k, a=a, b=b, gi=gi, ge=ge, dAqk=dAqk, dAqb=dAqb, dAak=dAak, dAab=dAab, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg, dgk_last=dgk_last, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, safe_gate=safe_gate, chunk_indices=chunk_indices)
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'), map_location='cpu', weights_only=False)
    dq_gpu_fp64 = outputs_dict['dq_fp64'].to(device)
    dk_gpu_fp64 = outputs_dict['dk_fp64'].to(device)
    da_gpu_fp64 = outputs_dict['da_fp64'].to(device)
    db_gpu_fp64 = outputs_dict['db_fp64'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    da_gpu = outputs_dict['da'].to(device)
    db_gpu = outputs_dict['db'].to(device)
    print('\nComparing dq with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq, dq_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dq 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dq 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dq 均方根误差比 check failed'
    print('\nComparing dk with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk, dk_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dk 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dk 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dk 均方根误差比 check failed'
    print('\nComparing da with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(da, da_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(da_gpu, da_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'da 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'da 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'da 均方根误差比 check failed'
    print('\nComparing db with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(db, db_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(db_gpu, db_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'db 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'db 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'db 均方根误差比 check failed'
    print('Test chunk_dplr_bwd_kernel_intra_tensorcore passed!')
