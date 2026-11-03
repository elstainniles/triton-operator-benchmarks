
from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import triton
from typing import Optional
from yb_8b45f3294e9cfe import chunk_kda_bwd_kernel_intra
from yb_74f55558e08fc1 import IS_GATHER_SUPPORTED

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

def get_test_config():
    """
    Generate test configuration with large shapes for performance testing
    """
    config = {'B': 2, 'T': 500, 'H': 3, 'HV': 3, 'D': 60, 'scale': 1.0, 'gate_logit_normalizer': 1.0, 'mask_p': 0.0, 'use_qk_l2norm_in_kernel': True, 'use_gate_in_kernel': False, 'dtype': torch.float16, 'device': 'npu', 'safe_gate': False, 'disable_recompute': False, 'chunk_size': 64}
    return config

def load_inputs(config):
    """
    Load input tensors from saved pt file
    """
    device = config['device']
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_kda_bwd_kernel_intra_fp16_v2'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    g = inputs_dict['g'].to(device)
    beta = inputs_dict['beta'].to(device)
    dAqk = inputs_dict['dAqk'].to(device)
    dAkk = inputs_dict['dAkk'].to(device)
    dq = inputs_dict['dq'].to(device)
    dk = inputs_dict['dk'].to(device)
    db = inputs_dict['db'].to(device)
    dg = inputs_dict['dg'].to(device)
    chunk_size = inputs_dict['chunk_size']
    safe_gate = inputs_dict['safe_gate']
    print(f'Inputs loaded from {load_dir}/inputs.pt')
    return (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size, safe_gate)

def test_kernel():
    """
    Test chunk_kda_bwd_kernel_intra kernel with profiling
    """
    config = get_test_config()
    print(f'Test config: {config}')
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size, safe_gate = load_inputs(config)
    B, T, H, K, HV = (*k.shape, g.shape[2])
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float32)
    dg2 = torch.empty_like(dg, dtype=torch.float32)
    grid = (NK * NC, NT, B * HV)
    chunk_kda_bwd_kernel_intra[grid](q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2, cu_seqlens=None, chunk_indices=None, B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK, NC=NC, SAFE_GATE=safe_gate, USE_GATHER=IS_GATHER_SUPPORTED)
    torch.npu.synchronize()
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_kda_bwd_kernel_intra_fp16_v2'
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    expected_dq2 = expected_outputs['dq2'].to(config['device'])
    expected_dk2 = expected_outputs['dk2'].to(config['device'])
    expected_db2 = expected_outputs['db2'].to(config['device'])
    expected_dg2 = expected_outputs['dg2'].to(config['device'])
    print('Comparing dq2 outputs...')
    validate_cmp('float32', dq2, expected_dq2)
    print('Comparing dk2 outputs...')
    validate_cmp('float32', dk2, expected_dk2)
    print('Comparing db2 outputs...')
    validate_cmp('float32', db2, expected_db2)
    print('Comparing dg2 outputs...')
    validate_cmp('float32', dg2, expected_dg2)
    print('All outputs match within tolerance!')

    def kernel_call():
        grid = (NK * NC, NT, B * HV)
        chunk_kda_bwd_kernel_intra[grid](q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2, cu_seqlens=None, chunk_indices=None, B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK, NC=NC, SAFE_GATE=safe_gate, USE_GATHER=IS_GATHER_SUPPORTED)
        torch.npu.synchronize()
    print('Running warmup...')
    for _ in range(5):
        kernel_call()
    print('Running profiling...')
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result_dir_chunk_kda_bwd_kernel_intra')) as prof:
        prof.start()
        for i in range(35):
            kernel_call()
            prof.step()
        prof.stop()
    print('Profiling completed. Results saved to ./result_dir_chunk_kda_bwd_kernel_intra')
    print(f'Output shapes: dq2={dq2.shape}, dk2={dk2.shape}, db2={db2.shape}, dg2={dg2.shape}')
