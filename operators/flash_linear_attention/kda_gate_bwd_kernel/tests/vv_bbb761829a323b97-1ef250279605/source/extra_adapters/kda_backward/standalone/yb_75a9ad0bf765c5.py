
from opbench.fla_fixtures import load_fixture
from typing import Optional
import torch
import torch_npu
import os
from yb_18dc009e5fff3c import kda_gate_bwd

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

def test_kda_gate_bwd_kernel():
    """
    Test kda_gate_bwd_kernel with large shape configuration
    """
    B = 2
    T = 1024
    H = 8
    K = 64
    device = torch.device('npu:0')
    dtype = torch.float16
    input_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/kda_gate_bwd_kernel'
    print(f'Loading inputs from: {input_dir}')
    g = load_fixture(f'{input_dir}/g.pt').to(device)
    A_log = load_fixture(f'{input_dir}/A_log.pt').to(device)
    dt_bias = load_fixture(f'{input_dir}/dt_bias.pt').to(device)
    dyg = load_fixture(f'{input_dir}/dyg.pt').to(device)
    print('\nExecuting kernel...')
    dg, dA, dbias = kda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=None)
    torch.npu.synchronize()
    print('\nLoading expected outputs and comparing...')
    expected_dg = load_fixture(f'{input_dir}/dg.pt').to(device)
    expected_dA = load_fixture(f'{input_dir}/dA.pt').to(device)
    expected_dbias = load_fixture(f'{input_dir}/dbias.pt').to(device) if os.path.exists(f'{input_dir}/dbias.pt') else None
    validate_cmp('float16', dg, expected_dg)
    validate_cmp('float16', dA, expected_dA)
    if expected_dbias is not None and dbias is not None:
        validate_cmp('float16', dbias, expected_dbias)
    print('✓ All outputs matched')
    print('\nWarmup runs...')
    for _ in range(5):
        dg, dA, dbias = kda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=None)
        torch.npu.synchronize()
    print('\nStarting profiling...')
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result_dir_original')) as prof:
        prof.start()
        for i in range(35):
            dg, dA, dbias = kda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=None)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nTest completed successfully!')
    print(f'Input shapes: g={g.shape}, A_log={A_log.shape}, dt_bias={dt_bias.shape}, dyg={dyg.shape}')
    print(f'Output shapes: dg={dg.shape}, dA={dA.shape}, dbias={(dbias.shape if dbias is not None else None)}')
    print(f'Profiling results saved to ./result_dir_original')
