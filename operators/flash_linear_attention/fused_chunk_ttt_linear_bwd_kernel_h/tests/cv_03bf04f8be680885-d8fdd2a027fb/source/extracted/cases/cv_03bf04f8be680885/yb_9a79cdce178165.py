
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_14d6a471f4f932 import fused_chunk_ttt_linear_bwd_kernel_h
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))

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

@pytest.mark.functiontest
def test_fused_chunk_ttt_linear_bwd_kernel_h_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_ttt_linear_bwd_kernel_h'
    try:
        datafp64 = load_fixture(os.path.join(load_dir, 'test_fused_chunk_ttt_linear_bwd_kernel_h_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        golden_output = datafp64['gpu_output']
    except Exception as e:
        pytest.fail(f'load fp64 file failed: {str(e)}')
    try:
        data = load_fixture(os.path.join(load_dir, 'test_fused_chunk_ttt_linear_bwd_kernel_h_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        gpu_fp32_output = data['gpu_output']
        input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    except Exception as e:
        pytest.fail(f'load fp32 file failed: {str(e)}')
    T = data['input_data']['T']
    BT = data['input_data']['BT']
    print(f'T: {T}, BT: {BT}, NT: {(T + BT - 1) // BT}')
    fused_chunk_ttt_linear_bwd_kernel_h[data['grid']](**input_data, inject_barrier_all=True)
    npu_output = {'dq': input_data['dq'].cpu().clone(), 'h': input_data['h'].cpu().clone(), 'v2': input_data['v2'].cpu().clone()}

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

    def cal_precision(input, golden):
        relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
        relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
        rmse = torch.sqrt(torch.mean((input - golden) ** 2))
        mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
        small_val_error = torch.sum(torch.abs(input - golden)[mask])
        return (relative_max.item(), relative_mean.item(), rmse.item(), small_val_error.item())
    print('\n' + '=' * 60)
    print('NPU vs GPU fp64 Validation & Comparison')
    print('=' * 60)
    for key in ['dq', 'h', 'v2']:
        dtype_str = str(npu_output[key].dtype).replace('torch.', '')
        try:
            validate_cmp(dtype_str, npu_output[key].clone(), golden_output[key].clone().float())
            print(f'\n{key}: PASSED (validate_cmp)')
        except (AssertionError, RuntimeError) as e:
            print(f'\n{key}: FAILED validate_cmp - {e}')
            npu_rel_max, npu_rel_mean, npu_rmse, npu_small = cal_precision(npu_output[key].double(), golden_output[key].double())
            print(f'  NPU vs GPU fp64:')
            print(f'    Max relative error:  {npu_rel_max:.6e}')
            print(f'    Mean relative error: {npu_rel_mean:.6e}')
            print(f'    RMSE:                {npu_rmse:.6e}')
            print(f'    Small val abs error: {npu_small:.6e}')
            if gpu_fp32_output is not None:
                gpu_rel_max, gpu_rel_mean, gpu_rmse, gpu_small = cal_precision(gpu_fp32_output[key].double(), golden_output[key].double())
                print(f'  GPU fp32 vs GPU fp64:')
                print(f'    Max relative error:  {gpu_rel_max:.6e}')
                print(f'    Mean relative error: {gpu_rel_mean:.6e}')
                print(f'    RMSE:                {gpu_rmse:.6e}')
                print(f'    Small val abs error: {gpu_small:.6e}')
                print(f'  NPU/GPU fp32 ratio:')
                print(f'    Max relative error:  {npu_rel_max / (gpu_rel_max + 1e-10):.2f}x')
                print(f'    Mean relative error: {npu_rel_mean / (gpu_rel_mean + 1e-10):.2f}x')
                print(f'    RMSE:                {npu_rmse / (gpu_rmse + 1e-10):.2f}x')
                print(f'    Small val abs error: {npu_small / (gpu_small + 1e-10):.2f}x')
                assert npu_rel_max / (gpu_rel_max + 1e-10) < 2, f'{key} Max relative error NPU/GPU fp32 ratio too high'
                assert npu_rel_mean / (gpu_rel_mean + 1e-10) < 1.2, f'{key} Mean relative error NPU/GPU fp32 ratio too high'
                assert npu_rmse / (gpu_rmse + 1e-10) < 1.2, f'{key} RMSE NPU/GPU fp32 ratio too high'
                print(f'\n{key}: PASSED (third_cmp)')
    print('=' * 60 + '\n')
