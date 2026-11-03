
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_3e77d2739161a0 import fused_grpo_loss
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
BETA = 0.04
CONFIGS = [{'save_kl': False, 'desc': 'save_kl=False'}, {'save_kl': True, 'desc': 'save_kl=True'}]
kernel_name = 'grpo_fwd_kernel'

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

def load_data(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    logits = inputs['logits'].to(DEVICE)
    logits.requires_grad_(True)
    ref_log_probs = inputs['ref_log_probs'].to(DEVICE)
    input_ids = inputs['input_ids'].to(DEVICE)
    advantages = inputs['advantages'].to(DEVICE)
    completion_mask = inputs['completion_mask'].to(DEVICE)
    return (logits, ref_log_probs, input_ids, advantages, completion_mask)

@pytest.mark.functiontest
def test_grpo_fwd_kernel_functiontest(load_dir):
    for cfg in CONFIGS:
        save_kl, desc = (cfg['save_kl'], cfg['desc'])
        logits, ref_logp, input_ids, advantages, completion_mask = load_data(load_dir)
        result = fused_grpo_loss(logits, ref_logp, input_ids, advantages, beta=BETA, completion_mask=completion_mask, save_kl=save_kl, inplace=False)
        if save_kl:
            loss, kl = result
        else:
            loss = result
        expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
        expected_loss = expected_outputs['loss'].to(DEVICE)
        validate_cmp('bfloat16', loss, expected_loss)
        print(f'Accuracy check passed for grpo_fwd (save_kl={save_kl})')
    print(f'[{kernel_name}] Precision check completed successfully')
