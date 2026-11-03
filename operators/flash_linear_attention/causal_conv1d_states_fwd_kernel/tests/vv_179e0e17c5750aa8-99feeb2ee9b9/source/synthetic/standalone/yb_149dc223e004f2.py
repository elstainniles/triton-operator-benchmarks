
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'causal_conv1d_states_fwd_kernel'

def _get_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

@pytest.mark.functiontest
def test_causal_conv1d_states_fwd_kernel_functiontest(load_dir):
    from yb_717c6a81dac3a7 import causal_conv1d_update_states
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to(DEVICE)
    initial_state = inputs['initial_state'].to(DEVICE)
    state_len = _get_value(inputs['state_len'])
    final_state = causal_conv1d_update_states(x, state_len=state_len, initial_state=initial_state)
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = x.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(final_state, expected_output, atol=atol, rtol=rtol)
    print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
