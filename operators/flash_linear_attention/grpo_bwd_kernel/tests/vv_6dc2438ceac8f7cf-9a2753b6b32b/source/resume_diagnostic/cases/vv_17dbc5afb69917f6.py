
from opbench.fla_fixtures import load_fixture
from yb_5e1590de33c45a import *

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    grpo_bwd_kernel[data['grid']](**input_data)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; original GPU-error-ratio acceptance unavailable; no accuracy or timing success claimed')
