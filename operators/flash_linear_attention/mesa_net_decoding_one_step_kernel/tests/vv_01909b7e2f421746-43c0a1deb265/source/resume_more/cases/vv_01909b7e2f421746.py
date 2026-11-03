
from opbench.fla_fixtures import load_fixture
from yb_175c0a23d94883 import *

def test_diagnostic(load_dir):
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_mesa_net_decoding_one_step_kernel/test_mesa_net_decoding_one_step_kernel.pt'
    try:
        data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file {ptfile_path} failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    mesa_net_decoding_one_step_kernel[data['grid']](**input_data)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; independent CPU reference retained; no accuracy or timing success claimed')
