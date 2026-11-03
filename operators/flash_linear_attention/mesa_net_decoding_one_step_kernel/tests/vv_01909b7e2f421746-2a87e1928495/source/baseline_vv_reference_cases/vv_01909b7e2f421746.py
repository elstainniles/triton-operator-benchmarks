
from opbench.fla_fixtures import load_fixture
from yb_175c0a23d94883 import *

def test_reference(load_dir, reference_path):
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_mesa_net_decoding_one_step_kernel/test_mesa_net_decoding_one_step_kernel.pt'
    try:
        data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file {ptfile_path} failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    mesa_net_decoding_one_step_kernel[data['grid']](**input_data)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'o': input_data['o'], 'curr_h_kk': input_data['curr_h_kk'], 'curr_h_kv': input_data['curr_h_kv']}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
