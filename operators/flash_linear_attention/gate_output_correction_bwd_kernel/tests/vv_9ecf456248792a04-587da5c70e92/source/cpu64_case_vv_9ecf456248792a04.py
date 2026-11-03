
from opbench.fla_fixtures import load_fixture
from yb_62f460003bf370 import *

def launch_only(load_dir, reference_path):
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    data = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_bwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    r = input_data['r']
    k = input_data['k']
    r_k = input_data['r_k']
    v = input_data['v']
    o = input_data['o']
    grad_output = input_data['grad_output']
    g = input_data['g']
    grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g = gate_output_correction_backward_triton(grad_output, o, r, k, r_k, v, g)
    return (dict(zip(('grad_o', 'grad_r', 'grad_k', 'grad_r_k', 'grad_v', 'grad_g'), (grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g))), reference)

def test_cpu64(load_dir, reference_path):
    from cpu64_absolute import compare_named
    values, reference = launch_only(load_dir, reference_path)
    compare_named(torch, values, reference)
