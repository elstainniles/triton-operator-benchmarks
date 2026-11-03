
from opbench.fla_fixtures import load_fixture
from yb_bbe41b015a64a9 import *

def launch_only(load_dir, reference_path):
    import sys
    from dual.autotune_selection import apply
    from triton.runtime.autotuner import Autotuner
    apply(sys.modules[__name__], 'gate_output_correction_fwd_kernel', Autotuner, {'BT': 4, 'num_warps': 4, 'num_ctas': 1, 'num_stages': 3})
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    data = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_fwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    r = input_data['r']
    k = input_data['k']
    r_k = input_data['r_k']
    v = input_data['v']
    o = input_data['o']
    g = input_data['g']
    output = gate_output_correction_fwd_triton(o, r, k, r_k, v, g)
    return ({'output': output}, reference)

def test_cpu64(load_dir, reference_path):
    from cpu64_absolute import compare_named
    values, reference = launch_only(load_dir, reference_path)
    compare_named(torch, values, reference)
