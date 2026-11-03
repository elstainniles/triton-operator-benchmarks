
from opbench.fla_fixtures import load_fixture
from yb_eeed3605e3bd12 import *

def launch_only(load_dir, reference_path):
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    log2_kernel[data['grid']](**input_data)
    return ({'output_ptr': input_data['output_ptr']}, reference)

def test_cpu64(load_dir, reference_path):
    from cpu64_absolute import compare_named
    values, reference = launch_only(load_dir, reference_path)
    compare_named(torch, values, reference)
