
from opbench.fla_fixtures import load_fixture
from yb_5e1590de33c45a import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    grpo_bwd_kernel[data['grid']](**input_data)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'dlogits_written_rows': input_data['dlogits_ptr'][:, :input_data['L'], :]}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
