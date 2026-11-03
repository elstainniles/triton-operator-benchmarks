
from opbench.fla_fixtures import load_fixture
from yb_e7a3f4d28ef6ad import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o, final_state = fused_recurrent_iplr_delta_rule(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], scale=input_data['scale'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], cu_seqlens=input_data['cu_seqlens'])
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'o': o, 'final_state': final_state}
    from baseline_reference_checks import compare_original
    compare_original(values, reference, compare_data_precision)
