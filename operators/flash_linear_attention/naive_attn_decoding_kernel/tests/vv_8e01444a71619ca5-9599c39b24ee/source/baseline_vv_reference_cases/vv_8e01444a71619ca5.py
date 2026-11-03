
from opbench.fla_fixtures import load_fixture
from yb_0f209d06292290 import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, 'test_naive_attn_decoding_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o = attn_decoding_one_step(input_data['q'], input_data['k'], input_data['v'], g=input_data['g'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'])
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'o': o}
    from baseline_reference_checks import compare_original
    compare_original(values, reference, compare_data_precision)
