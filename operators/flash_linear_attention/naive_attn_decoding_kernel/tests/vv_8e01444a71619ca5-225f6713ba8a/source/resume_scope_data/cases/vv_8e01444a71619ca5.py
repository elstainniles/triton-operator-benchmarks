
from opbench.fla_fixtures import load_fixture
from resume_precision_contract import compare_raw
from yb_0f209d06292290 import *

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_naive_attn_decoding_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o = attn_decoding_one_step(input_data['q'], input_data['k'], input_data['v'], g=input_data['g'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'])
    reference = load_fixture(os.path.join(load_dir, 'reference_cpu64.pt'), map_location='cpu', weights_only=False)
    compare_raw('vv_8e01444a71619ca5', reference, {'o': o}, compare_data_precision)
