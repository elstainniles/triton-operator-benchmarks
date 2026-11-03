
from opbench.fla_fixtures import load_fixture
from yb_ecb4498f323529 import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_bwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dq, dk, dv, da, db, dh0 = fused_recurrent_iplr_delta_rule_bwd(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], ha=input_data['ha'], do=input_data['do'], dht=input_data['dht'], initial_state=input_data['initial_state'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'])
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'dq': dq, 'dk': dk, 'dv': dv, 'da': da, 'db': db, 'dh0': dh0}
    from baseline_reference_checks import compare_original
    compare_original(values, reference, compare_data_precision)
