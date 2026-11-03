
from opbench.fla_fixtures import load_fixture
from yb_51b53fec3731f3 import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    kl_div_kernel[data['grid']](logits=input_data['logits'], target_logits=input_data['target_logits'], loss=input_data['loss'], s_logits=input_data['s_logits'], s_loss=input_data['s_loss'], reduction=input_data['reduction'], N=input_data['N'], V=input_data['V'], BV=input_data['BV'])
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'logits_gradient': input_data['logits'], 'loss': input_data['loss']}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
