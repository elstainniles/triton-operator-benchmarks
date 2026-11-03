
from opbench.fla_fixtures import load_fixture
from yb_51b53fec3731f3 import *

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    kl_div_kernel[data['grid']](logits=input_data['logits'], target_logits=input_data['target_logits'], loss=input_data['loss'], s_logits=input_data['s_logits'], s_loss=input_data['s_loss'], reduction=input_data['reduction'], N=input_data['N'], V=input_data['V'], BV=input_data['BV'])
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; original GPU-error-ratio acceptance unavailable; no accuracy or timing success claimed')
